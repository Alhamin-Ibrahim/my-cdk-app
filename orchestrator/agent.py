from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Literal

import boto3
import httpx
from langgraph.graph import END, StateGraph

from memory import ConversationMemory
from state import AgentState, initial_state

logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "eu-west-1")
MODEL_ID = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"

# Endpoint of the retriever service (injected by ECS via environment variable)
RETRIEVER_ENDPOINT = os.environ.get("RETRIEVER_ENDPOINT", "http://localhost:8080")

# Queries that are clearly conversational — skip the retriever entirely
GENERAL_PATTERNS = [
    "what is today",
    "what time is it",
    "who are you",
    "hello",
    "hi ",
    "thanks",
    "thank you",
    "help",
    "what can you do",
]


# Intent classification
def classify_intent_fast(query: str) -> str | None:
    q_lower = query.lower().strip()
    for pattern in GENERAL_PATTERNS:
        if pattern in q_lower:
            return "direct"
    return None


def classify_intent_llm(query: str, history_text: str) -> str:
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)

    prompt = f"""Classify this user query. Respond ONLY with JSON:
{{"intent": "retrieve", "reason": "one sentence"}}
or
{{"intent": "direct", "reason": "one sentence"}}

Use "retrieve" if the query asks about specific content in technical documents.
Use "direct" for greetings, meta questions, or things answerable without documents.

{f"Recent conversation:{chr(10)}{history_text}{chr(10)}" if history_text else ""}
User query: {query}

JSON response:"""

    try:
        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 100, "temperature": 0.0},
        )
        raw = response["output"]["message"]["content"][0]["text"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        intent = parsed.get("intent", "retrieve")
        logger.info("Intent: %s — %s", intent, parsed.get("reason"))
        return intent if intent in ("retrieve", "direct") else "retrieve"

    except Exception as exc:
        logger.warning("Intent classification failed (%s), defaulting to 'retrieve'", exc)
        return "retrieve"


# Graph nodes
def orchestrator_node(state: AgentState) -> dict:
    session_id = state["session_id"]
    query = state["query"]

    memory = ConversationMemory(region=REGION)
    history = memory.load_history(session_id)

    # Persist the new user query immediately — ensures it's in the history for intent classification
    memory.save_turn(session_id, "user", query)

    history_text = memory.format_for_prompt(history)
    intent = classify_intent_fast(query) or classify_intent_llm(query, history_text)

    logger.info(
        "Orchestrator: session=%s intent=%s history_turns=%d",
        session_id, intent, len(history),
    )

    return {"history": history, "intent": intent}


def retriever_node(state: AgentState) -> dict:
    """
    Calls the retriever HTTP service to fetch relevant document chunks.
    """
    query = state["query"]
    logger.info("Retriever node: calling retriever service for '%s'", query[:80])

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                f"{RETRIEVER_ENDPOINT}/retrieve",
                json={"query": query, "top_k": 5},
            )
            resp.raise_for_status()
            result = resp.json()

        chunks = result.get("chunks", [])
        logger.info("Retriever node: received %d chunks", len(chunks))
        return {"chunks": chunks}

    except httpx.ConnectError:
        logger.error("Retriever service unreachable at %s", RETRIEVER_ENDPOINT)
        return {"chunks": [], "error": "Retriever service unavailable."}
    except Exception as exc:
        logger.exception("Retriever node failed")
        return {"chunks": [], "error": f"Retrieval failed: {exc}"}


def generator_node(state: AgentState) -> dict:
    query = state["query"]
    chunks = state.get("chunks") or []
    history = state.get("history") or []

    history_text = ConversationMemory.format_for_prompt_static(history)

    context_block = "\n\n".join(
        f"[{i+1}] (Source: {c.get('source', 'unknown')})\n{c.get('text', c) if isinstance(c, dict) else c}"
        for i, c in enumerate(chunks)
    )

    history_block = f"\n\nConversation so far:\n{history_text}" if history_text else ""

    prompt = f"""You are a helpful assistant that answers questions based on provided document context.
{history_block}

Context documents:
{context_block}

Question: {query}

Instructions:
- Answer based ONLY on the context above. Do not use outside knowledge.
- If the answer is not in the context, say "I don't have enough information in the provided documents to answer this."
- At the end, list the source documents used as: Sources: [1], [2], etc.
- Be concise and direct.

Answer:"""

    logger.info("Generator: calling Haiku with %d chunks, %d history turns", len(chunks), len(history))

    try:
        bedrock = boto3.client("bedrock-runtime", region_name=REGION)
        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.1},
        )
        answer = response["output"]["message"]["content"][0]["text"]
        sources = _parse_sources(answer, chunks)

        logger.info("Generator: answer generated, %d sources cited", len(sources))
        return {"answer": answer, "sources": sources}

    except Exception as exc:
        logger.exception("Generator node failed")
        return {
            "answer": "Sorry, I encountered an error generating the answer.",
            "sources": [],
            "error": f"Generation failed: {exc}",
        }


def direct_answer_node(state: AgentState) -> dict:
    query = state["query"]
    history = state.get("history") or []
    history_text = ConversationMemory.format_for_prompt_static(history)

    prompt = f"""You are a helpful assistant with access to a document retrieval system.
{f"Conversation so far:{chr(10)}{history_text}{chr(10)}" if history_text else ""}
Question: {query}

Answer based on the conversation history if relevant. Be helpful and concise."""

    try:
        bedrock = boto3.client("bedrock-runtime", region_name=REGION)
        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.3},
        )
        answer = response["output"]["message"]["content"][0]["text"]
        return {"answer": answer, "sources": []}
    except Exception as exc:
        logger.exception("Direct answer node failed")
        return {"answer": "Sorry, I couldn't generate an answer.", "sources": [], "error": str(exc)}


def save_result_node(state: AgentState) -> dict:
    session_id = state["session_id"]
    answer = state.get("answer", "")

    if answer:
        memory = ConversationMemory(region=REGION)
        memory.save_turn(session_id, "assistant", answer)

    return {}


# Routing
def route_by_intent(state: AgentState) -> Literal["retrieve", "direct"]:
    return state.get("intent", "retrieve")


# Graph construction
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("retriever", retriever_node)      # HTTP call to retriever service
    graph.add_node("generator", generator_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("save_result", save_result_node)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        source="orchestrator",
        path=route_by_intent,
        path_map={"retrieve": "retriever", "direct": "direct_answer"},
    )

    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "save_result")
    graph.add_edge("direct_answer", "save_result")
    graph.add_edge("save_result", END)

    return graph.compile()


# Singleton — avoids recompiling on every request
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str, session_id: str | None = None) -> dict:
    """
    Run a user query through the full agent graph.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info("New session: %s", session_id)

    state = initial_state(session_id=session_id, query=query)
    final_state = get_graph().invoke(state)

    return {
        "answer": final_state.get("answer", "No answer generated."),
        "sources": final_state.get("sources", []),
        "session_id": session_id,
        "intent": final_state.get("intent"),
        "error": final_state.get("error"),
    }


def _parse_sources(answer: str, chunks: list) -> list[str]:
    import re

    citation_numbers = re.findall(r"\[(\d+)\]", answer.split("Sources:")[-1])
    sources = []
    for num_str in citation_numbers:
        idx = int(num_str) - 1
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            source = chunk.get("source", "unknown") if isinstance(chunk, dict) else "unknown"
            if source not in sources:
                sources.append(source)
    return sources
