from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Literal

import boto3
from langgraph.graph import StateGraph, END

from orchestrator.state import AgentState, initial_state
from orchestrator.memory import ConversationMemory
from retriever.agent import retriever_node
from generator.agent import generator_node, direct_answer_node

logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "eu-west-1")
MODEL_ID = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"

# Queries that are clearly general (no document retrieval needed).
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


def classify_intent_fast(query: str) -> str | None:
    q_lower = query.lower().strip()
    for pattern in GENERAL_PATTERNS:
        if pattern in q_lower:
            return "direct"
    return None


def classify_intent_llm(query: str, history_text: str) -> str:
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)

    classification_prompt = f"""Classify this user query. Respond ONLY with JSON in this exact format:
{{"intent": "retrieve", "reason": "one sentence"}}
or
{{"intent": "direct", "reason": "one sentence"}}

Use "retrieve" if the query is asking about specific content, facts, or information
that would be found in technical documents (e.g. "How does X work?", "What is Y?",
"Show me the steps for Z?").

Use "direct" if it is a greeting, meta question about the assistant, or a request
that can be answered without consulting any documents.

{f"Recent conversation:{chr(10)}{history_text}{chr(10)}" if history_text else ""}
User query: {query}

JSON response:"""

    try:
        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": classification_prompt}]}],
            inferenceConfig={"maxTokens": 100, "temperature": 0.0},
        )
        raw = response["output"]["message"]["content"][0]["text"].strip()
        # Strip markdown code fences if present
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        intent = parsed.get("intent", "retrieve")
        logger.info("Intent classification: %s (reason: %s)", intent, parsed.get("reason"))
        return intent if intent in ("retrieve", "direct") else "retrieve"

    except Exception as exc:  # noqa: BLE001
        logger.warning("Intent classification failed (%s), defaulting to 'retrieve'", exc)
        return "retrieve"

def orchestrator_node(state: AgentState) -> dict:
    session_id = state["session_id"]
    query = state["query"]

    # Load conversation history
    memory = ConversationMemory(region=REGION)
    history = memory.load_history(session_id)

    # Persist the user's message now so it's in DynamoDB even if we crash later
    memory.save_turn(session_id, "user", query)

    # Format history for the intent classifier
    history_text = memory.format_for_prompt(history)

    # Fast path first (free), then LLM path
    intent = classify_intent_fast(query) or classify_intent_llm(query, history_text)

    logger.info(
        "Orchestrator: session=%s intent=%s history_turns=%d",
        session_id, intent, len(history)
    )

    return {"history": history, "intent": intent}


def save_result_node(state: AgentState) -> dict:
    session_id = state["session_id"]
    answer = state.get("answer", "")

    if answer:
        memory = ConversationMemory(region=REGION)
        memory.save_turn(session_id, "assistant", answer)

    return {}  


def route_by_intent(state: AgentState) -> Literal["retrieve", "direct"]:
    return state.get("intent", "retrieve")


# ── Build the graph ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("save_result", save_result_node)

    # Entry point
    graph.set_entry_point("orchestrator")

    # Conditional branch: orchestrator → retriever OR direct_answer
    graph.add_conditional_edges(
        source="orchestrator",
        path=route_by_intent,
        path_map={
            "retrieve": "retriever",
            "direct": "direct_answer",
        },
    )

    # Unconditional edges
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "save_result")
    graph.add_edge("direct_answer", "save_result")
    graph.add_edge("save_result", END)

    return graph.compile()


# Singleton compiled graph (avoids recompiling on every request)
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

# public api for running a query through the agent
def run_query(query: str, session_id: str | None = None) -> dict:
    """
    Run a user query through the full agent graph.

    Args:
        query:      The user's question.
        session_id: Existing session UUID for multi-turn, or None for new session.

    Returns:
        dict with keys: answer, sources, session_id, intent
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info("New session started: %s", session_id)

    state = initial_state(session_id=session_id, query=query)
    graph = get_graph()

    final_state = graph.invoke(state)

    return {
        "answer": final_state.get("answer", "No answer generated."),
        "sources": final_state.get("sources", []),
        "session_id": session_id,
        "intent": final_state.get("intent"),
        "error": final_state.get("error"),
    }