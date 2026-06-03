from __future__ import annotations

import logging
import os
from typing import Any

import boto3

from orchestrator.state import AgentState

logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "eu-west-1")
MODEL_ID = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
MAX_TOKENS = 1024


def build_prompt(
    query: str,
    chunks: list[dict[str, Any]],
    history_text: str,
) -> str:
    # Construct the grounded RAG prompt.
    context_block = "\n\n".join(
        f"[{i+1}] (Source: {c['source']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    history_block = f"\n\nConversation so far:\n{history_text}" if history_text else ""

    return f"""You are a helpful assistant that answers questions based on provided document context.
{history_block}

Context documents:
{context_block}

Question: {query}

Instructions:
- Answer based ONLY on the context above. Do not use outside knowledge.
- If the answer is not in the context, say "I don't have enough information in the provided documents to answer this."
- At the end of your answer, list the source documents you used as: Sources: [1], [2], etc.
- Be concise and direct.

Answer:"""


def parse_sources(answer: str, chunks: list[dict[str, Any]]) -> list[str]:
    import re

    # Find all citation numbers like [1], [2], [3]
    citation_numbers = re.findall(r"\[(\d+)\]", answer.split("Sources:")[-1])

    sources = []
    for num_str in citation_numbers:
        idx = int(num_str) - 1   # convert to 0-indexed
        if 0 <= idx < len(chunks):
            source = chunks[idx]["source"]
            if source not in sources:
                sources.append(source)

    return sources


def call_haiku(prompt: str) -> str:
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)

    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[
            {"role": "user", "content": [{"text": prompt}]}
        ],
        inferenceConfig={
            "maxTokens": MAX_TOKENS,
            "temperature": 0.1, 
        },
    )

    return response["output"]["message"]["content"][0]["text"]


def generator_node(state: AgentState) -> dict:
    query = state["query"]
    chunks = state.get("chunks") or []
    history = state.get("history") or []

    # Format history for the prompt
    from orchestrator.memory import ConversationMemory
    history_text = ConversationMemory.format_for_prompt(None, history)

    logger.info(
        "Generator: calling Haiku with %d chunks, %d history turns",
        len(chunks), len(history)
    )

    try:
        prompt = build_prompt(query, chunks, history_text)
        answer = call_haiku(prompt)
        sources = parse_sources(answer, chunks)

        logger.info("Generator: answer generated, %d sources cited", len(sources))
        return {"answer": answer, "sources": sources}

    except Exception as exc:  # noqa: BLE001
        logger.exception("Generator node failed")
        return {
            "answer": "Sorry, I encountered an error generating the answer.",
            "sources": [],
            "error": f"Generation failed: {exc}",
        }


def direct_answer_node(state: AgentState) -> dict:
    query = state["query"]
    history = state.get("history") or []

    from orchestrator.memory import ConversationMemory
    history_text = ConversationMemory.format_for_prompt(None, history)

    general_prompt = f"""You are a helpful assistant with access to a document retrieval system.
{f"Conversation so far:{chr(10)}{history_text}{chr(10)}" if history_text else ""}
Question: {query}

Answer based on the conversation history above if relevant. Be helpful and concise."""

    try:
        answer = call_haiku(general_prompt)
        return {"answer": answer, "sources": []}
    except Exception as exc:
        logger.exception("Direct answer node failed")
        return {"answer": "Sorry, I couldn't generate an answer.", "sources": [], "error": str(exc)}