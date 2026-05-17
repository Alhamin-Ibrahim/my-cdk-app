from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    The complete state object passed between every node in the graph.

    Fields:
        session_id:     UUID for this conversation session. Persists across turns.
        query:          The raw user message for the current turn.
        intent:         What the orchestrator decided: "retrieve" or "direct".
        chunks:         Top-5 document chunks returned by the retriever.
        answer:         The final generated answer string.
        sources:        Document names cited in the answer.
        history:        Last N conversation turns loaded from DynamoDB.
                        Annotated with operator.add so parallel nodes can
                        append to it without overwriting each other.
        error:          Any error message — lets us handle failures gracefully
                        without crashing the whole graph.
    """

    session_id: str
    query: str
    intent: Optional[str]                          
    chunks: Optional[list[dict[str, Any]]]    
    answer: Optional[str]
    sources: Optional[list[str]]
    history: Annotated[list[dict[str, str]], operator.add]
    error: Optional[str]


def initial_state(session_id: str, query: str) -> AgentState:
    return AgentState(
        session_id=session_id,
        query=query,
        intent=None,
        chunks=None,
        answer=None,
        sources=None,
        history=[],
        error=None,
    )