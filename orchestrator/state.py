from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Complete state passed between every node in the LangGraph graph.

    Fields
    ------
    session_id  UUID for this conversation. Persists across turns.
    query       The raw user message for this turn.
    intent      Routing decision: "retrieve" or "direct".
    chunks      Top-N document chunks returned by the retriever service.
    answer      The final generated answer string.
    sources     Document names cited in the answer.
    history     Last N turns loaded from DynamoDB.
                Annotated with operator.add so parallel nodes can append
                without overwriting each other.
    error       Non-fatal error message — lets us degrade gracefully.
    """

    session_id: str
    query: str
    intent: Optional[str]
    chunks: Optional[list[Any]]
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
