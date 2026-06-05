"""
Orchestrator FastAPI service.

Entry point for ECS. All RAG logic lives in agent.py — this file is purely
the HTTP layer. It delegates to run_query() which drives the LangGraph graph.
"""
import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from agent import run_query

# X-Ray tracing — must be configured before any AWS SDK clients are created
from aws_xray_sdk.core import xray_recorder, patch_all

xray_recorder.configure(context_missing="LOG_ERROR")
patch_all()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

app = FastAPI(title="Agent Orchestrator", version="1.0.0")


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str]
    intent: Optional[str] = None


@app.get("/health")
async def health_check():
    return {"status": "ok", "environment": ENVIRONMENT}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Run a RAG query through the full agent graph.

    Session IDs are UUIDs. Omit session_id to start a new conversation;
    pass the returned session_id on follow-up turns for multi-turn memory.
    """
    try:
        result = run_query(query=request.query, session_id=request.session_id)
    except Exception as exc:
        logger.exception("run_query failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return QueryResponse(
        answer=result["answer"],
        session_id=result["session_id"],
        sources=result.get("sources", []),
        intent=result.get("intent"),
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

