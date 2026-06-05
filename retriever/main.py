import logging

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from aws_xray_sdk.core import xray_recorder, patch_all

xray_recorder.configure(context_missing="LOG_ERROR")
patch_all()

from agent import retrieve  # noqa: E402 — must come after patch_all

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Agent Retriever", version="1.0.0")


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
async def health_check():
    """ALB health check — must return HTTP 200."""
    return {"status": "ok"}


@app.post("/retrieve")
async def retrieve_endpoint(request: RetrieveRequest):
    """
    Embed the query, run kNN search against OpenSearch, apply BM25 re-ranking,
    and return the top-N chunks with source metadata.
    """
    result = retrieve(query=request.query)
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
