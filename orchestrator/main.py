import os
import json
import httpx
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from aws_xray_sdk.core import xray_recorder, patch_all

# Configure X-Ray to not throw errors when no segment is open
patch_all()

app = FastAPI(title="Agent Orchestrator")

# Use environment variables for configuration
RETRIEVER_ENDPOINT = os.environ.get("RETRIEVER_ENDPOINT", "http://retriever:8080")
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

bedrock_runtime = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
dynamodb = boto3.resource("dynamodb", region_name=BEDROCK_REGION)


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str]


# health check endpoint for ALB target group
@app.get("/health")
async def health_check():
    """ALB target group health check — must return 200."""
    return {"status": "ok", "environment": ENVIRONMENT}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    import uuid
    session_id = request.session_id or str(uuid.uuid4())

    # Step 1: Retrieve context from the retriever service
    with xray_recorder.capture("retriever-call"):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                retrieve_resp = await client.post(
                    f"{RETRIEVER_ENDPOINT}/retrieve",
                    json={"query": request.query, "top_k": 5},
                )
                retrieve_resp.raise_for_status()
                retrieval_result = retrieve_resp.json()
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Retriever service unavailable. Is it running?",
            )

    context_chunks = retrieval_result.get("chunks", [])
    sources = retrieval_result.get("sources", [])
    context_text = "\n\n".join(context_chunks)

    # Step 2: Build prompt and call Bedrock
    prompt = f"""You are a helpful assistant. Use only the following context to answer the question.
If the context doesn't contain the answer, say so.

Context:
{context_text}

Question: {request.query}

Answer:"""

    with xray_recorder.capture("bedrock-invoke"):
        bedrock_response = bedrock_runtime.invoke_model(
            modelId="eu.anthropic.claude-haiku-4-5-20251001-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }),
            contentType="application/json",
            accept="application/json",
        )

    response_body = json.loads(bedrock_response["body"].read())
    answer = response_body["content"][0]["text"]

    # Step 3: Store in DynamoDB (if table configured and exists)
    if DYNAMODB_TABLE and DYNAMODB_TABLE != "REPLACE_WITH_YOUR_TABLE_NAME":
        try:
            with xray_recorder.capture("dynamodb-put"):
                table = dynamodb.Table(DYNAMODB_TABLE)
                table.put_item(Item={
                    "session_id": session_id,
                    "timestamp": str(__import__("time").time()),
                    "query": request.query,
                    "answer": answer,
                })
        except Exception as e:
            # Non-fatal: log and continue — answer is still returned
            import logging
            logging.getLogger(__name__).warning(f"DynamoDB write failed (non-fatal): {e}")

    return QueryResponse(answer=answer, session_id=session_id, sources=sources)