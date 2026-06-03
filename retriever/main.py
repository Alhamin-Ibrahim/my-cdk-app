import os
import json
import boto3
from fastapi import FastAPI
from pydantic import BaseModel
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from aws_xray_sdk.core import xray_recorder, patch_all
from aws_xray_sdk.core import xray_recorder

# Configure X-Ray to not throw errors when no segment is open
xray_recorder.configure(context_missing="LOG_ERROR")
patch_all()

app = FastAPI(title="Agent Retriever")

OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "")
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
INDEX_NAME = os.environ.get("INDEX_NAME", "documents")

bedrock_runtime = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
credentials = boto3.Session().get_credentials()

# AWS4Auth handles signing requests to OpenSearch with AWS credentials
auth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    BEDROCK_REGION,
    "aoss",
    session_token=credentials.token,
)

opensearch_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://", ""), "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    pool_maxsize=20,
)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    # Step 1: Embed the query
    with xray_recorder.capture("titan-embed"):
        embed_response = bedrock_runtime.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": request.query}),
            contentType="application/json",
            accept="application/json",
        )
        embedding = json.loads(embed_response["body"].read())["embedding"]

    # Step 2: k-NN search in OpenSearch
    with xray_recorder.capture("opensearch-knn"):
        search_body = {
            "size": request.top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": request.top_k,
                    }
                }
            },
            "_source": ["text", "source", "metadata"],
        }

        response = opensearch_client.search(
            index=INDEX_NAME,
            body=search_body,
        )

    hits = response["hits"]["hits"]
    chunks = [hit["_source"].get("text", "") for hit in hits]
    sources = list({hit["_source"].get("source", "") for hit in hits})

    return {"chunks": chunks, "sources": sources, "count": len(chunks)}