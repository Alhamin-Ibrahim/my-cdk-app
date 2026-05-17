from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

from orchestrator.state import AgentState

logger = logging.getLogger(__name__)

# set the same as ecs task env vars
OPENSEARCH_ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]   # e.g. https://abc.eu-west-1.aoss.amazonaws.com
INDEX_NAME = os.environ.get("OPENSEARCH_INDEX", "documents")
TOP_K = 10          # fetch 10 from kNN, then re-rank to top 5
FINAL_TOP_K = 5
REGION = os.environ.get("AWS_REGION", "eu-west-1")


def _get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=REGION)


def _get_opensearch_client() -> OpenSearch:
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, REGION, "aoss")

    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://", ""), "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=10,
    )


def embed_query(query: str) -> list[float]:
    bedrock = _get_bedrock_client()

    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": query}),
    )

    body = json.loads(response["body"].read())
    return body["embedding"] 


def knn_search(client: OpenSearch, query_vector: list[float]) -> list[dict[str, Any]]:
    query_body = {
        "size": TOP_K,
        "query": {
            "knn": {
                "embedding": {    
                    "vector": query_vector,
                    "k": TOP_K,
                }
            }
        },
        "_source": ["text", "source_file", "chunk_index"],
    }

    response = client.search(index=INDEX_NAME, body=query_body)
    hits = response["hits"]["hits"]

    return [
        {
            "text": h["_source"]["text"],
            "source": h["_source"].get("source_file", "unknown"),
            "chunk_index": h["_source"].get("chunk_index", 0),
            "knn_score": h["_score"],
            "knn_rank": i + 1,   
        }
        for i, h in enumerate(hits)
    ]


def bm25_score(query: str, text: str) -> float:
    query_terms = re.findall(r"\w+", query.lower())
    doc_terms = re.findall(r"\w+", text.lower())

    if not doc_terms or not query_terms:
        return 0.0

    doc_len = len(doc_terms)
    score = 0.0

    for term in set(query_terms):
        tf = doc_terms.count(term) / doc_len
        idf = 1.0 / len(set(query_terms))
        score += tf * (1 + idf) 

    return score


def reciprocal_rank_fusion(
    chunks: list[dict[str, Any]], query: str
) -> list[dict[str, Any]]:
    K = 60

    # Add BM25 scores and ranks
    scored = []
    for chunk in chunks:
        bm25 = bm25_score(query, chunk["text"])
        chunk["bm25_score"] = bm25
        scored.append((chunk, bm25))

    # Sort by BM25 to get BM25 ranks
    scored.sort(key=lambda x: x[1], reverse=True)
    for bm25_rank, (chunk, _) in enumerate(scored, start=1):
        chunk["bm25_rank"] = bm25_rank

    # Compute RRF score
    for chunk in chunks:
        rrf = 1 / (K + chunk["knn_rank"]) + 1 / (K + chunk["bm25_rank"])
        chunk["rrf_score"] = rrf

    # Sort by final RRF score
    chunks.sort(key=lambda c: c["rrf_score"], reverse=True)
    return chunks[:FINAL_TOP_K]


# langgraph node function
def retriever_node(state: AgentState) -> dict:
    query = state["query"]
    logger.info("Retriever: embedding query '%s'", query[:80])

    try:
        # Step 1: embed the query
        query_vector = embed_query(query)

        # Step 2: kNN search
        client = _get_opensearch_client()
        raw_chunks = knn_search(client, query_vector)

        if not raw_chunks:
            logger.warning("Retriever: no chunks found for query")
            return {"chunks": [], "error": "No relevant documents found."}

        # Step 3: re-rank
        reranked = reciprocal_rank_fusion(raw_chunks, query)

        logger.info("Retriever: returning %d chunks after re-ranking", len(reranked))
        return {"chunks": reranked}

    except Exception as exc:  # noqa: BLE001
        logger.exception("Retriever node failed")
        return {"chunks": [], "error": f"Retrieval failed: {exc}"}