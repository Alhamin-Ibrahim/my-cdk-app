from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

logger = logging.getLogger(__name__)

OPENSEARCH_ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]
INDEX_NAME = os.environ.get("OPENSEARCH_INDEX", "documents")
REGION = os.environ.get("AWS_REGION", "eu-west-1")

TOP_K = 10       # fetch more candidates, then re-rank to FINAL_TOP_K
FINAL_TOP_K = 5


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
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": query}),
    )
    return json.loads(response["body"].read())["embedding"]


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
        "_source": ["text", "source", "chunk_index"],
    }

    response = client.search(index=INDEX_NAME, body=query_body)
    hits = response["hits"]["hits"]

    return [
        {
            "text": h["_source"].get("text", ""),
            "source": h["_source"].get("source", "unknown"),   # consistent field name
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


def reciprocal_rank_fusion(chunks: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    K = 60

    # Assign BM25 ranks
    scored = [(chunk, bm25_score(query, chunk["text"])) for chunk in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    for bm25_rank, (chunk, _) in enumerate(scored, start=1):
        chunk["bm25_rank"] = bm25_rank

    # Compute RRF score and re-sort
    for chunk in chunks:
        chunk["rrf_score"] = (
            1 / (K + chunk["knn_rank"]) + 1 / (K + chunk["bm25_rank"])
        )

    chunks.sort(key=lambda c: c["rrf_score"], reverse=True)
    return chunks[:FINAL_TOP_K]


def retrieve(query: str) -> dict[str, Any]:
    """
    Full retrieval pipeline: embed → kNN search → BM25 re-rank.

    Returns a dict compatible with the retriever HTTP response schema.
    """
    logger.info("Retrieving for query: '%s'", query[:80])

    query_vector = embed_query(query)
    client = _get_opensearch_client()
    raw_chunks = knn_search(client, query_vector)

    if not raw_chunks:
        logger.warning("No chunks found for query")
        return {"chunks": [], "sources": [], "count": 0}

    reranked = reciprocal_rank_fusion(raw_chunks, query)
    sources = list({c["source"] for c in reranked})

    logger.info("Returning %d chunks after re-ranking", len(reranked))
    return {"chunks": reranked, "sources": sources, "count": len(reranked)}
