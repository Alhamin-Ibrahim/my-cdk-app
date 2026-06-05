from __future__ import annotations

import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from pypdf import PdfReader
from requests_aws4auth import AWS4Auth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "eu-west-1")
OPENSEARCH_ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]
OPENSEARCH_INDEX = os.environ.get("OPENSEARCH_INDEX", "documents")

bedrock = boto3.client("bedrock-runtime", region_name=REGION)


# OpenSearch client
def get_opensearch_client() -> OpenSearch:
    credentials = boto3.Session().get_credentials().get_frozen_credentials()
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        REGION,
        "aoss",
        session_token=credentials.token,
    )
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://", ""), "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


# Document loading
def load_document(bucket: str, key: str) -> list[tuple[int, str]]:
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = obj["Body"].read()

    if key.lower().endswith(".pdf"):
        return _extract_pdf_pages(file_bytes)
    else:
        text = file_bytes.decode("utf-8")
        return [(0, text)]


def _extract_pdf_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def clean_text(text: str) -> str:
    return " ".join(text.split())


# Chunking
def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = []
    for page_num, text in pages:
        for chunk_text in splitter.split_text(text):
            chunks.append({"text": chunk_text, "page": page_num})
    return chunks


# Embedding
def get_embedding(text: str) -> list[float]:
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


def get_embeddings_concurrent(
    chunks: list[dict], max_workers: int = 10
) -> list[dict]:
    results = [None] * len(chunks)

    def _embed(idx: int, chunk: dict) -> tuple[int, list[float]]:
        return idx, get_embedding(chunk["text"])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_embed, i, c): i for i, c in enumerate(chunks)}
        for future in as_completed(futures):
            try:
                idx, embedding = future.result()
                results[idx] = embedding
            except Exception as exc:
                logger.error("Embedding failed for chunk %d: %s", futures[future], exc)

    return results


# Indexing
def index_chunks(
    client: OpenSearch,
    index_name: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    source_key: str,
) -> None:
    actions = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        if embedding is None:
            logger.warning("Skipping chunk %d — embedding failed", i)
            continue
        actions.append({
            "_index": index_name,
            "_source": {
                "text": chunk["text"],
                "embedding": embedding,
                "source": source_key,
                "page": chunk["page"],
                "chunk_index": i,
            },
        })

    if actions:
        bulk(client, actions)
        logger.info("Indexed %d chunks", len(actions))
    else:
        logger.warning("No chunks to index")


# Entry point for Lambda
def main() -> None:
    bucket = os.environ["BUCKET_NAME"]
    key = os.environ["OBJECT_KEY"]

    logger.info("Processing s3://%s/%s", bucket, key)

    pages = load_document(bucket, key)
    pages = [(p, clean_text(t)) for p, t in pages if t.strip()]
    logger.info("Pages after cleaning: %d", len(pages))

    chunks = chunk_pages(pages)
    logger.info("Chunks created: %d", len(chunks))

    if not chunks:
        logger.warning("No chunks — nothing to index")
        return

    logger.info("Generating embeddings concurrently…")
    embeddings = get_embeddings_concurrent(chunks)

    client = get_opensearch_client()
    index_chunks(client, OPENSEARCH_INDEX, chunks, embeddings, key)

    logger.info("Ingestion complete for %s", key)


if __name__ == "__main__":
    main()