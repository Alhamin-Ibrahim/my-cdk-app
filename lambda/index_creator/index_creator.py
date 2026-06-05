import logging
import time

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INDEX_NAME = "documents"
DIMENSION = 1024           # Titan Embeddings v2 output dimension
MAX_WAIT_SECONDS = 240    # give up after 4 minutes
POLL_INTERVAL = 10         # check every 10 seconds


def _build_client(host: str, region: str = "eu-west-1") -> OpenSearch:
    session = boto3.Session()
    creds = session.get_credentials().get_frozen_credentials()
    auth = AWS4Auth(
        creds.access_key,
        creds.secret_key,
        region,
        "aoss",
        session_token=creds.token,
    )
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


def _wait_for_collection(client: OpenSearch) -> None:
    deadline = time.time() + MAX_WAIT_SECONDS
    attempt = 0

    while time.time() < deadline:
        attempt += 1

        try:
            client.indices.exists(index=INDEX_NAME)

            logger.info(
                "Collection reachable after %d attempt(s)",
                attempt,
            )
            return

        except Exception as exc:
            logger.info(
                "Waiting for collection (attempt %d): %s",
                attempt,
                exc,
            )
            time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        f"Collection not reachable after {MAX_WAIT_SECONDS}s"
    )


def handler(event, context):
    request_type = event["RequestType"]

    # Stack deletion: the collection itself will be cleaned up by CDK
    if request_type == "Delete":
        logger.info("Delete event — no action needed")
        return {"Status": "SUCCESS"}

    host = event["ResourceProperties"]["CollectionEndpoint"].replace("https://", "")
    client = _build_client(host)

    # Block until the collection is accepting requests
    _wait_for_collection(client)

    index_body = {
        "settings": {
            "index": {"knn": True}
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": DIMENSION,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                    },
                },
                "text": {"type": "text"},
                "source": {"type": "keyword"},
                "page": {"type": "integer"},
                "chunk_index": {"type": "integer"},
            }
        },
    }

    if client.indices.exists(index=INDEX_NAME):
        logger.info("Index '%s' already exists — skipping creation", INDEX_NAME)
    else:
        client.indices.create(index=INDEX_NAME, body=index_body)
        logger.info("Index '%s' created successfully", INDEX_NAME)

    return {"Status": "SUCCESS"}
