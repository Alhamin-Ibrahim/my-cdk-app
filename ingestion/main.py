from langchain_text_splitters import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from opensearchpy.helpers import bulk
from pypdf import PdfReader
import boto3
import json
import os
import io

bedrock = boto3.client("bedrock-runtime", region_name="eu-west-1")

def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    chunked_data = []

    for page_num, text in pages:
        chunks = splitter.split_text(text)

        for chunk in chunks:
            chunked_data.append({
                "text": chunk,
                "page": page_num
            })

    return chunked_data

def get_embedding(text):
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({
            "inputText": text
        })
    )

    result = json.loads(response["body"].read())
    return result["embedding"]

def get_opensearch_client(host):
    region = "eu-west-1"
    service = "aoss"

    credentials = boto3.Session().get_credentials()

    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token,
    )

    return OpenSearch(
        hosts=[{"host": host.replace("https://", ""), "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

def index_chunks(client, index_name, chunks, source_key):
    actions = []

    for i, chunk_obj in enumerate(chunks):
        embedding = get_embedding(chunk_obj["text"])

        actions.append({
            "_index": index_name,
            "_source": {
                "text": chunk_obj["text"],
                "embedding": embedding,
                "source": source_key,
                "page": chunk_obj["page"],
                "chunk_id": i
            }
        })

    bulk(client, actions)

def load_document(bucket, key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = obj["Body"].read()

    if key.endswith(".pdf"):
        return extract_pdf_pages(file_bytes)  # <-- now returns [(page_num, text)]
    else:
        text = file_bytes.decode("utf-8")
        return [(0, text)]  # treat txt as single "page"

def clean_text(text):
    return " ".join(text.split())

def extract_pdf_pages(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))

    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))

    return pages

def main():
    bucket = os.environ["BUCKET_NAME"]
    key = os.environ["OBJECT_KEY"]
    host = os.environ["OPENSEARCH_ENDPOINT"]

    print(f"Processing file: {key}")

    pages = load_document(bucket, key)
    print(f"Total pages (raw): {len(pages)}")

    # clean each page
    pages = [(p, clean_text(t)) for p, t in pages if t.strip()]
    print(f"Total pages (after cleaning): {len(pages)}")

    chunks = chunk_pages(pages)
    print(f"Total chunks created: {len(chunks)}")

    client = get_opensearch_client(host)

    index_chunks(client, "documents", chunks, key)

    print("Ingestion complete")


if __name__ == "__main__":
    main()