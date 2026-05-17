import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import time

def handler(event, context):

    request_type = event["RequestType"]

    if request_type == "Delete":
        return {"Status": "SUCCESS"}
    
    time.sleep(60)

    host = event["ResourceProperties"]["CollectionEndpoint"].replace("https://", "")

    region = "eu-west-1"
    service = "aoss"

    session = boto3.Session()
    credentials = session.get_credentials().get_frozen_credentials()

    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token,
    )

    client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

    index_body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1024
                },
                "text": {
                    "type": "text"
                }
            }
        }
    }

    index_name = "documents"

    # Create index if it doesn't exist
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=index_body)

    return {"Status": "SUCCESS"}