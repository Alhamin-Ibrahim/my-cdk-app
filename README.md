# Multi-Agent RAG System on AWS

A Retrieval-Augmented Generation (RAG) system built with LangGraph, deployed on ECS Fargate, and fully provisioned with AWS CDK. Upload a PDF to S3 — the system automatically ingests it, chunks and embeds it into OpenSearch, and exposes a conversational API that retrieves relevant context and generates grounded answers via Amazon Bedrock.

## Architecture overview

```
User → ALB → Orchestrator (ECS) → Retriever (ECS) → OpenSearch Serverless
                    ↓                                        ↑
              Generator (Bedrock)               Ingestion (ECS, event-driven)
                    ↓                                        ↑
              DynamoDB (memory)                    S3 → EventBridge
```

Two CDK stacks are deployed in order:

- **InfraStack** — VPC, IAM roles, ECR repos, S3 bucket, OpenSearch Serverless collection + kNN index, DynamoDB conversation table, EventBridge rule, ingestion ECS cluster
- **EcsStack** — ECS Fargate cluster, orchestrator and retriever services, public and internal ALBs, auto-scaling, CloudWatch log groups

## Prerequisites

- AWS account with programmatic access (IAM user or role)
- Python 3.11+
- Docker (running locally)
- Node.js 18+ (required by CDK CLI)
- AWS CDK CLI: `npm install -g aws-cdk`
- AWS CLI configured: `aws configure`

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Alhamin-Ibrahim/my-cdk-app.git
cd my-cdk-app
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Bootstrap CDK (first time only)

```bash
cdk bootstrap aws://YOUR_ACCOUNT_ID/eu-west-1
```

### 3. Deploy infrastructure

```bash
cdk deploy InfraStack \
  --context account=YOUR_ACCOUNT_ID \
  --context region=eu-west-1 \
  --outputs-file infra-outputs.json
```

This takes 5–15 minutes. The OpenSearch Serverless collection must activate before the Lambda index creator runs. Watch progress in the CloudFormation console.

### 4. Build and push Docker images

Build and push manually:

```bash
# Log in to ECR
aws ecr get-login-password --region eu-west-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com

# Build and push each image
docker build --platform linux/amd64 -t agent-orchestrator ./orchestrator && \
  docker tag agent-orchestrator:latest \
  YOUR_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/agent-orchestrator:latest && \
  docker push YOUR_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/agent-orchestrator:latest

docker build --platform linux/amd64 -t agent-retriever ./retriever && \
  docker tag agent-retriever:latest \
  YOUR_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/agent-retriever:latest && \
  docker push YOUR_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/agent-retriever:latest

docker build --platform linux/amd64 -t agent-ingestion ./ingestion && \
  docker tag agent-ingestion:latest \
  YOUR_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/agent-ingestion:latest && \
  docker push YOUR_ACCOUNT_ID.dkr.ecr.eu-west-1.amazonaws.com/agent-ingestion:latest
```

### 5. Deploy ECS services

```bash
cdk deploy EcsStack \
  --context account=YOUR_ACCOUNT_ID \
  --context region=eu-west-1 \
  --context active=true \
  --outputs-file ecs-outputs.json
```

### 6. Ingest a document

```bash
BUCKET=$(cat infra-outputs.json | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(d['InfraStack']['DocumentBucketName'])")

aws s3 cp your-document.pdf s3://$BUCKET/your-document.pdf
```

EventBridge automatically triggers the ingestion task. Wait ~60 seconds, then query the system. 
Or you can upload a file on the S3 bucket to trigger the EventBridge.

### 7. Query the API

```bash
ALB=$(cat ecs-outputs.json | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(d['EcsStack']['AlbEndpoint'])")

# New conversation
curl -X POST $ALB/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the document say about X?"}'

# Follow-up using the returned session_id
curl -X POST $ALB/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Can you elaborate?", "session_id": "PASTE_SESSION_ID"}'
```

## API reference

### `POST /query`

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | The user's question |
| `session_id` | string | no | Omit to start a new session; include to continue a multi-turn conversation |

Response:

```json
{
  "answer": "Based on the documents...",
  "session_id": "uuid",
  "sources": ["your-document.pdf"],
  "intent": "retrieve"
}
```

### `GET /health`

Returns `{"status": "ok"}`. Used by the ALB health check.

## Running locally

Start the retriever first (it must be running before the orchestrator):

```bash
# Terminal 1 — retriever
export OPENSEARCH_ENDPOINT=https://YOUR_COLLECTION.eu-west-1.aoss.amazonaws.com
export AWS_REGION=eu-west-1
cd retriever && uvicorn main:app --port 8080

# Terminal 2 — orchestrator
export RETRIEVER_ENDPOINT=http://localhost:8080
export DYNAMODB_TABLE=rag-conversation-history
export AWS_REGION=eu-west-1
cd orchestrator && uvicorn main:app --port 8000
```

Then query at `http://localhost:8000/query`.

## Cost management

OpenSearch Serverless charges ~$0.24/hr per OCU from the moment the collection exists. To pause Fargate tasks when not using it (Fargate billing stops; everything else is minimal):

```bash
cdk deploy EcsStack \
  --context account=YOUR_ACCOUNT_ID \
  --context region=eu-west-1 \
  --context active=false
```

To destroy all resources completely:

```bash
cdk destroy --all
```

## Project structure

```
├── app.py                          # CDK app entry point
├── my_cdk_app/
│   ├── infra_stack.py              # VPC, IAM, ECR, S3, AOSS, DynamoDB
│   └── ecs_stack.py                # ECS cluster, services, ALBs
├── orchestrator/                   # LangGraph agent — routes, retrieves, generates
│   ├── agent.py                    # Graph definition and node logic
│   ├── main.py                     # FastAPI service (port 8000)
│   ├── memory.py                   # DynamoDB conversation history
│   └── state.py                    # LangGraph AgentState TypedDict
├── retriever/                      # Retrieval service
│   ├── agent.py                    # Embed → kNN → BM25/RRF re-rank
│   └── main.py                     # FastAPI service (port 8080)
├── ingestion/
│   └── main.py                     # PDF → chunks → embeddings → OpenSearch
├── generator/
│   └── agent.py                    # Prompt builder + Bedrock Haiku call
├── lambda/index_creator/
│   └── index_creator.py            # Custom resource: creates kNN index on deploy
└── .github/workflows/deploy.yml    # CI: build images → push ECR → rolling deploy
```

## AWS services used

| Service | Purpose |
|---|---|
| ECS Fargate | Runs orchestrator and retriever as long-lived services; ingestion as a run-task |
| Application Load Balancer | Public ALB for the orchestrator; internal ALB for the retriever |
| Amazon Bedrock | Titan Embeddings v2 (1024-dim vectors); Claude Haiku (answer generation) |
| OpenSearch Serverless | Vector index with kNN (HNSW, cosine similarity) |
| S3 | Raw document store with versioning and Glacier lifecycle |
| EventBridge | Triggers ingestion on S3 `Object Created` events |
| DynamoDB | Conversation history (session_id + turn_number, 24h TTL) |
| ECR | Container registry for all three Docker images |
| Lambda | Custom CDK resource that creates the OpenSearch index post-deploy |
| CloudWatch | Logs for all services; 1-week retention |
| X-Ray | Distributed tracing across the full call chain |
| IAM | Least-privilege task roles per service; separate execution role |