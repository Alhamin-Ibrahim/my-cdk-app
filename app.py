import aws_cdk as cdk
from my_cdk_app.infra_stack import InfraStack
from my_cdk_app.ecs_stack import EcsStack

app = cdk.App()

env = cdk.Environment(
    account=app.node.try_get_context("account"),
    region=app.node.try_get_context("region") or "eu-west-1",
)

# VPC, IAM, ECR, S3, OpenSearch, DynamoDB, Lambda index creator
infra_stack = InfraStack(app, "InfraStack", env=env)

# ECS Fargate services (orchestrator + retriever)
# Cross-stack values come from InfraStack *properties*
ecs_stack = EcsStack(
    app,
    "EcsStack",
    vpc=infra_stack.vpc,
    opensearch_endpoint=infra_stack.opensearch_endpoint,
    dynamodb_table_name=infra_stack.dynamodb_table_name,
    orchestrator_repo=infra_stack.orchestrator_repo,
    retriever_repo=infra_stack.retriever_repo,
    env=env,
)

# CDK will deploy InfraStack first, then EcsStack
ecs_stack.add_dependency(infra_stack)

app.synth()