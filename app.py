import aws_cdk as cdk
from my_cdk_app.my_cdk_app_stack import MyCdkAppStack
from my_cdk_app.ecs_stack import EcsStack

app = cdk.App()

infra_stack = MyCdkAppStack(
    app,
    "MyCdkAppStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "eu-west-1",
    ),
)

ecs_stack = EcsStack(
    app,
    "EcsStack",
    # Pass cross-stack values as constructor props
    # Replace these with actual property references from your infra_stack
    opensearch_endpoint=infra_stack.node.try_get_context("opensearch_endpoint")
        or "https://l622892vevh1z8rvwjll.eu-west-1.aoss.amazonaws.com",
    dynamodb_table_name=infra_stack.node.try_get_context("dynamodb_table_name")
        or "rag-conversation-history",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "eu-west-1",
    ),
)

# Declare that ECS stack depends on infra stack — CDK will deploy in order
ecs_stack.add_dependency(infra_stack)

app.synth()