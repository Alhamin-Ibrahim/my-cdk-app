from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_ecr as ecr,
    RemovalPolicy,
    aws_s3 as s3,
    Duration,
    aws_opensearchserverless as opensearchserverless,
    aws_lambda as _lambda,
    custom_resources as cr,
    CustomResource,
    aws_ecs as ecs,
    aws_events as events,
    aws_events_targets as targets,
    aws_dynamodb as dynamodb,
)
from constructs import Construct
import json


class MyCdkAppStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create a VPC with public and private subnets across two AZs and a NAT gateway
        vpc = ec2.Vpc(self, "MyVPC",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="PublicSubnet",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="PrivateSubnet",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                )
            ]
        )

        # S3 bucket for storing documents
        document_bucket = s3.Bucket(
            self,
            "DocumentBucket",
            versioned=True,
            event_bridge_enabled=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90)
                        )
                    ]
                )
            ]
        )

        # ECS execution role: pulls images from ECR and writes CloudWatch logs
        execution_role = iam.Role(
            self,
            "ExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AmazonECSTaskExecutionRolePolicy"
            )
        )

        # ECS task role: used by app containers to call AWS services
        task_role = iam.Role(
            self,
            "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )

        # S3 read/write on the document bucket
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[f"{document_bucket.bucket_arn}/*"]
            )
        )

        # ECR repositories
        orchestrator_repo = ecr.Repository(  # noqa: F841
            self,
            "OrchestratorRepo",
            repository_name="orchestrator",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_images=True
        )

        retriever_repo = ecr.Repository(  # noqa: F841
            self,
            "RetrieverRepo",
            repository_name="retriever",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_images=True
        )

        ingestion_repo = ecr.Repository(
            self,
            "IngestionRepo",
            repository_name="ingestion",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_images=True
        )

        # ── OpenSearch Serverless ──────────────────────────────────────────

        encryption_policy = opensearchserverless.CfnSecurityPolicy(
            self,
            "EncryptionPolicy",
            name="vector-encryption-policy",
            type="encryption",
            policy=json.dumps({
                "Rules": [
                    {
                        "Resource": ["collection/vector-collection"],
                        "ResourceType": "collection"
                    }
                ],
                "AWSOwnedKey": True
            })
        )

        network_policy = opensearchserverless.CfnSecurityPolicy(
            self,
            "NetworkPolicy",
            name="vector-network-policy",
            type="network",
            policy=json.dumps([
                {
                    "Rules": [
                        {
                            "Resource": ["collection/vector-collection"],
                            "ResourceType": "collection"
                        },
                        {
                            "Resource": ["collection/vector-collection"],
                            "ResourceType": "dashboard"
                        }
                    ],
                    "AllowFromPublic": True
                }
            ])
        )

        collection = opensearchserverless.CfnCollection(
            self,
            "VectorCollection",
            name="vector-collection",
            type="VECTORSEARCH"
        )
        collection.add_dependency(encryption_policy)
        collection.add_dependency(network_policy)

        # ── Lambda: creates the kNN index on first deploy ──────────────────

        index_lambda = _lambda.Function(
            self,
            "IndexCreatorFunction",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="index_creator.handler",
            code=_lambda.Code.from_asset(
                "lambda/index_creator",
                bundling={
                    "image": _lambda.Runtime.PYTHON_3_11.bundling_image,
                    "command": [
                        "bash", "-c",
                        "pip install -r requirements.txt -t /asset-output && cp -au . /asset-output"
                    ],
                },
            ),
            timeout=Duration.minutes(5),
        )

        index_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["aoss:*"],
                resources=["*"]
            )
        )

        # Data access policy: grants task role, Lambda role, and your IAM user
        data_policy = opensearchserverless.CfnAccessPolicy(
            self,
            "DataAccessPolicy",
            name="vector-access-policy",
            type="data",
            policy=json.dumps([
                {
                    "Rules": [
                        {
                            "Resource": ["collection/vector-collection"],
                            "Permission": [
                                "aoss:CreateCollectionItems",
                                "aoss:DeleteCollectionItems",
                                "aoss:UpdateCollectionItems",
                                "aoss:DescribeCollectionItems"
                            ],
                            "ResourceType": "collection"
                        },
                        {
                            "Resource": ["index/vector-collection/*"],
                            "Permission": [
                                "aoss:CreateIndex",
                                "aoss:DeleteIndex",
                                "aoss:UpdateIndex",
                                "aoss:DescribeIndex",
                                "aoss:ReadDocument",
                                "aoss:WriteDocument"
                            ],
                            "ResourceType": "index"
                        }
                    ],
                    "Principal": [
                        task_role.role_arn,
                        index_lambda.role.role_arn,
                        f"arn:aws:iam::{self.account}:user/alhaminibrahim"
                    ]
                }
            ])
        )
        data_policy.add_dependency(collection)

        # Task role also needs aoss:APIAccessAll for the data plane
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["aoss:APIAccessAll"],
                resources=[collection.attr_arn]
            )
        )

        # Custom resource: triggers the Lambda to create the index after deploy
        provider = cr.Provider(
            self,
            "IndexProvider",
            on_event_handler=index_lambda
        )

        index_resource = CustomResource(
            self,
            "CreateIndex",
            service_token=provider.service_token,
            properties={
                "CollectionEndpoint": collection.attr_collection_endpoint
            }
        )
        index_resource.node.add_dependency(collection)
        index_resource.node.add_dependency(data_policy)

        # ── ECS cluster + ingestion task definition ────────────────────────

        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)

        task_definition = ecs.FargateTaskDefinition(
            self,
            "IngestionTask",
            cpu=256,
            memory_limit_mib=512,
            execution_role=execution_role,
            task_role=task_role,
        )

        ingestion_container = task_definition.add_container(
            "IngestionContainer",
            image=ecs.ContainerImage.from_ecr_repository(ingestion_repo),
            logging=ecs.LogDrivers.aws_logs(stream_prefix="ingestion"),
            command=["python", "main.py"],
        )

        # ── EventBridge rule: trigger ingestion task on S3 upload ──────────

        events_role = iam.Role(
            self,
            "EventsEcsRole",
            assumed_by=iam.ServicePrincipal("events.amazonaws.com"),
        )
        events_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecs:RunTask"],
                resources=[task_definition.task_definition_arn]
            )
        )
        events_role.add_to_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=[execution_role.role_arn, task_role.role_arn]
            )
        )

        rule = events.Rule(
            self,
            "S3UploadRule",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {
                        "name": [document_bucket.bucket_name]
                    }
                }
            )
        )

        rule.add_target(
            targets.EcsTask(
                cluster=cluster,
                task_definition=task_definition,
                role=events_role,
                subnet_selection=ec2.SubnetSelection(
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
                ),
                task_count=1,
                container_overrides=[
                    targets.ContainerOverride(
                        container_name="IngestionContainer",
                        environment=[
                            targets.TaskEnvironmentVariable(
                                name="BUCKET_NAME",
                                value=events.EventField.from_path("$.detail.bucket.name")
                            ),
                            targets.TaskEnvironmentVariable(
                                name="OBJECT_KEY",
                                value=events.EventField.from_path("$.detail.object.key")
                            ),
                            targets.TaskEnvironmentVariable(
                                name="OPENSEARCH_ENDPOINT",
                                value=collection.attr_collection_endpoint
                            ),
                        ]
                    )
                ]
            )
        )

        # ── DynamoDB conversation memory ──────────────────────────

        conversation_table = dynamodb.Table(
            self,
            "ConversationHistory",
            table_name="rag-conversation-history",
            partition_key=dynamodb.Attribute(
                name="session_id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="turn_number",
                type=dynamodb.AttributeType.NUMBER,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Grant task role read/write on the conversation table
        conversation_table.grant_read_write_data(task_role)

        # Bedrock: Claude Haiku + Titan Embeddings
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-haiku-4-5",
                    f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v2",
                ],
            )
        )

        # Inject runtime config into the ingestion container as environment variables
        # Secrets (API keys, passwords) would use secretsFrom + Secrets Manager instead
        ingestion_container.add_environment(
            "OPENSEARCH_ENDPOINT", collection.attr_collection_endpoint
        )
        ingestion_container.add_environment("OPENSEARCH_INDEX", "documents")
        ingestion_container.add_environment(
            "DYNAMODB_TABLE_NAME", conversation_table.table_name
        )