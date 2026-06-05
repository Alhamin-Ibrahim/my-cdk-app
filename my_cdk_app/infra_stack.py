from __future__ import annotations

import json

from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_dynamodb as dynamodb,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_opensearchserverless as aoss,
    aws_s3 as s3,
    custom_resources as cr,
    CustomResource,
    CfnOutput,
)
from constructs import Construct


class InfraStack(Stack):
    """
    Exposes the following properties for EcsStack to consume:
      - vpc
      - opensearch_endpoint  (string)
      - dynamodb_table_name  (string)
      - orchestrator_repo    (ecr.Repository)
      - retriever_repo       (ecr.Repository)
      - ingestion_repo       (ecr.Repository)
      - task_execution_role  (iam.Role)
      - task_role            (iam.Role)
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # VPC
        # Single VPC shared by both stacks.
        # 2 public subnets host the ALB; 2 private subnets host Fargate tasks.
        self.vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        # IAM roles
        # Execution role: ECS control-plane — pulls images, writes CW logs.
        self.task_execution_role = iam.Role(
            self,
            "TaskExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ],
        )

        # Task role: used by application code inside containers.
        self.task_role = iam.Role(
            self,
            "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )

        # S3 document bucket
        document_bucket = s3.Bucket(
            self,
            "DocumentBucket",
            versioned=True,
            event_bridge_enabled=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90),
                        )
                    ]
                )
            ],
        )

        # Least-privilege S3 access for the task role
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[f"{document_bucket.bucket_arn}/*"],
            )
        )

        # ECR repositories
        def _make_repo(id_: str, name: str) -> ecr.Repository:
            repo = ecr.Repository(
                self,
                id_,
                repository_name=name,
                removal_policy=RemovalPolicy.DESTROY,
                auto_delete_images=True,
            )
            repo.add_lifecycle_rule(
                description="Expire untagged images after 7 days",
                tag_status=ecr.TagStatus.UNTAGGED,
                max_image_age=Duration.days(7),
            )
            repo.add_lifecycle_rule(
                description="Keep only the last 3 tagged images",
                tag_status=ecr.TagStatus.ANY,
                max_image_count=3,
            )
            # Grant the execution role pull access to every repo
            repo.grant_pull(self.task_execution_role)
            return repo

        self.orchestrator_repo = _make_repo("OrchestratorRepo", "agent-orchestrator")
        self.retriever_repo = _make_repo("RetrieverRepo", "agent-retriever")
        self.ingestion_repo = _make_repo("IngestionRepo", "agent-ingestion")

        # OpenSearch Serverless
        encryption_policy = aoss.CfnSecurityPolicy(
            self,
            "EncryptionPolicy",
            name="vector-encryption-policy",
            type="encryption",
            policy=json.dumps(
                {
                    "Rules": [
                        {
                            "Resource": ["collection/vector-collection"],
                            "ResourceType": "collection",
                        }
                    ],
                    "AWSOwnedKey": True,
                }
            ),
        )

        network_policy = aoss.CfnSecurityPolicy(
            self,
            "NetworkPolicy",
            name="vector-network-policy",
            type="network",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "Resource": ["collection/vector-collection"],
                                "ResourceType": "collection",
                            },
                            {
                                "Resource": ["collection/vector-collection"],
                                "ResourceType": "dashboard",
                            },
                        ],
                        "AllowFromPublic": True,
                    }
                ]
            ),
        )

        collection = aoss.CfnCollection(
            self,
            "VectorCollection",
            name="vector-collection",
            type="VECTORSEARCH",
        )
        collection.add_dependency(encryption_policy)
        collection.add_dependency(network_policy)

        # Lambda: creates the kNN index after the collection is active
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
                        "bash",
                        "-c",
                        "pip install -r requirements.txt -t /asset-output && cp -au . /asset-output",
                    ],
                },
            ),
            timeout=Duration.minutes(5),
        )

        # Least-privilege: only the specific AOSS actions the Lambda needs
        index_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["aoss:APIAccessAll"],
                resources=[collection.attr_arn],
            )
        )

        # Data access policy grants the task role + Lambda role access to the index
        data_policy = aoss.CfnAccessPolicy(
            self,
            "DataAccessPolicy",
            name="vector-access-policy",
            type="data",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "Resource": ["collection/vector-collection"],
                                "Permission": [
                                    "aoss:CreateCollectionItems",
                                    "aoss:DeleteCollectionItems",
                                    "aoss:UpdateCollectionItems",
                                    "aoss:DescribeCollectionItems",
                                ],
                                "ResourceType": "collection",
                            },
                            {
                                "Resource": ["index/vector-collection/*"],
                                "Permission": [
                                    "aoss:CreateIndex",
                                    "aoss:DeleteIndex",
                                    "aoss:UpdateIndex",
                                    "aoss:DescribeIndex",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                ],
                                "ResourceType": "index",
                            },
                        ],
                        "Principal": [
                            self.task_role.role_arn,
                            index_lambda.role.role_arn,
                        ],
                    }
                ]
            ),
        )
        data_policy.add_dependency(collection)

        # task_role also needs aoss:APIAccessAll for data-plane HTTP requests
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["aoss:APIAccessAll"],
                resources=[collection.attr_arn],
            )
        )

        # Custom resource triggers the Lambda once after deploy to create the index
        provider = cr.Provider(
            self,
            "IndexProvider",
            on_event_handler=index_lambda,
        )

        index_resource = CustomResource(
            self,
            "CreateIndex",
            service_token=provider.service_token,
            properties={
                "CollectionEndpoint": collection.attr_collection_endpoint,
            },
        )
        index_resource.node.add_dependency(collection)
        index_resource.node.add_dependency(data_policy)

        # DynamoDB conversation memory table
        conversation_table = dynamodb.Table(
            self,
            "ConversationHistory",
            table_name="rag-conversation-history",
            partition_key=dynamodb.Attribute(
                name="session_id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                # turn_number is a millisecond epoch integer written by memory.py
                name="turn_number",
                type=dynamodb.AttributeType.NUMBER,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY,
        )

        conversation_table.grant_read_write_data(self.task_role)

        # Bedrock permissions
        self.task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v2:0",
                    f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
                ],
            )
        )

        # ECS cluster + ingestion task
        cluster = ec2.Cluster if False else None  # cluster created in EcsStack from vpc

        ingestion_task_def = __import__(
            "aws_cdk", fromlist=["aws_ecs"]
        ).aws_ecs.FargateTaskDefinition

        # We define the ingestion task definition here so it shares the same
        # task_role and can be referenced by the EventBridge rule below.
        from aws_cdk import aws_ecs as ecs

        ingestion_ecs_cluster = ecs.Cluster(self, "IngestionCluster", vpc=self.vpc)

        ingestion_task = ecs.FargateTaskDefinition(
            self,
            "IngestionTaskDef",
            cpu=256,
            memory_limit_mib=512,
            execution_role=self.task_execution_role,
            task_role=self.task_role,
        )

        ingestion_container = ingestion_task.add_container(
            "IngestionContainer",
            image=ecs.ContainerImage.from_ecr_repository(
                self.ingestion_repo, tag="latest"
            ),
            logging=ecs.LogDrivers.aws_logs(stream_prefix="ingestion"),
            command=["python", "main.py"],
            environment={
                "OPENSEARCH_ENDPOINT": collection.attr_collection_endpoint,
                "OPENSEARCH_INDEX": "documents",
                "DYNAMODB_TABLE_NAME": conversation_table.table_name,
            },
        )

        # EventBridge rule: S3 → ingestion ECS task
        events_role = iam.Role(
            self,
            "EventsEcsRole",
            assumed_by=iam.ServicePrincipal("events.amazonaws.com"),
        )
        events_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecs:RunTask"],
                resources=[ingestion_task.task_definition_arn],
            )
        )
        events_role.add_to_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=[
                    self.task_execution_role.role_arn,
                    self.task_role.role_arn,
                ],
            )
        )

        rule = events.Rule(
            self,
            "S3UploadRule",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={"bucket": {"name": [document_bucket.bucket_name]}},
            ),
        )

        rule.add_target(
            targets.EcsTask(
                cluster=ingestion_ecs_cluster,
                task_definition=ingestion_task,
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
                                value=events.EventField.from_path(
                                    "$.detail.bucket.name"
                                ),
                            ),
                            targets.TaskEnvironmentVariable(
                                name="OBJECT_KEY",
                                value=events.EventField.from_path(
                                    "$.detail.object.key"
                                ),
                            ),
                        ],
                    )
                ],
            )
        )

        # Expose outputs as stack properties (consumed by EcsStack)
        self.opensearch_endpoint: str = collection.attr_collection_endpoint
        self.dynamodb_table_name: str = conversation_table.table_name

        # CloudFormation outputs 
        CfnOutput(self, "DocumentBucketName", value=document_bucket.bucket_name)
        CfnOutput(self, "OpenSearchEndpoint", value=collection.attr_collection_endpoint)
        CfnOutput(self, "DynamoTableName", value=conversation_table.table_name)
        CfnOutput(self, "OrchestratorRepoUri", value=self.orchestrator_repo.repository_uri)
        CfnOutput(self, "RetrieverRepoUri", value=self.retriever_repo.repository_uri)
        CfnOutput(self, "IngestionRepoUri", value=self.ingestion_repo.repository_uri)
