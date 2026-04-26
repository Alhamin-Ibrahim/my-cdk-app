from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_ecr as ecr,
    RemovalPolicy,
    aws_s3 as s3,
    Duration,
    aws_s3_notifications as s3n,
    aws_opensearchserverless as opensearchserverless,
    aws_lambda as _lambda,
    custom_resources as cr,
    CustomResource
)
from constructs import Construct

class MyCdkAppStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create a VPC with public and private subnets across two availability zones and a NAT gateway
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

        # ECS to pull images from ECR and write CloudWatch logs
        execution_role = iam.Role(
            self,
            "ExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )

        #
        execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AmazonECSTaskExecutionRolePolicy"
            )
        )

        # App containers to call Bedrock, S3, OpenSearch
        task_role = iam.Role(
            self,
            "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )

        # Permissions for S3 access
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:PutObject"  
                ],
                resources=[
                    "arn:aws:s3:::my-bucket-name/*"
                ]
            )
        )

        # Permissions for Bedrock access   
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel"
                ],
                resources=["*"]
            )
        )

        # Permissions for OpenSearch access
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "es:ESHttpGet",
                    "es:ESHttpPost",
                    "es:ESHttpPut",
                    "es:ESHttpDelete",
                ],
                resources=[
                    f"arn:aws:es:{self.region}:{self.account}:domain/my-domain/*"
                ]
            )
        )

        #ECR repositories for app containers
        orchestrator_repo = ecr.Repository(
            self,
            "OrchestratorRepo",
            repository_name="orchestrator",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_images=True
        )

        retriever_repo = ecr.Repository(
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

        # S3 bucket for storing data
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

        encryption_policy = opensearchserverless.CfnSecurityPolicy(
            self,
            "EncryptionPolicy",
            name="vector-encryption-policy",
            type="encryption",
            policy="""
            {
            "Rules": [
                {
                "Resource": ["collection/vector-collection"],
                "ResourceType": "collection"
                }
            ],
            "AWSOwnedKey": true
            }
            """
        )

        network_policy = opensearchserverless.CfnSecurityPolicy(
            self,
            "NetworkPolicy",
            name="vector-network-policy",
            type="network",
            policy="""
            [
            {
                "Rules": [
                {
                    "Resource": ["collection/vector-collection"],   
                    "ResourceType": "collection"
                }
                ],
                "AllowFromPublic": true
            }
            ]
            """
        )

        collection = opensearchserverless.CfnCollection(
            self,
            "VectorCollection",
            name="vector-collection",
            type="VECTORSEARCH"
        )
        collection.add_dependency(encryption_policy)
        collection.add_dependency(network_policy)

        data_policy = opensearchserverless.CfnAccessPolicy(
            self,
            "DataAccessPolicy",
            name="vector-access-policy",
            type="data",
            policy=f"""
            [
            {{
                "Rules": [
                {{
                    "Resource": ["collection/vector-collection"],
                    "Permission": [
                    "aoss:CreateCollectionItems",
                    "aoss:DeleteCollectionItems",
                    "aoss:UpdateCollectionItems",
                    "aoss:DescribeCollectionItems"
                    ],
                    "ResourceType": "collection"
                }},
                {{
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
                }}
                ],
                "Principal": [
                    "{task_role.role_arn}",
                    "arn:aws:iam::{self.account}:root"
                ]
            }}
            ]
            """
        )

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
