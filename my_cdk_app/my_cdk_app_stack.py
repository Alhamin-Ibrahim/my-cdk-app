from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_ecr as ecr,
    RemovalPolicy,
    aws_s3 as s3,
    Duration,
    aws_s3_notifications as s3n,
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

        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel"
                ],
                resources=["*"]
            )
        )

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
