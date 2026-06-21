from __future__ import annotations

import json

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_applicationautoscaling as autoscaling,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_elasticloadbalancingv2 as elbv2,
    aws_iam as iam,
    aws_logs as logs,
    aws_opensearchserverless as aoss,
)
from constructs import Construct


class EcsStack(Stack):
    """
    Receives the VPC, ECR repos, and config strings from InfraStack via
    constructor props

    Cost-saving toggle:
        cdk deploy --context active=false   → desired_count = 0 (zero Fargate cost)
        cdk deploy --context active=true    → desired_count = 1
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.Vpc,
        opensearch_endpoint: str,
        dynamodb_table_name: str,
        orchestrator_repo: ecr.Repository,
        retriever_repo: ecr.Repository,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        is_active = self.node.try_get_context("active")
        desired_count = 0 if is_active == "false" else 1

        # ECS cluster
        # Reuses the VPC from InfraStack
        cluster = ecs.Cluster(
            self,
            "AgentCluster",
            cluster_name="AgentCluster",  # stable name used by CI `aws ecs update-service`
            vpc=vpc,
            enable_fargate_capacity_providers=True,
            container_insights=True,
        )

        # Shared execution role
        execution_role = iam.Role(
            self,
            "EcsExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ],
        )
        for repo in [orchestrator_repo, retriever_repo]:
            repo.grant_pull(execution_role)

        # Orchestrator task role with permissions for Bedrock, DynamoDB, and X-Ray
        orchestrator_task_role = iam.Role(
            self,
            "OrchestratorTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        orchestrator_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/*",
                ],
            )
        )
        orchestrator_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:Query",
                ],
                resources=[
                    f"arn:aws:dynamodb:{self.region}:{self.account}:table/{dynamodb_table_name}"
                ],
            )
        )
        orchestrator_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["xray:PutTraceSegments", "xray:PutTelemetryRecords"],
                resources=["*"],
            )
        )

        # Retriever task role with permissions for OpenSearch, Bedrock, and X-Ray
        retriever_task_role = iam.Role(
            self,
            "RetrieverTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        retriever_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["aoss:APIAccessAll"],
                resources=[
                    f"arn:aws:aoss:{self.region}:{self.account}:collection/*"
                ],
            )
        )
        retriever_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0"
                ],
            )
        )
        retriever_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["xray:PutTraceSegments", "xray:PutTelemetryRecords"],
                resources=["*"],
            )
        )

        # AOSS data access policy for the retriever 
        retriever_data_policy = aoss.CfnAccessPolicy(
            self,
            "RetrieverDataAccessPolicy",
            name="retriever-access-policy",
            type="data",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "Resource": ["collection/vector-collection"],
                                "Permission": ["aoss:DescribeCollectionItems"],
                                "ResourceType": "collection",
                            },
                            {
                                "Resource": ["index/vector-collection/*"],
                                "Permission": [
                                    "aoss:DescribeIndex",
                                    "aoss:ReadDocument",
                                ],
                                "ResourceType": "index",
                            },
                        ],
                        "Principal": [retriever_task_role.role_arn],
                    }
                ]
            ),
        )

        # CloudWatch log groups
        def _log_group(name: str) -> logs.LogGroup:
            return logs.LogGroup(
                self,
                f"{name}LogGroup",
                log_group_name=f"/ecs/agents/{name}",
                retention=logs.RetentionDays.ONE_WEEK,
                removal_policy=RemovalPolicy.DESTROY,
            )

        orchestrator_logs = _log_group("orchestrator")
        retriever_logs = _log_group("retriever")

        # Security groups
        alb_sg = ec2.SecurityGroup(
            self,
            "AlbSg",
            vpc=vpc,
            description="Public ALB - accept HTTP from internet",
            allow_all_outbound=True,
        )
        alb_sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(80))

        ecs_sg = ec2.SecurityGroup(
            self,
            "EcsSg",
            vpc=vpc,
            description="ECS tasks - accept from ALB and each other",
            allow_all_outbound=True,
        )
        ecs_sg.add_ingress_rule(alb_sg, ec2.Port.tcp(8000))   # orchestrator
        ecs_sg.add_ingress_rule(ecs_sg, ec2.Port.tcp(8080))   # retriever (from orchestrator)

        retriever_alb_sg = ec2.SecurityGroup(
            self,
            "RetrieverAlbSg",
            vpc=vpc,
            description="Internal ALB for retriever - ECS SG only",
            allow_all_outbound=True,
        )
        retriever_alb_sg.add_ingress_rule(ecs_sg, ec2.Port.tcp(8080))

        # Public ALB (orchestrator)
        alb = elbv2.ApplicationLoadBalancer(
            self,
            "AgentAlb",
            vpc=vpc,
            internet_facing=True,
            security_group=alb_sg,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
        )

        listener = alb.add_listener("HttpListener", port=80, open=False)

        # Internal ALB (retriever)
        retriever_alb = elbv2.ApplicationLoadBalancer(
            self,
            "RetrieverAlb",
            vpc=vpc,
            internet_facing=False,
            security_group=retriever_alb_sg,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
        )

        retriever_listener = retriever_alb.add_listener(
            "RetrieverListener",
            port=8080,
            protocol=elbv2.ApplicationProtocol.HTTP,
            open=False,
        )

        retriever_tg = elbv2.ApplicationTargetGroup(
            self,
            "RetrieverTg",
            vpc=vpc,
            port=8080,
            protocol=elbv2.ApplicationProtocol.HTTP,
            target_type=elbv2.TargetType.IP,
            health_check=elbv2.HealthCheck(
                path="/health",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
            deregistration_delay=Duration.seconds(30),
        )

        retriever_listener.add_target_groups(
            "RetrieverInternalTargets",
            target_groups=[retriever_tg],
        )

        # Stable DNS — doesn't change when tasks restart
        retriever_endpoint = f"http://{retriever_alb.load_balancer_dns_name}:8080"

        # Orchestrator task definition 
        orchestrator_task_def = ecs.FargateTaskDefinition(
            self,
            "OrchestratorTaskDef",
            cpu=256,
            memory_limit_mib=512,
            execution_role=execution_role,
            task_role=orchestrator_task_role,
        )

        orchestrator_task_def.add_container(
            "OrchestratorContainer",
            image=ecs.ContainerImage.from_ecr_repository(
                orchestrator_repo, tag="latest"
            ),
            environment={
                "ENVIRONMENT": "production",
                "RETRIEVER_ENDPOINT": retriever_endpoint,
                "OPENSEARCH_ENDPOINT": opensearch_endpoint,
                "BEDROCK_REGION": self.region,
                "DYNAMODB_TABLE": dynamodb_table_name,
                "AWS_DEFAULT_REGION": self.region,
            },
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="orchestrator",
                log_group=orchestrator_logs,
            ),
            port_mappings=[ecs.PortMapping(container_port=8000)],
        )

        # Retriever task definition
        retriever_task_def = ecs.FargateTaskDefinition(
            self,
            "RetrieverTaskDef",
            cpu=256,
            memory_limit_mib=512,
            execution_role=execution_role,
            task_role=retriever_task_role,
        )

        retriever_task_def.add_container(
            "RetrieverContainer",
            image=ecs.ContainerImage.from_ecr_repository(
                retriever_repo, tag="latest"
            ),
            environment={
                "ENVIRONMENT": "production",
                "OPENSEARCH_ENDPOINT": opensearch_endpoint,
                "BEDROCK_REGION": self.region,
                "AWS_DEFAULT_REGION": self.region,
                "OPENSEARCH_INDEX": "documents",
            },
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="retriever",
                log_group=retriever_logs,
            ),
            port_mappings=[ecs.PortMapping(container_port=8080)],
        )

        # ECS services
        # min_healthy_percent=50 allows rolling deploys with desired_count=1.
        orchestrator_service = ecs.FargateService(
            self,
            "OrchestratorService",
            service_name="OrchestratorService",  # stable name for CLI / CI
            cluster=cluster,
            task_definition=orchestrator_task_def,
            desired_count=desired_count,
            security_groups=[ecs_sg],
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            assign_public_ip=False,
            min_healthy_percent=50,
            max_healthy_percent=200,
        )

        retriever_service = ecs.FargateService(
            self,
            "RetrieverService",
            service_name="RetrieverService",
            cluster=cluster,
            task_definition=retriever_task_def,
            desired_count=desired_count,
            security_groups=[ecs_sg],
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            assign_public_ip=False,
            min_healthy_percent=50,
            max_healthy_percent=200,
        )

        retriever_service.attach_to_application_target_group(retriever_tg)

        # Orchestrator target group + listener
        orchestrator_tg = elbv2.ApplicationTargetGroup(
            self,
            "OrchestratorTg",
            vpc=vpc,
            port=8000,
            protocol=elbv2.ApplicationProtocol.HTTP,
            target_type=elbv2.TargetType.IP,
            health_check=elbv2.HealthCheck(
                path="/health",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
            deregistration_delay=Duration.seconds(30),
        )

        orchestrator_service.attach_to_application_target_group(orchestrator_tg)
        listener.add_target_groups("OrchestratorTargets", target_groups=[orchestrator_tg])

        # Auto-scaling (orchestrator only)
        scalable = orchestrator_service.auto_scale_task_count(
            min_capacity=1,
            max_capacity=3,
        )
        scalable.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=60,
            scale_in_cooldown=Duration.seconds(300),
            scale_out_cooldown=Duration.seconds(60),
        )
        scalable.scale_on_memory_utilization(
            "MemoryScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(300),
            scale_out_cooldown=Duration.seconds(60),
        )

        # CloudFormation outputs
        CfnOutput(
            self,
            "AlbEndpoint",
            value=f"http://{alb.load_balancer_dns_name}",
            description="Curl this to reach the orchestrator",
        )
        CfnOutput(
            self,
            "RetrieverInternalEndpoint",
            value=retriever_endpoint,
            description="Internal ALB used by orchestrator",
        )
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "OrchestratorServiceName", value=orchestrator_service.service_name)

        self.alb_dns_name = alb.load_balancer_dns_name