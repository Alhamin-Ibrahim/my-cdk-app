import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecr as ecr,
    aws_elasticloadbalancingv2 as elbv2,
    aws_iam as iam,
    aws_logs as logs,
    aws_applicationautoscaling as autoscaling,
    CfnOutput,
    Duration,
    RemovalPolicy,
)
from constructs import Construct


class EcsStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        opensearch_endpoint: str,
        dynamodb_table_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Usage:  cdk deploy --context active=false
        # When active=false → desired_count=0 on all services (zero Fargate cost)
        # When active=true  → desired_count=1
        is_active = self.node.try_get_context("active")
        desired_count = 0 if is_active == "false" else 1

        vpc = ec2.Vpc(
            self,
            "AgentVpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                # Public subnets host the ALB (internet-facing)
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                # Private subnets host Fargate tasks
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        cluster = ecs.Cluster(
            self,
            "AgentCluster",
            vpc=vpc,
            # Fargate capacity providers let you use FARGATE_SPOT for cheaper, non-critical tasks
            enable_fargate_capacity_providers=True,
            # Container Insights gives you detailed metrics for auto-scaling and monitoring in CloudWatch
            container_insights=True,
        )

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

        # Orchestrator task role
        orchestrator_task_role = iam.Role(
            self,
            "OrchestratorTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        # Allow Bedrock InvokeModel for Claude and Titan
        orchestrator_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/*",
                ],
            )
        )
        # Allow DynamoDB operations on the conversation history table
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
        # Allow X-Ray tracing
        orchestrator_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["xray:PutTraceSegments", "xray:PutTelemetryRecords"],
                resources=["*"],
            )
        )

        # Retriever task role
        retriever_task_role = iam.Role(
            self,
            "RetrieverTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        # OpenSearch Serverless uses aoss:APIAccessAll
        retriever_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["aoss:APIAccessAll"],
                resources=[
                    f"arn:aws:aoss:{self.region}:{self.account}:collection/*"
                ],
            )
        )
        # Bedrock Titan embeddings
        retriever_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=["arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0"],
            )
        )
        retriever_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["xray:PutTraceSegments", "xray:PutTelemetryRecords"],
                resources=["*"],
            )
        )

        # ecr repositories for orchestrator, retriever, and ingestion containers
        def make_ecr_repo(id: str, name: str) -> ecr.Repository:
            repo = ecr.Repository(
                self,
                id,
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
                description="Keep only last 3 tagged images",
                tag_status=ecr.TagStatus.ANY,
                max_image_count=3,
            )
            return repo

        orchestrator_repo = make_ecr_repo("OrchestratorRepo", "agent-orchestrator")
        retriever_repo = make_ecr_repo("RetrieverRepo", "agent-retriever")
        ingestion_repo = make_ecr_repo("IngestionRepo", "agent-ingestion")

        # Grant the execution role permission to pull from all three repos
        for repo in [orchestrator_repo, retriever_repo, ingestion_repo]:
            repo.grant_pull(execution_role)

        # CloudWatch Log Groups for ECS task logging (one for each service)
        def make_log_group(service_name: str) -> logs.LogGroup:
            return logs.LogGroup(
                self,
                f"{service_name}LogGroup",
                log_group_name=f"/ecs/agents/{service_name}",
                retention=logs.RetentionDays.ONE_WEEK,
                removal_policy=RemovalPolicy.DESTROY,
            )

        orchestrator_logs = make_log_group("orchestrator")
        retriever_logs = make_log_group("retriever")

        # Orchestrator task definition
        orchestrator_task_def = ecs.FargateTaskDefinition(
            self,
            "OrchestratorTaskDef",
            cpu=256,      
            memory_limit_mib=512,
            execution_role=execution_role,
            task_role=orchestrator_task_role,
        )

        orchestrator_container = orchestrator_task_def.add_container(
            "OrchestratorContainer",
            image=ecs.ContainerImage.from_ecr_repository(
                orchestrator_repo, tag="latest"
            ),
            environment={
                "ENVIRONMENT": "production",
                "RETRIEVER_ENDPOINT": "PLACEHOLDER",  # overridden below after ALB is created
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
            },
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="retriever",
                log_group=retriever_logs,
            ),
            port_mappings=[ecs.PortMapping(container_port=8080)],
        )

        # security groups for ALB and ECS tasks
        alb_sg = ec2.SecurityGroup(
            self,
            "AlbSecurityGroup",
            vpc=vpc,
            description="ALB: accept HTTP from internet",
            allow_all_outbound=True,
        )
        alb_sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(80))

        ecs_sg = ec2.SecurityGroup(
            self,
            "EcsSecurityGroup",
            vpc=vpc,
            description="ECS tasks: accept traffic from ALB and each other",
            allow_all_outbound=True,
        )
        # Only the ALB can reach the orchestrator on port 8000
        ecs_sg.add_ingress_rule(alb_sg, ec2.Port.tcp(8000))
        # Retriever internal ALB reaches the retriever container on 8080
        ecs_sg.add_ingress_rule(ecs_sg, ec2.Port.tcp(8080))

        # an internet-facing ALB for the orchestrator (demo endpoint)
        alb = elbv2.ApplicationLoadBalancer(
            self,
            "AgentAlb",
            vpc=vpc,
            internet_facing=True,
            security_group=alb_sg,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
        )

        # HTTP listener (port 80)
        listener = alb.add_listener(
            "HttpListener",
            port=80,
            open=False,  # We manage SG rules above
        )

        # an internal ALB for the retriever service — not reachable from the internet
        retriever_alb_sg = ec2.SecurityGroup(
            self, "RetrieverAlbSg",
            vpc=vpc,
            description="Internal ALB for retriever - accept from ECS SG only",
            allow_all_outbound=True,
        )
        # Only the orchestrator ECS tasks can reach the retriever ALB
        retriever_alb_sg.add_ingress_rule(ecs_sg, ec2.Port.tcp(8080))

        retriever_alb = elbv2.ApplicationLoadBalancer(
            self, "RetrieverAlb",
            vpc=vpc,
            internet_facing=False,   # internal only — not reachable from internet
            security_group=retriever_alb_sg,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
        )

        retriever_listener = retriever_alb.add_listener(
            "RetrieverListener",
            port=8080,
            protocol=elbv2.ApplicationProtocol.HTTP,
            open=False,
        )

        retriever_internal_tg = elbv2.ApplicationTargetGroup(
            self, "RetrieverInternalTg",
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
            target_groups=[retriever_internal_tg],
        )

        # This DNS name is stable — it never changes even when tasks are replaced
        retriever_alb_dns = f"http://{retriever_alb.load_balancer_dns_name}:8080"

        orchestrator_container.add_environment("RETRIEVER_ENDPOINT", retriever_alb_dns)

        # orchestrator and retriever services (backed by Fargate tasks)
        orchestrator_service = ecs.FargateService(
            self,
            "OrchestratorService",
            cluster=cluster,
            task_definition=orchestrator_task_def,
            desired_count=desired_count,
            security_groups=[ecs_sg],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            assign_public_ip=False,
            min_healthy_percent=100,

        )

        retriever_service = ecs.FargateService(
            self,
            "RetrieverService",
            cluster=cluster,
            task_definition=retriever_task_def,
            desired_count=desired_count,
            security_groups=[ecs_sg],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            assign_public_ip=False,
            min_healthy_percent=100,

        )

        retriever_service.attach_to_application_target_group(retriever_internal_tg)

        # orchestrator target group for the public ALB
        orchestrator_target_group = elbv2.ApplicationTargetGroup(
            self,
            "OrchestratorTargetGroup",
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

        # Register the ECS service as the target
        orchestrator_service.attach_to_application_target_group(orchestrator_target_group)

        # Default action: forward all requests to the orchestrator
        listener.add_target_groups(
            "OrchestratorTargets",
            target_groups=[orchestrator_target_group],
        )

        #auto-scaling for orchestrator service based on CPU and memory utilization
        scalable_target = orchestrator_service.auto_scale_task_count(
            min_capacity=1,
            max_capacity=3,
        )

        # Scale out quickly (60s cooldown) to handle sudden traffic spikes, 
        # but scale in slowly (300s cooldown) to avoid killing tasks too 
        # aggressively during short lulls.
        scalable_target.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=60,
            scale_in_cooldown=Duration.seconds(300),
            scale_out_cooldown=Duration.seconds(60),
        )

        # Memory-based scaling can help when CPU isn't the bottleneck
        scalable_target.scale_on_memory_utilization(
            "MemoryScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(300),
            scale_out_cooldown=Duration.seconds(60),
        )

        # outputs to easily find important info after deployment
        CfnOutput(
            self,
            "RetrieverInternalAlb",
            value=retriever_alb_dns,
            description="Internal ALB for retriever (used by orchestrator)",
        )

        CfnOutput(
            self,
            "AlbDnsName",
            value=f"http://{alb.load_balancer_dns_name}",
            description="ALB endpoint - curl this to test the orchestrator",
        )

        CfnOutput(
            self,
            "OrchestratorServiceName",
            value=orchestrator_service.service_name,
            description="ECS service name (for CloudWatch / CLI)",
        )

        CfnOutput(
            self,
            "ClusterName",
            value=cluster.cluster_name,
            description="ECS cluster name",
        )

        CfnOutput(
            self,
            "OrchestratorRepoUri",
            value=orchestrator_repo.repository_uri,
            description="ECR URI for orchestrator image pushes",
        )

        CfnOutput(
            self,
            "RetrieverRepoUri",
            value=retriever_repo.repository_uri,
            description="ECR URI for retriever image pushes",
        )

        self.alb_dns_name = alb.load_balancer_dns_name