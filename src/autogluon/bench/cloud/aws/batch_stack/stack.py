#!/usr/bin/env python3

import importlib.resources
import logging
import os

import aws_cdk as core
import boto3
from aws_cdk import aws_batch_alpha as batch
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from constructs import Construct

from autogluon.bench.cloud.aws.batch_stack.constructs.batch_lambda_function import BatchLambdaFunction
from autogluon.bench.cloud.aws.batch_stack.constructs.instance_profile import InstanceProfile

"""
Sample CDK code for creating the required infrastructure for running a AWS Batch job.
AWS Batch as the compute enviroment in which a docker image runs the benchmarking script.
"""

with importlib.resources.path("autogluon.bench", "Dockerfile") as file_path:
    docker_base_dir = os.path.dirname(file_path)

with importlib.resources.path("autogluon.bench.cloud.aws.batch_stack.lambdas", "lambda_function.py") as file_path:
    lambda_script_dir = os.path.dirname(file_path)

logger = logging.getLogger(__name__)


class StaticResourceStack(core.Stack):
    """
    Defines a stack for creating and importing static resources, such as S3 buckets and VPCs.
    """

    def import_or_create_bucket(self, resource, bucket_name):
        """
        Imports an S3 bucket if it already exists or creates a new one if it doesn't exist.

        Args:
            resource: A boto3 S3 resource object.
            bucket_name: The name of the S3 bucket.

        Returns:
            An S3 bucket object.
        """
        bucket = resource.Bucket(bucket_name)
        if bucket.creation_date:
            logger.warning("The bucket %s already exists, importing to the stack...", bucket_name)
            bucket = s3.Bucket.from_bucket_name(self, bucket_name, bucket_name=bucket_name)
        else:
            logger.warning("The bucket %s does not exist, creating a new bucket...", bucket_name)
            bucket = s3.Bucket(
                self,
                bucket_name,
                bucket_name=bucket_name,
                removal_policy=core.RemovalPolicy.RETAIN,
            )
        return bucket

    def create_s3_resources(self):
        """
        Creates S3 bucket resources.
        """
        region = os.environ["CDK_DEPLOY_REGION"]
        s3_resource = boto3.resource(service_name="s3", region_name=region)
        self.metrics_bucket = self.import_or_create_bucket(resource=s3_resource, bucket_name=self.metrics_bucket_name)
        if self.data_bucket_name:
            self.data_bucket = s3.Bucket.from_bucket_name(
                self, self.data_bucket_name, bucket_name=self.data_bucket_name
            )
        else:
            self.data_bucket = None

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        self.metrics_bucket_name = self.node.try_get_context("METRICS_BUCKET")
        self.data_bucket_name = self.node.try_get_context("DATA_BUCKET")
        self.vpc_name = self.node.try_get_context("VPC_NAME")
        self.prefix = self.node.try_get_context("STACK_NAME_PREFIX")

        self.create_s3_resources()
        self.vpc = ec2.Vpc.from_lookup(self, f"{self.prefix}-vpc", vpc_name=self.vpc_name) if self.vpc_name else None


class BatchJobStack(core.Stack):
    def __init__(self, scope: Construct, id: str, static_stack: StaticResourceStack, **kwargs) -> None:
        """
        Defines a stack with the following:
        - A new Compute Environment with
            - Batch Instance Role (read access to S3)
            - Launch Template
            - Security Group
            - VPC
            - Job Definition (with customized container)
            - Job Queue
        - An existing or new S3 bucket
        - A new Lambda function to run training.

        Args:
            scope (constructs.Construct): The scope of the stack.
            id (str): The ID of the stack.
            static_stack (StaticResourceStack): A StaticResourceStack object.
            **kwargs: Keyword arguments.
        """
        super().__init__(scope, id, **kwargs)
        prefix = self.node.try_get_context("STACK_NAME_PREFIX")
        tag = self.node.try_get_context("STACK_NAME_TAG")
        instance_types = self.node.try_get_context("INSTANCE_TYPES")
        compute_env_maxv_cpus = int(self.node.try_get_context("COMPUTE_ENV_MAXV_CPUS"))
        container_gpu = self.node.try_get_context("CONTAINER_GPU")
        container_vcpu = self.node.try_get_context("CONTAINER_VCPU")
        container_memory = self.node.try_get_context("CONTAINER_MEMORY")
        block_device_volume = self.node.try_get_context("BLOCK_DEVICE_VOLUME")
        lambda_function_name = self.node.try_get_context("LAMBDA_FUNCTION_NAME") + "-" + prefix

        instances = []
        for instance in instance_types:
            instances.append(ec2.InstanceType(instance))

        vpc = static_stack.vpc

        if vpc is None:
            vpc = ec2.Vpc(
                self,
                f"{prefix}-vpc",
                max_azs=2,  # You can increase this number for high availability
                nat_gateways=1,
                subnet_configuration=[
                    ec2.SubnetConfiguration(
                        name=f"{prefix}-PublicSubnet",
                        subnet_type=ec2.SubnetType.PUBLIC,
                    ),
                    ec2.SubnetConfiguration(
                        name=f"{prefix}-PrivateSubnet",
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    ),
                ],
            )
        sg = ec2.SecurityGroup(
            self,
            f"{prefix}-security-group",
            vpc=vpc,
        )

        # Add inbound rule for ssh access
        # sg.add_ingress_rule(
        #     peer=ec2.Peer.any_ipv4(),
        #     connection=ec2.Port.tcp(22),
        #     description="Allow SSH access from Internet"
        # )

        # Currently CDK can only push to the default repo aws-cdk/assets
        # https://github.com/aws/aws-cdk/issues/12597
        # TODO: use https://github.com/cdklabs/cdk-docker-image-deployment
        docker_image_asset = ecr_assets.DockerImageAsset(
            self,
            f"{prefix}-ecr-docker-image-asset",
            directory=docker_base_dir,
            follow_symlinks=core.SymlinkFollowMode.ALWAYS,
            build_args={
                "AG_BENCH_VERSION": os.getenv("AG_BENCH_VERSION", "latest"),
                "AG_BENCH_DEV_URL": os.getenv("AG_BENCH_DEV_URL", ""),
                "CDK_DEPLOY_REGION": os.environ["CDK_DEPLOY_REGION"],
                "FRAMEWORK_PATH": os.environ["FRAMEWORK_PATH"],
                "GIT_URI": os.environ["GIT_URI"],
                "GIT_BRANCH": os.environ["GIT_BRANCH"],
                "BENCHMARK_DIR": os.environ["BENCHMARK_DIR"],
                "AMLB_FRAMEWORK": os.getenv("AMLB_FRAMEWORK", ""),
                "AMLB_USER_DIR": os.getenv("AMLB_USER_DIR", ""),
            },
        )

        docker_container_image = ecs.ContainerImage.from_docker_image_asset(docker_image_asset)

        container = batch.JobDefinitionContainer(
            image=docker_container_image,
            gpu_count=container_gpu,
            vcpus=container_vcpu,
            memory_limit_mib=container_memory,
            # Bug that this parameter is not rending in the CF stack under cdk.out
            # https://github.com/aws/aws-cdk/issues/13023
            linux_params=ecs.LinuxParameters(self, f"{prefix}-linux_params", shared_memory_size=container_memory),
            environment={
                "AG_BENCH_VERSION": os.getenv("AG_BENCH_VERSION", "latest"),
                "AG_BENCH_DEV_URL": os.getenv("AG_BENCH_DEV_URL", ""),
            },
        )

        job_definition = batch.JobDefinition(
            self,
            "job-definition",
            container=container,
            retry_attempts=3,
            timeout=core.Duration.minutes(1500),
        )

        # LaunchTemplate.launch_template_name returns Null https://github.com/aws/aws-cdk/issues/19405
        # so we are defining the name here instead of tagging
        batch_launch_template_name = f"{prefix}-launch-template"
        launch_template = ec2.LaunchTemplate(
            self,
            f"{prefix}-launch-template",
            launch_template_name=batch_launch_template_name,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(block_device_volume),
                )
            ],
            http_tokens=ec2.LaunchTemplateHttpTokens.OPTIONAL,
            http_endpoint=True,
        )

        cloudwatch_policy = iam.Policy(
            self,
            f"{prefix}-cloudwatch-policy",
            policy_name=f"{prefix}-cloudwatch-policy",
            statements=[
                iam.PolicyStatement(
                    actions=["cloudwatch:PutMetricData"],
                    effect=iam.Effect.ALLOW,
                    resources=["*"],
                )
            ],
        )
        batch_instance_role = iam.Role(
            self,
            f"{prefix}-instance-role",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("ec2.amazonaws.com"),
                iam.ServicePrincipal("ecs.amazonaws.com"),
                iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonEC2ContainerServiceforEC2Role"),
            ],
        )
        batch_instance_role.attach_inline_policy(cloudwatch_policy)

        metrics_bucket = static_stack.metrics_bucket
        data_bucket = static_stack.data_bucket
        if data_bucket is not None:
            data_bucket.grant_read(batch_instance_role)
        metrics_bucket.grant_read_write(batch_instance_role)

        batch_instance_profile = InstanceProfile(self, f"{prefix}-instance-profile", prefix=prefix)
        batch_instance_profile.attach_role(batch_instance_role)

        compute_environment = batch.ComputeEnvironment(
            self,
            f"{prefix}-compute-environment",
            compute_resources=batch.ComputeResources(
                allocation_strategy=batch.AllocationStrategy.BEST_FIT_PROGRESSIVE,
                vpc=vpc,
                vpc_subnets=ec2.SubnetSelection(subnets=vpc.private_subnets),
                # vpc_subnets=ec2.SubnetSelection(subnets=vpc.public_subnets),  # use public subnet for ssh
                maxv_cpus=compute_env_maxv_cpus,
                instance_role=batch_instance_profile.profile_arn,
                instance_types=instances,
                security_groups=[sg],
                type=batch.ComputeResourceType.ON_DEMAND,
                # ec2_key_pair=f"{prefix}-perm-key", # set this if you need ssh into instance
                launch_template=batch.LaunchTemplateSpecification(
                    launch_template_name=batch_launch_template_name  # LaunchTemplate.launch_template_name returns None
                ),
            ),
        )

        job_queue = batch.JobQueue(
            self,
            f"{prefix}-job-queue",
            priority=1,
            compute_environments=[batch.JobQueueComputeEnvironment(compute_environment=compute_environment, order=1)],
        )

        lambda_function = BatchLambdaFunction(
            self,
            lambda_function_name,
            prefix=prefix,
            tag=tag,
            function_name=lambda_function_name,
            code_path=lambda_script_dir,
            environment={
                "BATCH_JOB_QUEUE": job_queue.job_queue_name,
                "BATCH_JOB_DEFINITION": job_definition.job_definition_name,
                "METRICS_BUCKET": metrics_bucket.bucket_name,
                "STACK_NAME_PREFIX": prefix,
            },
        )
        metrics_bucket.grant_read_write(lambda_function._lambda_function)

        # Output the ARN for manually updating tagging
        core.CfnOutput(
            self,
            "ComputeEnvironmentARN",
            value=compute_environment.compute_environment_arn,
        )
        core.CfnOutput(self, "JobQueueARN", value=job_queue.job_queue_arn)
        core.CfnOutput(self, "JobDefinitionARN", value=job_definition.job_definition_arn)
        core.CfnOutput(
            self,
            "EcrRepositoryName",
            value=docker_image_asset.repository.repository_name,
        )
        core.CfnOutput(self, "ImageUri", value=docker_image_asset.image_uri)
