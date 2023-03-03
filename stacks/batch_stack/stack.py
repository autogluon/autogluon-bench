#!/usr/bin/env python3

import os

import aws_cdk as core
import boto3
from aws_cdk import aws_batch_alpha as batch
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as aws_lambda
from aws_cdk import aws_s3 as s3
from botocore.exceptions import ClientError
from constructs import Construct

from batch_job_cdk.constructs.batch_lambda_function import BatchLambdaFunction
from batch_job_cdk.constructs.instance_profile import InstanceProfile

"""
Sample CDK code for creating the required infrastructure for running a AWS Batch job.
AWS Batch as the compute enviroment in which a docker image runs the benchmarking script.
"""

# Relative path to the source code for the aws batch job, from the project root
docker_base_dir = "benchmarks"

# Relative path to the source for the AWS lambda, from the project root
lambda_script_dir = "automm_lambda"


class StaticResourceStack(core.Stack):
    def import_or_create_bucket(self, resource, bucket_name):
        bucket = resource.Bucket(bucket_name)
        if bucket.creation_date:
            print(f"The bucket {bucket_name} already exists, importing to the stack...")
            bucket = s3.Bucket.from_bucket_name(
                self, bucket_name, bucket_name=bucket_name
            )
        else:
            print(f"The bucket {bucket_name} does not exist, creating a new bucket...")
            bucket = s3.Bucket(
                self,
                bucket_name,
                bucket_name=bucket_name,
                removal_policy=core.RemovalPolicy.RETAIN,
            )
        return bucket

    def import_or_create_dynamo_db(self, resource, table_name):
        table = resource.Table(table_name)
        try:
            _table_exists = table.table_status
            print(f"The table {table_name} already exists, importing to the stack...")
            table = dynamodb.Table.from_table_name(
                self,
                table_name,
                table_name=table_name,
            )
        except ClientError:
            print(f"The table {table_name} does not exist, creating a new table...")
            table = dynamodb.Table(
                self,
                table_name,
                table_name=table_name,
                partition_key=dynamodb.Attribute(
                    name="uuid", type=dynamodb.AttributeType.STRING
                ),
                billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
                point_in_time_recovery=True,
                removal_policy=core.RemovalPolicy.RETAIN,
            )
        return table
    
    def import_or_create_vpc(self, resource, vpc_name, prefix):
        filters = [
            {
                "Name": "tag:Name",
                "Values": [vpc_name],
            }
        ]
        response = resource.describe_vpcs(Filters=filters)
        if response["Vpcs"]:
            vpc_id = response["Vpcs"][0]["VpcId"]
            vpc = ec2.Vpc.from_lookup(self, f"{prefix}-vpc", vpc_id=vpc_id)
        else:
            vpc = ec2.Vpc(self, f"{prefix}-vpc", vpc_name=vpc_name, max_azs=1)
        return vpc


    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        experiment_table = self.node.try_get_context("EXPERIMENT_TABLE")
        model_bucket_name = self.node.try_get_context("MODEL_BUCKET")
        data_bucket_name = self.node.try_get_context("DATA_BUCKET")
        vpc_name = self.node.try_get_context("VPC_NAME")
        prefix = self.node.try_get_context("STACK_NAME_PREFIX")

        region = os.environ["CDK_DEPLOY_REGION"]
        s3_resource = boto3.resource(service_name="s3", region_name=region)
        self.model_bucket = self.import_or_create_bucket(
            resource=s3_resource, bucket_name=model_bucket_name
        )
        self.data_bucket = s3.Bucket.from_bucket_name(
            self, data_bucket_name, bucket_name=data_bucket_name
        )

        db = boto3.resource(service_name="dynamodb", region_name=region)
        self.db_table = self.import_or_create_dynamo_db(
            resource=db, table_name=experiment_table
        )
        ec2_client = boto3.client("ec2", region_name=region)
        self.vpc = self.import_or_create_vpc(resource=ec2_client, vpc_name=vpc_name, prefix=prefix)
        

class BatchJobStack(core.Stack):
    """Defines stack with:
    - New Compute Environment with
        - Batch Instance Role (read access to S3, read-write access to DynamoDB)
        - Launch Template
        - Security Group
        - VPC
        - Job Definition (with customized container)
        - Job Queue
    - Use existing or create new DynamoDB table
    - Use existing or create new S3 bucket
    - New Lambda function to run training
    """

    def __init__(
        self, scope: Construct, id: str, static_stack: StaticResourceStack, **kwargs
    ) -> None:
        super().__init__(scope, id, **kwargs)
        prefix = self.node.try_get_context("STACK_NAME_PREFIX")
        tag = self.node.try_get_context("STACK_NAME_TAG")
        instance_types = self.node.try_get_context("INSTANCE_TYPES")
        compute_env_maxv_cpus = int(self.node.try_get_context("COMPUTE_ENV_MAXV_CPUS"))
        container_gpu = self.node.try_get_context("CONTAINER_GPU")
        container_vcpu = self.node.try_get_context("CONTAINER_VCPU")
        container_memory = self.node.try_get_context("CONTAINER_MEMORY")
        block_device_volume = self.node.try_get_context("BLOCK_DEVICE_VOLUME")
        lambda_function_name = self.node.try_get_context("LAMBDA_FUNCTION_NAME")

        instances = []
        for instance in instance_types:
            instances.append(ec2.InstanceType(instance))

        vpc = static_stack.vpc

        sg = ec2.SecurityGroup(
            self,
            f"{prefix}-security-group",
            vpc=vpc,
        )

        # Currently CDK can only push to the default repo aws-cdk/assets
        # https://github.com/aws/aws-cdk/issues/12597
        # TODO: use https://github.com/cdklabs/cdk-docker-image-deployment
        docker_image_asset = aws_ecr_assets.DockerImageAsset(
            self,
            f"{prefix}-ecr-docker-image-asset",
            directory=docker_base_dir,
            follow_symlinks=core.SymlinkFollowMode.ALWAYS,
        )

        docker_container_image = ecs.ContainerImage.from_docker_image_asset(
            docker_image_asset
        )

        container = batch.JobDefinitionContainer(
            image=docker_container_image,
            gpu_count=container_gpu,
            vcpus=container_vcpu,
            memory_limit_mib=container_memory,
            # Bug that this parameter is not rending in the CF stack under cdk.out
            # https://github.com/aws/aws-cdk/issues/13023
            linux_params=ecs.LinuxParameters(
                self, f"{prefix}-linux_params", shared_memory_size=container_memory
            ),
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
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonEC2ContainerServiceforEC2Role"
                ),
            ],
        )

        db_table = static_stack.db_table
        model_bucket = static_stack.model_bucket
        data_bucket = static_stack.data_bucket
        db_table.grant_read_write_data(batch_instance_role)
        data_bucket.grant_read(batch_instance_role)
        model_bucket.grant_read_write(batch_instance_role)

        batch_instance_profile = InstanceProfile(
            self, f"{prefix}-instance-profile", prefix=prefix
        )
        batch_instance_profile.attach_role(batch_instance_role)

        compute_environment = batch.ComputeEnvironment(
            self,
            f"{prefix}-compute-environment",
            compute_resources=batch.ComputeResources(
                allocation_strategy=batch.AllocationStrategy.BEST_FIT_PROGRESSIVE,
                vpc=vpc,
                vpc_subnets=ec2.SubnetSelection(subnets=vpc.private_subnets),
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
            compute_environments=[
                batch.JobQueueComputeEnvironment(
                    compute_environment=compute_environment, order=1
                )
            ],
        )

        batch_lambda_function = BatchLambdaFunction(
            self,
            lambda_function_name,
            prefix=prefix,
            tag=tag,
            function_name=lambda_function_name,
            code_path=lambda_script_dir,
            environment={
                "BATCH_JOB_QUEUE": job_queue.job_queue_name,
                "BATCH_JOB_DEFINITION": job_definition.job_definition_name,
                "DYNAMODB_TABLE_NAME": db_table.table_name,
                "DYNAMODB_TABLE_REGION": os.environ["CDK_DEPLOY_REGION"],
                "MODEL_BUCKET": model_bucket.bucket_name,
                "STACK_NAME_PREFIX": prefix,
            },
        )

        # Output the ARN for manually updating tagging
        core.CfnOutput(
            self,
            "ComputeEnvironmentARN",
            value=compute_environment.compute_environment_arn,
        )
        core.CfnOutput(self, "JobQueueARN", value=job_queue.job_queue_arn)
        core.CfnOutput(
            self, "JobDefinitionARN", value=job_definition.job_definition_arn
        )
