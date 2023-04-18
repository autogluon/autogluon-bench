from unittest.mock import MagicMock, patch

import aws_cdk.aws_ec2 as ec2
import aws_cdk.aws_s3 as s3
from aws_cdk import App, Stack
from aws_cdk.aws_batch_alpha import ComputeEnvironment, JobDefinition, JobQueue
from aws_cdk.aws_ec2 import LaunchTemplate, SecurityGroup
from aws_cdk.aws_ecr_assets import DockerImageAsset
from aws_cdk.aws_iam import Role
from conftest import context_values, env

from autogluon.bench.cloud.aws.batch_stack.constructs.batch_lambda_function import BatchLambdaFunction
from autogluon.bench.cloud.aws.batch_stack.constructs.instance_profile import InstanceProfile
from autogluon.bench.cloud.aws.batch_stack.stack import BatchJobStack, StaticResourceStack


def test_static_resource_stack():
    app = App()
    for key, value in context_values.items():
        app.node.set_context(key, value)

    with patch.object(StaticResourceStack, "create_s3_resources", MagicMock()) as mock_s3_resources, patch.object(
        StaticResourceStack, "create_vpc_resources", MagicMock()
    ) as mock_vpc_resources:
        stack = StaticResourceStack(app, "TestStaticResourceStack", env=env)

        mock_s3_resources.assert_called_once()
        mock_vpc_resources.assert_called_once()


def test_batch_job_stack():
    app = App()
    for key, value in context_values.items():
        app.node.set_context(key, value)

    with patch("autogluon.bench.cloud.aws.batch_stack.stack.StaticResourceStack.create_s3_resources"), patch(
        "autogluon.bench.cloud.aws.batch_stack.stack.StaticResourceStack.create_vpc_resources"
    ):
        static_resource_stack = StaticResourceStack(app, "TestStaticResourceStack", env=env)
        dummy_stack = Stack(app, "DummyVpcStack")
        static_resource_stack.metrics_bucket = s3.Bucket(dummy_stack, "DummyMetricsBucket")
        static_resource_stack.data_bucket = s3.Bucket(dummy_stack, "DummyDataBucket")
        static_resource_stack.vpc = ec2.Vpc(dummy_stack, "DummyVpc")

        batch_job_stack = BatchJobStack(app, "TestBatchJobStack", static_stack=static_resource_stack, env=env)

        constructs = [
            ("test-security-group", SecurityGroup),
            ("test-ecr-docker-image-asset", DockerImageAsset),
            ("job-definition", JobDefinition),
            ("test-launch-template", LaunchTemplate),
            ("test-instance-role", Role),
            ("test-instance-profile", InstanceProfile),
            ("test-compute-environment", ComputeEnvironment),
            ("test-job-queue", JobQueue),
            ("test-batch-job-function", BatchLambdaFunction),
        ]

        for construct_id, construct_class in constructs:
            construct = batch_job_stack.node.try_find_child(construct_id)
            assert construct is not None, f"{construct_id} not found"
            assert isinstance(
                construct, construct_class
            ), f"{construct_id} is not an instance of {construct_class.__name__}"
