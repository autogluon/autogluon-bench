import os
from unittest.mock import patch

import aws_cdk.aws_ec2 as ec2
import aws_cdk.aws_s3 as s3
from aws_cdk import App, Stack
from aws_cdk.aws_batch_alpha import ComputeEnvironment, JobDefinition, JobQueue
from aws_cdk.aws_ecr_assets import DockerImageAsset
from aws_cdk.aws_iam import Role
from conftest import context_values, env

from autogluon.bench.cloud.aws.batch_stack.constructs.batch_lambda_function import BatchLambdaFunction
from autogluon.bench.cloud.aws.batch_stack.constructs.instance_profile import InstanceProfile
from autogluon.bench.cloud.aws.batch_stack.stack import BatchJobStack, StaticResourceStack


def test_static_resource_stack_without_vpc():
    app = App()
    for key, value in context_values.items():
        app.node.set_context(key, value)

    with patch.object(
        StaticResourceStack,
        "create_s3_resources",
        return_value=s3.Bucket(Stack(app, "TestBucketStack"), "DummyBucket"),
    ) as mock_s3_resources, patch.dict(os.environ, {"CDK_DEPLOY_REGION": "dummy_region"}):
        stack = StaticResourceStack(app, "TestStaticResourceStack", env=env)

        mock_s3_resources.assert_called_once()

        assert stack.vpc is None


def test_static_resource_stack_with_vpc():
    app = App()
    for key, value in context_values.items():
        app.node.set_context(key, value)
    app.node.set_context("VPC_NAME", "ProvidedVpcName")

    with patch.object(
        StaticResourceStack,
        "create_s3_resources",
        return_value=s3.Bucket(Stack(app, "TestBucketStack"), "DummyBucket"),
    ) as mock_s3_resources, patch.dict(os.environ, {"CDK_DEPLOY_REGION": "dummy_region"}):
        stack = StaticResourceStack(app, "TestStaticResourceStack", env=env)

        mock_s3_resources.assert_called_once()

        assert stack.vpc is not None


@patch.dict("os.environ", {"CDK_DEPLOY_REGION": "dummy_region", "CDK_DEPLOY_ACCOUNT": "dummy_account"}, clear=True)
def test_batch_job_stack():
    app = App()
    for key, value in context_values.items():
        app.node.set_context(key, value)

    os.environ["FRAMEWORK_PATH"] = "frameworks/tabular"
    os.environ["GIT_URI"] = "https://github.com/openml/automlbenchmark.git"
    os.environ["GIT_BRANCH"] = "master"
    os.environ["BENCHMARK_DIR"] = "benchmark_name_20230725"

    with patch.object(
        StaticResourceStack,
        "create_s3_resources",
        return_value=s3.Bucket(Stack(app, "TestBucketStack"), "DummyBucket"),
    ):
        static_resource_stack = StaticResourceStack(app, "TestStaticResourceStack", env=env)

        assert static_resource_stack.vpc is None

        dummy_stack = Stack(app, "DummyVpcStack")
        static_resource_stack.metrics_bucket = s3.Bucket(dummy_stack, "DummyMetricsBucket")
        static_resource_stack.data_bucket = s3.Bucket(dummy_stack, "DummyDataBucket")

        batch_job_stack = BatchJobStack(app, "TestBatchJobStack", static_stack=static_resource_stack, env=env)
        prefix = app.node.try_get_context("STACK_NAME_PREFIX")
        lambda_function_name = app.node.try_get_context("LAMBDA_FUNCTION_NAME")
        constructs = [
            (f"{prefix}-security-group", ec2.SecurityGroup),
            (f"{prefix}-ecr-docker-image-asset", DockerImageAsset),
            ("job-definition", JobDefinition),
            (f"{prefix}-launch-template", ec2.LaunchTemplate),
            (f"{prefix}-instance-role", Role),
            (f"{prefix}-instance-profile", InstanceProfile),
            (f"{prefix}-compute-environment", ComputeEnvironment),
            (f"{prefix}-job-queue", JobQueue),
            (f"{lambda_function_name}-{prefix}", BatchLambdaFunction),
            ("vpc", ec2.Vpc),
        ]

        for construct_id, construct_class in constructs:
            if construct_id == "vpc":
                constructs = [c for c in batch_job_stack.node.children if isinstance(c, ec2.Vpc)]
                assert constructs, "No VPC found in BatchJobStack"
                assert isinstance(
                    constructs[0], construct_class
                ), f"First VPC in BatchJobStack is not an instance of {construct_class.__name__}"
            else:
                construct = batch_job_stack.node.try_find_child(construct_id)
                assert construct is not None, f"{construct_id} not found"
                assert isinstance(
                    construct, construct_class
                ), f"{construct_id} is not an instance of {construct_class.__name__}"
