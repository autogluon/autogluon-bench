import pytest
from aws_cdk import Environment

env = Environment(account="123456789012", region="us-west-1")

default_configs = {
    "CDK_DEPLOY_ACCOUNT": "dummy",
    "CDK_DEPLOY_REGION": "us-west-2",
    "PREFIX": "test-prefix",
    "METRICS_BUCKET": "test-metrics-bucket",
    "DATA_BUCKET": "test-data-bucket",
    "INSTANCE": "g4dn.2xlarge",
    "MAX_MACHINE_NUM": 2,
    "RESERVED_MEMORY_SIZE": 10000,
    "BLOCK_DEVICE_VOLUME": 10,
    "VPC_NAME": "test-vpc",
}

context_values = {
    "STACK_NAME_PREFIX": "test",
    "STACK_NAME_TAG": "benchmark",
    "STATIC_RESOURCE_STACK_NAME": "test-static-resource-stack",
    "BATCH_STACK_NAME": "test-batch-stack",
    "METRICS_BUCKET": "test-metrics-bucket",
    "DATA_BUCKET": "test-data-bucket",
    "INSTANCE_TYPES": ["g4dn.2xlarge"],
    "COMPUTE_ENV_MAXV_CPUS": 160,
    "CONTAINER_GPU": 1,
    "CONTAINER_VCPU": 8,
    "CONTAINER_MEMORY": 10000,
    "BLOCK_DEVICE_VOLUME": 100,
    "LAMBDA_FUNCTION_NAME": "test-batch-job-function",
}


@pytest.fixture
def default_context_values():
    return context_values
