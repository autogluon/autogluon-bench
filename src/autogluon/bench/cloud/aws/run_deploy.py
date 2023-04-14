import json
import os
import subprocess

import yaml

from autogluon.bench.cloud.aws.constants import gpu_map, memory_map, vcpu_map

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
CONTEXT_FILE = "./cdk.context.json"


def construct_context(custom_configs: dict = {}):
    default_config_file = CURRENT_DIR + "/default_config.yaml"
    configs = {}
    with open(default_config_file, "r") as f:
        configs = yaml.safe_load(f)
    configs.update(custom_configs)
    prefix = configs["PREFIX"]
    context_to_parse = {
        "CDK_DEPLOY_ACCOUNT": configs["CDK_DEPLOY_ACCOUNT"],
        "CDK_DEPLOY_REGION": configs["CDK_DEPLOY_REGION"],
        "STACK_NAME_PREFIX": prefix,  # aws resource tag key, also used as name prefix for resources created
        "STACK_NAME_TAG": "benchmark",  # aws resource tag value
        "STATIC_RESOURCE_STACK_NAME": f"{prefix}-static-resource-stack",
        "BATCH_STACK_NAME": f"{prefix}-batch-stack",
        "METRICS_BUCKET": configs["METRICS_BUCKET"],  # bucket to upload metrics
        "DATA_BUCKET": configs["DATA_BUCKET"],  # bucket to download data
        "INSTANCE_TYPES": [configs["INSTANCE"]],  # can be a list of instance families or instance types
        "COMPUTE_ENV_MAXV_CPUS": vcpu_map[configs["INSTANCE"]]
        * configs["MAX_MACHINE_NUM"],  # total max v_cpus in batch compute environment
        "CONTAINER_GPU": gpu_map[configs["INSTANCE"]],  # GPU reserved for container
        "CONTAINER_VCPU": vcpu_map[configs["INSTANCE"]],  # v_cpus reserved for container
        "CONTAINER_MEMORY": memory_map[configs["INSTANCE"]]
        - configs[
            "RESERVED_MEMORY_SIZE"
        ],  # memory in MB reserved for container, also used for shm_size, i.e. `shared_memory_size`
        "BLOCK_DEVICE_VOLUME": configs["BLOCK_DEVICE_VOLUME"],  # device attached to instance, in GB
        "LAMBDA_FUNCTION_NAME": f"{prefix}-batch-job-function",
        "VPC_NAME": configs[
            "VPC_NAME"
        ],  # it's recommended to share a vpc for all benchmark infra, you can lookup an existing VPC name under aws console -> VPC, if you want to create a new one, assign a new name
    }
    with open(CONTEXT_FILE, "w+") as f:
        try:
            cdk_config = json.load(f)
        except:
            cdk_config = {}
        cdk_config.update(context_to_parse)
        json.dump(cdk_config, f, indent=2)
        f.close()
    # set environment variables
    os.environ["CDK_DEPLOY_ACCOUNT"] = configs["CDK_DEPLOY_ACCOUNT"]
    os.environ["CDK_DEPLOY_REGION"] = configs["CDK_DEPLOY_REGION"]

    return context_to_parse


def deploy_stack(configs: dict = {}):
    infra_configs = construct_context(custom_configs=configs)

    subprocess.check_call(
        [
            CURRENT_DIR + "/deploy.sh",
            infra_configs["STACK_NAME_PREFIX"],
            infra_configs["STACK_NAME_TAG"],
            infra_configs["STATIC_RESOURCE_STACK_NAME"],
            infra_configs["BATCH_STACK_NAME"],
            str(infra_configs["CONTAINER_MEMORY"]),
        ]
    )
    return infra_configs


def destroy_stack(configs: dict):
    subprocess.check_call(
        [
            CURRENT_DIR + "/destroy.sh",
            configs["STATIC_RESOURCE_STACK_NAME"],
            configs["BATCH_STACK_NAME"],
        ]
    )
