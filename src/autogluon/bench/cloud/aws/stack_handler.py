import importlib.resources
import json
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import typer
import yaml

from autogluon.bench.cloud.aws.constants import gpu_map, memory_map, vcpu_map

with importlib.resources.path("autogluon.bench.cloud.aws", "stack_handler.py") as file_path:
    module_base_dir = os.path.dirname(file_path)
CONTEXT_FILE = "./cdk.context.json"

app = typer.Typer()


def _get_temp_cdk_app_path():
    temp_dir = tempfile.mkdtemp()
    temp_cdk_app_path = os.path.join(temp_dir, "app.py")
    with importlib.resources.path("autogluon.bench.cloud.aws", "app.py") as cdk_app_path:
        shutil.copy2(cdk_app_path, temp_cdk_app_path)
    os.chmod(temp_cdk_app_path, 0o755)
    return temp_cdk_app_path


def construct_context(custom_configs: dict) -> dict:
    """
    Constructs the AWS Cloud Development Kit (CDK) context using a combination of default configuration
    settings and custom settings, and writes the context to a JSON file. Also sets environment variables for
    the CDK deployment account and region.

    Args:
        custom_configs (dict, optional): A dictionary containing custom configuration settings. Defaults to {}.

    Returns:
        dict: A dictionary containing the constructed CDK context settings.
    """
    default_config_file = os.path.join(module_base_dir, "default_config.yaml")
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


def deploy_stack(configs: Optional[dict] = None) -> dict:
    """
    Deploys the AWS CloudFormation stack containing the benchmarking infrastructure by calling the deploy.sh
    script and passing it the required command line arguments. Constructs the CDK context using the custom
    configuration settings specified in the configs parameter, or the default configuration settings if no
    custom settings are provided.

    Args:
        configs (dict, optional): A dictionary containing custom configuration settings. Defaults to None.

    Returns:
        dict: A dictionary containing the CDK context settings used for the deployment.
    """
    cdk_path = _get_temp_cdk_app_path()
    custom_configs = {} if configs is None else configs
    infra_configs = construct_context(custom_configs=custom_configs)

    subprocess.check_call(
        [
            os.path.join(module_base_dir, "deploy.sh"),
            infra_configs["STACK_NAME_PREFIX"],
            infra_configs["STACK_NAME_TAG"],
            infra_configs["STATIC_RESOURCE_STACK_NAME"],
            infra_configs["BATCH_STACK_NAME"],
            str(infra_configs["CONTAINER_MEMORY"]),
            infra_configs["CDK_DEPLOY_REGION"],
            cdk_path,
        ]
    )
    shutil.rmtree(os.path.dirname(cdk_path))

    return infra_configs


@app.command()
def destroy_stack(
    static_resource_stack: str = typer.Argument(..., help="The static resource stack name."),
    batch_stack: str = typer.Argument(..., help="The batch stack name."),
    cdk_deploy_account: str = typer.Argument(..., help="The CDK deploy account ID."),
    cdk_deploy_region: str = typer.Argument(..., help="The CDK deploy region."),
):
    """
    This function destroys AWS CloudFormation stacks using the AWS Cloud Development Kit (CDK).

    It first sets up the necessary environment variables for the CDK, then calls a shell script
    that uses the CDK to destroy the specified static resource stack and batch stack. Finally, it
    removes the temporary directory that was used to deploy the CDK app.

    If you have previously deployed with `agbench run CONFIG_FILE,`
    you can find the AWS configs saved under {root_dir}/{module}/{prefix}_{timestamp}/aws_configs.yaml"
    """
    cdk_path = _get_temp_cdk_app_path()
    os.environ["CDK_DEPLOY_ACCOUNT"] = cdk_deploy_account
    os.environ["CDK_DEPLOY_REGION"] = cdk_deploy_region
    subprocess.check_call(
        [
            os.path.join(module_base_dir, "destroy.sh"),
            static_resource_stack,
            batch_stack,
            cdk_deploy_region,
            cdk_path,
        ]
    )
    shutil.rmtree(os.path.dirname(cdk_path))
