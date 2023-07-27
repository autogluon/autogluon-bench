import importlib.resources
import json
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import boto3
import typer
import yaml

with importlib.resources.path("autogluon.bench.cloud.aws", "stack_handler.py") as file_path:
    module_base_dir = os.path.dirname(file_path)
CONTEXT_FILE = "./cdk.context.json"

app = typer.Typer()


def get_instance_type_specs(instance_type, region):
    ec2_client = boto3.client("ec2", region_name=region)
    response = ec2_client.describe_instance_types(InstanceTypes=[instance_type])

    instance_type_info = response["InstanceTypes"][0]

    gpu_info_list = instance_type_info.get("GpuInfo", [])
    gpu_count = sum(gpu_info.get("Count", 0) for gpu_info in gpu_info_list["Gpus"])

    vcpu_info = instance_type_info.get("VCpuInfo", {})
    vcpu_count = vcpu_info.get("DefaultVCpus", 0)

    memory_info = instance_type_info.get("MemoryInfo", {})
    memory = memory_info.get("SizeInMiB", 0)

    return gpu_count, vcpu_count, memory


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
    gpu_count, vcpu_count, memory = get_instance_type_specs(
        instance_type=configs["INSTANCE"], region=configs["CDK_DEPLOY_REGION"]
    )
    context_to_parse = {
        "CDK_DEPLOY_ACCOUNT": configs["CDK_DEPLOY_ACCOUNT"],
        "CDK_DEPLOY_REGION": configs["CDK_DEPLOY_REGION"],
        "STACK_NAME_PREFIX": prefix,  # aws resource tag key, also used as name prefix for resources created
        "STACK_NAME_TAG": "benchmark",  # aws resource tag value
        "STATIC_RESOURCE_STACK_NAME": f"{prefix}-static-resource-stack",
        "BATCH_STACK_NAME": f"{prefix}-batch-stack",
        "METRICS_BUCKET": configs["METRICS_BUCKET"],  # bucket to upload metrics
        "DATA_BUCKET": configs.get("DATA_BUCKET", None),  # bucket to download data
        "INSTANCE_TYPES": [configs["INSTANCE"]],  # can be a list of instance families or instance types
        "COMPUTE_ENV_MAXV_CPUS": vcpu_count
        * configs["MAX_MACHINE_NUM"],  # total max v_cpus in batch compute environment
        "CONTAINER_GPU": gpu_count,  # GPU reserved for container
        "CONTAINER_VCPU": vcpu_count,  # v_cpus reserved for container
        "CONTAINER_MEMORY": memory
        - configs[
            "RESERVED_MEMORY_SIZE"
        ],  # memory in MB reserved for container, also used for shm_size, i.e. `shared_memory_size`
        "BLOCK_DEVICE_VOLUME": configs["BLOCK_DEVICE_VOLUME"],  # device attached to instance, in GB
        "LAMBDA_FUNCTION_NAME": f"{prefix}-batch-job-function",
        "VPC_NAME": configs.get(
            "VPC_NAME", None
        ),  # it's recommended to share a vpc for all benchmark infra, you can lookup an existing VPC name under aws console -> VPC, if you want to create a new one, assign a new name
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
    os.environ["CDK_DEPLOY_ACCOUNT"] = str(configs["CDK_DEPLOY_ACCOUNT"])
    os.environ["CDK_DEPLOY_REGION"] = configs["CDK_DEPLOY_REGION"]

    return context_to_parse


def deploy_stack(custom_configs: dict) -> dict:
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
    custom_infra_configs = custom_configs.get("cdk_context", {})
    infra_configs = construct_context(custom_configs=custom_infra_configs)
    command = [
        os.path.join(module_base_dir, "deploy.sh"),
        infra_configs["STACK_NAME_PREFIX"],
        infra_configs["STACK_NAME_TAG"],
        infra_configs["STATIC_RESOURCE_STACK_NAME"],
        infra_configs["BATCH_STACK_NAME"],
        str(infra_configs["CONTAINER_MEMORY"]),
        infra_configs["CDK_DEPLOY_REGION"],
        cdk_path,
    ]

    subprocess.check_call(command)
    shutil.rmtree(os.path.dirname(cdk_path))

    return infra_configs


@app.command()
def destroy_stack(
    static_resource_stack: Optional[str] = typer.Option(None, help="The static resource stack name."),
    batch_stack: Optional[str] = typer.Option(None, help="The batch stack name."),
    cdk_deploy_account: Optional[str] = typer.Option(None, help="The CDK deploy account ID."),
    cdk_deploy_region: Optional[str] = typer.Option(None, help="The CDK deploy region."),
    config_file: Optional[str] = typer.Option(None, help="Path to YAML config file containing stack information."),
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

    if config_file is not None:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            static_resource_stack = config.get("STATIC_RESOURCE_STACK_NAME", static_resource_stack)
            batch_stack = config.get("BATCH_STACK_NAME", batch_stack)
            cdk_deploy_account = config.get("CDK_DEPLOY_ACCOUNT", cdk_deploy_account)
            cdk_deploy_region = config.get("CDK_DEPLOY_REGION", cdk_deploy_region)

    if static_resource_stack is None or batch_stack is None or cdk_deploy_account is None or cdk_deploy_region is None:
        raise ValueError(
            "static_resource_stack, batch_stack, cdk_deploy_account and cdk_deploy_region must be specified or configured in the config_file."
        )

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
