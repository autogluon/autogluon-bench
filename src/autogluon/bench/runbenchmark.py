import json
import logging
import os
import re
import subprocess
import tempfile
import time
from typing import List, Optional

import boto3
import botocore
import typer
import yaml

from autogluon.bench import __path__ as agbench_path
from autogluon.bench import __version__ as agbench_version
from autogluon.bench.cloud.aws.stack_handler import deploy_stack, destroy_stack
from autogluon.bench.eval.hardware_metrics.hardware_metrics import get_hardware_metrics
from autogluon.bench.frameworks.multimodal.multimodal_benchmark import MultiModalBenchmark
from autogluon.bench.frameworks.tabular.tabular_benchmark import TabularBenchmark
from autogluon.bench.frameworks.timeseries.timeseries_benchmark import TimeSeriesBenchmark
from autogluon.bench.utils.general_utils import (
    download_dir_from_s3,
    download_file_from_s3,
    formatted_time,
    upload_to_s3,
)

app = typer.Typer()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
AMLB_DEPENDENT_MODULES = ["tabular", "timeseries"]


def get_kwargs(module: str, configs: dict):
    """Returns a dictionary of keyword arguments to be used for setting up and running the benchmark.

    Args:
        module (str): The name of the module to benchmark (either "multimodal", "tabular", or "timeseires").
        configs (dict): A dictionary of configuration options for the benchmark.

    Returns:
        A dictionary containing the keyword arguments to be used for setting up and running the benchmark.
    """

    if module == "multimodal":
        framework_configs = get_framework_configs(configs=configs)
        return {
            "setup_kwargs": {
                "git_uri": framework_configs["repo"],
                "git_branch": framework_configs.get("version", "stable"),
            },
            "run_kwargs": {
                "dataset_name": configs["dataset_name"],
                "framework": configs["framework"],
                "constraint": configs.get("constraint"),
                "params": framework_configs.get("params"),
                "custom_dataloader": configs.get("custom_dataloader"),
                "custom_metrics": configs.get("custom_metrics"),
            },
        }
    elif module in AMLB_DEPENDENT_MODULES:
        git_uri, git_branch = _get_git_info(configs["git_uri#branch"])
        return {
            "setup_kwargs": {
                "git_uri": git_uri,
                "git_branch": git_branch,
                "framework": configs["framework"],
                "user_dir": configs.get("amlb_user_dir"),
            },
            "run_kwargs": {
                "framework": configs["framework"],
                "benchmark": configs.get("amlb_benchmark", "test"),
                "constraint": configs.get("amlb_constraint", "test"),
                "task": configs.get("amlb_task"),
                "fold": configs.get("fold_to_run"),
                "user_dir": configs.get("amlb_user_dir"),
            },
        }


def _get_benchmark_name(configs: dict) -> str:
    default_benchmark_name = "ag_bench"
    benchmark_name = configs.get("benchmark_name", default_benchmark_name) or default_benchmark_name
    return benchmark_name


def run_benchmark(
    benchmark_name: str,
    benchmark_dir: str,
    configs: dict,
    benchmark_dir_s3: str = None,
    skip_setup: str = False,
):
    """Runs a benchmark based on the provided configuration options.

    Args:
        configs (dict): A dictionary of configuration options for the benchmark.

    Returns:
        None
    """

    module_to_benchmark = {
        "multimodal": MultiModalBenchmark,
        "tabular": TabularBenchmark,
        "timeseries": TimeSeriesBenchmark,
    }
    module_name = configs["module"]

    benchmark_class = module_to_benchmark.get(module_name, None)
    if benchmark_class is None:
        raise NotImplementedError

    benchmark = benchmark_class(benchmark_name=benchmark_name, benchmark_dir=benchmark_dir)

    module_kwargs = get_kwargs(module=module_name, configs=configs)
    if not skip_setup:
        benchmark.setup(**module_kwargs["setup_kwargs"])

    benchmark.run(**module_kwargs["run_kwargs"])
    logger.info(f"Backing up benchmarking configs to {benchmark.metrics_dir}/configs.yaml")
    _dump_configs(benchmark_dir=benchmark.metrics_dir, configs=configs, file_name="configs.yaml")

    if configs.get("METRICS_BUCKET", None):
        benchmark.upload_metrics(s3_bucket=configs["METRICS_BUCKET"], s3_dir=benchmark_dir_s3)


def invoke_lambda(configs: dict, config_file: str) -> dict:
    """Invokes an AWS Lambda function to run benchmarks based on the provided configuration options.

    Args:
        configs (dict): A dictionary of configuration options for the AWS infrastructure.
        config_file (str): The path of the configuration file to use for running the benchmarks.

    Returns:
        A dictionary containing Lambda response payload.
    """

    lambda_client = boto3.client("lambda", configs["CDK_DEPLOY_REGION"])
    payload = {"config_file": config_file}
    lambda_function_name = configs["LAMBDA_FUNCTION_NAME"] + "-" + configs["STACK_NAME_PREFIX"]
    response = lambda_client.invoke(
        FunctionName=lambda_function_name, InvocationType="RequestResponse", Payload=json.dumps(payload)
    )
    response_payload = json.loads(response["Payload"].read().decode("utf-8"))

    if "FunctionError" in response:
        error_payload = response_payload
        raise Exception(f"Lambda function error: {error_payload}")

    logger.info("AWS Batch jobs submitted by %s.", configs["LAMBDA_FUNCTION_NAME"])

    return response_payload


@app.command()
def get_job_status(
    job_ids: Optional[List[str]] = typer.Option(None, "--job-ids", help="List of job ids, separated by space."),
    cdk_deploy_region: Optional[str] = typer.Option(None, help="AWS region that the Batch jobs run in."),
    config_file: Optional[str] = typer.Option(None, help="Path to YAML config file containing job ids."),
):
    """
    Query the status of AWS Batch job ids.
    The job ids can either be passed in directly or read from a YAML configuration file.

    Args:
        job_ids (List[str], optional):
            A list of job ids to query the status for.
        cdk_deploy_region (str, optional):
            AWS region that the Batch jobs run in.
        config_file (str, optional):
            A path to a YAML config file containing job ids. The YAML file should have the structure:
                job_configs:
                    <job_id>: <job_config>
                    <job_id>: <job_config>
                    ...

    Returns:
        A dictionary containing the status of the queried job ids.
    """
    if config_file is not None:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            job_ids = list(config.get("job_configs", {}).keys())
            cdk_deploy_region = config.get("CDK_DEPLOY_REGION", cdk_deploy_region)
    if job_ids is None or cdk_deploy_region is None:
        raise ValueError("Either job_ids or cdk_deploy_region must be provided or configured in the config_file.")

    batch_client = boto3.client("batch", region_name=cdk_deploy_region)

    status_dict = {}

    for job_id in job_ids:
        response = batch_client.describe_jobs(jobs=[job_id])
        job_detail = response["jobs"][0]

        # Check if the job is an array job
        if "arrayProperties" in job_detail and "size" in job_detail["arrayProperties"]:
            status_dict[job_id] = job_detail["arrayProperties"]["statusSummary"]
        else:
            status_dict[job_id] = job_detail["status"]

    logger.info(status_dict)
    return status_dict


def wait_for_jobs(
    config_file: Optional[str] = None,
    job_ids: Optional[List[str]] = None,
    aws_region: Optional[str] = None,
    quit_statuses: Optional[list] = ["SUCCEEDED", "FAILED"],
    frequency: Optional[int] = 120,
):
    if config_file is not None:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            job_ids = list(config.get("job_configs", {}).keys())
            aws_region = config.get("CDK_DEPLOY_REGION", aws_region)

    batch_client = boto3.client("batch", region_name=aws_region)
    while True:
        all_jobs_completed = True
        failed_jobs = set()

        try:
            job_status = get_job_status(job_ids=job_ids, cdk_deploy_region=aws_region, config_file=None)

            for job_id, status in job_status.items():
                if isinstance(status, str):
                    if status == "FAILED":
                        failed_jobs.append(job_id)
                    elif status not in quit_statuses:
                        all_jobs_completed = False
                elif isinstance(status, dict):
                    for status, num in status.items():
                        if status == "FAILED" and num > 0:
                            paginator = batch_client.get_paginator("list_jobs")
                            for page in paginator.paginate(arrayJobId=job_id, jobStatus="FAILED"):
                                for job in page["jobSummaryList"]:
                                    failed_jobs.add(job["jobId"])
                        if status not in quit_statuses and num > 0:
                            all_jobs_completed = False
        except botocore.exceptions.ClientError as e:
            logger.error(f"An error occurred: {e}.")
            return

        if all_jobs_completed:
            break
        else:
            time.sleep(frequency)  # Poll job statuses every 120 seconds

    return failed_jobs


def _dump_configs(benchmark_dir: str, configs: dict, file_name: str):
    os.makedirs(benchmark_dir, exist_ok=True)
    config_path = os.path.join(benchmark_dir, file_name)
    with open(config_path, "w") as file:
        yaml.dump(configs, file)
        logger.info(f"Configs have been saved to {config_path}")
    return config_path


def _get_git_info(git_uri_branch: str):
    git_info = git_uri_branch.split("#")
    if len(git_info) == 2:
        git_uri, git_branch = git_info
    elif len(git_info) == 1:
        git_uri = git_info[0]
        git_branch = "stable"
    return git_uri, git_branch


def _is_mounted(path: str):
    with open("/proc/mounts", "r") as mounts:
        for line in mounts:
            parts = line.split()
            if len(parts) > 1 and parts[1] == path:
                return True


def _umount_if_needed(path: str = None):
    if not path:
        return
    if _is_mounted(path):
        logging.info(f"{path} is a mount point, attempting to umount.")
        subprocess.run(["sudo", "umount", path])
        logging.info(f"Successfully umounted {path}.")


def _mount_dir(orig_path: str, new_path: str):
    logging.info(f"Mounting from {orig_path} to {new_path}.")
    subprocess.run(["sudo", "mount", "--bind", orig_path, new_path])


def update_custom_dataloader(configs: dict):
    custom_dataloader_file = configs["custom_dataloader"]["dataloader_file"]
    original_path = os.path.dirname(custom_dataloader_file)
    custom_dataset_config = configs["custom_dataloader"]["dataset_config_file"]
    if original_path != os.path.dirname(custom_dataset_config):
        raise ValueError(
            "Custom dataloader dataset definition <config_file> and dataloader definition <file_path> need to be placed under the same parent directory."
        )
    dataloader_file_name = os.path.basename(custom_dataloader_file)
    dataset_config_file_name = os.path.basename(custom_dataset_config)
    current_path = os.path.dirname(os.path.abspath(__file__))
    custom_dataloader_path = os.path.join(current_path, "custom_configs", "dataloaders")

    configs["custom_dataloader"]["dataloader_file"] = f"custom_configs/dataloaders/{dataloader_file_name}"
    configs["custom_dataloader"]["dataset_config_file"] = f"custom_configs/dataloaders/{dataset_config_file_name}"

    return original_path, custom_dataloader_path


def update_custom_metrics(configs: dict):
    custom_metrics_path = configs["custom_metrics"]["metrics_path"]
    original_path = os.path.dirname(custom_metrics_path)

    metrics_file_name = os.path.basename(custom_metrics_path)
    current_path = os.path.dirname(os.path.abspath(__file__))
    custom_metrics_path = os.path.join(current_path, "custom_configs", "metrics")

    configs["custom_metrics"]["metrics_path"] = f"custom_configs/metrics/{metrics_file_name}"

    return original_path, custom_metrics_path


def get_resource(configs: dict, resource_name: str):
    ag_path = agbench_path[0]
    default_resource_file = os.path.join(ag_path, "resources", f"{resource_name}.yaml")
    with open(default_resource_file, "r") as f:
        resources = yaml.safe_load(f)

    current_path = os.getcwd()
    if configs.get("custom_resource_dir") is not None:
        custom_resource_dir = configs["custom_resource_dir"]
        resource_file = os.path.join(current_path, custom_resource_dir, f"{resource_name}.yaml")
        if os.path.exists(resource_file):
            with open(resource_file, "r") as f:
                resources = yaml.safe_load(f)
    return resources


def update_resource_constraint(configs: dict):
    constraint_name = configs.get("constraint", "test")
    constraints = get_resource(configs=configs, resource_name="multimodal_constraints")
    constraint_configs = constraints[constraint_name]
    configs["cdk_context"].update(constraint_configs)


def get_framework_configs(configs: dict):
    framework_name = configs.get("framework", "stable")
    frameworks = get_resource(configs=configs, resource_name="multimodal_frameworks")
    framework_configs = frameworks[framework_name]
    return framework_configs


@app.command()
def run(
    config_file: str = typer.Argument(..., help="Path to custom config file."),
    remove_resources: bool = typer.Option(False, help="Remove resources after run."),
    wait: bool = typer.Option(False, help="Whether to block and wait for the benchmark to finish, default to False."),
    skip_setup: bool = typer.Option(
        False, help="Whether to skip setting up framework in local mode, default to False."
    ),
    save_hardware_metrics: bool = typer.Option(False, help="Whether to query and save the hardware metrics."),
):
    """Main function that runs the benchmark based on the provided configuration options."""
    configs = {}
    if config_file.startswith("s3"):
        config_file = download_file_from_s3(s3_path=config_file)
    with open(config_file, "r") as f:
        configs = yaml.safe_load(f)
        if isinstance(configs, list) and os.environ.get(
            "AWS_BATCH_JOB_ARRAY_INDEX"
        ):  # AWS array job sets ARRAY_INDEX environment variable for each child job
            configs = configs[int(os.environ["AWS_BATCH_JOB_ARRAY_INDEX"])]

    benchmark_name = _get_benchmark_name(configs=configs)
    timestamp_pattern = r"\d{8}T\d{6}"  # Timestamp that matches YYYYMMDDTHHMMSS
    if not re.search(timestamp_pattern, benchmark_name):
        benchmark_name += "_" + formatted_time()

    root_dir = configs.get("root_dir", "ag_bench_runs")
    module = configs["module"]
    benchmark_dir = os.path.join(root_dir, module, benchmark_name)
    tmpdir = None

    if configs["mode"] == "aws":
        current_path = os.path.dirname(os.path.abspath(__file__))
        paths = []
        try:
            configs["benchmark_name"] = benchmark_name
            # setting environment variables for docker build ARG
            os.environ["AG_BENCH_VERSION"] = agbench_version

            os.environ["FRAMEWORK_PATH"] = f"frameworks/{module}/"

            if module in AMLB_DEPENDENT_MODULES:
                os.environ["AMLB_FRAMEWORK"] = configs["framework"]
                os.environ["GIT_URI"], os.environ["GIT_BRANCH"] = _get_git_info(configs["git_uri#branch"])

                if configs.get("amlb_constraint") is None:
                    configs["amlb_constraint"] = "test"

                amlb_user_dir = configs.get("amlb_user_dir")
                tmpdir = None
                if amlb_user_dir is not None:
                    if amlb_user_dir.startswith("s3://"):
                        tmpdir = tempfile.TemporaryDirectory()
                        amlb_user_dir_local = download_dir_from_s3(s3_path=amlb_user_dir, local_path=tmpdir.name)
                    else:
                        amlb_user_dir_local = amlb_user_dir

                    default_user_dir = "custom_configs/amlb_configs"
                    custom_configs_path = os.path.join(current_path, default_user_dir)
                    lambda_custom_configs_path = os.path.join(
                        current_path, "cloud/aws/batch_stack/lambdas", default_user_dir
                    )
                    original_path = amlb_user_dir_local
                    paths += [custom_configs_path, lambda_custom_configs_path]
                    for path in paths:
                        # mounting custom directory to a predefined directory in the package
                        # to make it available for Docker build
                        _umount_if_needed(path)
                        _mount_dir(orig_path=original_path, new_path=path)
                    os.environ["AMLB_USER_DIR"] = default_user_dir  # For Docker build
                    configs["amlb_user_dir"] = default_user_dir  # For Lambda job config
            elif module == "multimodal":
                if configs.get("custom_dataloader") is not None:
                    original_path, custom_dataloader_path = update_custom_dataloader(configs=configs)
                    paths.append(custom_dataloader_path)
                    _umount_if_needed(custom_dataloader_path)
                    _mount_dir(orig_path=original_path, new_path=custom_dataloader_path)

                if configs.get("custom_metrics") is not None:
                    original_path, custom_metrics_path = update_custom_metrics(configs=configs)
                    paths.append(custom_metrics_path)
                    _umount_if_needed(custom_metrics_path)
                    _mount_dir(orig_path=original_path, new_path=custom_metrics_path)

                update_resource_constraint(configs=configs)
                framework_configs = get_framework_configs(configs=configs)
                if configs.get("custom_resource_dir") is not None:
                    custom_resource_path = os.path.join(current_path, "custom_configs", "resources")
                    paths.append(custom_resource_path)
                    _umount_if_needed(custom_resource_path)
                    _mount_dir(orig_path=configs["custom_resource_dir"], new_path=custom_resource_path)
                    configs["custom_resource_dir"] = "custom_configs/resources"

                os.environ["GIT_URI"] = framework_configs["repo"]
                os.environ["GIT_BRANCH"] = framework_configs.get("version", "stable")

            infra_configs = deploy_stack(custom_configs=configs)

            cloud_config_path = _dump_configs(
                benchmark_dir=benchmark_dir, configs=configs, file_name=os.path.basename(config_file)
            )
            config_s3_path = upload_to_s3(
                s3_bucket=infra_configs["METRICS_BUCKET"],
                s3_dir=f"configs/{module}/{benchmark_name}",
                local_path=cloud_config_path,
            )

            response = invoke_lambda(configs=infra_configs, config_file=config_s3_path)

            job_configs = {
                "job_configs": response,
            }
            aws_configs = {**infra_configs, **job_configs}
            logger.info(f"Saving infra configs and submitted job configs under {benchmark_dir}.")
            aws_config_path = _dump_configs(
                benchmark_dir=benchmark_dir, configs=aws_configs, file_name="aws_configs.yaml"
            )

            if remove_resources:
                wait = True
            if wait or save_hardware_metrics:
                logger.info(
                    "Waiting for jobs to complete. You can quit at anytime and the benchmark will continue to run on the cloud"
                )
                if remove_resources:
                    logger.info(
                        "Resources will be deleted after the jobs are finished. You can also call \n"
                        f"`agbench destroy-stack --config-file {aws_config_path}` "
                        "to delete the stack after jobs have run to completion if you choose to quit now."
                    )
                time.sleep(10)  # wait for Batch to load
                failed_jobs = wait_for_jobs(config_file=aws_config_path)
                if len(failed_jobs) > 0:
                    logger.warning("Some jobs have failed: %s.", failed_jobs)
                    if remove_resources:
                        logger.warning("Resources will be kept so error logs can be accessed")
                elif failed_jobs is None:
                    if remove_resources:
                        logger.error("Resources are not being removed due to errors.")
                else:
                    logger.info("All job succeeded.")
                    if save_hardware_metrics:
                        get_hardware_metrics(
                            config_file=aws_config_path,
                            s3_bucket=infra_configs["METRICS_BUCKET"],
                            module=module,
                            benchmark_name=benchmark_name,
                        )
                    if remove_resources:
                        logger.info("Removing resoureces...")
                        destroy_stack(
                            static_resource_stack=infra_configs["STATIC_RESOURCE_STACK_NAME"],
                            batch_stack=infra_configs["BATCH_STACK_NAME"],
                            cdk_deploy_account=infra_configs["CDK_DEPLOY_ACCOUNT"],
                            cdk_deploy_region=infra_configs["CDK_DEPLOY_REGION"],
                            config_file=None,
                        )
                        logger.info("Resources removed successfully.")
        finally:
            for path in paths:
                _umount_if_needed(path)

    elif configs["mode"] == "local":
        split_id = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 0)
        benchmark_dir_s3 = f"{module}/{benchmark_name}"
        if split_id is not None:
            benchmark_dir_s3 += f"/{benchmark_name}_{split_id}"

        if module in AMLB_DEPENDENT_MODULES:
            amlb_user_dir = configs.get("amlb_user_dir")
            if amlb_user_dir and amlb_user_dir.startswith("s3://"):
                tmpdir = tempfile.TemporaryDirectory()
                configs["amlb_user_dir"] = download_dir_from_s3(s3_path=amlb_user_dir, local_path=tmpdir.name)

        logger.info(f"Running benchmark {benchmark_name} at {benchmark_dir}.")

        run_benchmark(
            benchmark_name=benchmark_name,
            benchmark_dir=benchmark_dir,
            configs=configs,
            benchmark_dir_s3=benchmark_dir_s3,
            skip_setup=skip_setup,
        )
    else:
        raise NotImplementedError

    if tmpdir:
        tmpdir.cleanup()
