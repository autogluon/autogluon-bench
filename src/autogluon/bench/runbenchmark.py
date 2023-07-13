import json
import logging
import os
import re
import time
from typing import List, Optional

import boto3
import botocore
import typer
import yaml

from autogluon.bench import __version__ as agbench_version
from autogluon.bench.cloud.aws.stack_handler import deploy_stack, destroy_stack
from autogluon.bench.frameworks.multimodal.multimodal_benchmark import MultiModalBenchmark
from autogluon.bench.frameworks.tabular.tabular_benchmark import TabularBenchmark
from autogluon.bench.utils.general_utils import formatted_time

app = typer.Typer()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_kwargs(module: str, configs: dict, agbench_dev_url: str):
    """Returns a dictionary of keyword arguments to be used for setting up and running the benchmark.

    Args:
        module (str): The name of the module to benchmark (either "multimodal" or "tabular").
        configs (dict): A dictionary of configuration options for the benchmark.

    Returns:
        A dictionary containing the keyword arguments to be used for setting up and running the benchmark.
    """

    if module == "multimodal":
        git_uri, git_branch = configs["git_uri#branch"].split("#")
        return {
            "setup_kwargs": {
                "git_uri": git_uri,
                "git_branch": git_branch,
                "agbench_dev_url": agbench_dev_url,
            },
            "run_kwargs": {
                "dataset_name": configs["dataset_name"],
                "framework": configs["git_uri#branch"].split("/")[-1],
                "presets": configs.get("presets"),
                "hyperparameters": configs.get("hyperparameters"),
                "time_limit": configs.get("time_limit"),
            },
        }
    elif module == "tabular":
        return {
            "setup_kwargs": {},
            "run_kwargs": {
                "benchmark": configs["amlb_benchmark"],
                "constraint": configs["amlb_constraint"],
                "task": configs.get("amlb_task") if configs.get("amlb_task") else None,
                "framework": configs.get("framework"),
                "custom_branch": configs.get("amlb_custom_branch"),
            },
        }


def _get_benchmark_name(configs: dict) -> str:
    default_benchmark_name = "ag_bench"
    benchmark_name = configs.get("benchmark_name", default_benchmark_name) or default_benchmark_name
    return benchmark_name


def run_benchmark(benchmark_name: str, benchmark_dir: str, configs: dict, agbench_dev_url: str = None):
    """Runs a benchmark based on the provided configuration options.

    Args:
        configs (dict): A dictionary of configuration options for the benchmark.

    Returns:
        None
    """

    module_to_benchmark = {
        "multimodal": MultiModalBenchmark,
        "tabular": TabularBenchmark,
    }
    module_name = configs["module"]

    benchmark_class = module_to_benchmark.get(module_name, None)
    if benchmark_class is None:
        raise NotImplementedError

    benchmark = benchmark_class(benchmark_name=benchmark_name, benchmark_dir=benchmark_dir)

    module_kwargs = get_kwargs(module=module_name, configs=configs, agbench_dev_url=agbench_dev_url)
    benchmark.setup(**module_kwargs.get("setup_kwargs", {}))
    benchmark.run(**module_kwargs.get("run_kwargs", {}))
    logger.info(f"Backing up benchmarking configs to {benchmark.metrics_dir}/configs.yaml")
    _dump_configs(benchmark_dir=benchmark.metrics_dir, configs=configs, file_name="configs.yaml")

    if configs.get("METRICS_BUCKET", None):
        s3_dir = f"{module_name}{benchmark_dir.split(module_name, 1)[-1]}"
        benchmark.upload_metrics(s3_bucket=configs["METRICS_BUCKET"], s3_dir=s3_dir)


def upload_config(bucket: str, benchmark_name: str, file: str):
    """Uploads a configuration file to an S3 bucket.

    Args:
        bucket (str): The name of the S3 bucket to upload the file to.
        file (str): The path to the local file to upload.

    Returns:
        The S3 path of the uploaded file.
    """
    s3 = boto3.client("s3")
    config_file_name = os.path.basename(file)
    file_name = benchmark_name + "_" + config_file_name

    s3_path = f"configs/{file_name}"
    s3.upload_file(file, bucket, s3_path)
    logger.info(f"Config file has been uploaded to S3://{bucket}/{s3_path}")
    return f"s3://{bucket}/{s3_path}"


def download_config(s3_path: str, dir: str = "/tmp"):
    """Downloads a configuration file from an S3 bucket.

    Args:
        s3_path (str): The S3 path of the file to download.
        dir (str): The local directory to download the file to (default: "/tmp").

    Returns:
        The local path of the downloaded file.
    """

    s3 = boto3.client("s3")
    file_path = os.path.join(dir, s3_path.split("/")[-1])
    bucket = s3_path.strip("s3://").split("/")[0]
    s3_path = s3_path.split(bucket)[-1].lstrip("/")
    s3.download_file(bucket, s3_path, file_path)
    return file_path


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
    response = json.loads(response["Payload"].read().decode("utf-8"))
    logger.info("AWS Batch jobs submitted by %s.", configs["LAMBDA_FUNCTION_NAME"])

    return response


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
        job = response["jobs"][0]
        status_dict[job_id] = job["status"]

    logger.info(status_dict)
    return status_dict


def wait_for_jobs_to_complete(
    config_file: Optional[str] = None, job_ids: Optional[List[str]] = None, aws_region: Optional[str] = None
):
    while True:
        all_jobs_completed = True
        failed_jobs = []

        try:
            job_status = get_job_status(job_ids=job_ids, cdk_deploy_region=aws_region, config_file=config_file)

            for job_id, job_status in job_status.items():
                if job_status == "FAILED":
                    failed_jobs.append(job_id)
                elif job_status not in ["SUCCEEDED", "FAILED"]:
                    all_jobs_completed = False
        except botocore.exceptions.ClientError as e:
            logger.error(f"An error occurred: {e}.")
            return

        if all_jobs_completed:
            break
        else:
            time.sleep(120)  # Poll job statuses every 60 seconds

    return failed_jobs


def _get_split_id(file_name: str):
    if "split" in file_name:
        file_name = os.path.basename(file_name)
        match = re.search(r"([a-f0-9]{32})", file_name)
        if match:
            return match.group(1)
        else:
            return None

    return None


def _dump_configs(benchmark_dir: str, configs: dict, file_name: str):
    os.makedirs(benchmark_dir, exist_ok=True)
    config_path = os.path.join(benchmark_dir, file_name)
    with open(config_path, "w") as file:
        yaml.dump(configs, file)
        logger.info(f"Configs have been saved to {config_path}")
    return config_path


@app.command()
def run(
    config_file: str = typer.Argument(..., help="Path to custom config file."),
    remove_resources: bool = typer.Option(False, help="Remove resources after run."),
    wait: bool = typer.Option(False, help="Whether to block and wait for the benchmark to finish."),
    dev_branch: Optional[str] = typer.Option(None, help="Path to a development AutoGluon-Bench branch."),
):
    """Main function that runs the benchmark based on the provided configuration options."""
    configs = {}
    if config_file.startswith("s3"):
        config_file = download_config(s3_path=config_file)
    with open(config_file, "r") as f:
        configs = yaml.safe_load(f)

    benchmark_name = _get_benchmark_name(configs=configs)
    timestamp_pattern = r"\d{8}T\d{6}"  # Timestamp that matches YYYYMMDDTHHMMSS
    if not re.search(timestamp_pattern, benchmark_name):
        benchmark_name += "_" + formatted_time()

    root_dir = configs.get("root_dir", ".ag_bench_runs")
    benchmark_dir = os.path.join(root_dir, configs["module"], benchmark_name)

    if configs["mode"] == "aws":
        configs["benchmark_name"] = benchmark_name
        cloud_config_path = _dump_configs(
            benchmark_dir=benchmark_dir, configs=configs, file_name=os.path.basename(config_file)
        )
        if dev_branch is not None:
            os.environ["AG_BENCH_DEV_URL"] = dev_branch  # pull dev branch from GitHub
        else:
            os.environ["AG_BENCH_VERSION"] = agbench_version  # set the installed version for Dockerfile to align with
        infra_configs = deploy_stack(custom_configs=configs.get("cdk_context", {}))
        config_s3_path = upload_config(
            bucket=infra_configs["METRICS_BUCKET"], benchmark_name=benchmark_name, file=cloud_config_path
        )
        lambda_response = invoke_lambda(configs=infra_configs, config_file=config_s3_path)
        aws_configs = {**infra_configs, **lambda_response}
        logger.info(f"Saving infra configs and submitted job configs under {benchmark_dir}.")
        aws_config_path = _dump_configs(benchmark_dir=benchmark_dir, configs=aws_configs, file_name="aws_configs.yaml")

        if remove_resources:
            wait = True
        if wait:
            logger.info(
                "Waiting for jobs to complete. You can quit at anytime and the benchmark will continue to run on the cloud"
            )
            if remove_resources:
                logger.info(
                    "Resources will be deleted after the jobs are finished. You can also call \n"
                    f"`agbench destroy-stack --config-file {aws_config_path}` "
                    "to delete the stack after jobs have run to completion if you choose to quit now."
                )

            failed_jobs = wait_for_jobs_to_complete(config_file=aws_config_path)
            if len(failed_jobs) > 0:
                logger.warning("Some jobs have failed: %s.", failed_jobs)
                if remove_resources:
                    logger.warning("Resources will be kept so error logs can be accessed")
            elif failed_jobs is None:
                if remove_resources:
                    logger.error("Resources are not being removed due to errors.")
            else:
                logger.info("All job succeeded.")
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

    elif configs["mode"] == "local":
        split_id = _get_split_id(config_file)
        if split_id is not None:
            benchmark_name += "_" + split_id
            benchmark_dir = os.path.join(benchmark_dir, benchmark_name)
        logger.info(f"Running benchmark {benchmark_name} at {benchmark_dir}.")
        if dev_branch is not None:
            logger.info(f"Using dev branch at {dev_branch}...")
        run_benchmark(
            benchmark_name=benchmark_name, benchmark_dir=benchmark_dir, configs=configs, agbench_dev_url=dev_branch
        )
    else:
        raise NotImplementedError
