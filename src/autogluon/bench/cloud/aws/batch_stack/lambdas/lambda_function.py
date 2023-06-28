import logging
import os
import uuid
from itertools import product

import boto3
import yaml

aws_batch = boto3.client("batch")
s3 = boto3.client("s3")

logger = logging.getLogger(__name__)


def submit_batch_job(env: list, job_name: str, job_queue: str, job_definition: str):
    """
    Submits a Batch job with the given environment variables, job name, job queue and job definition.

    Args:
        env (List[Dict[str, Any]]): List of dictionaries containing environment variables.
        job_name (str): Name of the job.
        job_queue (str): Name of the job queue.
        job_definition (str): Name of the job definition.

    Returns:
        str: Job ID.
    """
    container_overrides = {"environment": env}
    response = aws_batch.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides=container_overrides,
    )
    logger.info("Job %s submitted to AWS Batch queue %s.", job_name, job_queue)
    logger.info(response)
    return response["jobId"]


def download_config(s3_path: str):
    """
    Downloads a file from S3 bucket with the given path.

    Args:
        s3_path (str): Path of the file in the S3 bucket.

    Returns:
        str: Local path of the downloaded file.
    """
    file_path = os.path.join("/tmp", s3_path.split("/")[-1])
    bucket = s3_path.strip("s3://").split("/")[0]
    s3_path = s3_path.split(bucket)[-1].lstrip("/")
    s3.download_file(bucket, s3_path, file_path)
    return file_path


def upload_config(bucket: str, benchmark_name: str, file: str):
    """
    Uploads a file to the given S3 bucket.

    Args:
        bucket (str): Name of the bucket.
        file (str): Local path of the file to upload.

    Returns:
        str: S3 path of the uploaded file.
    """
    file_name = f'{file.split("/")[-1].split(".")[0]}.yaml'
    s3_path = f"configs/{benchmark_name}/{file_name}"
    s3.upload_file(file, bucket, s3_path)
    return f"s3://{bucket}/{s3_path}"


def save_configs(configs: dict, uid: str):
    """
    Saves the given dictionary of configs to a YAML file with the given UID as a part of the filename.

    Args:
        configs (Dict[str, Any]): Dictionary of configurations to be saved.
        uid (str): UID to be added to the filename of the saved file.

    Returns:
        str: Local path of the saved file.
    """
    benchmark_name = configs["benchmark_name"]
    config_file_path = os.path.join("/tmp", f"{benchmark_name}_split_{uid}.yaml")
    with open(config_file_path, "w+") as f:
        yaml.dump(configs, f, default_flow_style=False)
    return config_file_path


def process_combination(combination, keys, metrics_bucket, batch_job_queue, batch_job_definition):
    """
    Processes a combination of configurations by generating and submitting Batch jobs.

    Args:
        combination (Tuple): tuple of configurations to process.
        keys (List[str]): list of keys of the configurations.
        metrics_bucket (str): name of the bucket to upload metrics to.
        batch_job_queue (str): name of the Batch job queue to submit jobs to.
        batch_job_definition (str): name of the Batch job definition to use for submitting jobs.

    Returns:
        str: job id of the submitted batch job.
    """
    local_configs = dict(zip(keys, combination))
    config_uid = uuid.uuid1().hex
    config_local_path = save_configs(configs=local_configs, uid=config_uid)
    config_s3_path = upload_config(
        bucket=metrics_bucket, benchmark_name=local_configs["benchmark_name"], file=config_local_path
    )
    job_name = f"{local_configs['benchmark_name']}-{local_configs['module']}-{config_uid}"
    env = [{"name": "config_file", "value": config_s3_path}]

    job_id = submit_batch_job(
        env=env,
        job_name=job_name,
        job_queue=batch_job_queue,
        job_definition=batch_job_definition,
    )
    return job_id, config_s3_path


def handler(event, context):
    """
    Execution entrypoint for AWS Lambda.
    Triggers batch jobs with hyperparameter combinations.
    ENV variables are set by the AWS CDK infra code.

    Sample of cloud_configs.yaml to be supplied by user

    # Infra configurations
    cdk_context:
        CDK_DEPLOY_ACCOUNT: dummy
        CDK_DEPLOY_REGION: dummy

    # Benchmark configurations
    module: multimodal
    mode: aws
    benchmark_name: test_yaml
    metrics_bucket: autogluon-benchmark-metrics

    # Module specific configurations
    module_configs:
        # Multimodal specific
        multimodal:
            git_uri#branch: https://github.com/autogluon/autogluon#master
            dataset_name: melbourne_airbnb
            presets: medium_quality
            hyperparameters:
            optimization.learning_rate: 0.0005
            optimization.max_epochs: 5
            time_limit: 10


        # Tabular specific
        # You can refer to AMLB (https://github.com/openml/automlbenchmark#quickstart) for more details
        tabular:
            framework:
                - AutoGluon
            label:
                - stable
            amlb_benchmark:
                - test
                - small
            amlb_task:
                test: null
                small:
                    - credit-g
                    - vehicle
            amlb_constraint:
                - test
    """
    if "config_file" not in event or not event["config_file"].startswith("s3"):
        raise KeyError("S3 path of config file is required.")

    config_file_path = download_config(s3_path=event["config_file"])
    with open(config_file_path, "r") as f:
        configs = yaml.safe_load(f)

    metrics_bucket = configs["cdk_context"]["METRICS_BUCKET"]
    del configs["cdk_context"]

    batch_job_queue = os.environ.get("BATCH_JOB_QUEUE")
    batch_job_definition = os.environ.get("BATCH_JOB_DEFINITION")

    module_configs = configs["module_configs"].pop(configs["module"])
    del configs["module_configs"]
    common_configs = configs
    common_configs["METRICS_BUCKET"] = metrics_bucket
    common_configs["mode"] = "local"
    common_configs = {key: [value] if not isinstance(value, list) else value for key, value in common_configs.items()}

    if common_configs["module"][0] == "tabular":
        amlb_benchmarks = module_configs.pop("amlb_benchmark", [])
        amlb_tasks = module_configs.pop("amlb_task", {})

    # Generate all combinations and submit jobs
    job_configs = {}
    for common_combination in product(*[common_configs[key] for key in common_configs.keys()]):
        for module_combination in product(*[module_configs[key] for key in module_configs.keys()]):
            keys = list(common_configs.keys()) + list(module_configs.keys())
            combination = common_combination + module_combination

            if common_configs["module"][0] == "tabular":
                for amlb_benchmark in amlb_benchmarks:
                    amlb_task_values = amlb_tasks.get(amlb_benchmark)
                    extended_combination = combination + (amlb_benchmark, amlb_task_values)
                    extended_keys = keys + ["amlb_benchmark", "amlb_task"]
                    job_id, config_s3_path = process_combination(
                        extended_combination,
                        extended_keys,
                        metrics_bucket,
                        batch_job_queue,
                        batch_job_definition,
                    )
                    job_configs[job_id] = config_s3_path
            else:
                job_id, config_s3_path = process_combination(
                    combination,
                    keys,
                    metrics_bucket,
                    batch_job_queue,
                    batch_job_definition,
                )
                job_configs[job_id] = config_s3_path

    response = {
        "job_configs": job_configs,
    }
    return response
