import io
import itertools
import logging
import os
import zipfile

import requests
import yaml
from boto3 import client

aws_batch = client("batch")
s3 = client("s3")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AMLB_DEPENDENT_MODULES = ["tabular", "timeseries"]


def submit_batch_job(env: list, job_name: str, job_queue: str, job_definition: str, array_size: int):
    """
    Submits a Batch job with the given environment variables, job name, job queue and job definition.

    Args:
        env (List[Dict[str, Any]]): List of dictionaries containing environment variables.
        job_name (str): Name of the job.
        job_queue (str): Name of the job queue.
        job_definition (str): Name of the job definition.
        array_size (int): Number of jobs to submit.

    Returns:
        str: Job ID.
    """
    container_overrides = {"environment": env}
    job_params = {
        "jobName": job_name,
        "jobQueue": job_queue,
        "jobDefinition": job_definition,
        "containerOverrides": container_overrides,
    }
    if array_size > 1:
        job_params["arrayProperties"] = {"size": array_size}

    response = aws_batch.submit_job(**job_params)

    logger.info("Job %s submitted to AWS Batch queue %s.", job_name, job_queue)
    logger.info(response)
    return response["jobId"]


def download_file_from_s3(s3_path: str, local_path: str) -> str:
    """Downloads a file from an S3 bucket.

    Args:
        s3_path (str): The S3 path of the file.
        local_path (str): The local path where the file will be downloaded.

    Returns:
        str: The local path of the downloaded file.
    """
    bucket = s3_path.strip("s3://").split("/")[0]
    s3_path = s3_path[len(f"s3://{bucket}/") :]

    local_file_path = os.path.join(local_path, s3_path.split("/")[-1])
    s3.download_file(bucket, s3_path, local_file_path)

    return local_file_path


def download_dir_from_s3(s3_path: str, local_path: str) -> str:
    """Downloads a directory from an S3 bucket.

    Args:
        s3_path (str): The S3 path of the directory.
        local_path (str): The local path where the directory will be downloaded.

    Returns:
        str: The local path of the downloaded directory.
    """
    bucket = s3_path.strip("s3://").split("/")[0]
    s3_path = s3_path[len(f"s3://{bucket}/") :]

    response = s3.list_objects(Bucket=bucket, Prefix=s3_path)

    for content in response.get("Contents", []):
        s3_obj_path = content["Key"]
        relative_path = os.path.relpath(s3_obj_path, s3_path)
        local_obj_path = os.path.join(local_path, relative_path)

        os.makedirs(os.path.dirname(local_obj_path), exist_ok=True)
        s3.download_file(bucket, s3_obj_path, local_obj_path)

    return local_path


def upload_config(config_list: list, bucket: str, benchmark_name: str):
    """
    Uploads a file to the given S3 bucket.

    Args:
        bucket (str): Name of the bucket.
        file (str): Local path of the file to upload.

    Returns:
        str: S3 path of the uploaded file.
    """
    s3_key = f"configs/{benchmark_name}/{benchmark_name}_job_configs.yaml"
    s3.put_object(Body=yaml.dump(config_list), Bucket=bucket, Key=s3_key)
    return f"s3://{bucket}/{s3_key}"


def download_automlbenchmark_resources():
    """
    Clones the stable version of the AutoML Benchmark repository from GitHub as a zip file,
    and extracts all files under 'resources' directory into a local path.

    Returns:
        str: The local path of the cloned repository.
    """

    amlb_zip = "https://github.com/openml/automlbenchmark/archive/refs/tags/stable.zip"
    resources_path = "automlbenchmark-stable/resources"
    automlbenchmark_repo_path = "/tmp"
    response = requests.get(amlb_zip)

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for filename in z.namelist():
            if filename.startswith(resources_path):
                destination_path = os.path.join(automlbenchmark_repo_path, filename)
                if filename.endswith("/"):
                    # This is a directory, need to create it
                    os.makedirs(destination_path, exist_ok=True)
                else:
                    # Extract the file to the destination path
                    with z.open(filename) as zf, open(destination_path, "wb") as f:
                        f.write(zf.read())
    return str(os.path.join(automlbenchmark_repo_path, "automlbenchmark-stable"))


def load_benchmark_from_yaml(filepath):
    with open(filepath, "r") as file:
        benchmark_yaml = yaml.safe_load(file)
    return [task["name"] for task in benchmark_yaml]


def find_benchmark_yaml(benchmark, search_dirs):
    for dir in search_dirs:
        filepath = os.path.join(dir, benchmark + ".yaml")
        if os.path.isfile(filepath):
            return filepath
    return None


def get_run_folds(file: str, default_max_folds: int = 10):
    configs = {}
    with open(file, "r") as f:
        amlb_benchmark_configs = yaml.safe_load(f)
        for item in amlb_benchmark_configs:
            folds = min(item.get("folds", default_max_folds), default_max_folds)
            configs[item["name"]] = [i for i in range(folds)]
    return configs


def get_max_fold(amlb_constraint_search_files: list, constraint: str):
    default_folds = 10
    for file in amlb_constraint_search_files:
        with open(file, "r") as f:
            constraints = yaml.safe_load(f)
            if constraint in constraints.keys():
                return constraints[constraint].get("folds", default_folds)
    return default_folds


def process_benchmark_runs(module_configs: dict, amlb_benchmark_search_dirs: list, default_max_folds: int = 10):
    """
    Updates module_configs["fold_to_run"] with tasks and folds defined in config files

    Only amlb_benchmark was required,
    if amlb_task does not have keys, we assume all tasks in the amlb_benchmark will be run;
    similarly, if folds_to_run does not have any key, we assume all folds in amlb_task will be run.
    If the aforementioned have keys, we only populate folds_to_run for the existing keys.
    """
    module_configs.setdefault("amlb_task", {})
    module_configs.setdefault("fold_to_run", {})

    amlb_task_folds = {}
    for benchmark in module_configs["amlb_benchmark"]:
        filepath = find_benchmark_yaml(benchmark, amlb_benchmark_search_dirs)
        if filepath is not None:
            amlb_task_folds[benchmark] = get_run_folds(file=filepath, default_max_folds=default_max_folds)
        else:
            raise FileNotFoundError(f"Benchmark config for {benchmark} is not found.")

        if module_configs["amlb_task"].get(benchmark) is None:
            module_configs["amlb_task"][benchmark] = list(amlb_task_folds[benchmark].keys())

        module_configs["fold_to_run"].setdefault(benchmark, {})
        for task in module_configs["amlb_task"][benchmark]:
            if module_configs["fold_to_run"][benchmark].get(task):
                tasks = module_configs["fold_to_run"][benchmark][task]
                module_configs["fold_to_run"][benchmark][task] = [t for t in tasks if t < default_max_folds]
            else:
                module_configs["fold_to_run"][benchmark][task] = amlb_task_folds[benchmark][task]


def get_cloudwatch_logs_url(region: str, job_id: str, log_group_name: str = "aws/batch/job"):
    base_url = f"https://console.aws.amazon.com/cloudwatch/home?region={region}"
    job_response = aws_batch.describe_job(jobs=[job_id])
    log_stream_name = job_response["jobs"][0]["attempts"][0]["container"]["logStreamName"]
    return f"{base_url}#logsV2:log-groups/log-group/{log_group_name.replace('/', '%2F')}/log-events/{log_stream_name.replace('/', '%2F')}"


def generate_config_combinations(config, metrics_bucket, batch_job_queue, batch_job_definition):
    job_configs = []
    if config["module"] in AMLB_DEPENDENT_MODULES:
        job_configs = generate_amlb_module_config_combinations(config)
    elif config["module"] == "multimodal":
        job_configs = generate_multimodal_config_combinations(config)
    else:
        raise ValueError("Invalid module. Choose either 'tabular', 'timeseries', or 'multimodal'.")

    benchmark_name = config["benchmark_name"]
    config_s3_path = upload_config(config_list=job_configs, bucket=metrics_bucket, benchmark_name=benchmark_name)
    env = [{"name": "config_file", "value": config_s3_path}]
    job_name = f"{benchmark_name}-array-job"
    parent_job_id = submit_batch_job(
        env=env,
        job_name=job_name,
        job_queue=batch_job_queue,
        job_definition=batch_job_definition,
        array_size=len(job_configs),
    )
    return {parent_job_id: config_s3_path}


def generate_multimodal_config_combinations(config):
    common_keys = []
    specific_keys = []
    for key in config.keys():
        if isinstance(config[key], list):
            specific_keys.append(key)
        else:
            common_keys.append(key)

    specific_value_combinations = list(
        itertools.product(*(config[key] for key in specific_keys if key in config.keys()))
    ) or [None]

    all_configs = []
    for combo in specific_value_combinations:
        new_config = {key: config[key] for key in common_keys}
        if combo is not None:
            new_config.update(dict(zip(specific_keys, combo)))
        all_configs.append(new_config)

    return all_configs


def generate_amlb_module_config_combinations(config):
    specific_keys = ["git_uri#branch", "framework", "amlb_constraint", "amlb_user_dir"]
    exclude_keys = ["amlb_benchmark", "amlb_task", "fold_to_run"]
    common_keys = []
    specific_keys = []
    for key in config.keys():
        if key in exclude_keys:
            continue

        if isinstance(config[key], list):
            specific_keys.append(key)
        else:
            common_keys.append(key)

    specific_value_combinations = list(
        itertools.product(*(config[key] for key in specific_keys if key in config.keys()))
    ) or [None]

    # Iterate through the combinations and the amlb benchmark task keys
    # Generates a config for each combination of specific key and keys in `fold_to_run`
    all_configs = []
    for combo in specific_value_combinations:
        for benchmark, tasks in config["fold_to_run"].items():
            for task, fold_numbers in tasks.items():
                for fold_num in fold_numbers:
                    new_config = {key: config[key] for key in common_keys}
                    if combo is not None:
                        new_config.update(dict(zip(specific_keys, combo)))
                    new_config["amlb_benchmark"] = benchmark
                    new_config["amlb_task"] = task
                    new_config["fold_to_run"] = fold_num
                    all_configs.append(new_config)

    return all_configs


def handler(event, context):
    """
    Execution entrypoint for AWS Lambda.
    Triggers batch jobs with hyperparameter combinations.
    ENV variables are set by the AWS CDK infra code.
    """
    if "config_file" not in event or not event["config_file"].startswith("s3"):
        raise KeyError("S3 path of config file is required.")

    config_file_path = download_file_from_s3(s3_path=event["config_file"], local_path="/tmp")
    with open(config_file_path, "r") as f:
        configs = yaml.safe_load(f)

    metrics_bucket = configs["cdk_context"]["METRICS_BUCKET"]

    batch_job_queue = os.environ.get("BATCH_JOB_QUEUE")
    batch_job_definition = os.environ.get("BATCH_JOB_DEFINITION")

    configs["METRICS_BUCKET"] = metrics_bucket
    configs["mode"] = "local"

    if configs["module"] in AMLB_DEPENDENT_MODULES:
        # download the almb repo resources/ to process the default resources
        amlb_repo_path = download_automlbenchmark_resources()
        amlb_benchmark_search_dirs = []
        amlb_constraint_search_files = []
        amlb_user_dir = configs.get("amlb_user_dir")
        if amlb_user_dir is not None:
            if amlb_user_dir.startswith("s3://"):
                amlb_user_dir_local = download_dir_from_s3(
                    s3_path=amlb_user_dir, local_path=f"/tmp/amlb_custom_configs"
                )
            else:
                amlb_user_dir_local = amlb_user_dir

            # check if default amlb resources are required
            user_config_file = os.path.join(amlb_user_dir_local, "config.yaml")
            with open(user_config_file, "r") as f:
                # check the user_dir config.yaml and append search directories accordingly
                user_configs = yaml.safe_load(f)
                if user_configs.get("benchmarks"):
                    if user_configs["benchmarks"].get("definition_dir"):
                        for definition_dir in user_configs["benchmarks"]["definition_dir"]:
                            if "{user}" in definition_dir:
                                ext = definition_dir.split("{user}")[-1].lstrip("/")
                                amlb_benchmark_search_dirs.append(os.path.join(amlb_user_dir_local, ext))
                            elif "{root}" in definition_dir:
                                ext = definition_dir.split("{root}")[-1].lstrip("/")
                                amlb_benchmark_search_dirs.append(os.path.join(amlb_repo_path, ext))
                    else:
                        amlb_benchmark_search_dirs.append(os.path.join(amlb_repo_path, "resources/benchmarks"))

                    if user_configs["benchmarks"].get("constraints_file"):
                        for constraints_file in user_configs["benchmarks"]["constraints_file"]:
                            if "{user}" in constraints_file:
                                ext = constraints_file.split("{user}")[-1].lstrip("/")
                                amlb_constraint_search_files.append(os.path.join(amlb_user_dir_local, ext))
                            elif "{root}" in constraints_file:
                                ext = constraints_file.split("{root}")[-1].lstrip("/")
                                amlb_constraint_search_files.append(os.path.join(amlb_repo_path, ext))
                    else:
                        amlb_constraint_search_files.append(os.path.join(amlb_repo_path, "constraints.yaml"))
        else:
            # if no amlb_user_dir is specified, default resources/ from amlb stable version is used
            amlb_benchmark_search_dirs.append(os.path.join(amlb_repo_path, "resources/benchmarks"))
            amlb_constraint_search_files.append(os.path.join(amlb_repo_path, "resources/constraints.yaml"))

        if configs.get("amlb_constraint") is None:
            configs["amlb_constraint"] = "test"
        fold_constraint = get_max_fold(
            amlb_constraint_search_files=amlb_constraint_search_files, constraint=configs["amlb_constraint"]
        )
        process_benchmark_runs(
            module_configs=configs,
            amlb_benchmark_search_dirs=amlb_benchmark_search_dirs,
            default_max_folds=fold_constraint,
        )

    response = generate_config_combinations(
        config=configs,
        metrics_bucket=metrics_bucket,
        batch_job_queue=batch_job_queue,
        batch_job_definition=batch_job_definition,
    )

    return response
