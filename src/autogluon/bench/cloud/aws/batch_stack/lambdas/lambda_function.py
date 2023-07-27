import io
import itertools
import logging
import os
import uuid
import zipfile

import requests
import yaml
from boto3 import client

aws_batch = client("batch")
s3 = client("s3")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def process_combination(configs, metrics_bucket, batch_job_queue, batch_job_definition):
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
    logger.info(f"Generating config with: {configs}")
    config_uid = uuid.uuid1().hex
    config_local_path = save_configs(configs=configs, uid=config_uid)
    config_s3_path = upload_config(
        bucket=metrics_bucket, benchmark_name=configs["benchmark_name"], file=config_local_path
    )
    job_name = f"{configs['benchmark_name']}-{configs['module']}-{config_uid}"
    env = [{"name": "config_file", "value": config_s3_path}]

    job_id = submit_batch_job(
        env=env,
        job_name=job_name,
        job_queue=batch_job_queue,
        job_definition=batch_job_definition,
    )
    return job_id, config_s3_path


def generate_config_combinations(config, metrics_bucket, batch_job_queue, batch_job_definition):
    job_configs = {}
    config
    if config["module"] == "tabular":
        job_configs = generate_tabular_config_combinations(
            config, metrics_bucket, batch_job_queue, batch_job_definition
        )
    elif config["module"] == "multimodal":
        job_configs = generate_multimodal_config_combinations(
            config, metrics_bucket, batch_job_queue, batch_job_definition
        )
    else:
        raise ValueError("Invalid module. Choose either 'tabular' or 'multimodal'.")

    response = {
        "job_configs": job_configs,
    }
    return response


def generate_multimodal_config_combinations(config, metrics_bucket, batch_job_queue, batch_job_definition):
    common_keys = ["module", "mode", "benchmark_name", "root_dir", "METRICS_BUCKET"]
    specific_keys = ["git_uri#branch", "dataset_name", "presets", "hyperparameters", "time_limit"]

    job_configs = {}
    specific_value_combinations = list(
        itertools.product(
            *(
                config["module_configs"]["multimodal"][key]
                for key in specific_keys
                if key in config["module_configs"]["multimodal"]
            )
        )
    )

    for combo in specific_value_combinations:
        new_config = {key: config[key] for key in common_keys}
        new_config.update(dict(zip(specific_keys, combo)))

        job_id, config_s3_path = process_combination(new_config, metrics_bucket, batch_job_queue, batch_job_definition)
        job_configs[job_id] = config_s3_path

    return job_configs


def generate_tabular_config_combinations(config, metrics_bucket, batch_job_queue, batch_job_definition):
    common_keys = ["module", "mode", "benchmark_name", "root_dir", "METRICS_BUCKET"]
    specific_keys = ["git_uri#branch", "framework", "amlb_constraint", "amlb_user_dir"]

    job_configs = {}

    # Generate combinations for the specific keys
    specific_value_combinations = list(
        itertools.product(
            *(
                config["module_configs"]["tabular"][key]
                for key in specific_keys
                if key in config["module_configs"]["tabular"]
            )
        )
    )

    # Iterate through the combinations and the amlb benchmark task keys
    # Generates a config for each combination of specific key and keys in `fold_to_run`
    for combo in specific_value_combinations:
        for benchmark, tasks in config["module_configs"]["tabular"]["fold_to_run"].items():
            for task, fold_numbers in tasks.items():
                for fold_num in fold_numbers:
                    new_config = {key: config[key] for key in common_keys}
                    new_config.update(dict(zip(specific_keys, combo)))
                    new_config["amlb_benchmark"] = benchmark
                    new_config["amlb_task"] = task
                    new_config["fold_to_run"] = fold_num
                    job_id, config_s3_path = process_combination(
                        new_config, metrics_bucket, batch_job_queue, batch_job_definition
                    )
                    job_configs[job_id] = config_s3_path
    return job_configs


def _validate_single_value(configs: dict, key: str):
    value = configs[key]
    if isinstance(value, str):
        configs[key] = [value]
    elif isinstance(value, list) and len(value) != 1:
        raise ValueError(f"Only single value (str, list[str]) is supported for {key}.")


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

    config_file_path = download_file_from_s3(s3_path=event["config_file"], local_path="/tmp")
    with open(config_file_path, "r") as f:
        configs = yaml.safe_load(f)

    metrics_bucket = configs["cdk_context"]["METRICS_BUCKET"]

    batch_job_queue = os.environ.get("BATCH_JOB_QUEUE")
    batch_job_definition = os.environ.get("BATCH_JOB_DEFINITION")

    module_configs = configs["module_configs"][configs["module"]]
    configs["METRICS_BUCKET"] = metrics_bucket
    configs["mode"] = "local"

    if configs["module"] == "tabular":
        # download the almb repo resources/ to process the default resources
        amlb_repo_path = download_automlbenchmark_resources()
        amlb_benchmark_search_dirs = []
        amlb_constraint_search_files = []
        amlb_user_dir = module_configs.get("amlb_user_dir")
        if amlb_user_dir is not None:
            _validate_single_value(module_configs, "amlb_user_dir")
            if amlb_user_dir[0].startswith("s3://"):
                amlb_user_dir_local = download_dir_from_s3(
                    s3_path=amlb_user_dir[0], local_path=f"/tmp/amlb_custom_configs"
                )
            else:
                amlb_user_dir_local = amlb_user_dir[0]

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

        if module_configs.get("amlb_constraint"):
            _validate_single_value(configs=module_configs, key="amlb_constraint")
        else:
            module_configs["amlb_constraint"] = ["test"]
        fold_constraint = get_max_fold(
            amlb_constraint_search_files=amlb_constraint_search_files, constraint=module_configs["amlb_constraint"][0]
        )
        process_benchmark_runs(
            module_configs=module_configs,
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
