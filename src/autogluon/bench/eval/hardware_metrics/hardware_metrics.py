import csv
import logging
import os
import tempfile
from datetime import datetime, timedelta
from typing import List, Optional

import boto3
import pandas as pd
import typer
import yaml

from autogluon.bench.utils.general_utils import upload_to_s3

aws_account_id = None
aws_account_region = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_s3_file(s3_bucket: str, prefix: str, file: str):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=s3_bucket, Prefix=prefix)

    for page in page_iterator:
        if "Contents" in page:
            for obj in page["Contents"]:
                if obj["Key"].endswith("results.csv"):
                    return f"s3://{s3_bucket}/{obj['Key']}"
    return None


def get_job_ids(config: dict):
    """
    This function returns a list of job IDs of all jobs ran for a benchmark run
    Parameters
    ----------
    config_file: str,
        Path to config file containing job IDs
    """
    job_ids = list(config.get("job_configs", {}).keys())
    return job_ids


def get_instance_id(job_id):
    """
    This function returns the instance ID (ARN) of the EC2 instance that was used to run a job with given job ID.
    Parameters
    ----------
    job_id: str
    """
    batch_client = boto3.client("batch", region_name=aws_account_region)
    ecs_client = boto3.client("ecs", region_name=aws_account_region)

    response = batch_client.describe_jobs(jobs=[f"{job_id}"])
    if response:
        container_arn = response["jobs"][0]["container"]["containerInstanceArn"]
        cluster_arn = response["jobs"][0]["container"]["taskArn"].split("/")
        cluster = f"arn:aws:ecs:{aws_account_region}:{aws_account_id}:cluster/" + cluster_arn[1]

    response = ecs_client.describe_container_instances(cluster=cluster, containerInstances=[container_arn])
    instance_id = response["containerInstances"][0]["ec2InstanceId"]
    return instance_id


def get_instance_util(
    namespace: str,
    instance_id: str,
    metric: str,
    start_time: datetime,
    end_time: datetime,
    cloudwatch_client: boto3.client,
    period: int = 360,
    statistics: Optional[List[str]] = ["Average"],
) -> dict:
    """
    This function returns the instance ID of the EC2 instance that was used to run a job with given job ID.
    Refer to https://docs.aws.amazon.com/cli/latest/reference/cloudwatch/get-metric-statistics.html for docs on how to interact with the CloudWatch API
    Also refer to https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cloudwatch/client/get_metric_statistics.html for documentation on how to interact with the API through Python
    Parameters
    ----------
    instance_id: str,
        EC2 instance ARN
    metric: str,
        Name of metric to pass into the CloudWatch API. Example: CPUUtilization
        Refer to https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/viewing_metrics_with_cloudwatch.html#ec2-cloudwatch-metrics
    start_time: datetime,
    end_time: datetime,
    statistics: Optional[List[str]] = ["Average"],
        The metric statistics, other than percentile. For percentile statistics, use `ExtendedStatistics` . When calling `get_metric_statistics` , you must specify either `Statistics` or `ExtendedStatistics` , but not both.
        Examples: Average, Maximum, Minimum
    """
    return cloudwatch_client.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric,
        Dimensions=[
            {"Name": "InstanceId", "Value": instance_id},
        ],
        Statistics=statistics,
        StartTime=start_time,
        EndTime=end_time,
        Period=period,
    )


def format_metrics(
    instance_metrics: dict,
    framework: str,
    dataset: str,
    fold: int,
    mode: str,
    statistics: Optional[List[str]] = ["Average"],
):
    """
    This function returns a formatted version of the dictionary of metrics provided by the CloudWatch API so it can be easily added to a CSV file and passed into `autogluon-dashboard`.
    Parameters
    ----------
    instance_metrics: dict,
        Dictionary of instance metrics for a given EC2 instance provided by CloudWatch
    framework: str,
        Name of the framework
    dataset: str,
        Name of the dataset
    fold: int,
        Fold #
    mode: str,
        Mode -> Training or Prediction
    statistics: Optional[List[str]] = ["Average"],
        The metric statistics, other than percentile. For percentile statistics, use `ExtendedStatistics` . When calling `get_metric_statistics` , you must specify either `Statistics` or `ExtendedStatistics` , but not both.
        Examples: Average, Maximum, Minimum
    """
    output_dict = {}
    output_dict["framework"] = framework
    output_dict["dataset"] = dataset
    output_dict["mode"] = mode
    output_dict["fold"] = fold
    output_dict["metric"] = instance_metrics["Label"]
    for i in range(len(instance_metrics["Datapoints"])):
        for stat in statistics:
            output_dict["framework"] = framework
            output_dict["dataset"] = dataset
            output_dict["mode"] = mode
            output_dict["fold"] = fold
            output_dict["metric"] = instance_metrics["Label"]
            output_dict["statistic_type"] = stat
            output_dict["statistic_value"] = instance_metrics["Datapoints"][i][f"{stat}"]
            output_dict["unit"] = instance_metrics["Datapoints"][i]["Unit"]
    return output_dict


def get_metrics(
    job_id: str,
    s3_bucket: str,
    module: str,
    benchmark_name: str,
    sub_folder: str,
    cloudwatch_client: boto3.client,
    namespace: str = "EC2",  # CloudWatch "Custom" namespace, i.e. Custom/EC2
):
    """
    Parameters
    ----------
    job_id: str,
    metrics: list,
        List of metrics to pass into the CloudWatch API. Example: CPUUtilization
        Refer to https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/viewing_metrics_with_cloudwatch.html#ec2-cloudwatch-metrics
    s3_bucket: str,
    module: str,
    benchmark_name: str,
    sub_folder: str,
        Sub folder for results.csv file.
        Passed in from `get_hardware_metrics` function
    namespace: str,
        CloudWatch Metrics Namespace, default: AWS/EC2
    """
    path_prefix = f"{module}/{benchmark_name}/{sub_folder}/"
    s3_path_to_csv = find_s3_file(s3_bucket=s3_bucket, prefix=path_prefix, file="results.csv")
    results = pd.read_csv(s3_path_to_csv)
    metrics_list = []
    instance_id = get_instance_id(job_id)
    metrics_data = cloudwatch_client.list_metrics(
        Dimensions=[
            {"Name": "InstanceId", "Value": instance_id},
        ],
        Namespace=namespace,
    )["Metrics"]
    metrics_pool = [item["MetricName"] for item in metrics_data]

    for metric in metrics_pool:
        for i in results.index:
            framework, dataset, utc, train_time, predict_time, fold = (
                results["framework"][i],
                results["task"][i],
                results["utc"][i],
                results["training_duration"][i],
                results["predict_duration"][i],
                results["fold"][i],
            )
            utc_dt = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%S")
            period = int((timedelta(seconds=train_time) + timedelta(seconds=predict_time)).total_seconds())
            if period < 60:
                period = 60
            elif period % 60 != 0:
                period = (period // 60) * 60  # Round down to the nearest multiple of 60

            training_util = get_instance_util(
                namespace=namespace,
                instance_id=instance_id,
                metric=metric,
                start_time=utc_dt,
                end_time=utc_dt + timedelta(seconds=train_time) + timedelta(seconds=predict_time),
                period=period,
                cloudwatch_client=cloudwatch_client,
            )
            predict_util = get_instance_util(
                namespace=namespace,
                instance_id=instance_id,
                metric=metric,
                start_time=utc_dt - timedelta(minutes=predict_time),
                end_time=utc_dt,
                period=period,
                cloudwatch_client=cloudwatch_client,
            )
            if training_util["Datapoints"]:
                metrics_list.append(format_metrics(training_util, framework, dataset, fold, "Training"))
            if predict_util["Datapoints"]:
                metrics_list.append(format_metrics(predict_util, framework, dataset, fold, "Prediction"))
    return metrics_list


def save_results(metrics_list: list, path: str):
    """
    Writes the formatted dictionary of metrics to a csv to pass into `autogluon-dashboard`.
    Parameters
    ----------
    metrics_list: list,
        List of hardware metrics to write to CSV
    path: str:
        Path to save file
    """
    csv_headers = ["framework", "dataset", "mode", "fold", "metric", "statistic_type", "statistic_value", "unit"]
    csv_location = os.path.join(path, "hardware_metrics.csv")
    with open(csv_location, "w", newline="") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(metrics_list)
    return csv_location


def get_hardware_metrics(
    config_file: str = typer.Argument(help="Path to YAML config file containing job ids."),
    s3_bucket: str = typer.Argument(help="Name of the S3 bucket to which the benchmark results were outputted."),
    module: str = typer.Argument(help="Can be one of ['tabular', 'timeseries', 'multimodal']."),
    benchmark_name: str = typer.Argument(
        help="Folder name of benchmark run in which all objects with path 'scores/results.csv' get aggregated."
    ),
):
    """
    External API function to interact with the script.
    Parameters
    ----------
    config_file: str,
        Path to config file containing job IDs
    s3_bucket: str,
        Name of the S3 bucket to which the benchmark results were outputted.
    module: str,
        Benchmark module: tabular or multimodal
    benchmark_name: str,
        Name of the benchmark
        Example: ag_bench_20230817T123456
    """
    if not config_file:
        raise ValueError("Invalid Config File")
    logger.info(f"Getting hardware metrics for jobs under config file: {config_file}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    job_ids = get_job_ids(config)

    global aws_account_id, aws_account_region
    aws_account_id = config.get("CDK_DEPLOY_ACCOUNT")
    aws_account_region = config.get("CDK_DEPLOY_REGION")

    cloudwatch_client = boto3.client("cloudwatch", region_name=aws_account_region)
    batch_client = boto3.client("batch", region_name=aws_account_region)

    metrics_list = []
    for job_id in job_ids:
        response = batch_client.describe_jobs(jobs=[job_id])
        job_detail = response["jobs"][0]

        # Check if the job is an array job
        if "arrayProperties" in job_detail and "size" in job_detail["arrayProperties"]:
            size = job_detail["arrayProperties"]["size"]
            sub_ids = [f"{job_id}:{i}" for i in range(size)]
        else:
            sub_ids = [job_id]

        for sub_id in sub_ids:
            id = sub_id.split(":")[-1]
            if id != "":
                id = "_" + id
            sub_folder = f"{benchmark_name}{id}"
            metrics_list += get_metrics(
                job_id=sub_id,
                s3_bucket=s3_bucket,
                module=module,
                benchmark_name=benchmark_name,
                sub_folder=sub_folder,
                cloudwatch_client=cloudwatch_client,
            )
    if metrics_list:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = save_results(metrics_list, temp_dir)
            upload_to_s3(s3_bucket=s3_bucket, s3_dir=f"{module}/{benchmark_name}", local_path=local_path)
