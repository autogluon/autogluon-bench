import csv
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

import boto3
import pandas as pd
import typer
import yaml

aws_account_id = None
aws_account_region = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_job_ids(config_file: str):
    """
    This function returns a list of job IDs of all jobs ran for a benchmark run
    Parameters
    ----------
    config_file: str,
        Path to config file containing job IDs
    """
    job_ids = list(config_file.get("job_configs", {}).keys())
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
    period: int = 300,
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
    cloudwatch_client = boto3.client("cloudwatch", region_name=aws_account_region)
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
        Unit="Percent",
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
    metrics: list,
    s3_bucket: str,
    module: str,
    benchmark_name: str,
    sub_folder: str,
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
    """
    result_path = f"{module}/{benchmark_name}/{sub_folder}"
    path_prefix = f"s3://{s3_bucket}/{result_path}"
    metrics_list = []
    instance_id = get_instance_id(job_id)
    s3_path_to_csv = f"{path_prefix}/results.csv"
    results = pd.read_csv(s3_path_to_csv)
    for metric in metrics:
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
            training_util = get_instance_util(
                "AWS/EC2",
                instance_id,
                f"{metric}",
                utc_dt - timedelta(minutes=train_time),
                utc_dt - timedelta(minutes=predict_time),
            )
            predict_util = get_instance_util(
                "AWS/EC2", instance_id, f"{metric}", utc_dt - timedelta(minutes=predict_time), utc_dt
            )
            # print(training_util, predict_util)
            if training_util["Datapoints"]:
                metrics_list.append(format_metrics(training_util, framework, dataset, fold, "Training"))
            if predict_util["Datapoints"]:
                metrics_list.append(format_metrics(predict_util, framework, dataset, fold, "Prediction"))
    return metrics_list


def results_to_csv(metrics_list: list):
    """
    Writes the formatted dictionary of metrics to a csv to pass into `autogluon-dashboard`.
    Parameters
    ----------
    metrics_list: list,
        List of EC2 metrics to write to CSV
    """
    csv_headers = ["framework", "dataset", "mode", "fold", "metric", "statistic_type", "statistic_value", "unit"]
    file_dir = os.path.dirname(__file__)
    csv_location = os.path.join(file_dir, "hardware_metrics.csv")
    with open(csv_location, "w", newline="") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(metrics_list)


def get_hardware_metrics(
    config_file: str = typer.Argument(help="Path to YAML config file containing job ids."),
    s3_bucket: str = typer.Argument(help="Name of the S3 bucket to which the benchmark results were outputted."),
    module: str = typer.Argument(help="Can be one of ['tabular', 'multimodal']."),
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
    metrics_list = []
    for job_id in job_ids:
        sub_folder = config["job_configs"][f"{job_id}"].split("/")[5].replace("_split", "").replace(".yaml", "")
        metrics_list.append(
            get_metrics(
                job_id, ["CPUUtilization", "EBSWriteOps", "EBSReadOps"], s3_bucket, module, benchmark_name, sub_folder
            )
        )
    results_to_csv(metrics_list)
