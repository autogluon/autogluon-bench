import csv
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

import boto3
import pandas as pd
import typer
import yaml

metrics_list = []
aws_account_id = None
aws_account_region = None


def get_job_ids(config_file: str):
    job_ids = list(config_file.get("job_configs", {}).keys())
    return job_ids


def get_instance_id(job_id):
    batch_client = boto3.client("batch", region_name=f"{aws_account_region}")
    ecs_client = boto3.client("ecs", region_name=f"{aws_account_region}")

    response = batch_client.describe_jobs(jobs=[f"{job_id}"])
    if response:
        container_arn = response["jobs"][0]["container"]["containerInstanceArn"]
        cluster_arn = response["jobs"][0]["container"]["taskArn"].split("/")
        cluster = f"arn:aws:ecs:{aws_account_region}:{aws_account_id}:cluster/" + cluster_arn[1]

    response = ecs_client.describe_container_instances(cluster=cluster, containerInstances=[container_arn])
    instance_id = response["containerInstances"][0]["ec2InstanceId"]
    return instance_id


def get_instance_util(
    instance_id: str,
    metric: str,
    start_time: datetime,
    end_time: datetime,
    statistics: Optional[List[str]] = ["Average"],
) -> dict:
    cloudwatch_client = boto3.client("cloudwatch", region_name=f"{aws_account_region}")
    return cloudwatch_client.get_metric_statistics(
        Namespace="AWS/EC2",
        MetricName=metric,
        Dimensions=[
            {"Name": "InstanceId", "Value": instance_id},
        ],
        Statistics=statistics,
        StartTime=start_time,
        EndTime=end_time,
        Period=120,
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

    result_path = f"{module}/{benchmark_name}/{sub_folder}"
    path_prefix = f"s3://{s3_bucket}/{result_path}"
    global metrics_list
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
                instance_id,
                f"{metric}",
                utc_dt - timedelta(minutes=train_time),
                utc_dt - timedelta(minutes=predict_time),
            )
            predict_util = get_instance_util(
                instance_id, f"{metric}", utc_dt - timedelta(minutes=predict_time), utc_dt
            )
            # print(training_util, predict_util)
            if training_util["Datapoints"]:
                metrics_list.append(format_metrics(training_util, framework, dataset, fold, "Training"))
            if predict_util["Datapoints"]:
                metrics_list.append(format_metrics(predict_util, framework, dataset, fold, "Prediction"))


def results_to_csv():
    csv_headers = ["framework", "dataset", "mode", "fold", "metric", "statistic_type", "statistic_value", "unit"]
    file_dir = os.path.dirname(__file__)
    csv_location = os.path.join(file_dir, "hardware_metrics.csv")
    with open(csv_location, "w", newline="") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(metrics_list)


def get_hardware_metrics(
    config_file: str = typer.Argument(help="Path to YAML config file containing job ids."),
    s3_bucket: str = typer.Argument(help="Name of the S3 bucket to which the aggregated results will be outputted."),
    module: str = typer.Argument(help="Can be one of ['tabular', 'multimodal']."),
    benchmark_name: str = typer.Argument(
        help="Folder name of benchmark run in which all objects with path 'scores/results.csv' get aggregated."
    ),
):
    if not config_file:
        raise ValueError("Invalid Config File")
    logger.info(f"Getting hardware metrics for jobs under config file: {config_file}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    job_ids = get_job_ids(config)

    global metrics_list, aws_account_id, aws_account_region
    aws_account_id = config.get("CDK_DEPLOY_ACCOUNT")
    aws_account_region = config.get("CDK_DEPLOY_REGION")
    for job_id in job_ids:
        sub_folder = config["job_configs"][f"{job_id}"].split("/")[5].replace("_split", "").replace(".yaml", "")
        get_metrics(
            job_id, ["CPUUtilization", "EBSWriteOps", "EBSReadOps"], s3_bucket, module, benchmark_name, sub_folder
        )
    results_to_csv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    typer.run(get_hardware_metrics)
