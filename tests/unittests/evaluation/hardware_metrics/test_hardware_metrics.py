import datetime
import os
import unittest
from unittest.mock import ANY, MagicMock, call, patch

import pandas as pd
import yaml

from autogluon.bench.eval.hardware_metrics import hardware_metrics
from autogluon.bench.eval.hardware_metrics.hardware_metrics import (
    get_hardware_metrics,
    get_instance_id,
    get_instance_util,
    get_job_ids,
    get_metrics,
)

test_dir = os.path.dirname(__file__)
config_file = os.path.join(test_dir, "resources/test_config.yaml")
if not config_file:
    raise ValueError("Invalid Config File")
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

hardware_metrics.aws_account_id = config["CDK_DEPLOY_ACCOUNT"]
hardware_metrics.aws_account_region = config["CDK_DEPLOY_REGION"]

from resources.expected_metrics import metrics

mock_cloudwatch_response = {
    "Label": "CPUUtilization",
    "Datapoints": [
        {"Timestamp": datetime.datetime(2023, 7, 12, 17, 39), "Average": 11.472356376239336, "Unit": "Percent"}
    ],
    "ResponseMetadata": {
        "RequestId": "93ed0de6-7f3c-4af8-8650-2310042c97f8",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "93ed0de6-7f3c-4af8-8650-2310042c97f8",
            "content-type": "text/xml",
            "content-length": "512",
            "date": "Wed, 12 Jul 2023 18:19:56 GMT",
        },
        "RetryAttempts": 0,
    },
}

mock_results_df = pd.read_csv(os.path.join(test_dir, "resources/results.csv"))


class TestHardwareMetrics(unittest.TestCase):
    def test_get_job_ids(self):
        job_ids = get_job_ids(config)
        self.assertEqual(job_ids, ["123456-abc-efg", "010101-xxx-zzz"])

    @patch("boto3.client")
    def test_get_instance_id(self, mock_client):
        job_ids = get_job_ids(config)
        batch_client, ecs_client = MagicMock(), MagicMock()
        mock_client.side_effect = [batch_client, ecs_client]
        mock_batch_response = {
            "jobs": [
                {
                    "container": {
                        "containerInstanceArn": "abc",
                        "taskArn": "arn:aws:ecs:us-east-2:123456789:task/agbenchcomputeenvironmen-DhbZ6yaLr_Batch/b3bb44aa78f",
                    }
                }
            ]
        }
        batch_client.describe_jobs.return_value = mock_batch_response

        mock_ecs_response = {"containerInstances": [{"ec2InstanceId": 12345}]}
        ecs_client.describe_container_instances.return_value = mock_ecs_response
        instance_id = get_instance_id(job_ids[0])
        self.assertEqual(instance_id, 12345)
        cluster = f"arn:aws:ecs:{hardware_metrics.aws_account_region}:{hardware_metrics.aws_account_id}:cluster/agbenchcomputeenvironmen-DhbZ6yaLr_Batch"
        ecs_client.describe_container_instances.assert_called_once_with(cluster=cluster, containerInstances=["abc"])

    @patch("boto3.client")
    def test_get_instance_util(self, mock_client):
        cloudwatch_client = MagicMock()
        mock_client.side_effect = [cloudwatch_client]
        cloudwatch_client.get_metric_statistics.return_value = mock_cloudwatch_response
        self.assertEqual(
            get_instance_util(
                namespace="namespace",
                instance_id="1234",
                metric="CPUUtilization",
                start_time=datetime.datetime(2023, 7, 12, 17, 39),
                end_time=datetime.datetime(2023, 7, 12, 16, 39),
                cloudwatch_client=cloudwatch_client,
            ),
            cloudwatch_client.get_metric_statistics.return_value,
        )

    @patch("pandas.read_csv")
    @patch("autogluon.bench.eval.hardware_metrics.hardware_metrics.get_instance_id")
    @patch("autogluon.bench.eval.hardware_metrics.hardware_metrics.get_instance_util")
    @patch("autogluon.bench.eval.hardware_metrics.hardware_metrics.find_s3_file")
    def test_get_metrics(self, mock_s3_file, mock_instance_util, mock_instance_id, mock_csv):
        mock_csv.return_value = mock_results_df
        mock_instance_id.return_value = "12345"
        job_id = list(config.get("job_configs", {}).keys())[0]
        mock_instance_util.return_value = mock_cloudwatch_response
        mock_list_metrics_return = {
            "Metrics": [
                {"MetricName": "CPUUtilization"},
            ]
        }
        mock_cloudwatch_client = MagicMock()
        mock_cloudwatch_client.list_metrics.return_value = mock_list_metrics_return

        metrics_list = get_metrics(
            job_id=job_id,
            s3_bucket="some bucket",
            module="tabular",
            benchmark_name="some_benchmark",
            sub_folder="test_folder",
            cloudwatch_client=mock_cloudwatch_client,
        )
        self.assertEqual(metrics_list, metrics)

    @patch("tempfile.TemporaryDirectory")
    @patch("autogluon.bench.eval.hardware_metrics.hardware_metrics.save_results")
    @patch("autogluon.bench.eval.hardware_metrics.hardware_metrics.upload_to_s3")
    @patch("autogluon.bench.eval.hardware_metrics.hardware_metrics.get_metrics")
    @patch("boto3.client")
    def test_get_hardware_metrics(self, mock_boto3, mock_metrics, mock_upload_to_s3, mock_save_results, mock_temp_dir):
        mock_metrics.return_value = ["metrics"]
        get_hardware_metrics(config_file, "some bucket", "tabular", "some_benchmark")
        job_ids = get_job_ids(config)
        calls = [
            call(
                job_id=job_ids[0],
                s3_bucket="some bucket",
                module="tabular",
                benchmark_name="some_benchmark",
                sub_folder="some_benchmark_123456-abc-efg",
                cloudwatch_client=ANY,
            ),
            call(
                job_id=job_ids[1],
                s3_bucket="some bucket",
                module="tabular",
                benchmark_name="some_benchmark",
                sub_folder="some_benchmark_010101-xxx-zzz",
                cloudwatch_client=ANY,
            ),
        ]
        mock_metrics.assert_has_calls(calls, any_order=False)
        mock_save_results.assert_called_once()
        mock_upload_to_s3.assert_called_once()

    def test_invalid_config_file(self):
        with self.assertRaises(ValueError):
            get_hardware_metrics(None, "some bucket", "tabular", "some_benchmark")
        with self.assertRaises(FileNotFoundError):
            get_hardware_metrics("incorrect config path", "some bucket", "tabular", "some_benchmark")


if __name__ == "__main__":
    unittest.main()
