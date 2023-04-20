import json
import os
import unittest
from unittest.mock import mock_open, patch

from conftest import default_configs

from autogluon.bench.cloud.aws.run_deploy import construct_context, deploy_stack


class TestRunDeploy(unittest.TestCase):
    def test_construct_context(self):
        custom_configs = {"CDK_DEPLOY_ACCOUNT": "123456789012", "CDK_DEPLOY_REGION": "us-west-1", "PREFIX": "test"}

        with patch("builtins.open", mock_open(read_data=json.dumps(default_configs))) as mock_file:
            context = construct_context(custom_configs=custom_configs)

        prefix = custom_configs["PREFIX"]
        self.assertEqual(context["CDK_DEPLOY_ACCOUNT"], custom_configs["CDK_DEPLOY_ACCOUNT"])
        self.assertEqual(context["CDK_DEPLOY_REGION"], custom_configs["CDK_DEPLOY_REGION"])
        self.assertEqual(context["STACK_NAME_PREFIX"], prefix)
        self.assertEqual(context["METRICS_BUCKET"], default_configs["METRICS_BUCKET"])
        self.assertEqual(context["DATA_BUCKET"], default_configs["DATA_BUCKET"])
        self.assertEqual(context["VPC_NAME"], default_configs["VPC_NAME"])
        self.assertEqual(context["STATIC_RESOURCE_STACK_NAME"], f"{prefix}-static-resource-stack")
        self.assertEqual(context["BATCH_STACK_NAME"], f"{prefix}-batch-stack")

    @patch("autogluon.bench.cloud.aws.run_deploy.construct_context")
    @patch("subprocess.check_call")
    def test_deploy_stack(self, mock_check_call, mock_construct_context):
        mock_construct_context.return_value = {
            "STACK_NAME_PREFIX": "test-prefix",
            "STACK_NAME_TAG": "benchmark",
            "STATIC_RESOURCE_STACK_NAME": "test-prefix-static-resource-stack",
            "BATCH_STACK_NAME": "test-prefix-batch-stack",
            "CONTAINER_MEMORY": 10000,
        }

        deploy_stack()
        mock_construct_context.assert_called_once()
        mock_check_call.assert_called_once()
