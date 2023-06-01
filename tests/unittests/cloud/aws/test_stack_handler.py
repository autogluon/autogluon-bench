import importlib.resources
import json
import os
import unittest
from unittest import mock
from unittest.mock import mock_open, patch

from conftest import default_configs

from autogluon.bench.cloud.aws.stack_handler import (
    _get_temp_cdk_app_path,
    construct_context,
    deploy_stack,
    destroy_stack,
)


class TestStackHandler(unittest.TestCase):
    @patch("os.path.dirname")
    @patch("os.path.join")
    @patch("tempfile.mkdtemp")
    @patch("shutil.copy2")
    @patch("os.chmod")
    def test_get_temp_cdk_app_path(self, mock_chmod, mock_copy2, mock_mkdtemp, mock_join, mock_dirname):
        mock_mkdtemp.return_value = "/temp/dir"
        mock_join.return_value = "/temp/dir/app.py"
        mock_dirname.return_value = "/dir"
        result = _get_temp_cdk_app_path()
        self.assertEqual(result, "/temp/dir/app.py")

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

    @patch("autogluon.bench.cloud.aws.stack_handler.construct_context")
    @patch("subprocess.check_call")
    @patch("tempfile.mkdtemp")
    @patch("shutil.copy2")
    @patch("os.chmod")
    @patch("shutil.rmtree")
    def test_deploy_stack(
        self, mock_rmtree, mock_chmod, mock_copy2, mock_mktemp, mock_check_call, mock_construct_context
    ):
        infra_configs_dict = {
            "STACK_NAME_PREFIX": "test-prefix",
            "STACK_NAME_TAG": "benchmark",
            "STATIC_RESOURCE_STACK_NAME": "test-prefix-static-resource-stack",
            "BATCH_STACK_NAME": "test-prefix-batch-stack",
            "CONTAINER_MEMORY": 10000,
            "CDK_DEPLOY_REGION": "us-west-2",
        }
        mock_construct_context.return_value = infra_configs_dict
        mock_mktemp.return_value = "/path/to/temp_dir"
        with importlib.resources.path("autogluon.bench.cloud.aws", "stack_handler.py") as file_path:
            module_base_dir = os.path.dirname(file_path)

        deploy_stack()
        mock_construct_context.assert_called_once()
        mock_check_call.assert_called_once_with(
            [
                os.path.join(module_base_dir, "deploy.sh"),
                infra_configs_dict["STACK_NAME_PREFIX"],
                infra_configs_dict["STACK_NAME_TAG"],
                infra_configs_dict["STATIC_RESOURCE_STACK_NAME"],
                infra_configs_dict["BATCH_STACK_NAME"],
                str(infra_configs_dict["CONTAINER_MEMORY"]),
                infra_configs_dict["CDK_DEPLOY_REGION"],
                "/path/to/temp_dir/app.py",
            ]
        )
        mock_rmtree.assert_called_once_with("/path/to/temp_dir")

    @mock.patch("subprocess.check_call")
    @mock.patch("shutil.rmtree")
    def test_destroy_stack(self, mock_rmtree, mock_check_call):
        static_resource_stack = "static_resource_stack"
        batch_stack = "batch_stack"
        cdk_deploy_account = "cdk_deploy_account"
        cdk_deploy_region = "cdk_deploy_region"

        cdk_path = "/path/to/temp_dir/app.py"
        with mock.patch("autogluon.bench.cloud.aws.stack_handler._get_temp_cdk_app_path", return_value=cdk_path):
            destroy_stack(static_resource_stack, batch_stack, cdk_deploy_account, cdk_deploy_region)

        self.assertEqual(os.environ["CDK_DEPLOY_ACCOUNT"], cdk_deploy_account)
        self.assertEqual(os.environ["CDK_DEPLOY_REGION"], cdk_deploy_region)
        with importlib.resources.path("autogluon.bench.cloud.aws", "stack_handler.py") as file_path:
            module_base_dir = os.path.dirname(file_path)

        expected_call_args = [
            os.path.join(module_base_dir, "destroy.sh"),
            static_resource_stack,
            batch_stack,
            cdk_deploy_region,
            cdk_path,
        ]
        mock_check_call.assert_called_once_with(expected_call_args)
        mock_rmtree.assert_called_once_with(os.path.dirname(cdk_path))
