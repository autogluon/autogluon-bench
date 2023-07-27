import importlib.resources
import json
import os
import tempfile
from unittest.mock import mock_open, patch

import yaml
from conftest import default_configs

from autogluon.bench.cloud.aws.stack_handler import (
    _get_temp_cdk_app_path,
    construct_context,
    deploy_stack,
    destroy_stack,
)


def test_get_temp_cdk_app_path(mocker):
    mock_mkdtemp = mocker.patch("tempfile.mkdtemp")
    mock_copy2 = mocker.patch("shutil.copy2")
    mock_chmod = mocker.patch("os.chmod")
    mock_join = mocker.patch("os.path.join")
    mock_resources_path = mocker.patch("importlib.resources.path")

    mock_mkdtemp.return_value = "/temp/dir"
    mock_join.return_value = "/temp/dir/app.py"

    mock_resources_path.return_value.__enter__.return_value = "path_to_cdk_app"

    result = _get_temp_cdk_app_path()

    assert result == "/temp/dir/app.py"
    mock_mkdtemp.assert_called_once()
    mock_join.assert_called_once_with("/temp/dir", "app.py")
    mock_copy2.assert_called_once_with("path_to_cdk_app", "/temp/dir/app.py")
    mock_chmod.assert_called_once_with("/temp/dir/app.py", 0o755)


def test_construct_context():
    custom_configs = {"CDK_DEPLOY_ACCOUNT": "123456789012", "CDK_DEPLOY_REGION": "us-west-1", "PREFIX": "test"}

    with patch("builtins.open", mock_open(read_data=json.dumps(default_configs))) as mock_file:
        with patch("autogluon.bench.cloud.aws.stack_handler.get_instance_type_specs") as mock_instance_specs:
            mock_instance_specs.return_value = (1, 8, 32000)
            context = construct_context(custom_configs=custom_configs)

    prefix = custom_configs["PREFIX"]
    assert context["CDK_DEPLOY_ACCOUNT"] == custom_configs["CDK_DEPLOY_ACCOUNT"]
    assert context["CDK_DEPLOY_REGION"] == custom_configs["CDK_DEPLOY_REGION"]
    assert context["STACK_NAME_PREFIX"] == prefix
    assert context["METRICS_BUCKET"] == default_configs["METRICS_BUCKET"]
    assert context["DATA_BUCKET"] == default_configs["DATA_BUCKET"]
    assert context["VPC_NAME"] == default_configs["VPC_NAME"]
    assert context["STATIC_RESOURCE_STACK_NAME"] == f"{prefix}-static-resource-stack"
    assert context["BATCH_STACK_NAME"] == f"{prefix}-batch-stack"
    assert context["CONTAINER_GPU"] == 1
    assert context["CONTAINER_VCPU"] == 8
    assert context["CONTAINER_MEMORY"] == 22000
    assert context["COMPUTE_ENV_MAXV_CPUS"] == 16


def test_deploy_stack(mocker):
    infra_configs_dict = {
        "STACK_NAME_PREFIX": "test-prefix",
        "STACK_NAME_TAG": "benchmark",
        "STATIC_RESOURCE_STACK_NAME": "test-prefix-static-resource-stack",
        "BATCH_STACK_NAME": "test-prefix-batch-stack",
        "CONTAINER_MEMORY": 10000,
        "CDK_DEPLOY_REGION": "us-west-2",
    }
    custom_configs = {
        "cdk_context": {
            "METRICS_BUCKET": "test-metrics-bucket",
            "DATA_BUCKET": "test-data-bucket",
        },
    }

    mock_subprocess = mocker.patch("subprocess.check_call")
    with importlib.resources.path("autogluon.bench.cloud.aws", "stack_handler.py") as file_path:
        module_base_dir = os.path.dirname(file_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_file:
            mock_cdk_path = mocker.patch(
                "autogluon.bench.cloud.aws.stack_handler._get_temp_cdk_app_path", return_value=temp_file.name
            )
            mock_get_context = mocker.patch(
                "autogluon.bench.cloud.aws.stack_handler.construct_context", return_value=infra_configs_dict
            )

            deploy_stack(custom_configs=custom_configs)

    assert not os.path.exists(temp_dir)
    mock_cdk_path.assert_called_once()
    mock_get_context.assert_called_with(custom_configs=custom_configs["cdk_context"])
    mock_subprocess.assert_called_once_with(
        [
            os.path.join(module_base_dir, "deploy.sh"),
            infra_configs_dict["STACK_NAME_PREFIX"],
            infra_configs_dict["STACK_NAME_TAG"],
            infra_configs_dict["STATIC_RESOURCE_STACK_NAME"],
            infra_configs_dict["BATCH_STACK_NAME"],
            str(infra_configs_dict["CONTAINER_MEMORY"]),
            infra_configs_dict["CDK_DEPLOY_REGION"],
            temp_file.name,
        ]
    )


def test_destroy_stack(mocker):
    mock_subprocess = mocker.patch("subprocess.check_call")
    static_resource_stack = "static_resource_stack"
    batch_stack = "batch_stack"
    cdk_deploy_account = "cdk_deploy_account"
    cdk_deploy_region = "cdk_deploy_region"

    with importlib.resources.path("autogluon.bench.cloud.aws", "stack_handler.py") as file_path:
        module_base_dir = os.path.dirname(file_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_file:
            mock_cdk_path = mocker.patch(
                "autogluon.bench.cloud.aws.stack_handler._get_temp_cdk_app_path", return_value=temp_file.name
            )

            destroy_stack(
                static_resource_stack=static_resource_stack,
                batch_stack=batch_stack,
                cdk_deploy_account=cdk_deploy_account,
                cdk_deploy_region=cdk_deploy_region,
                config_file=None,
            )

    assert not os.path.exists(temp_dir)
    assert os.environ["CDK_DEPLOY_ACCOUNT"] == cdk_deploy_account
    assert os.environ["CDK_DEPLOY_REGION"] == cdk_deploy_region

    mock_cdk_path.assert_called_once()
    mock_subprocess.assert_called_once_with(
        [
            os.path.join(module_base_dir, "destroy.sh"),
            static_resource_stack,
            batch_stack,
            cdk_deploy_region,
            temp_file.name,
        ]
    )


def test_destroy_stack_with_config(mocker):
    mock_subprocess = mocker.patch("subprocess.check_call")
    configs = {
        "STATIC_RESOURCE_STACK_NAME": "static_resource_stack",
        "BATCH_STACK_NAME": "batch_stack",
        "CDK_DEPLOY_ACCOUNT": "cdk_deploy_account",
        "CDK_DEPLOY_REGION": "cdk_deploy_region",
    }

    with importlib.resources.path("autogluon.bench.cloud.aws", "stack_handler.py") as file_path:
        module_base_dir = os.path.dirname(file_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_file:
            mock_cdk_path = mocker.patch(
                "autogluon.bench.cloud.aws.stack_handler._get_temp_cdk_app_path", return_value=temp_file.name
            )
            config_file_path = os.path.join(temp_dir, "configs.yaml")
            with open(config_file_path, "w") as f:
                yaml.dump(configs, f)

            destroy_stack(
                static_resource_stack=None,
                batch_stack=None,
                cdk_deploy_account=None,
                cdk_deploy_region=None,
                config_file=config_file_path,
            )

    assert not os.path.exists(temp_dir)
    assert os.environ["CDK_DEPLOY_ACCOUNT"] == "cdk_deploy_account"
    assert os.environ["CDK_DEPLOY_REGION"] == "cdk_deploy_region"

    mock_cdk_path.assert_called_once()
    mock_subprocess.assert_called_once_with(
        [
            os.path.join(module_base_dir, "destroy.sh"),
            "static_resource_stack",
            "batch_stack",
            "cdk_deploy_region",
            temp_file.name,
        ]
    )
