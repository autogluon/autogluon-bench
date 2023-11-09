import io
import os

import yaml

from autogluon.bench.runbenchmark import get_job_status, get_kwargs, invoke_lambda, run, run_benchmark


def setup_mock(
    mocker, tmp_path: str, module: str = "tabular", custom_dataloader: dict = None, custom_metrics: dict = None
):
    config_file = tmp_path / "run_configs.yaml"
    config_file.touch()
    # mock_open = mocker.patch("builtins.open", new_callable=mocker.mock_open)
    mocker.patch("re.search", return_value=False)
    cdk_context = {
        "METRICS_BUCKET": "test_bucket",
    }
    job_configs = {
        "job_id_1": "config_spit_1",
        "job_id_2": "config_spit_2",
    }

    infra_configs = {
        "STATIC_RESOURCE_STACK_NAME": "test_static_stack",
        "BATCH_STACK_NAME": "test_batch_stack",
        "CDK_DEPLOY_ACCOUNT": "test_account",
        "CDK_DEPLOY_REGION": "test_region",
        "METRICS_BUCKET": "test_bucket",
    }

    if module in ["tabular", "timeseries"]:
        module_configs = {
            "git_uri#branch": "https://github.com/openml/automlbenchmark.git#master",
            "framework": "AutoGluon:stable",
            "amlb_benchmark": "small",
            "amlb_user_dir": "sample_configs/amlb_configs",
        }
    elif module == "multimodal":
        module_configs = {
            "dataset_name": "data",
            "framework": "AutoGluon_stable",
            "custom_dataloader": custom_dataloader,
            "custom_metrics": custom_metrics,
        }

    yaml_value = {
        "mode": "aws",
        "module": module,
        "cdk_context": cdk_context,
        "job_configs": job_configs,
    }
    yaml_value.update(infra_configs)
    yaml_value.update(module_configs)

    def mock_yaml_side_effect(file_obj):
        if hasattr(file_obj, "name"):
            if "run_configs" in file_obj.name:
                return yaml_value
            else:
                return original_safe_load(file_obj)
        return {}

    original_safe_load = yaml.safe_load
    mock_yaml = mocker.patch("yaml.safe_load", side_effect=mock_yaml_side_effect)
    mocker.patch("autogluon.bench.runbenchmark._get_benchmark_name", return_value="test_benchmark")
    mocker.patch("autogluon.bench.runbenchmark.formatted_time", return_value="test_time")
    mocker.patch("autogluon.bench.runbenchmark._dump_configs", return_value="test_dump")
    mocker.patch("autogluon.bench.runbenchmark._dump_configs", return_value="test_dump")
    mocker.patch("os.environ.__setitem__")

    mock_deploy_stack = mocker.patch("autogluon.bench.runbenchmark.deploy_stack", return_value=infra_configs)
    mock_upload_to_s3 = mocker.patch("autogluon.bench.runbenchmark.upload_to_s3", return_value="test_s3_path")
    mock_invoke_lambda = mocker.patch("autogluon.bench.runbenchmark.invoke_lambda", return_value={})

    mock_wait_for_jobs = mocker.patch("autogluon.bench.runbenchmark.wait_for_jobs", return_value=[])
    mock_get_hardware_metrics = mocker.patch("autogluon.bench.runbenchmark.get_hardware_metrics")
    mock_destroy_stack = mocker.patch("autogluon.bench.runbenchmark.destroy_stack")
    mock_mount = mocker.patch("autogluon.bench.runbenchmark._mount_dir")
    mock_umount = mocker.patch("autogluon.bench.runbenchmark._umount_if_needed")

    return {
        "config_file": str(config_file),
        "infra_configs": infra_configs,
        "mock_deploy_stack": mock_deploy_stack,
        "mock_upload_to_s3": mock_upload_to_s3,
        "mock_invoke_lambda": mock_invoke_lambda,
        "mock_wait_for_jobs": mock_wait_for_jobs,
        "mock_destroy_stack": mock_destroy_stack,
        "mock_get_hardware_metrics": mock_get_hardware_metrics,
        "mock_mount": mock_mount,
        "mock_umount": mock_umount,
        "cdk_context": cdk_context,
        "job_configs": job_configs,
        "custom_configs": yaml_value,
    }


def test_get_kwargs_multimodal():
    module = "multimodal"
    configs = {
        "framework": "AutoGluon_stable",
        "dataset_name": "dataset",
    }

    expected_result = {
        "setup_kwargs": {
            "git_uri": "https://github.com/autogluon/autogluon.git",
            "git_branch": "master",
        },
        "run_kwargs": {
            "dataset_name": "dataset",
            "framework": "AutoGluon_stable",
            "constraint": None,
            "params": {"presets": "best_quality", "hyperparameters": {"optimization.max_epochs": 10}},
            "custom_dataloader": None,
            "custom_metrics": None,
        },
    }

    assert get_kwargs(module, configs) == expected_result


def test_get_kwargs_tabular():
    module = "tabular"
    configs = {
        "git_uri#branch": "https://github.com/openml/automlbenchmark.git#stable",
        "framework": "AutoGluon:stable",
        "amlb_benchmark": "test_bench",
        "amlb_task": "iris",
        "amlb_constraint": "test_constraint",
        "fold_to_run": 6,
        "amlb_user_dir": "sample_configs/amlb_configs",
    }

    expected_result = {
        "setup_kwargs": {
            "git_uri": "https://github.com/openml/automlbenchmark.git",
            "git_branch": "stable",
            "framework": "AutoGluon:stable",
            "user_dir": "sample_configs/amlb_configs",
        },
        "run_kwargs": {
            "framework": "AutoGluon:stable",
            "benchmark": "test_bench",
            "constraint": "test_constraint",
            "task": "iris",
            "fold": 6,
            "user_dir": "sample_configs/amlb_configs",
        },
    }

    assert get_kwargs(module, configs) == expected_result


def test_invoke_lambda(mocker):
    configs = {
        "CDK_DEPLOY_REGION": "us-east-1",
        "LAMBDA_FUNCTION_NAME": "test_function",
        "STACK_NAME_PREFIX": "prefix",
    }
    config_file = "test_config_file"

    mock_response = {"StatusCode": 200, "Payload": io.BytesIO(b'{"key": "value"}')}
    mock_lambda_client = mocker.MagicMock()
    mock_lambda_client.invoke.return_value = mock_response

    mocker.patch("boto3.client", return_value=mock_lambda_client)
    invoke_lambda(configs, config_file)
    mock_lambda_client.invoke.assert_called_once_with(
        FunctionName=configs["LAMBDA_FUNCTION_NAME"] + "-" + configs["STACK_NAME_PREFIX"],
        InvocationType="RequestResponse",
        Payload='{"config_file": "test_config_file"}',
    )


def test_run_aws_mode(mocker, tmp_path):
    setup = setup_mock(mocker, tmp_path)

    run(
        setup["config_file"],
        remove_resources=False,
        wait=False,
        skip_setup=True,
        save_hardware_metrics=False,
    )

    setup["mock_deploy_stack"].assert_called_once_with(custom_configs=setup["custom_configs"])
    setup["mock_upload_to_s3"].assert_called_once_with(
        s3_bucket="test_bucket", s3_dir="configs/tabular/test_benchmark_test_time", local_path="test_dump"
    )
    setup["mock_invoke_lambda"].assert_called_once_with(configs=setup["infra_configs"], config_file="test_s3_path")
    setup["mock_wait_for_jobs"].assert_not_called()
    setup["mock_get_hardware_metrics"].assert_not_called(),
    setup["mock_destroy_stack"].assert_not_called()


def test_run_aws_mode_remove_resources(mocker, tmp_path):
    setup = setup_mock(mocker, tmp_path)

    run(
        setup["config_file"],
        remove_resources=True,
        wait=False,
        skip_setup=True,
        save_hardware_metrics=False,
    )

    setup["mock_deploy_stack"].assert_called_once_with(custom_configs=setup["custom_configs"])
    setup["mock_upload_to_s3"].assert_called_once_with(
        s3_bucket="test_bucket", s3_dir="configs/tabular/test_benchmark_test_time", local_path="test_dump"
    )
    setup["mock_invoke_lambda"].assert_called_once_with(configs=setup["infra_configs"], config_file="test_s3_path")

    setup["mock_wait_for_jobs"].assert_called_once_with(config_file="test_dump")
    setup["mock_destroy_stack"].assert_called_once_with(
        static_resource_stack=setup["infra_configs"]["STATIC_RESOURCE_STACK_NAME"],
        batch_stack=setup["infra_configs"]["BATCH_STACK_NAME"],
        cdk_deploy_account=setup["infra_configs"]["CDK_DEPLOY_ACCOUNT"],
        cdk_deploy_region=setup["infra_configs"]["CDK_DEPLOY_REGION"],
        config_file=None,
    )


def test_run_aws_mode_wait(mocker, tmp_path):
    setup = setup_mock(mocker, tmp_path)

    run(
        setup["config_file"],
        remove_resources=False,
        wait=True,
        skip_setup=True,
        save_hardware_metrics=False,
    )

    setup["mock_deploy_stack"].assert_called_once_with(custom_configs=setup["custom_configs"])
    setup["mock_upload_to_s3"].assert_called_once_with(
        s3_bucket="test_bucket", s3_dir="configs/tabular/test_benchmark_test_time", local_path="test_dump"
    )
    setup["mock_invoke_lambda"].assert_called_once_with(configs=setup["infra_configs"], config_file="test_s3_path")

    setup["mock_wait_for_jobs"].assert_called_once_with(config_file="test_dump")
    setup["mock_get_hardware_metrics"].assert_not_called()


def test_run_aws_tabular_user_dir(mocker, tmp_path):
    setup = setup_mock(mocker, tmp_path)
    temp_dir_mock = mocker.patch("tempfile.TemporaryDirectory")
    s3_mock = mocker.patch("autogluon.bench.utils.general_utils.download_dir_from_s3")

    run(
        setup["config_file"],
        remove_resources=False,
        wait=False,
        skip_setup=True,
        save_hardware_metrics=False,
    )
    assert os.environ["FRAMEWORK_PATH"] == "frameworks/tabular/"
    assert os.environ["GIT_URI"] == "https://github.com/openml/automlbenchmark.git"
    assert os.environ["GIT_BRANCH"] == "master"
    assert os.environ["AMLB_FRAMEWORK"] == "AutoGluon:stable"
    assert os.environ["AMLB_USER_DIR"] == "custom_configs/amlb_configs"
    temp_dir_mock.assert_not_called()
    s3_mock.assert_not_called()
    assert setup["mock_umount"].call_count == 4
    assert setup["mock_mount"].call_count == 2


def test_run_aws_multimodal_custom_dataloader(mocker, tmp_path):
    custom_dataloader = {
        "dataloader_file": "path_to/dataset.py",
        "class_name": "CustomDataset",
        "dataset_config_file": "path_to/datasets.yaml",
    }
    setup = setup_mock(mocker, tmp_path, module="multimodal", custom_dataloader=custom_dataloader)
    mount_mock = mocker.patch("autogluon.bench.runbenchmark._mount_dir")
    umount_mock = mocker.patch("autogluon.bench.runbenchmark._umount_if_needed")

    run(
        setup["config_file"],
        remove_resources=False,
        wait=False,
        skip_setup=True,
        save_hardware_metrics=False,
    )
    assert setup["custom_configs"]["custom_dataloader"]["dataloader_file"] == "custom_configs/dataloaders/dataset.py"
    assert (
        setup["custom_configs"]["custom_dataloader"]["dataset_config_file"]
        == "custom_configs/dataloaders/datasets.yaml"
    )
    assert umount_mock.call_count == 2
    assert mount_mock.call_count == 1


def test_run_aws_multimodal_custom_metrics(mocker, tmp_path):
    custom_metrics = {
        "metrics_path": "path_to/metrics.py",
        "func_name": "custom_score",
    }
    setup = setup_mock(mocker, tmp_path, module="multimodal", custom_metrics=custom_metrics)
    mount_mock = mocker.patch("autogluon.bench.runbenchmark._mount_dir")
    umount_mock = mocker.patch("autogluon.bench.runbenchmark._umount_if_needed")

    run(
        setup["config_file"],
        remove_resources=False,
        wait=False,
        skip_setup=True,
        save_hardware_metrics=False,
    )
    assert setup["custom_configs"]["custom_metrics"]["metrics_path"] == "custom_configs/metrics/metrics.py"
    assert umount_mock.call_count == 2
    assert mount_mock.call_count == 1


def test_run_local_mode(mocker, tmp_path):
    config_file = tmp_path / "config_split_test.yaml"
    config_file.touch()
    mock_open = mocker.patch("builtins.open", new_callable=mocker.mock_open)

    configs = {
        "mode": "local",
        "metrics_bucket": "test_bucket",
        "module": "tabular",
    }
    mock_yaml = mocker.patch("yaml.safe_load")
    mock_yaml.return_value = configs
    mocker.patch("autogluon.bench.runbenchmark._get_benchmark_name", return_value="test_benchmark")
    mocker.patch("autogluon.bench.runbenchmark.formatted_time", return_value="test_time")
    mock_run_benchmark = mocker.patch("autogluon.bench.runbenchmark.run_benchmark")

    run(
        str(config_file),
        remove_resources=False,
        wait=False,
        skip_setup=False,
        save_hardware_metrics=False,
    )

    mock_open.assert_called_with(str(config_file), "r")
    mock_run_benchmark.assert_called_with(
        benchmark_name="test_benchmark_test_time",
        benchmark_dir="ag_bench_runs/tabular/test_benchmark_test_time",
        configs=configs,
        benchmark_dir_s3="tabular/test_benchmark_test_time/test_benchmark_test_time_0",
        skip_setup=False,
    )


def test_run_benchmark(mocker):
    benchmark_name = "test_benchmark"
    benchmark_dir = "test_dir"
    configs = {
        "module": "tabular",
        "METRICS_BUCKET": "test_bucket",
    }

    mocker.patch("autogluon.bench.runbenchmark.get_kwargs", return_value={"setup_kwargs": {}, "run_kwargs": {}})
    setup_mock = mocker.patch("autogluon.bench.runbenchmark.TabularBenchmark.setup")
    run_mock = mocker.patch("autogluon.bench.runbenchmark.TabularBenchmark.run")
    upload_metrics_mock = mocker.patch("autogluon.bench.runbenchmark.TabularBenchmark.upload_metrics")
    mocker.patch("autogluon.bench.runbenchmark._dump_configs")

    run_benchmark(
        benchmark_name=benchmark_name, benchmark_dir=benchmark_dir, configs=configs, benchmark_dir_s3="s3_dir"
    )

    setup_mock.assert_called_once_with()
    run_mock.assert_called_once_with()
    upload_metrics_mock.assert_called_once_with(s3_bucket=configs["METRICS_BUCKET"], s3_dir="s3_dir")


def test_run_benchmark_skip_setup(mocker):
    benchmark_name = "test_benchmark"
    benchmark_dir = "test_dir"
    configs = {
        "module": "tabular",
        "METRICS_BUCKET": "test_bucket",
    }

    mocker.patch("autogluon.bench.runbenchmark.get_kwargs", return_value={"setup_kwargs": {}, "run_kwargs": {}})
    setup_mock = mocker.patch("autogluon.bench.runbenchmark.TabularBenchmark.setup")
    run_mock = mocker.patch("autogluon.bench.runbenchmark.TabularBenchmark.run")
    upload_metrics_mock = mocker.patch("autogluon.bench.runbenchmark.TabularBenchmark.upload_metrics")
    mocker.patch("autogluon.bench.runbenchmark._dump_configs")

    run_benchmark(
        benchmark_name=benchmark_name,
        benchmark_dir=benchmark_dir,
        configs=configs,
        benchmark_dir_s3="s3_dir",
        skip_setup=True,
    )

    setup_mock.assert_not_called()
    run_mock.assert_called_once_with()
    upload_metrics_mock.assert_called_once_with(s3_bucket=configs["METRICS_BUCKET"], s3_dir="s3_dir")


def test_get_job_status_with_config_file(mocker, tmp_path):
    # Setup mock
    setup = setup_mock(mocker, tmp_path)

    # Additional mock for describe_jobs
    mock_boto_client = mocker.patch("boto3.client")
    mock_boto_client.return_value.describe_jobs.side_effect = [
        {"jobs": [{"status": "SUCCEEDED"}]},
        {"jobs": [{"status": "FAILED"}]},
    ]

    expected_status_dict = {"job_id_1": "SUCCEEDED", "job_id_2": "FAILED"}
    actual_status_dict = get_job_status(config_file=setup["config_file"])

    mock_boto_client.assert_called_once_with("batch", region_name="test_region")
    assert actual_status_dict == expected_status_dict


def test_get_job_status_with_job_ids(mocker, tmp_path):
    # Setup mock
    setup = setup_mock(mocker, tmp_path)

    # Additional mock for describe_jobs
    mock_boto_client = mocker.patch("boto3.client")
    mock_boto_client.return_value.describe_jobs.side_effect = [
        {"jobs": [{"status": "SUCCEEDED"}]},
        {"jobs": [{"status": "FAILED"}]},
    ]

    expected_status_dict = {"job_id_1": "SUCCEEDED", "job_id_2": "FAILED"}
    actual_status_dict = get_job_status(
        job_ids=["job_id_1", "job_id_2"], cdk_deploy_region="test_region", config_file=None
    )

    mock_boto_client.assert_called_once_with("batch", region_name="test_region")
    assert actual_status_dict == expected_status_dict


def test_get_kwargs_timeseries():
    module = "timeseries"
    configs = {
        "git_uri#branch": "https://github.com/openml/automlbenchmark.git#stable",
        "framework": "AutoGluon:stable",
        "amlb_benchmark": "test_bench",
        "amlb_task": "iris",
        "amlb_constraint": "test_constraint",
        "fold_to_run": 6,
        "amlb_user_dir": "sample_configs/amlb_configs",
    }

    expected_result = {
        "setup_kwargs": {
            "git_uri": "https://github.com/openml/automlbenchmark.git",
            "git_branch": "stable",
            "framework": "AutoGluon:stable",
            "user_dir": "sample_configs/amlb_configs",
        },
        "run_kwargs": {
            "framework": "AutoGluon:stable",
            "benchmark": "test_bench",
            "constraint": "test_constraint",
            "task": "iris",
            "fold": 6,
            "user_dir": "sample_configs/amlb_configs",
        },
    }

    assert get_kwargs(module, configs) == expected_result


def test_run_aws_timeseries_user_dir(mocker, tmp_path):
    setup = setup_mock(mocker, tmp_path, module="timeseries")
    temp_dir_mock = mocker.patch("tempfile.TemporaryDirectory")
    s3_mock = mocker.patch("autogluon.bench.utils.general_utils.download_dir_from_s3")

    run(
        setup["config_file"],
        remove_resources=False,
        wait=False,
        skip_setup=True,
    )
    assert os.environ["FRAMEWORK_PATH"] == "frameworks/timeseries/"
    assert os.environ["GIT_URI"] == "https://github.com/openml/automlbenchmark.git"
    assert os.environ["GIT_BRANCH"] == "master"
    assert os.environ["AMLB_FRAMEWORK"] == "AutoGluon:stable"
    assert os.environ["AMLB_USER_DIR"] == "custom_configs/amlb_configs"
    temp_dir_mock.assert_not_called()
    s3_mock.assert_not_called()
    assert setup["mock_umount"].call_count == 4
    assert setup["mock_mount"].call_count == 2


def test_run_benchmark_timeseries(mocker):
    benchmark_name = "test_benchmark"
    benchmark_dir = "test_dir"
    configs = {
        "module": "timeseries",
        "METRICS_BUCKET": "test_bucket",
    }

    mocker.patch("autogluon.bench.runbenchmark.get_kwargs", return_value={"setup_kwargs": {}, "run_kwargs": {}})
    setup_mock = mocker.patch("autogluon.bench.runbenchmark.TimeSeriesBenchmark.setup")
    run_mock = mocker.patch("autogluon.bench.runbenchmark.TimeSeriesBenchmark.run")
    upload_metrics_mock = mocker.patch("autogluon.bench.runbenchmark.TimeSeriesBenchmark.upload_metrics")
    mocker.patch("autogluon.bench.runbenchmark._dump_configs")

    run_benchmark(
        benchmark_name=benchmark_name, benchmark_dir=benchmark_dir, configs=configs, benchmark_dir_s3="s3_dir"
    )

    setup_mock.assert_called_once_with()
    run_mock.assert_called_once_with()
    upload_metrics_mock.assert_called_once_with(s3_bucket=configs["METRICS_BUCKET"], s3_dir="s3_dir")
