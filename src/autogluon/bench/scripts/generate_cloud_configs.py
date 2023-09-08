import typer
import yaml

app = typer.Typer()


@app.command()
def generate_cloud_config(
    module: str = typer.Option(..., help="The module to generate config for: 'multimodal', 'tabular' or 'timeseries'"),
    root_dir: str = typer.Option("ag_bench_runs", help="Root directory (optional, default = 'ag_bench_runs')"),
    benchmark_name: str = typer.Option("ag_bench", help="Benchmark name (optional, default = 'ag_bench')"),
    cdk_deploy_account: str = typer.Option(..., help="CDK deploy account"),
    cdk_deploy_region: str = typer.Option(..., help="CDK deploy region"),
    prefix: str = typer.Option(
        "ag-bench", help="Prefix used to identify infra resources (optional, default = 'ag-bench')"
    ),
    metrics_bucket: str = typer.Option(..., help="Name of the metrics bucket (required)"),
    data_bucket: str = typer.Option(None, help="S3 bucket to download private datasets (optional)"),
    max_machine_num: int = typer.Option(
        None, help="Maximum number of machines for AWS Batch (optional, default = 20)"
    ),
    block_device_volume: int = typer.Option(
        None, help="Block device volume for EC2 instances (optional, default = 100)"
    ),
    reserved_memory_size: int = typer.Option(
        None, help="Reserved memory size for Docker container (optional, default = 15000)"
    ),
    instance: str = typer.Option(None, help="EC2 Instance type (optional, default = 'g4dn.2xlarge')"),
    time_limit: str = typer.Option(
        None,
        help="AWS Batch job time out",
    ),
    vpc_name: str = typer.Option(None, help="Existing VPC name (optional, default: create a new VPC)"),
    git_uri_branch: str = typer.Option("", help="AMLB git_uri#branch"),
    dataset_names: str = typer.Option(
        "",
        help="AutoGluon MultiModal dataset names for '--module multimodal' (comma-separated, to get a list of dataset names, run `from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry\n multimodal_dataset_registry.list_keys()`)",
    ),
    custom_resource_dir: str = typer.Option(
        None,
        help="Path to custom resources definitions for '--module multimodal'",
    ),
    custom_dataloader: str = typer.Option(
        None,
        help="Custom dataloader for '--module multimodal', in the format '\"dataloader_file:value1;class_name:value2;dataset_config_file:value3\"'",
    ),
    framework: str = typer.Option(
        "AutoGluon:stable",
        help="Framework name",
    ),
    constraint: str = typer.Option("test", help="Resource constraint for '--module multimodal'"),
    amlb_constraint: str = typer.Option(
        "",
        help="AMLB Constraints for tabular or timeseries, in the format 'test,1h4c,...'. Refer to https://github.com/openml/automlbenchmark/blob/master/resources/constraints.yaml for details.",
    ),
    amlb_benchmark: str = typer.Option(
        "",
        help="AMLB Benchmarks for tabular or timeseries, in the format 'test,small,...'. Refer to https://github.com/openml/automlbenchmark/tree/master/resources/benchmarks for options.",
    ),
    amlb_task: str = typer.Option(
        None,
        help="AMLB Tasks for tabular or timeseries (in the format '\"benchmark1:task1,task2;benchmark2:task3,task4,task5;...\"')",
    ),
    amlb_fold_to_run: str = typer.Option(
        None,
        help="AMLB fold for tabular or timeseries (in the format '\"benchmark1:task1:fold1/fold2,task2:fold3/fold4;benchmark2:task3:fold1/fold2;...\"')",
    ),
    amlb_user_dir: str = typer.Option(
        None,
        help="Custom config directory.",
    ),
):
    config = {
        "module": module,
        "mode": "aws",
        "benchmark_name": benchmark_name,
        "root_dir": root_dir,
        "framework": framework,
    }
    cdk_context = {
        "CDK_DEPLOY_ACCOUNT": cdk_deploy_account,
        "CDK_DEPLOY_REGION": cdk_deploy_region,
        "PREFIX": prefix,
        "METRICS_BUCKET": metrics_bucket,
    }
    if data_bucket:
        cdk_context["DATA_BUCKET"] = data_bucket
    if vpc_name:
        cdk_context["VPC_NAME"] = vpc_name
    if max_machine_num:
        cdk_context["MAX_MACHINE_NUM"] = max_machine_num
    if block_device_volume:
        cdk_context["BLOCK_DEVICE_VOLUME"] = block_device_volume
    if reserved_memory_size:
        cdk_context["RESERVED_MEMORY_SIZE"] = reserved_memory_size
    if instance:
        cdk_context["INSTANCE"] = instance
    if time_limit:
        cdk_context["TIME_LIMIT"] = time_limit

    config["cdk_context"] = cdk_context

    if module == "multimodal":
        dataset_names = dataset_names.split(",") if dataset_names else None
        module_configs = {
            "constraint": constraint,
            "dataset_name": dataset_names,
        }
        if custom_resource_dir:
            module_configs["custom_resource_dir"] = custom_resource_dir

        if custom_dataloader:
            custom_dataloader_dict = {}
            for item in custom_dataloader.split(";"):
                k, v = item.split(":")
                custom_dataloader_dict[k] = v
            module_configs["custom_dataloader"] = custom_dataloader_dict

        config.update(module_configs)

    elif module in ["tabular", "timeseries"]:
        module_configs = {
            "git_uri#branch": git_uri_branch,
            "amlb_constraint": amlb_constraint,
        }

        if amlb_benchmark:
            amlb_benchmark = amlb_benchmark.split(",")
            module_configs["amlb_benchmark"] = amlb_benchmark

        if amlb_task:
            amlb_task_dict = {}
            for item in amlb_task.split(";"):
                if ":" in item:
                    benchmark, tasks = item.split(":")
                    task_list = tasks.split(",")
                    amlb_task_dict[benchmark] = task_list
            module_configs["amlb_task"] = amlb_task_dict

        if amlb_fold_to_run:
            fold_to_run_dict = {}
            for benchmark_item in amlb_fold_to_run.split(";"):
                benchmark, task_item = benchmark_item.split(":", 1)
                fold_to_run_dict[benchmark] = {}
                for task_folds in task_item.split(","):
                    task, folds = task_folds.split(":")
                    folds = [int(v) for v in folds.split("/")]
                    fold_to_run_dict[benchmark][task] = folds
            module_configs["fold_to_run"] = fold_to_run_dict

        if amlb_user_dir:
            module_configs["amlb_user_dir"] = amlb_user_dir

        config.update(module_configs)
    else:
        typer.echo("Invalid module. Please choose 'multimodal', 'tabular' or 'timeseries'.")
        return

    output_file = f"{module}_cloud_configs.yaml"
    with open(output_file, "w") as f:
        yaml.dump(config, f)

    typer.echo(f"Config file '{output_file}' generated successfully.")
