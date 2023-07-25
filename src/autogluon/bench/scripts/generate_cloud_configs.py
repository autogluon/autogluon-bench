import typer
import yaml

app = typer.Typer()


@app.command()
def generate_cloud_config(
    module: str = typer.Option(..., help="The module to generate config for: 'multimodal' or 'tabular'"),
    root_dir: str = typer.Option("ag_bench_runs", help="Root directory (optional, default = 'ag_bench_runs')"),
    benchmark_name: str = typer.Option("ag_bench", help="Benchmark name (optional, default = 'ag_bench')"),
    cdk_deploy_account: str = typer.Option(..., help="CDK deploy account"),
    cdk_deploy_region: str = typer.Option(..., help="CDK deploy region"),
    prefix: str = typer.Option(
        "ag-bench", help="Prefix used to identify infra resources (optional, default = 'ag-bench')"
    ),
    metrics_bucket: str = typer.Option(..., help="Name of the metrics bucket (required)"),
    data_bucket: str = typer.Option(None, help="S3 bucket to download private datasets (optional)"),
    vpc_name: str = typer.Option(None, help="Existing VPC name (optional, default: create a new VPC)"),
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
    git_uri_branch: str = typer.Option(
        "", help="AutoGluon MultiModal or AMLB git_uri#branch (in the format 'git_uri1#branch1,git_uri2#branch2,...')"
    ),
    dataset_names: str = typer.Option(
        "",
        help="AutoGluon MultiModal dataset names for '--module multimodal' (comma-separated, to get a list of dataset names, run `from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry\n multimodal_dataset_registry.list_keys()`)",
    ),
    presets: str = typer.Option(
        None,
        help="AutoGluon MultiModal presets for '--module multimodal' (comma-separated, can choose from ['medium_quality', 'high_quality', 'best_quality'])",
    ),
    time_limit: str = typer.Option(
        None,
        help="AutoGluon MultiModal time limits for '--module multimodal' (in the format 'time_limit1,time_limit2,...'",
    ),
    hyperparameters: str = typer.Option(
        None,
        help="AutoGluon MultiModal hyperparameters for '--module multimodal' (in the format '\"key1:value1,key2:value2;key1:value1,key2:value2;...\"'). Refer to https://auto.gluon.ai/stable/tutorials/multimodal/advanced_topics/customization.html for hyperparameter cutomization.",
    ),
    framework: str = typer.Option(
        "AutoGluon:stable",
        help="AMLB Frameworks for '--module tabular', in the format 'Framework1:label,Framework2:label,...'",
    ),
    amlb_benchmark: str = typer.Option(
        "",
        help="AMLB Benchmarks for '--module tabular', in the format 'test,small,...'. Refer to https://github.com/openml/automlbenchmark/tree/master/resources/benchmarks for options.",
    ),
    amlb_task: str = typer.Option(
        None,
        help="AMLB Tasks for '--module tabular' (in the format '\"benchmark1:task1,task2;benchmark2:task3,task4,task5;...\"')",
    ),
    amlb_constraint: str = typer.Option(
        "",
        help="AMLB Constraints for '--module tabular', in the format 'test,1h4c,...'. Refer to https://github.com/openml/automlbenchmark/blob/master/resources/constraints.yaml for details.",
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

    config["cdk_context"] = cdk_context
    git_uri_branch = git_uri_branch.split(",") if git_uri_branch else None

    if module == "multimodal":
        dataset_names = dataset_names.split(",") if dataset_names else None
        presets = presets.split(",") if presets else None
        time_limit = [int(t.strip()) for t in time_limit.split(",")] if time_limit else None

        hyperparameters_list = []
        if hyperparameters:
            hyperparameters_items = hyperparameters.split(";")
            for item in hyperparameters_items:
                item_dict = {}
                key_value_pairs = item.split(",")
                for kv in key_value_pairs:
                    key, value = kv.split(":")
                    item_dict[key.strip()] = float(value.strip())
                hyperparameters_list.append(item_dict)

        module_configs = {
            "git_uri#branch": git_uri_branch,
            "dataset_name": dataset_names,
        }
        if presets:
            module_configs["presets"] = presets
        if time_limit:
            module_configs["time_limit"] = time_limit
        if hyperparameters_list:
            module_configs["hyperparameters"] = hyperparameters_list

        config["module_configs"] = {"multimodal": module_configs}

    elif module == "tabular":
        framework = framework.split(",") if framework else None
        amlb_benchmark = amlb_benchmark.split(",") if amlb_benchmark else None

        if amlb_task:
            amlb_task_dict = {}
            for item in amlb_task.split(";"):
                if ":" in item:
                    benchmark, tasks = item.split(":")
                    task_list = tasks.split(",")
                    amlb_task_dict[benchmark] = task_list
            amlb_task = amlb_task_dict
        amlb_constraint = amlb_constraint.split(",") if amlb_constraint else None
        amlb_user_dir = amlb_user_dir.split(",") if amlb_user_dir else None

        module_configs = {
            "git_uri#branch": git_uri_branch,
            "framework": framework,
            "amlb_benchmark": amlb_benchmark,
            "amlb_constraint": amlb_constraint,
        }

        if amlb_task:
            module_configs["amlb_task"] = amlb_task
        if amlb_user_dir:
            module_configs["amlb_user_dir"] = amlb_user_dir

        config["module_configs"] = {"tabular": module_configs}
    else:
        typer.echo("Invalid module. Please choose 'multimodal' or 'tabular'.")
        return

    output_file = f"{module}_cloud_configs.yaml"
    with open(output_file, "w") as f:
        yaml.dump(config, f)

    typer.echo(f"Config file '{output_file}' generated successfully.")
