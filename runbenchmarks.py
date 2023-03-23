import argparse
import boto3
import json
import os
import time
import yaml
from autogluon.bench.cloud.aws.run_deploy import deploy_stack
from autogluon.bench.frameworks.multimodal.multimodal_benchmark import \
    MultiModalBenchmark
from autogluon.bench.frameworks.tabular.tabular_benchmark import \
    TabularBenchmark


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, help='Path to custom config file.')

    args = parser.parse_args()
    return args
    

def run_benchmark(configs: dict):
    if configs["module"] == "multimodal":
        benchmark = MultiModalBenchmark(benchmark_name=configs["benchmark_name"])
        git_uri, git_branch = configs["git_uri#branch"].split("#")
        benchmark.setup(git_uri=git_uri, git_branch=git_branch)
        benchmark.run(data_path=configs["data_path"])
        if configs.get("metrics_bucket", None):
            benchmark.upload_metrics(s3_bucket=configs["metrics_bucket"], s3_dir=f'{configs["module"]}/{benchmark.benchmark_name}')
    elif configs["module"] == "tabular":
        benchmark = TabularBenchmark(
            benchmark_name=configs["benchmark_name"], 
        )
        benchmark.setup()
        benchmark.run(
            framework=f'{configs["framework"]}:{configs["label"]}',
            benchmark=configs["amlb_benchmark"],
            constraint=configs["amlb_constraint"],
            task=configs["amlb_task"]
        )
        if configs["metrics_bucket"] is not None:
            benchmark.upload_metrics(s3_bucket=configs["metrics_bucket"], s3_dir=f'{configs["module"]}/{benchmark.benchmark_name}')
    elif configs["module"] == "ts":
        raise NotImplementedError
    

def upload_config(bucket: str, file: str):
    s3 = boto3.client("s3")
    file_name = f'{file.split("/")[-1].split(".")[0]}_{time.strftime("%Y%m%dT%H%M%S", time.localtime())}.yaml'
    s3_path = f"configs/{file_name}"
    s3.upload_file(file, bucket, s3_path)
    return f"s3://{bucket}/{s3_path}"


def download_config(s3_path: str):
    s3 = boto3.client("s3")
    file_path = os.path.join("/tmp", s3_path.split("/")[-1])
    bucket = s3_path.strip("s3://").split("/")[0]
    s3_path = s3_path.split(bucket)[-1].lstrip("/")
    s3.download_file(bucket, s3_path, file_path)
    return file_path


def invoke_lambda(configs: dict, config_file: str):
    lambda_client = boto3.client("lambda", configs["CDK_DEPLOY_REGION"])
    payload = {
        "config_file": config_file
    }
    lambda_client.invoke(
        FunctionName=configs["LAMBDA_FUNCTION_NAME"],
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
    print(f'AWS Batch jobs submitted by {configs["LAMBDA_FUNCTION_NAME"]}.')


def run():
    args = get_args()
    configs = {}
    if args.config_file.startswith("s3"):
        args.config_file = download_config(s3_path=args.config_file)
    with open(args.config_file, "r") as f:
        configs = yaml.safe_load(f)

    if configs["mode"] == "aws":
        infra_configs = deploy_stack(configs=configs.get("cdk_context", {}))
        config_s3_path = upload_config(bucket=configs["metrics_bucket"], file=args.config_file)
        invoke_lambda(configs=infra_configs, config_file=config_s3_path) # lambda_invoke script to run benchmarks
        # destroy_stack()
    elif configs["mode"] == "local":
        run_benchmark(configs)
    else:
        raise NotImplementedError
    
 
if __name__ == '__main__':
    run()
