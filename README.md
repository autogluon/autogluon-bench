<div align="left">
  <img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</div>

# AutoGluon-Bench

Welcome to AutoGluon-Bench, a suite for benchmarking your AutoML frameworks.

## Setup

Follow the steps below to set up autogluon-bench:

```bash
# create virtual env
python3.9 -m venv .venv_agbench
source .venv_agbench/bin/activate
```

Install `autogloun-bench` from PyPI:

```bash
python -m pip install autogluon.bench
```


Or install `autogluon-bench` from source:

```bash
git clone https://github.com/autogluon/autogluon-bench.git
cd autogluon-bench

# install from source in editable mode
pip install -e .
```
Also, replace all `pip install autogluon.bench` with `pip install -e .` in the source code.


## Run benchmarks locally

To run the benchmarks on your local machine, use the following command:

```
agbench run path/to/local_config_file
```

Check out our [sample configuration file](https://github.com/autogluon/autogluon-bench/blob/master/sample_configs/local_configs.yaml) for local runs.

The results are stored in the following directory: `{WORKING_DIR}/{root_dir}/{module}/{benchmark_name}_{timestamp}`.


### Tabular Benchmark

To perform tabular benchmarking, set the module to tabular. You must set both Benchmark Configurations and Tabular Specific configurations, and each should have a single value. Refer to the [sample configuration file](https://github.com/autogluon/autogluon-bench/blob/master/sample_configs/local_configs.yaml) for more details.

The tabular module leverages the [AMLB](https://github.com/openml/automlbenchmark) benchmarking framework. Required and optional AMLB arguments are specified via the configuration file mentioned previously.

To benchmark a custom branch of AutoGluon on `tabular` module, use `amlb_custom_branch: https://github.com/REPO/autogluon#BRANCH` in the configuration file.


### Multimodal Benchmark

For multimodal benchmarking, set the module to multimodal. We currently support benchmarking multimodal on a custom branch. Note that multimodal benchmarking directly calls the MultiModalPredictor, bypassing the extra layer of [AMLB](https://github.com/openml/automlbenchmark). Therefore, the required arguments are different from those for tabular.

You can add more datasets to your benchmarking jobs. We provided sample [multimodal datasets](https://github.com/autogluon/autogluon-bench/blob/master/src/autogluon/bench/datasets/multimodal_dataset.py) and [object detection dataset](https://github.com/autogluon/autogluon-bench/blob/master/src/autogluon/bench/datasets/object_detection_dataset.py). Follow these samples to add custom datasets, then specify dataset_name in your local config file. Please follow the section `Install From Source` for more instructions on how to develop with source.

## Run benchmarks on AWS

AutoGluon-Bench uses the AWS CDK to build an AWS Batch compute environment for benchmarking.

To get started, install [Node.js](https://nodejs.org/) and [AWS CDK](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html#getting_started_install) with the following commands:

```bash
curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash  # replace bash with other shell (e.g. zsh) if you are using a different one
source ~/.bashrc
nvm install 18.16.0  # install Node.js
npm install -g aws-cdk  # install aws-cdk
cdk --version  # verify the installation, you might need to update the Node.js version depending on the log.
```

To initiate benchmarking on the cloud, use the command below:

```
agbench run /path/to/cloud_config_file
```

This command automatically sets up an AWS Batch environment using instance specifications defined in the `cloud_config_file`. It also creates a lambda function named with your chosen `LAMBDA_FUNCTION_NAME`. This lambda function is automatically invoked with the cloud config file you provided, submitting multiple AWS Batch jobs to the job queue (named with the `PREFIX` you provided).

In order for the Lambda function to submit multiple jobs simultaneously, you need to specify a list of values for each module-specific key. Each combination of configurations is saved and uploaded to your specified `METRICS_BUCKET` in S3, stored under `S3://{METRICS_BUCKET}/configs/{BENCHMARK_NAME}_{timestamp}/{BENCHMARK_NAME}_split_{UID}.yaml`. Here, `UID` is a unique ID assigned to the split.

The AWS infrastructure configurations and submitted job IDs are saved locally at `{WORKING_DIR}/{root_dir}/{module}/{benchmark_name}_{timestamp}/aws_configs.yaml`. You can use this file to check the job status at any time:

```bash
agbench get-job-status --config-file /path/to/aws_configs.yaml
```

You can also check the job status using job IDs:

```bash
agbench get-job-status --job-ids JOB_ID_1 --job-ids JOB_ID_2 —cdk_deploy_region AWS_REGION

```

Job logs can be viewed on the AWS console. Each job has an `UID` attached to the name, which you can use to identify the respective config split. After the jobs are completed and reach the `SUCCEEDED` status in the job queue, you'll find metrics saved under `S3://{METRICS_BUCKET}/{module}/{benchmark_name}_{timestamp}/{benchmark_name}_{timestamp}_{UID}`.

By default, the infrastructure created is retained for future use. To automatically remove resources after the run, use the `--remove_resources` option:

```bash
agbench run path/to/cloud_config_file --remove_resources
```

This will check the job status every 2 minutes and remove resources after all jobs succeed. If any job fails, resources will be kept.

If you want to manually remove resources later, use:

```bash
agbench destroy-stack STATIC_RESOURCE_STACK_NAME BATCH_STACK_NAME CDK_DEPLOY_ACCOUNT CDK_DEPLOY_REGION
```

where you can find all argument values in `{WORKING_DIR}/{root_dir}/{module}/{benchmark_name}_{timestamp}/aws_configs.yaml`.


### Configure the AWS infrastructure

The default infrastructure configurations are located [here](https://github.com/autogluon/autogluon-bench/blob/master/src/autogluon/bench/cloud/aws/default_config.yaml).

```
CDK_DEPLOY_ACCOUNT: dummy
CDK_DEPLOY_REGION: dummy
PREFIX: ag-bench-test
RESERVED_MEMORY_SIZE: 15000
MAX_MACHINE_NUM: 20
BLOCK_DEVICE_VOLUME: 100
INSTANCE: g4dn.2xlarge
METRICS_BUCKET: autogluon-benchmark-metrics
DATA_BUCKET: automl-mm-bench
VPC_NAME: automm-batch-stack/automm-vpc
LAMBDA_FUNCTION_NAME: ag-bench-test-job-function
```
where:
- `CDK_DEPLOY_ACCOUNT` and `CDK_DEPLOY_REGION` should be overridden with your AWS account ID and desired region to create the stack.
- `PREFIX` is used as an identifier for the stack and resources created.
- `RESERVED_MEMORY_SIZE` is used together with the instance memory size to calculate the container shm_size.
- `BLOCK_DEVICE_VOLUME` is the size of storage device attached to instance.
- `METRICS_BUCKET` is the bucket to upload benchmarking metrics.
- `DATA_BUCKET` is the bucket to download dataset from.
- `VPC_NAME` is used to look up an existing VPC.
- `LAMBDA_FUNCTION_NAME` lambda function to submit jobs to AWS Batch.

To override these configurations, use the `cdk_context` key in your custom config file. See our [sample cloud config](https://github.com/autogluon/autogluon-bench/blob/master/sample_configs/cloud_configs.yaml) for reference.

## Evaluating bechmark runs

Innixma's `autogluon-benchmark` repository can be used to evaluate tabular benchmark runs whose results are in S3.
Using these utilities is ad-hoc at this time, but in a coming release we will integrate this capability into `autogluon-bench` and support evaulation of multimodal benchmarks.

### Evaluation Steps

Clone the `autogluon-benchmark` repository:
```
git clone https://github.com/gidler/autogluon-benchmark.git
```

Confirm that AWS credentials are setup for the AWS account that has the benchmark results in S3.

Run the `aggregate_all.py` script
```
python scripts/aggregate_all.py --s3_bucket {AWS_BUCKET} --s3_prefix {AWS_PREFIX} --version_name {BENCHMARK_VERSION_NAME}

# example: python scripts/aggregate_all.py --s3_bucket autogluon-benchmark-metrics --s3_prefix tabular/ --version_name test_local_20230330T180916
```

This will create a new file in S3 with this signature:
```
s3://{AWS_BUCKET}/aggregated/{AWS_PREFIX}/{BENCHMARK_VERSION_NAME}/results.csv
```

Run the `run_generate_clean_openml` python utility. You will need to manually set the `run_name_arg` and `path_prefix` variables in the script.
```
python autogluon_benchmark/evaluation/runners/run_generate_clean_openml.py 
```
This will create a local file of results in the `data/results/input/prepared/openml/` directory.

Run the `benchmark_evaluation` python script. You will need to manually update the `frameworks_run` and `paths` variables in the script.
```
python autogluon_benchmark/evaluation/runners/run_evaluation_openml.py
```

