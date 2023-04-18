<div align="left">
  <img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</div>

# AutoGluon-Bench

## Setup

```
git clone https://github.com/autogluon/autogluon-bench.git
cd autogluon-bench
git checkout poc_0.0.1

# create virtual env
python3.10 -m venv .venv
source .venv/bin/activate

# install autogluon-bench
pip install -e .
```

## Run benchmarkings locally

Currently, `tabular` makes use of the [AMLB](https://github.com/openml/automlbenchmark) benchmarking framework. Required and optional AMLB arguments are specified via configuration file. Sample configuration files are provided in the `sample_configs` directory.

A custom branch of autogluon can be benchmarked by specifying the `amlb_custom_branch` configuration of the form: `https://github.com/REPO/autogluon#BRANCH`

```
python ./runbenchmarks.py  --config_file path/to/local_config_file
```
A sample local config file is available for reference at `./sample_configs/local_configs.yaml`. All keys should have single value.

Note that `multimodal` benchmarking directly calls the `MultiModalPredictor` without the extra layer of [AMLB](https://github.com/openml/automlbenchmark), so the set of arguments we call is different from that of running `tabular`. Currently, we support benchmarking `multimodal` on custom branch of the main repository or any forked repository.

To customize the benchmarking experiment, including adding more hyperparameters, and evaluate on more metrics, you can refer to `./src/autogluon/bench/frameworks/multimodal/exec.py`.

Results are saved under `$WORKING_DIR/benchmark_runs/$module/{$benchmark_name}_{$timestamp}`


## Run benchmarkings on AWS

The infrastructure is built on AWS CDK, where a AWS Batch compute environment is setup to run the benchmarkings. 

In order to run AWS CDK and build containers, the following setup is required to install [Node.js](https://nodejs.org/) and [AWS CDK](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html#getting_started_install)

```
curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash  # replace bash with other shell (e.g. zsh) if you are using a different one
source ~/.bashrc
nvm install 18.13.0  # install Node.js
npm install -g aws-cdk  # install aws-cdk
cdk --version  # verify the installation
```

The default configurations of the infrastructure is located at `./src/autogluon/bench/cloud/aws/default_config.yaml`.
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

The configs can be overridden by a custom config file defined by `--config_file` under `cdk_context` key. Please refer to `./sample_configs/cloud_configs.yaml` for reference. Note that in `AWS` mode, we support running multiple benchmarking jobs at the same time, so you can have a list of values for each key in the module specific keys.

To deploy the stack and run the benchmarking jobs with one command:

```
python ./runbenchmarks.py  --config_file path/to/cloud_config_file
```

The above command will deploy the infrastructure automatically and create a lambda_function with the `LAMBDA_FUNCTION_NAME` of your choice. The lambda function will then be invoked automatically with the cloud config file you provided, and submit AWS Batch jobs to the job queue (named with the `PREFIX` you provided).
