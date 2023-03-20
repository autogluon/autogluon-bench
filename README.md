# AutoGluon-Bench

## Setup

```
git clone https://github.com/suzhoum/autogluon-bench.git
cd autogluon-bench
git checkout poc_0.0.1

# create virtual env
python3.9 -m venv .venv
source .venv/bin/activate

# install autogluon-bench
pip install -e .
```

## Run benchmarkings locally

Currently, `tabular` makes use of the [AMLB](https://github.com/openml/automlbenchmark) benchmarking framework, so we will supply the required and optional arguments when running benchmark for tabular. It can only run on the stable and latest(master) build of the framework at the moment. In order to benchmark on a custom branch, changes need to be made on AMLB.

```
python ./runbenchmarks.py  --module tabular --mode local  --benchmark_name local_test --framework AutoGluon --label stable --amlb_benchmark test --amlb_constraint test --amlb_task iris
```

where `--amlb_task` is optional, and corresponds to `--task` argument in AMLB. You can refer to [Quickstart of AMLB](https://github.com/openml/automlbenchmark#quickstart) for more details.

On the other hand, `multimodal` benchmarking directly calls the `MultiModalPredictor` without the extra layer of [AMLB](https://github.com/openml/automlbenchmark), so the set of arguments we call is different from that of running `tabular`. Currently, we support benchmarking `multimodal` on custom branch of the main repository or any forked repository.

```
python ./runbenchmarks.py --module multimodal --mode local --git_uri https://github.com/autogluon/autogluon.git --git_branch master --data_path MNIST --benchmark_name local_test
```
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


The configs can be overridden by a custom config file defined by `--config_file`:
```
python runbenchmarks.py --mode "aws" --config_file custom_configs.yaml
```

The above command will deploy the infrastructure automatically and create a lambda_function with the `LAMBDA_FUNCTION_NAME` of your choice.

At the moment, you can run the below command to start benchmarking jobs on AWS:
```
./lambda_invoke.sh
```
Integration with the API is under active development.
