<div align="left">
  <img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</div>

# AutoGluon-Bench

Welcome to AutoGluon-Bench, a suite for benchmarking your AutoML frameworks.

## Setup

Follow the steps below to set up autogluon-bench:

```bash
# create virtual env
python3 -m venv .venv_agbench
source .venv_agbench/bin/activate
```

Install `autogloun-bench` from PyPI:

```bash
python3 -m pip install autogluon.bench
```

Or install `autogluon-bench` from source:

```bash
git clone https://github.com/autogluon/autogluon-bench.git
cd autogluon-bench

# install from source in editable mode
pip install -e .
```

For development, please be aware that `autogluon.bench` is installed as a dependency in certain places, such as the [Dockerfile](https://github.com/autogluon/autogluon-bench/blob/master/src/autogluon/bench/Dockerfile) and [Multimodal Setup](https://github.com/autogluon/autogluon-bench/blob/master/src/autogluon/bench/frameworks/multimodal/setup.sh). To ensure that your local changes are reflected in the installed package, you may need to adjust the installation command as necessary. A recommended approach is to push your changes to a remote git branch and then pull from this branch, installing the package from source in the scripts.


## Run benchmarks locally

To run the benchmarks on your local machine, use the following command:

```
agbench run path/to/local_config_file
```

Check out our [sample local configuration files](https://github.com/autogluon/autogluon-bench/blob/master/sample_configs) for local runs.

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

To get started, install [Node.js](https://nodejs.org/) and [AWS CDK](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html#getting_started_install) with the following instructions:

1. Install [Node Version Manager](https://github.com/nvm-sh/nvm#installing-and-updating).
2. Source profile or restart the terminal.
3. Follow the `Prerequisites` section on the [AWS CDK Guide](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html) and install an appropriate version for your system:
```bash
nvm install $VERSION  # install Node.js
npm install -g aws-cdk  # install aws-cdk
cdk --version  # verify the installation, you might need to update the Node.js version depending on the log.
```

If it is the first time using CDK to deploy to an AWS environment (An AWS environment is a combination of an AWS account and Region), please run the following:

```bash
cdk bootstrap aws://CDK_DEPLOY_ACCOUNT/CDK_DEPLOY_REGION
```

To initiate benchmarking on the cloud, use the command below:

```
agbench run /path/to/cloud_config_file
```

You can edit the provided [sample cloud config files](https://github.com/autogluon/autogluon-bench/blob/master/sample_configs), or use the CLI tool to generate the cloud config files locally.

For multimodal:

```
agbench generate-cloud-config --module multimodal --cdk-deploy-account <AWS_ACCOUNT_ID> --cdk-deploy-region <AWS_ACCOUNT_REGION> --prefix <PREFIX> --metrics-bucket <METRICS_BUCKET> --git-uri-branch <GIT_URI#BRANCH> --dataset-names DATASET_1,DATASET_2 --presets <PRESET_1>,<PRESET_2> --time-limit <TIME_LIMIT_1>,<TIME_LIMIT_2> --hyperparameters "key_1:value_1,key_2:value_2;key_1:value_3,key_2:value_4"
```

For tabular:
```
agbench generate-cloud-config --module tabular --cdk-deploy-account <AWS_ACCOUNT_ID> --cdk-deploy-region <AWS_ACCOUNT_REGION> --prefix <PREFIX> --metrics-bucket <METRICS_BUCKET> --framework <FRAMEWORK>:<LABEL> --amlb-benchmark <BENCHMARK1>,<BENCHMARK2> --amlb-task "BENCHMARK1:DATASET1,DATASET2;BENCHMARK2:DATASET3" --amlb-constraint <CONSTRAINT>
```

For more details, you can run
```
agbench generate-cloud-config --help
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
agbench destroy-stack --config_file `{WORKING_DIR}/{root_dir}/{module}/{benchmark_name}_{timestamp}/aws_configs.yaml`
```

Or you can remove specific stacks by running:

```bash
agbench destroy-stack --static_resource_stack STATIC_RESOURCE_STACK_NAME --batch_stack BATCH_STACK_NAME --cdk_deploy_account CDK_DEPLOY_ACCOUNT --cdk_deploy_region CDK_DEPLOY_REGION
```
where you can find all argument values in `{WORKING_DIR}/{root_dir}/{module}/{benchmark_name}_{timestamp}/aws_configs.yaml`.





### Configure the AWS infrastructure

The default infrastructure configurations are located [here](https://github.com/autogluon/autogluon-bench/blob/master/src/autogluon/bench/cloud/aws/default_config.yaml).

where:
- `CDK_DEPLOY_ACCOUNT` and `CDK_DEPLOY_REGION` should be overridden with your AWS account ID and desired region to create the stack.
- `PREFIX` is used as an identifier for the stack and resources created.
- `RESERVED_MEMORY_SIZE` is used together with the instance memory size to calculate the container shm_size.
- `BLOCK_DEVICE_VOLUME` is the size of storage device attached to instance.
- `LAMBDA_FUNCTION_NAME` lambda function to submit jobs to AWS Batch.

To override these configurations, use the `cdk_context` key in your custom config file. See our [sample cloud config](https://github.com/autogluon/autogluon-bench/blob/master/sample_configs/cloud_configs.yaml) for reference.

## Evaluating bechmark runs

Tabular benchmark results can be evaluated using the tools in `src/autogluon/bench/eval/`. The evaluation logic will aggregate, clean, and produce evaluation results for runs stored in S3.
In a future release, we intend to add evaluation support for multimodal benchmark results.

### Evaluation Steps

Begin by setting up AWS credentials for the default profile for the AWS account that has the benchmark results in S3.

Step 1: Run the `aggregate_amlb_results.py` script
```
cd src/autogluon/bench/eval/
python scripts/aggregate_amlb_results.py --s3_bucket {AWS_BUCKET} --s3_prefix {AWS_PREFIX} --version_name {BENCHMARK_VERSION_NAME}

# example: python scripts/aggregate_amlb_results.py --s3_bucket autogluon-benchmark-metrics --s3_prefix tabular/ --version_name test_local_20230330T180916
```

This will create a new file in S3 with this signature:
```
s3://{AWS_BUCKET}/aggregated/{AWS_PREFIX}/{BENCHMARK_VERSION_NAME}/results.csv
# example: s3://autogluon-benchmark-metrics/aggregated/tabular/test_local_20230330T180916/results_automlbenchmark_None_test_local_20230330T180916.csv
```

Step 2: Run the `run_generate_clean_openml` script
```
python scipts/run_generate_clean_openml.py --run_name {NAME_OF_AGGREGATED_RUN} --file_prefix {S3_FILE_PREFIX_FROM_PREVIOUS_STEP} --results_input_dir {S3_PATH_PREFIX_FROM_LAST_STEP}
# example: python scripts/run_generate_clean_openml.py --run_name test_local_20230330T180916 --file_prefix results_automlbenchmark_None_test_local_20230330T180916 --results_input_dir s3://autogluon-benchmark-metrics/aggregated/tabular/test_local_20230330T180916/
```

This will create a local file of results in the `data/results/input/prepared/openml/` directory.
Like this: `./data/results/input/prepared/openml/openml_ag_test_local_20230330T180916.csv`

Step 3: Run the `benchmark_evaluation` python script
```
python scripts/run_evaluation_openml.py --frameworks_run {NAME_OF_FRAMEWORKS_TO_EVALUATE} --paths {PATHS_TO_CLEANED_FILES} --folds_to_keep {FOLD_IDENTIFIER}
# example: python scripts/run_evaluation_openml.py --frameworks_run AutoGluon_test_test_local_20230330T180916 --paths openml_ag_test_local_20230330T180916.csv --folds_to_keep 0
```

