<div align="left">
  <img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</div>

# AutoGluon-Bench

Welcome to AutoGluon-Bench, a suite for benchmarking your AutoML frameworks.

## Setup

Follow the steps below to set up autogluon-bench:

```bash
# create virtual env and update pip
python3 -m venv .venv_agbench
source .venv_agbench/bin/activate
python3 -m pip install --upgrade pip
```

Install `autogloun-bench` from PyPI:

```bash
python3 -m pip install autogluon.bench
```

Install `autogluon-bench` from source for development:

```bash
git clone https://github.com/autogluon/autogluon-bench.git
cd autogluon-bench

# install from source in editable mode
pip install -e ".[tests]"
```

For development, please be aware that `autogluon.bench` is installed as a dependency in certain places, such as the [Dockerfile](https://github.com/autogluon/autogluon-bench/blob/master/src/autogluon/bench/Dockerfile) and [Multimodal Setup](https://github.com/autogluon/autogluon-bench/blob/master/src/autogluon/bench/frameworks/multimodal/setup.sh). We made it possible to reflect the development changes by pushing the changes to a remote GitHub branch, and providing the URI when testing on benchmark runs:

```
agbench run sample_configs/multimodal_cloud_configs.yaml --dev-branch https://github.com/<username>/autogluon-bench.git#<dev_branch>
```


## Run benchmarks locally

To run the benchmarks on your local machine, use the following command:

```
agbench run path/to/local_config_file
```

Check out our [sample local configuration files](https://github.com/autogluon/autogluon-bench/blob/master/sample_configs) for local runs.

The results are stored in the following directory: `{WORKING_DIR}/{root_dir}/{module}/{benchmark_name}_{timestamp}`.


### Tabular Benchmark

To perform tabular benchmarking, set the module to tabular. You must set both Benchmark Configurations and Tabular Specific configurations, and each should have a single value. Refer to the [sample configuration file](https://github.com/autogluon/autogluon-bench/blob/master/sample_configs/tabluar_local_configs.yaml) for more details.

The tabular module leverages the [AMLB](https://github.com/openml/automlbenchmark) benchmarking framework. Required and optional AMLB arguments are specified via the configuration file mentioned previously.

Custom configuration is supported by providing a local directory to `amlb_user_dir` in the config, by which custom frameworks, constraints and datasets can be overriden. We have a minimum working [custom config](https://github.com/autogluon/autogluon-bench/blob/master/sample_configs/amlb_configs) setup for benchmarking on a custom framework (a `AutoGluon` dev branch). In the [sample configuration file](https://github.com/autogluon/autogluon-bench/blob/master/sample_configs/tabluar_local_configs.yaml), change the following field to:

```
framework: AutoGluon_dev:example
amlb_user_dir: path_to/sample_configs/amlb_configs 
```

For more customizations, please follow the [example custom configuration folder](https://github.com/openml/automlbenchmark/tree/master/examples/custom) provided by AMLB and their [documentation](https://github.com/openml/automlbenchmark/blob/master/docs/HOWTO.md#custom-configuration). 


### Multimodal Benchmark

For multimodal benchmarking, set the module to multimodal. We currently support benchmarking multimodal on a custom branch. Note that multimodal benchmarking directly calls the MultiModalPredictor, bypassing the extra layer of [AMLB](https://github.com/openml/automlbenchmark). Therefore, the required arguments are different from those for tabular.

You can add more datasets to your benchmarking jobs. We provided sample [multimodal datasets](https://github.com/autogluon/autogluon-bench/blob/master/src/autogluon/bench/datasets/multimodal_dataset.py) and [object detection dataset](https://github.com/autogluon/autogluon-bench/blob/master/src/autogluon/bench/datasets/object_detection_dataset.py). Follow these samples to add custom datasets, then specify dataset_name in your local config file. Please follow the section `Install From Source` for more instructions on how to develop with source.

## Run benchmarks on AWS

AutoGluon-Bench uses the AWS CDK to build an AWS Batch compute environment for benchmarking.

To get started, install [Node.js](https://nodejs.org/) and [AWS CDK](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html#getting_started_install) with the following instructions:

1. Install [Node Version Manager](https://github.com/nvm-sh/nvm#installing-and-updating).
2. Source profile or restart the terminal.
3. Follow the `Prerequisites` section on the [AWS CDK Guide](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html) and install an appropriate `Node.js` version for your system:
```bash
nvm install $VERSION  # install Node.js
npm install -g aws-cdk  # install aws-cdk
cdk --version  # verify the installation, you might need to update the Node.js version depending on the log.
```
4. Follow the [AWS CLI Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to install `awscliv2`. 

If it is the first time using CDK to deploy to an AWS environment (An AWS environment is a combination of an AWS account and Region), please run the following:

```bash
cdk bootstrap aws://CDK_DEPLOY_ACCOUNT/CDK_DEPLOY_REGION
```

You will need a cloud configuration file to run the benchmarks. You can edit the provided [sample cloud config files](https://github.com/autogluon/autogluon-bench/blob/master/sample_configs), or use the CLI tool to generate the cloud config files locally.

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

After having the configuration file ready, use the command below to initiate benchmark runs on cloud:

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

A cloud configuration file with time-stamped `benchmark_name` is also saved under `{WORKING_DIR}/{root_dir}/{module}/{benchmark_name}_{timestamp}/{module}_cloud_configs.yaml`

By default, the infrastructure created is retained for future use. To automatically remove resources after the run, use the `--remove_resources` option:

```bash
agbench run path/to/cloud_config_file --remove-resources
```

This will check the job status every 2 minutes and remove resources after all jobs succeed. If any job fails, resources will be kept.

If you want to manually remove resources later, use:

```bash
agbench destroy-stack --config-file `{WORKING_DIR}/{root_dir}/{module}/{benchmark_name}_{timestamp}/aws_configs.yaml`
```

Or you can remove specific stacks by running:

```bash
agbench destroy-stack --static-resource-stack STATIC_RESOURCE_STACK_NAME --batch-stack BATCH_STACK_NAME --cdk-deploy-account CDK_DEPLOY_ACCOUNT --cdk-deploy-region CDK_DEPLOY_REGION
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

To override these configurations, use the `cdk_context` key in your custom config file. See our [sample cloud config](https://github.com/autogluon/autogluon-bench/blob/master/sample_configs/tabular_cloud_configs.yaml) for reference.


### Monitoring metrics for your instances on AWS

A variety of metrics are available for the EC2 instances that are launched during benchmarking. These can be accessed through the AWS Console by following this navigation path: `CloudWatch` -> `All metrics` -> `AWS namespaces` -> `EC2`. For a comprehensive list of these metrics, refer to the [official AWS documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/viewing_metrics_with_cloudwatch.html).

In addition to the standard metrics, we also provide a custom metric for `GPUUtilization`. This can be found in the `CloudWatch` section under `All metrics` -> `Custom namespaces` -> `EC2`. Please note that the `GPUUtilization` metric is also updated every five minutes.


## Evaluating bechmark runs

Tabular benchmark results can be evaluated using the tools in `src/autogluon/bench/eval/`. The evaluation logic will aggregate, clean, and produce evaluation results for runs stored in S3.
In a future release, we intend to add evaluation support for multimodal benchmark results.


### Evaluation Steps

Begin by setting up AWS credentials for the default profile for the AWS account that has the benchmark results in S3.

Step 1: Aggregate AMLB results on S3. After running the benchmark in [AWS mode](#run-benchmarks-on-aws), take note of the `benchmark_name` with timestamp in `{WORKING_DIR}/{root_dir}/{module}/{benchmark_name}_{timestamp}/{module}_cloud_configs.yaml` and run the command below:
```
agbench aggregate-amlb-results {METRICS_BUCKET} {module} {benchmark_name} --constraint {constraint}
```

This will create a new file on S3 with this signature:
```
s3://{METRICS_BUCKET}/aggregated/{module}/{benchmark_name}/results_automlbenchmark_{constraint}_{benchmark_name}.csv
```

Currently, aggregation is also supported for multimodal benchmark results without the `--constratint` option.

For more details, run:
```
agbench aggregate-amlb-results --help
```

Step 2: Further clean the aggregated results.

If the file is still on S3 from the previous step, run:
```
agbench clean-amlb-results {benchmark_name} --results-dir-input s3://{METRICS_BUCKET}/aggregated/{module}/{benchmark_name}/ --benchmark-name-in-input-path --constraints constratint_1 --constraints constratint_2 --results-dir-output {results_dir_output} 
--out-path-prefix {out_path_prefix} --out-path-suffix {out_path_suffix}
```
where `{results_dir_input}` can also be a local directory. This will create a local file `{results_dir_output}/{out_path_prefix}{benchmark_name}{out_path_suffix}`.

For more details, run:
```
agbench clean-amlb-results --help
```

Step 3: Run evaluation on multiple cleaned files from `Step 2`

```
agbench evaluate-amlb-results --frameworks_run framework_1 --frameworks_run framework_2 --results-dir-input data/results/input/prepared/openml --paths file_name_1.csv --paths file_name_2.csv --no-clean-data
```
