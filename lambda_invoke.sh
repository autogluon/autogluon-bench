#!/bin/bash

get_lambda_function_name () {
  echo `grep "LAMBDA_FUNCTION_NAME" cdk.context.json | sed -r 's/^[^:]*:(.*),/\1/' | xargs`
}

# payload="{
#     \"benchmark_name\": \"test\",
#     \"module\": \"multimodal\",
#     \"data_paths\": [
#         \"MNIST\"
#     ],
#     \"git_uri#branch\": [
#         \"https://github.com/autogluon/autogluon.git#master\"
#     ],
#     \"s3_bucket\": \"autogluon-benchmark-metrics\"
# }"

payload="{
    \"benchmark_name\": \"test\",
    \"module\": \"tabular\",
    \"frameworks\": [
        \"AutoGluon\"
    ],
    \"labels\": [
        \"latest\"
    ],
    \"amlb_benchmarks\": [
        \"test\",
        \"small\"
    ],
    \"amlb_constraints\": [
        \"test\"
    ],
    \"amlb_tasks\": {
        \"test\": [\"iris\"],
        \"small\": [\"credit-g\"]
    },
    \"s3_bucket\": \"autogluon-benchmark-metrics\"
}"

lambda_function_name=$(get_lambda_function_name)

echo "Submitting jobs to $lambda_function_name..."

aws lambda invoke \
--function-name $lambda_function_name \
--payload "$payload" \
--cli-binary-format raw-in-base64-out \
/dev/stdout
