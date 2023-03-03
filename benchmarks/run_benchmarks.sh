#!/bin/bash

USER=${1:-"suzhoum"}
BRANCH=${2:-"master"}
MODULE=${3:-"multimodal"}
DATA_PATH=${4:-"MNIST"}
BENCHMARK_NAME=${5:-"bench_test_local"}


# # multimodal
# python3 ./scripts/run_benchmarks.py \
#     --git_user suzhoum \
#     --git_branch skip_validation \
#     --module multimodal \
#     --data_path MNIST \
#     --benchmark_name bench_test_local 

# tabular
python3 ./scripts/run_benchmarks.py \
    --git_user $USER \
    --git_branch $BRANCH \
    --module $MODULE \
    --data_path $DATA_PATH\
    --benchmark_name $BENCHMARK_NAME 