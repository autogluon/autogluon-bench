#!/bin/bash

# Default values
git_uri="https://github.com/autogluon/autogluon.git"
git_branch="master"
module="multimodal"
data_path="MNIST"
benchmark_name="test"
framework="AutoGluon" # for AMLB
label="latest"
amlb_benchmark="test"
amlb_constraint="test"


OPTIONS=$(getopt \
    -o "" \
    -l "git_uri:" \
    -l "git_branch:" \
    -l "module:" \
    -l "data_path:" \
    -l "benchmark_name:" \
    -l "framework:" \
    -l "label:" \
    -l "amlb_benchmark:" \
    -l "amlb_constraint:" \
    -l "amlb_task:" \
    -- \
    "$@"
)
eval set -- "$OPTIONS"

while true; do
  case $1 in
    --git_uri)
      git_uri="$2"
      shift 2
      ;;
    --git_branch)
      git_branch="$2"
      shift 2
      ;;
    --module)
      module="$2"
      shift 2
      ;;
    --data_path)
      data_path="$2"
      shift 2
      ;;
    --benchmark_name)
      benchmark_name="$2"
      shift 2
      ;;
    --framework)
      framework="$2"
      shift 2
      ;;
    --label)
      label="$2"
      shift 2
      ;;
    --amlb_benchmark)
      amlb_benchmark="$2"
      shift 2
      ;;
    --amlb_constraint)
      amlb_constraint="$2"
      shift 2
      ;;
    --amlb_task)
      amlb_task="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Invalid option: $1" >&2
      exit 1
      ;;
  esac
done

if [ $module = "multimodal" ]; then
    python ./run_benchmarks.py \
        --git_uri $git_uri \
        --git_branch $git_branch \
        --module $module \
        --data_path $data_path\
        --benchmark_name $benchmark_name 
elif [ $module = "tabular" ]; then
    command="python ./run_benchmarks.py \
        --module $module \
        --benchmark_name $benchmark_name \
        --framework $framework \
        --label $label \
        --amlb_benchmark $amlb_benchmark \
        --amlb_constraint $amlb_constraint \
    "
    if [ -n "$amlb_task" ]; then
        command="$command --amlb_task $amlb_task"
    fi
    $command
else
    echo "Invalid module type: $MODULE"
fi