#!/bin/bash

set -eo pipefail

framework=${1}
benchmark=${2}
constraint=${3}
benchmark_dir=${4}  # from root of project
metrics_dir=${5}
shift 5

while getopts "t:c:" opt; do
    case $opt in
        t) task=$OPTARG;;
        c) custom_dir=$OPTARG;;
        :\?) echo "Error: invaled option -$OPTARG"; exit1;;
        :) echo "Error: option -$OPTARG requires an argument"; exit1;;
    esac
done

amlb_args="$framework $benchmark $constraint -s force"

if [ -n "$task" ]; then
    amlb_args+=" -t $task"
fi

if [ -n "$custom_dir" ]; then
    cp -r $custom_dir $benchmark_dir
    amlb_args+=" -u $custom_dir"
fi

amlb_args+=" -o $metrics_dir"

source $benchmark_dir/.venv/bin/activate

python3 $benchmark_dir/automlbenchmark/runbenchmark.py $amlb_args
