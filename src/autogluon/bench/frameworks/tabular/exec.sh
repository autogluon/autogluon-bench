#!/bin/bash

set -eo pipefail

framework=$1
benchmark=$2
constraint=$3
venv_base_dir=$4 
metrics_dir=$5
shift 5

while getopts "t:f:u:" opt; do
    case $opt in
        t) task=$OPTARG;;
        f) fold=$OPTARG;;
        u) user_dir=$OPTARG;;
        :\?) echo "Error: invaled option -$OPTARG"; exit1;;
        :) echo "Error: option -$OPTARG requires an argument"; exit1;;
    esac
done

amlb_args="$framework $benchmark $constraint -o $metrics_dir -s skip"  # skip installing framework since it's been done in setup.sh

if [ -n "$task" ]; then
    amlb_args+=" -t $task"
fi

if [ -n "$fold" ]; then
    amlb_args+=" -f $fold"
fi

if [ -n "$user_dir" ]; then
    cp -r $user_dir $venv_base_dir
    amlb_args+=" -u $user_dir"
fi

source $venv_base_dir/.venv/bin/activate

echo "Running AMLB benchmark with args $amlb_args"
python3 $venv_base_dir/automlbenchmark/runbenchmark.py $amlb_args
