#!/bin/bash

set -eo pipefail

framework=${1}
benchmark=${2}
constraint=${3}
DIR=${4}  # from root of project
shift 4

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
    cp -r $custom_dir $DIR
    amlb_args+=" -u $custom_dir"
fi

cd $DIR
source .venv/bin/activate

python ./automlbenchmark/runbenchmark.py  $amlb_args
