#!/bin/bash

framework=${1:-"AutoGluon:latest"}
benchmark=${2:-"test"}
constraint=${3:-"test"}
output_dir=${4:-"./benchmarks/tabular/benchmark_run/test"}
DIR=${5:-"./benchmarks/tabular/benchmark_run/test"}  # from root of project

cd $DIR/automlbenchmark
python runbenchmark.py $framework $benchmark $constraint -o $output_dir -s force