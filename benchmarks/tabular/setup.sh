#!/bin/bash

DIR=${1:-"./benchmarks/tabular/benchmark_runs/test"}  # from root of project

if [ ! -d $DIR ]; then
  mkdir -p $DIR
fi

# create virtual env
cd $DIR
python3.8 -m venv .venv
source venv/bin/activate

# install latest AMLB
git clone --branch stable --depth 1 https://github.com/openml/automlbenchmark.git 
pip install -r ./automlbenchmark/requirements.txt
