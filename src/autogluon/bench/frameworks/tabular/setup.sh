#!/bin/bash

set -eo pipefail

DIR=${1:-"./benchmark_runs/tabular/test"}  # from root of project

if [ ! -d $DIR ]; then
  mkdir -p $DIR
fi

# create virtual env
python3 -m venv $DIR/.venv
source $DIR/.venv/bin/activate

# install latest AMLB
pip install --upgrade pip
pip install --upgrade setuptools wheel
git clone --depth 1 --branch stable https://github.com/openml/automlbenchmark.git $DIR/automlbenchmark
pip install -r $DIR/automlbenchmark/requirements.txt
