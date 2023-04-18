#!/bin/bash

GIT_URI=${1:-"https://github.com/autogluon/autogluon.git"}
BRANCH=${2:-"master"}
DIR=${3:-"./benchmark_runs/multimodal/test"}  # from root of benchmark run

if [ ! -d $DIR ]; then
  mkdir -p $DIR
fi
repo_name=$(basename -s .git $(echo $GIT_URI))
git clone --depth 1 --single-branch --branch ${BRANCH} --recurse-submodules ${GIT_URI} $DIR/$repo_name

# create virtual env
python3.10 -m venv $DIR/.venv
source $DIR/.venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools wheel

cd $DIR/autogluon

python3 -m pip install -e multimodal
python3 -m mim install -q mmcv-full



