#!/bin/bash

set -eo pipefail

GIT_URI=${1:-"https://github.com/autogluon/autogluon.git"}
BRANCH=${2:-"master"}
DIR=${3:-"./benchmark_runs/multimodal/test"}  # from root of benchmark run

if [ ! -d $DIR ]; then
  mkdir -p $DIR
fi
repo_name=$(basename -s .git $(echo $GIT_URI))
git clone --depth 1 --single-branch --branch ${BRANCH} --recurse-submodules ${GIT_URI} $DIR/$repo_name

# create virtual env
python3.9 -m venv $DIR/.venv
source $DIR/.venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools wheel

# install autogluon-bench as source, this script should be run from the root of path_to/autogluon-bench
python3 -m pip install -U -e .

cd $DIR/$repo_name

python3 -m pip install -e common/[tests]
python3 -m pip install -e core/[all,tests]
python3 -m pip install -e features/
python3 -m pip install -e multimodal/[tests]

python3 -m mim install -q mmcv-full
python3 -m pip install "mmdet>=2.28, <3.0.0"