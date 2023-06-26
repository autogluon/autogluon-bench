#!/bin/bash

set -eo pipefail

GIT_URI=$1
BRANCH=$2
DIR=$3  # from root of benchmark run
AG_BENCH_VER=$4

if [ ! -d $DIR ]; then
  mkdir -p $DIR
fi
repo_name=$(basename -s .git $(echo $GIT_URI))
git clone --depth 1 --single-branch --branch ${BRANCH} --recurse-submodules ${GIT_URI} $DIR/$repo_name

# create virtual env
python3 -m venv $DIR/.venv
source $DIR/.venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools wheel

python3 -m pip install autogluon.bench==$AG_BENCH_VER

cd $DIR/$repo_name

python3 -m pip install -e common/[tests]
python3 -m pip install -e core/[all,tests]
python3 -m pip install -e features/
python3 -m pip install -e multimodal/[tests]

python3 -m mim install mmcv
python3 -m pip install "mmdet==3.0.0"

