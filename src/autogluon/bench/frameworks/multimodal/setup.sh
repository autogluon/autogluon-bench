#!/bin/bash

set -eo pipefail

GIT_URI=$1
BRANCH=$2
venv_base_dir=$3  # from root of benchmark run
AG_BENCH_VERSION=$4

if [ ! -d $venv_base_dir ]; then
  mkdir -p $venv_base_dir
fi

echo "Cloning $GIT_URI#$BRANCH..."
repo_name=$(basename -s .git $(echo $GIT_URI))
git clone --depth 1 --single-branch --branch ${BRANCH} --recurse-submodules ${GIT_URI} $venv_base_dir/$repo_name

# create virtual env
python3 -m venv $venv_base_dir/.venv
source $venv_base_dir/.venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools wheel

if echo "$AG_BENCH_VERSION" | grep -q "dev"; then
  # install from local source or docker
  pip install .
else
  pip install autogluon.bench==$AG_BENCH_VERSION
fi

cd $venv_base_dir/$repo_name

python3 -m pip install -e common
python3 -m pip install -e core[all]
python3 -m pip install -e features
python3 -m pip install -e multimodal

python3 -m mim install "mmcv==2.0.1"
python3 -m pip install "mmdet==3.0.0"
