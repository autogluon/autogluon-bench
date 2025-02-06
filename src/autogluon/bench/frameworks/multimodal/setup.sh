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
python -m venv $venv_base_dir/.venv
source $venv_base_dir/.venv/bin/activate

python -m pip install --upgrade pip
python -m pip install uv
python -m uv pip install --upgrade setuptools wheel

if echo "$AG_BENCH_VERSION" | grep -q "dev"; then
  # install from local source or docker
  python -m uv pip install .
else
  python -m uv pip install autogluon.bench==$AG_BENCH_VERSION
fi

cd $venv_base_dir/$repo_name

python -m uv pip install -e common
python -m uv pip install -e core[all]
python -m uv pip install -e features
python -m uv pip install -e multimodal

python -m mim install "mmcv==2.1.0" --timeout 60
python -m uv pip install "mmdet==3.2.0"
