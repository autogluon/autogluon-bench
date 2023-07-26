#!/bin/bash

set -eo pipefail

GIT_URI=$1
BRANCH=$2
DIR=$3  # from root of benchmark run
ARG=$4

if [ ! -d $DIR ]; then
  mkdir -p $DIR
fi

echo "Cloning $GIT_URI#$BRANCH..."
repo_name=$(basename -s .git $(echo $GIT_URI))
git clone --depth 1 --single-branch --branch ${BRANCH} --recurse-submodules ${GIT_URI} $DIR/$repo_name

# create virtual env
python3 -m venv $DIR/.venv
source $DIR/.venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools wheel

if [[ "$ARG" == "--AGBENCH_DEV_URL="* ]]; then
  AGBENCH_DEV_URL="${ARG#*=}"
  echo "Installing Dev Branch $AGBENCH_DEV_URL"
  AGBENCH_URI=$(echo "$AGBENCH_DEV_URL" | cut -d '#' -f 1)
  AGBENCH_BRANCH=$(echo "$AGBENCH_DEV_URL" | cut -d '#' -f 2)
  agbench_repo_name=$(basename -s .git $(echo $AGBENCH_URI))
  git clone --single-branch --branch ${AGBENCH_BRANCH} ${AGBENCH_URI} $DIR/$agbench_repo_name
  cd $DIR/$agbench_repo_name
  python3 -m pip install -e .
  cd -
elif [[ "$ARG" == "--AG_BENCH_VER="* ]]; then
  AG_BENCH_VER="${ARG#*=}"
  output=$(python3 -m pip install autogluon.bench==$AG_BENCH_VER 2>&1) || {
    err_message=$output
    if [[ $err_message == *"No matching distribution"* ]]; then
      echo -e "ERROR: No matching distribution found for autogluon.bench==$AG_BENCH_VER\n \
      To resolve the issue, try 'agbench run <config_file> --dev-branch <autogluon_bench_uri>#<git_branch>"
    fi
    exit 1
  }
else
  echo "Invalid argument: $ARG"
  exit 1
fi

cd $DIR/$repo_name

python3 -m pip install -e common
python3 -m pip install -e core[all]
python3 -m pip install -e features
python3 -m pip install -e multimodal

python3 -m mim install mmcv
python3 -m pip install "mmdet==3.0.0"
