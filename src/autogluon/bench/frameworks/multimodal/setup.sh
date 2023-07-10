#!/bin/bash

set -eo pipefail

GIT_URI=$1
BRANCH=$2
DIR=$3  # from root of benchmark run
ARG=$4

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
  python3 -m pip install autogluon.bench==$AG_BENCH_VER
else
  echo "Invalid argument: $ARG"
  exit 1
fi

cd $DIR/$repo_name

python3 -m pip install -e common/[tests]
python3 -m pip install -e core/[all,tests]
python3 -m pip install -e features/
python3 -m pip install -e multimodal/[tests]

python3 -m mim install mmcv
python3 -m pip install "mmdet==3.0.0"

