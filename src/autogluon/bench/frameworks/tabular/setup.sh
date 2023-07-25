#!/bin/bash

set -eo pipefail

GIT_URI=$1
BRANCH=$2
DIR=$3  # from root of benchmark run


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
pip install -r $DIR/automlbenchmark/requirements.txt
