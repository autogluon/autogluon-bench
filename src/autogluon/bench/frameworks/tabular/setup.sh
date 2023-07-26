#!/bin/bash

set -eo pipefail

GIT_URI=$1  # AMLB Git URI
BRANCH=$2  # AMLB branch
DIR=$3  # from root of benchmark run
AMLB_FRAMEWORK=$4  # e.g. AutoGluon_dev:test
AMLB_USER_DIR=$5  # directory where AMLB customizations are located


if [ ! -d $DIR ]; then
  mkdir -p $DIR
fi

# create virtual env
python3 -m venv $DIR/.venv
source $DIR/.venv/bin/activate

echo "Cloning $GIT_URI#$BRANCH..."
repo_name=$(basename -s .git $(echo $GIT_URI))
git clone --depth 1 --branch ${BRANCH} ${GIT_URI} $DIR/$repo_name

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools wheel
pip install -r $DIR/automlbenchmark/requirements.txt

# install amlb framework only
echo "Installing framework $AMLB_FRAMEWORK"
amlb_args="$AMLB_FRAMEWORK -s only"

if [ -n "$AMLB_USER_DIR" ]; then
    echo "using user_dir $AMLB_USER_DIR"
    amlb_args+=" -u $AMLB_USER_DIR"
fi
python3 $DIR/automlbenchmark/runbenchmark.py $amlb_args
