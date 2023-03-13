#!/bin/bash

GIT_URI=${1:-"https://github.com/suzhoum/autogluon.git"}
BRANCH=${2:-"master"}
DIR=${3:-"./benchmark_runs/multimodal/test"}  # from root of benchmark run

if [ ! -d $DIR ]; then
  mkdir -p $DIR
fi
repo_name=$(basename -s .git $(echo $GIT_URI))
git clone --depth 1 --single-branch --branch ${BRANCH} --recurse-submodules ${GIT_URI} $DIR/$repo_name

# create virtual env
echo $DIR
python3.8 -m venv $DIR/.venv
source $DIR/.venv/bin/activate

pip install --upgrade pip
pip install --upgrade setuptools wheel

cd $DIR/autogluon
./full_install.sh


