#!/bin/bash

USER=${1:-"suzhoum"}
BRANCH=${2:-"master"}
DIR=${3:-"./benchmarks/multimodal/benchmark_runs/test"}  # from root of project

if [ ! -d $DIR ]; then
  mkdir -p $DIR
fi


# create virtual env
cd $DIR
python3.8 -m venv .venv
source venv/bin/activate

REPO="https://github.com/${USER}/autogluon.git"
pip install --upgrade pip
pip install --upgrade setuptools wheel
pip install -r ./autogluon_bench/multimodal/requirements.txt
git clone --depth 1 --single-branch --branch ${BRANCH} --recurse-submodules ${REPO}
./autogluon/full_install.sh


