#!/bin/bash

framework=${1:-"AutoGluon:latest"}
benchmark=${2:-"test"}
constraint=${3:-"test"}
DIR=${4:-"./benchmark_runs/tabular/test"}  # from root of project

OPTIONS=$(getopt -o "" -l "task:" --name "$0" -- "$@")
eval set -- "$OPTIONS"
while true; do
  case $1 in
    --task)
      task="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Invalid option: $1"
      exit 1
      ;;
  esac
done

cd $DIR
source .venv/bin/activate
if [ -n "$task" ]; then
    python ./automlbenchmark/runbenchmark.py $framework $benchmark $constraint -t $task -s force
else
    python ./automlbenchmark/runbenchmark.py $framework $benchmark $constraint -s force
fi
