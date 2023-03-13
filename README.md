# AutoGluon-Bench

## Setup

```
git clone https://github.com/suzhoum/autogluon-bench.git
cd autogluon-bench
git checkout poc_0.0.1

# create virtual env
python3.9 -m venv .venv
source .venv/bin/activate

# install autogluon-bench
pip install -e .
```

## Run benchmarkings

Currently, `tabular` makes use of the [AMLB](https://github.com/openml/automlbenchmark) benchmarking framework, so we will supply the required and optional arguments when running benchmark for tabular. It can only run on the stable and latest(master) build of the framework at the moment. In order to benchmark on a custom branch, changes need to be made on AMLB.

```
python ./runbenchmarks.py  --module tabular --mode local  --benchmark_name local_test --framework AutoGluon --label stable --amlb_benchmark test --amlb_constraint test --amlb_task iris
```

where `--amlb_task` is optional, and corresponds to `--task` argument in AMLB. You can refer to [Quickstart of AMLB](https://github.com/openml/automlbenchmark#quickstart) for more details.

On the other hand, `multimodal` benchmarking directly calls the `MultiModalPredictor` without the extra layer of [AMLB](https://github.com/openml/automlbenchmark), so the set of arguments we call is different from that of running `tabular`. Currently, we support benchmarking `multimodal` on custom branch of the main repository or any forked repository.

```
python ./runbenchmarks.py --module multimodal --mode local --git_uri https://github.com/autogluon/autogluon.git --git_branch master --data_path MNIST --benchmark_name local_test
```
To customize the benchmarking experiment, including adding more hyperparameters, and evaluate on more metrics, you can refer to `./src/autogluon/bench/frameworks/multimodal/exec.py`.


Results are saved under `$WORKING_DIR/benchmark_runs/$module/{$benchmark_name}_{$timestamp}`

