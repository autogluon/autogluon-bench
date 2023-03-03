import argparse
import sys

sys.path.insert(0, "/home/ubuntu/src/autogluon_bench/")

from benchmarks.multimodal.multimodal_benchmark import MultiModalBenchmark
from benchmarks.tabular.tabular_benchmark import TabularBenchmark

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--git_user', type=str, help='GitHub username for cloning the repository')
    parser.add_argument('--git_branch', type=str, default='main', help='Git branch to checkout (default: main)')
    parser.add_argument('--module', type=str, choices=['tabular', 'multimodal', 'ts'], help='AutoGluon modules.')
    parser.add_argument('--data_path', type=str, help='Can be one of: dataset name, local path, S3 path, AMLB task ID/name')
    parser.add_argument('--benchmark_name', type=str, help='A unique name for the benchmark run.')


    args = parser.parse_args()
    return args


def run():
    args = get_args()
    module = args.module
    data_path = args.data_path
    benchmark_name = args.benchmark_name

    if module == "multimodal":
        benchmark = MultiModalBenchmark(
            benchmark_name=benchmark_name, 
        )
        benchmark.setup(git_user=args.git_user, git_branch=args.git_branch)
        metrics = benchmark.run(
            data_path=data_path,
            time_limit=10,
        )
        benchmark.save_metrics(metrics=metrics)
    elif module == "tabular":
        benchmark = TabularBenchmark(
            benchmark_name=benchmark_name, 
        )
        benchmark.setup()
        benchmark.run(
            framework=f"AutoGluon:{args.git_branch}",
            benchmark="test",
            constraint="test"
        )
    elif module == "ts":
        raise NotImplementedError


if __name__ == '__main__':
    run()