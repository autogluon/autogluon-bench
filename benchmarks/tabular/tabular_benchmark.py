from ..benchmark import Benchmark

import subprocess


class TabularBenchmark(Benchmark):
    def __init__(
        self, 
        benchmark_name: str, 
        root_dir: str = "./benchmarks/tabular/benchmark_runs",
    ):
        super().__init__(
            module="tabular",
            benchmark_name=benchmark_name,
            root_dir=root_dir,
        )
    
    def setup(
        self, 
    ):
        setup_command = ["./autogluon_bench/tabular/setup.sh", self.benchmark_dir]
        result = subprocess.run(setup_command, capture_output=True, text=True)
        if result.stdout:
            print(f"Successfully set up the environment under {self.benchmark_dir}/.venv")
        elif result.stderr:
            print(result.stderr)

    
    def run(
        self,
        framework: str = "AutoGluon:latest",
        benchmark: str = "test",
        constraint: str = "test",
    ):
        command = ["./autogluon_bench/tabular/benchmark_runner.sh", framework, benchmark, constraint, self.metrics_path, self.benchmark_dir]
        subprocess.run(command, stdout=subprocess.PIPE)


    def save_metrics(self, metrics):
        return super().save_metrics(metrics)