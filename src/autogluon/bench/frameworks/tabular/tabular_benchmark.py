import os
import subprocess

from autogluon.bench.benchmark import Benchmark


class TabularBenchmark(Benchmark):
    def __init__(self, benchmark_name: str, root_dir: str = "./benchmark_runs/tabular/"):
        super().__init__(
            benchmark_name=benchmark_name,
            root_dir=root_dir,
        )
        self.module = "tabular"

    def setup(
        self,
    ):
        setup_script_path = os.path.abspath(os.path.dirname(__file__)) + "/setup.sh"
        command = [setup_script_path, self.benchmark_dir]
        result = subprocess.run(command)
        if result.stdout:
            print(f"Successfully set up the environment under {self.benchmark_dir}/.venv")
        elif result.stderr:
            print(result.stderr)

    def run(
        self,
        framework: str = "AutoGluon:latest",
        benchmark: str = "test",
        constraint: str = "test",
        task: str = None,
    ):
        exec_script_path = os.path.abspath(os.path.dirname(__file__)) + "/exec.sh"
        command = [
            exec_script_path,
            framework,
            benchmark,
            constraint,
            self.benchmark_dir,
        ]
        if task is not None:
            command += ["--task", task]
        subprocess.run(command)
