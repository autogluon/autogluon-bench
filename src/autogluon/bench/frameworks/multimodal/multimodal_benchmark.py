import os
import subprocess

from autogluon.bench.benchmark import Benchmark


class MultiModalBenchmark(Benchmark):
    def __init__(
        self, benchmark_name: str, root_dir: str = "./benchmark_runs/multimodal/"
    ):
        super().__init__(
            benchmark_name=benchmark_name,
            root_dir=root_dir,
        )
        self.module = "multimodal"

    def setup(
        self,
        git_uri: str = "https://github.com/autogluon/autogluon.git",
        git_branch: str = "master",
    ):
        setup_script_path = os.path.abspath(os.path.dirname(__file__)) + "/setup.sh"
        command = [setup_script_path, git_uri, git_branch, self.benchmark_dir]
        result = subprocess.run(command)
        if result.stdout:
            print(
                f"Successfully set up the environment under {self.benchmark_dir}/.venv"
            )
        elif result.stderr:
            print(result.stderr)

    def run(self, data_path: str):
        PY_EXC_PATH = self.benchmark_dir + "/.venv/bin/python"
        exec_path = os.path.abspath(os.path.dirname(__file__)) + "/exec.py"
        command = [
            PY_EXC_PATH,
            exec_path,
            "--data_path",
            data_path,
            "--metrics_dir",
            self.metrics_dir,
        ]
        subprocess.run(command)
