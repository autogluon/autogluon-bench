import os
import subprocess
import yaml
import tempfile

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
        custom_branch: str = None,
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
            command += ["-t", task]

        if custom_branch is not None:

            custom_repo, custom_branch_name = tuple(custom_branch.split('#'))

            temp_dirpath = tempfile.mkdtemp()
            custom_framework_name = "AutoGluon_dev"
            command[1] = custom_framework_name

            custom_config_contents = {
                "frameworks": {
                    "definition_file": ["{root}/resources/frameworks.yaml", "{user}/frameworks.yaml"],
                    "allow_duplicates": "true"
                }
            }

            with open(os.path.join(temp_dirpath, "config.yaml"), "w") as fo:
               yaml.dump(custom_config_contents, fo)

            custom_framework_contents = {
                custom_framework_name: {
                    "extends": "AutoGluon",
                    "repo": custom_repo,
                    "version": custom_branch_name
                }
            }

            with open(os.path.join(temp_dirpath, "frameworks.yaml"), "w") as fo:
               yaml.dump(custom_framework_contents, fo)

            command += ["-c", temp_dirpath]

        subprocess.run(command)

