import logging
import os
import subprocess
import tempfile

import yaml

from autogluon.bench.frameworks.benchmark import Benchmark

logger = logging.getLogger(__name__)


class TabularBenchmark(Benchmark):
    def setup(
        self,
    ):
        """Sets up the virtual environment for tabular benchmark."""
        setup_script_path = os.path.abspath(os.path.dirname(__file__)) + "/setup.sh"
        command = [setup_script_path, self.benchmark_dir]
        result = subprocess.run(command)
        if result.stdout:
            logging.info("Successfully set up the environment under %s/.venv.", self.benchmark_dir)
        elif result.stderr:
            logging.error(result.stderr)

    def run(
        self,
        framework: str = "AutoGluon:latest",
        benchmark: str = "test",
        constraint: str = "test",
        task: str = None,
        custom_branch: str = None,
    ):
        """Runs the tabular benchmark.

        Args:
            framework (str): The name of the framework to use (default: "AutoGluon:latest").
            benchmark (str): The name of the benchmark to run (default: "test").
            constraint (str): The name of the constraint to use (default: "test").
            task (str): The name of the task to run (default: None).
            custom_branch (str): The name of the custom branch to use (default: None).

        Returns:
            None
        """
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
            custom_repo, custom_branch_name = tuple(custom_branch.split("#"))

            temp_dirpath = tempfile.mkdtemp()
            custom_framework_name = "AutoGluon_dev"
            command[1] = custom_framework_name

            custom_config_contents = {
                "frameworks": {
                    "definition_file": ["{root}/resources/frameworks.yaml", "{user}/frameworks.yaml"],
                    "allow_duplicates": "true",
                }
            }

            with open(os.path.join(temp_dirpath, "config.yaml"), "w") as fo:
                yaml.dump(custom_config_contents, fo)

            custom_framework_contents = {
                custom_framework_name: {"extends": "AutoGluon", "repo": custom_repo, "version": custom_branch_name}
            }

            with open(os.path.join(temp_dirpath, "frameworks.yaml"), "w") as fo:
                yaml.dump(custom_framework_contents, fo)

            command += ["-c", temp_dirpath]

        subprocess.run(command)
