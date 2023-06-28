import logging
import os
import subprocess
import sys
from typing import List

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
        if result.returncode != 0:
            logger.error(result.stderr)
            sys.exit(1)
        else:
            logger.info(result.stdout)
            logger.info("Successfully set up the environment under %s/.venv.", self.benchmark_dir)

    def run(
        self,
        benchmark: str = "test",
        constraint: str = "test",
        task: List[str] = None,
        framework: str = None,
        custom_branch: str = None,
    ):
        """Runs the tabular benchmark.

        Args:
            benchmark (str): The name of the benchmark to run (default: "test").
            constraint (str): The name of the constraint to use (default: "test").
            task (List[str]): The name of the task to run (default: None).
            framework (str): The name of the framework to use (default: None). Examples: "AutoGluon:latest", "AutoGluon:stable".
            custom_branch (str): The name of the custom branch to use (default: None).

        Returns:
            None
        """
        if framework is None and custom_branch is None:
            raise KeyError("Either 'framework' or 'custom_branch' should be provided.")

        custom_branch_dir = None
        if custom_branch is not None:
            custom_repo, custom_branch_name = tuple(custom_branch.split("#"))
            custom_branch_dir = self.benchmark_dir

            framework = "AutoGluon_dev"

            custom_config_contents = {
                "frameworks": {
                    "definition_file": ["{root}/resources/frameworks.yaml", "{user}/frameworks.yaml"],
                    "allow_duplicates": "true",
                }
            }

            with open(os.path.join(custom_branch_dir, "amlb_configs.yaml"), "w") as fo:
                yaml.dump(custom_config_contents, fo)

            custom_framework_contents = {
                framework: {"extends": "AutoGluon", "repo": custom_repo, "version": custom_branch_name}
            }

            with open(os.path.join(custom_branch_dir, "frameworks.yaml"), "w") as fo:
                yaml.dump(custom_framework_contents, fo)

        exec_script_path = os.path.abspath(os.path.dirname(__file__)) + "/exec.sh"
        command = [
            exec_script_path,
            framework,
            benchmark,
            constraint,
            self.benchmark_dir,
            self.metrics_dir,
        ]

        if custom_branch_dir is not None:
            command += ["-c", custom_branch_dir]

        if task is not None:
            command += ["-t", " ".join(task)]

        result = subprocess.run(command)
        if result.returncode != 0:
            logger.error(result.stderr)
            sys.exit(1)
        else:
            logger.info(result.stdout)
            logger.info(f"Benchmark {self.benchmark_name} is complete.")
