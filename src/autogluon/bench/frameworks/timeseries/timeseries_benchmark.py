import logging
import os
import subprocess
import sys

from autogluon.bench.frameworks.benchmark import Benchmark

logger = logging.getLogger(__name__)


class TimeSeriesBenchmark(Benchmark):
    def setup(
        self,
        git_uri: str = "https://github.com/openml/automlbenchmark.git",
        git_branch: str = "stable",
        framework: str = "AutoGluon:stable",
        user_dir: str = None,
    ):
        """Sets up the virtual environment for timeseries benchmark."""
        setup_script_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "setup.sh")
        command = [setup_script_path, git_uri, git_branch, self.benchmark_dir, framework]
        if user_dir is not None:
            command.append(user_dir)
        result = subprocess.run(command)
        if result.returncode != 0:
            logger.error(result.stderr)
            sys.exit(1)
        else:
            logger.info(result.stdout)
            logger.info("Successfully set up the environment under %s/.venv.", self.benchmark_dir)

    def run(
        self,
        framework: str,
        benchmark: str,
        constraint: str,
        task: str = None,
        fold: int = None,
        user_dir: str = None,
    ):
        """Runs the timeseries benchmark.

        Args:
            framework (str): The name of the framework to use. Examples: "AutoGluon:latest", "AutoGluon:stable".
            benchmark (str): The name of the benchmark to run.
            constraint (str): The name of the constraint to use.
            task (str): The name of the task to run (default: None).
            user_dir (str): Path to custom configs to use (default: None).

        Returns:
            None
        """
        if os.environ.get("RUNNING_IN_DOCKER", False):
            venv_base_dir = "/home/"
        else:
            venv_base_dir = self.benchmark_dir

        exec_script_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "exec.sh")
        command = [
            exec_script_path,
            framework,
            benchmark,
            constraint,
            venv_base_dir,
            self.metrics_dir,
        ]

        if task is not None:
            command += ["-t", task]

        if fold is not None:
            command += ["-f", str(fold)]

        if user_dir is not None:
            command += ["-u", user_dir]

        result = subprocess.run(command)
        if result.returncode != 0:
            sys.exit(1)
        else:
            logger.info(f"Benchmark {self.benchmark_name} is complete.")
