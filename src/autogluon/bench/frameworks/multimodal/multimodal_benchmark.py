import json
import logging
import os
import subprocess
import sys
from typing import Optional

from autogluon.bench import __version__ as agbench_version
from autogluon.bench.frameworks.benchmark import Benchmark

logger = logging.getLogger(__name__)


class MultiModalBenchmark(Benchmark):
    """
    A benchmark class for AutoGluon MultiModal.

    Attributes:
        benchmark_name (str): The name of the benchmark.
        root_dir (str): The root directory for the benchmark.
        module (str): The name of the module being benchmarked (multimodal).

    Methods:
        setup(): Sets up the virtual environment for running the benchmark.
        run(): Runs the benchmark on a given dataset.
    """

    def setup(
        self,
        git_uri: str = "https://github.com/autogluon/autogluon.git",
        git_branch: str = "master",
        agbench_dev_url: str = None,
    ):
        """
        Sets up the virtual environment for running the benchmark.

        Args:
            git_uri (str): The URI of the Git repository to clone (default: "https://github.com/autogluon/autogluon.git").
            git_branch (str): The branch of the Git repository to clone (default: "master").

        Returns:
            None
        """
        setup_script_path = os.path.abspath(os.path.dirname(__file__)) + "/setup.sh"
        command = [setup_script_path, git_uri, git_branch, self.benchmark_dir]
        if agbench_dev_url is not None:
            command.append(f"--AGBENCH_DEV_URL={agbench_dev_url}")
        else:
            command.append(f"--AG_BENCH_VER={agbench_version}")
        result = subprocess.run(command)
        if result.returncode != 0:
            sys.exit(1)
        else:
            logger.info("Successfully set up the environment under %s/.venv.", self.benchmark_dir)

    def run(
        self,
        dataset_name: str,
        framework: str,
        presets: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
        time_limit: Optional[int] = None,
    ):
        """
        Runs the benchmark on a given dataset.

        Args:
            dataset_name (str): Dataset that has been registered with multimodal_dataset_registry.

                                To get a list of datasets:

                                from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry
                                multimodal_dataset_registry.list_keys()
        Returns:
            None
        """
        PY_EXC_PATH = self.benchmark_dir + "/.venv/bin/python"
        exec_path = os.path.abspath(os.path.dirname(__file__)) + "/exec.py"
        command = [
            PY_EXC_PATH,
            exec_path,
            "--dataset_name",
            dataset_name,
            "--framework",
            framework,
            "--benchmark_dir",
            self.benchmark_dir,
            "--metrics_dir",
            self.metrics_dir,
        ]
        if presets is not None and len(presets) > 0:
            command += ["--presets", presets]
        if hyperparameters is not None:
            command += ["--hyperparameters", json.dumps(hyperparameters)]
        if time_limit is not None:
            command += ["--time_limit", str(time_limit)]
        result = subprocess.run(command)
        if result.returncode != 0:
            sys.exit(1)
        else:
            logger.info(f"Benchmark {self.benchmark_name} on dataset {dataset_name} is complete.")
