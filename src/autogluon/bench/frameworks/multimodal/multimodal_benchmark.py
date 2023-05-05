import json
import logging
import os
import subprocess
from typing import Optional

from autogluon.bench.benchmark import Benchmark

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

    def __init__(self, benchmark_name: str, root_dir: str = "./benchmark_runs/multimodal/"):
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
        result = subprocess.run(command)
        if result.stdout:
            logger.info("Successfully set up the environment under %s/.venv.", self.benchmark_dir)
        elif result.stderr:
            logger.error(result.stderr)

    def run(
        self,
        dataset_name: str,
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
            "--benchmark_dir",
            self.benchmark_dir,
        ]
        if presets is not None and len(presets) > 0:
            command += ["--presets", presets]
        if hyperparameters is not None:
            command += ["--hyperparameters", json.dumps(hyperparameters)]
        if time_limit is not None:
            command += ["--time_limit", str(time_limit)]
        subprocess.run(command)
