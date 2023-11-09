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
        command = [setup_script_path, git_uri, git_branch, self.benchmark_dir, agbench_version]
        result = subprocess.run(command)
        if result.returncode != 0:
            sys.exit(1)
        else:
            logger.info("Successfully set up the environment under %s/.venv.", self.benchmark_dir)

    def run(
        self,
        dataset_name: str,
        framework: str,
        constraint: Optional[str] = None,
        params: Optional[dict] = None,
        custom_dataloader: Optional[dict] = None,
        custom_metrics: Optional[dict] = None,
    ):
        """
        Runs the benchmark on a given dataset.

        Args:
            dataset_name (str): Dataset name, can be registered with multimodal_dataset_registry or a custom dataset.

                                To get a list of datasets:
                                from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry
                                multimodal_dataset_registry.list_keys()
            framework (str): The name of the framework to use for the benchmark.
            constraint (str): The resource constraint used by benchmarking during AWS mode.
            params (str): The multimodal params.
            custom_dataloader (Optional[dict], None): A dictionary containing information about a custom dataloader to use. Defaults to None.
                                To define a custom dataloader in the config file:

                                custom_dataloader:
                                    dataloader_file: path_to/dataloader.py   # relative path to WORKDIR
                                    class_name: DataLoaderClass
                                    dataset_config_file: path_to/dataset_config.yaml
                                    **kwargs (of DataLoaderClass)
            custom_metrics (Optional[dict], None): A dictionary containing information about a custom metrics to use. Defaults to None.
                                To define a custom metrics in the config file:

                                custom_metrics:
                                    metrics_path: path_to/metrics.py   # relative path to WORKDIR
                                    function_name: custom_metrics_function
                                    **kwargs (of )

        Returns:
            None
        """
        if os.environ.get("RUNNING_IN_DOCKER", False):
            venv_base_dir = "/home/"
        else:
            venv_base_dir = self.benchmark_dir
        PY_EXC_PATH = os.path.join(venv_base_dir, ".venv/bin/python")

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
        if constraint is not None:
            command += ["--constraint", constraint]
        if params is not None:
            command += ["--params", json.dumps(params)]
        if custom_dataloader is not None:
            command += ["--custom_dataloader", json.dumps(custom_dataloader)]
        if custom_metrics is not None:
            command += ["--custom_metrics", json.dumps(custom_metrics)]
        result = subprocess.run(command)
        if result.returncode != 0:
            sys.exit(1)
        else:
            logger.info(f"Benchmark {self.benchmark_name} on dataset {dataset_name} is complete.")
