import logging
import os
import subprocess

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

    def run(self, data_path: str):
        """
        Runs the benchmark on a given dataset.

        Args:
            data_path (str): The path to the dataset to use for training and evaluation.

        Returns:
            None
        """
        PY_EXC_PATH = self.benchmark_dir + "/.venv/bin/python"
        exec_path = os.path.abspath(os.path.dirname(__file__)) + "/exec.py"
        command = [
            PY_EXC_PATH,
            exec_path,
            "--data_path",
            data_path,
            "--benchmark_dir",
            self.benchmark_dir,
        ]
        subprocess.run(command)
