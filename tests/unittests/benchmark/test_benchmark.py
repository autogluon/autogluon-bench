import os
import unittest
from unittest.mock import MagicMock

import pytest

from autogluon.bench.frameworks.benchmark import Benchmark


class TempBenchmark(Benchmark):
    def run(self):
        pass


@pytest.fixture(scope="function")
def benchmark():
    return TempBenchmark("test_benchmark", "test_dir")


def test_attributes(benchmark):
    assert benchmark.benchmark_name.startswith("test_benchmark")
    assert benchmark.benchmark_dir.startswith("test_dir")
    assert benchmark.metrics_dir.startswith("test_dir")


def test_upload_metrics(benchmark, temp_dir):
    benchmark.metrics_dir = temp_dir

    temp_file = os.path.join(temp_dir, "file.txt")
    with open(temp_file, "w") as f:
        f.write("test")

    with unittest.mock.patch("boto3.client") as mock_client:
        mock_upload_file = MagicMock()
        mock_client.return_value = MagicMock(upload_file=mock_upload_file)

        s3_bucket = "my-bucket"
        s3_dir = "my-s3-dir"
        benchmark.upload_metrics(s3_bucket, s3_dir)

        assert mock_upload_file.call_count == 1

        s3_path = mock_upload_file.call_args[0][2]
        assert s3_path.startswith(f"{s3_dir}/file.txt")
