import os
import unittest
from unittest.mock import patch

from autogluon.bench.utils.dataset_utils import get_data_home_dir, get_home_dir, get_repo_url


class TestUtils(unittest.TestCase):
    def test_get_home_dir(self):
        expected_path = os.path.expanduser("~/.autogluon_bench")
        self.assertEqual(get_home_dir(), expected_path)

    def test_get_data_home_dir(self):
        with patch("autogluon.bench.utils.dataset_utils.get_home_dir", return_value="/home/user"):
            expected_path = "/home/user/datasets"
            self.assertEqual(get_data_home_dir(), expected_path)

    def test_get_repo_url_default(self):
        expected_url = "https://automl-mm-bench.s3.amazonaws.com/"
        self.assertEqual(get_repo_url(), expected_url)

    def test_get_repo_url_custom(self):
        custom_url = "https://custom-repo-url.com"
        expected_url = custom_url + "/"
        with patch.dict("os.environ", {"AUTOGLUON_BENCH_REPO": custom_url}):
            self.assertEqual(get_repo_url(), expected_url)
