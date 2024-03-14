import logging
import os

import pandas as pd
import yaml

from autogluon.bench.utils.dataset_utils import get_data_home_dir
from autogluon.common.loaders import load_zip
from autogluon.common.loaders._utils import download


def path_expander(path, base_folder):
    path_l = path.split(";")
    return ";".join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])


logger = logging.getLogger(__name__)


class TextTabularDataLoader:
    def __init__(self, dataset_name: str, dataset_config_file: str, split: str = "train"):
        with open(dataset_config_file, "r") as f:
            config = yaml.safe_load(f)

        self.dataset_config = config[dataset_name]
        if split not in self.dataset_config["splits"]:
            logger.warning(f"Data split {split} not available.")
            self.data = None
            return
        if split == "test" and self.dataset_config["test_split_name"] == "dev":
            split = "dev"

        self.name = dataset_name
        self.split = split
        self.image_columns = []
        self.text_columns = self.dataset_config["text_columns"] or []
        self.label_columns = self.dataset_config["label_columns"]
        self.columns_to_drop = self.dataset_config["columns_to_drop"] or []

        # url = self.dataset_config["url"].format(name=self.name)
        # base_dir = get_data_home_dir()
        # load_zip.unzip(url, unzip_dir=base_dir)
        # self.dataset_dir = os.path.join(base_dir, self.name)

        url = self.dataset_config["url"].format(split=self.split)
        file_extention = os.path.splitext(url)[-1]
        base_dir = get_data_home_dir()

        self.data_path = os.path.join(base_dir, self.name, f"{split}{file_extention}")
        download(url, path=self.data_path)
        if file_extention == ".csv":
            self.data = pd.read_csv(self.data_path)
        elif file_extention == ".pq":
            self.data = pd.read_parquet(self.data_path)
        else:
            raise NotImplementedError("Unsupported data type.")

        if self.columns_to_drop is not None:
            self.data.drop(columns=self.columns_to_drop, inplace=True)

    @property
    def problem_type(self):
        return self.dataset_config["problem_type"]

    @property
    def metric(self):
        return self.dataset_config["metric"]


