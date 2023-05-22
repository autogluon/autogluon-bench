import abc
import os

from autogluon.common.loaders import load_zip
from autogluon.multimodal.constants import (
    MAP,
    MAP_50,
    MAP_75,
    MAP_SMALL,
    MAP_MEDIUM,
    MAP_LARGE,
    MAR_1,
    MAR_10,
    MAR_100,
    MAR_SMALL,
    MAR_MEDIUM,
    MAR_LARGE,
)

from .constants import _OBJECT_DETECTION
from .utils import get_data_home_dir, get_repo_url

# Add dataset class names here
__all__ = ["TinyMotorbike", "Clipart", "ApparelBloggerInfluencer", "Comic", "Pothole"]


class BaseObjectDetectionDataset(abc.ABC):
    @property
    @abc.abstractmethod
    def base_folder(self):
        pass

    @property
    @abc.abstractmethod
    def data(self):
        pass

    @property
    def metric(self):
        return [
            MAP,
            MAP_50,
            MAP_75,
            MAP_SMALL,
            MAP_MEDIUM,
            MAP_LARGE,
            MAR_1,
            MAR_10,
            MAR_100,
            MAR_SMALL,
            MAR_MEDIUM,
            MAR_LARGE,
        ]

    @property
    def problem_type(self):
        return _OBJECT_DETECTION


class TinyMotorbike(BaseObjectDetectionDataset):
    _SOURCE = ""
    _INFO = {
        "data": {
            "url": get_repo_url() + "object_detection_dataset/tiny_motorbike_coco.zip",
            "sha1sum": "45c883b2feb0721d6eef29055fa28fb46b6e5346",
        },
    }
    _registry_name = "tiny_motorbike"

    def __init__(self, split="train"):
        self._split = f"{split}val" if split == "train" else split
        self._path = os.path.join(get_data_home_dir(), "tiny_motorbike")
        load_zip.unzip(self._INFO["data"]["url"], unzip_dir=self._path, sha1sum=self._INFO["data"]["sha1sum"])
        self._base_folder = os.path.join(self._path, "tiny_motorbike")
        self._data_path = os.path.join(self._base_folder, "Annotations", f"{self._split}_cocoformat.json")

    @property
    def base_folder(self):
        return self._base_folder

    @property
    def data(self):
        return self._data_path


class Clipart(BaseObjectDetectionDataset):
    _SOURCE = "https://github.com/naoto0804/cross-domain-detection/tree/master/datasets"
    _INFO = {
        "data": {
            "url": get_repo_url() + "few_shot_object_detection/clipart.zip",
            "sha1sum": "d25b2f905da597d7857297ac8e3efe4555e0bf32",
        },
    }
    _registry_name = "clipart"

    def __init__(self, split="train"):
        self._split = split
        self._path = os.path.join(get_data_home_dir(), "clipart")
        load_zip.unzip(self._INFO["data"]["url"], unzip_dir=self._path, sha1sum=self._INFO["data"]["sha1sum"])
        self._base_folder = os.path.join(self._path, "clipart")
        self._data_path = os.path.join(self._base_folder, "Annotations", f"{self._split}_cocoformat.json")

    @property
    def base_folder(self):
        return self._base_folder

    @property
    def data(self):
        return self._data_path


class AGDetBenchDataset(BaseObjectDetectionDataset):
    _SOURCE = ""
    _BENCHMARK_NAME = "AGDetBench"

    def __init__(self, dataset_name, split="train", sha1sum=None):
        self._dataset_name = dataset_name
        self._split = split
        self._sha1sum = sha1sum
        self._path = os.path.join(get_data_home_dir(), self._dataset_name)

        self._INFO = {
            "data": {
                "url": self.data_url,
                "sha1sum": self._sha1sum,
            },
        }

        load_zip.unzip(self._INFO["data"]["url"], unzip_dir=self._path, sha1sum=self._INFO["data"]["sha1sum"])
        self._base_folder = os.path.join(self._path, self._dataset_name)
        self._data_path = os.path.join(self._base_folder, "annotations", f"{self._split}.json")

    @property
    def data_url(self):
        return get_repo_url() + f"{self._BENCHMARK_NAME}/{self._dataset_name}.zip"

    @property
    def base_folder(self):
        return self._base_folder

    @property
    def data(self):
        return self._data_path


class ApparelBloggerInfluencer(AGDetBenchDataset):
    _registry_name = "apparel_blogger_influencer"

    def __init__(self, split="train"):
        super().__init__(dataset_name="apparel_blogger_influencer", split=split, sha1sum=None)


class Comic(AGDetBenchDataset):
    _registry_name = "comic"

    def __init__(self, split="train"):
        super().__init__(dataset_name="comic", split=split, sha1sum=None)


class Pothole(AGDetBenchDataset):
    _registry_name = "pothole"

    def __init__(self, split="train"):
        super().__init__(dataset_name="pothole", split=split, sha1sum=None)
