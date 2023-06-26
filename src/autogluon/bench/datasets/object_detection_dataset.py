import abc
import logging
import os

from autogluon.bench.utils.dataset_utils import get_data_home_dir, get_repo_url
from autogluon.common.loaders import load_zip

from .constants import _OBJECT_DETECTION

# Add dataset class names here
__all__ = ["TinyMotorbike", "Clipart"]

logger = logging.getLogger(__name__)


class BaseObjectDetectionDataset(abc.ABC):
    def __init__(self, split: str, dataset_name: str, data_info: dict):
        """
        Initializes the class.

        Args:
            split (str): Specifies the dataset split. It should be one of the following options: 'train', 'val', 'test'.
        """
        self._path = os.path.join(get_data_home_dir(), dataset_name)
        load_zip.unzip(data_info["data"]["url"], unzip_dir=self._path, sha1sum=data_info["data"]["sha1sum"])
        self._base_folder = os.path.join(self._path, dataset_name)
        self._data_path = os.path.join(self._base_folder, "Annotations", f"{split}_cocoformat.json")
        if not os.path.exists(self._data_path):
            logger.warn(f"No annotation found at {self._data_path}")
            self._data_path = None

    @property
    @abc.abstractmethod
    def base_folder(self):
        pass

    @property
    @abc.abstractmethod
    def data(self):
        pass

    @property
    @abc.abstractmethod
    def metric(self):
        pass

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
        super().__init__(split=self._split, dataset_name=self._registry_name, data_info=self._INFO)

    @property
    def base_folder(self):
        return self._base_folder

    @property
    def data(self):
        return self._data_path

    @property
    def metric(self):
        return "map"


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
        super().__init__(split=self._split, dataset_name=self._registry_name, data_info=self._INFO)

    @property
    def base_folder(self):
        return self._base_folder

    @property
    def data(self):
        return self._data_path

    @property
    def metric(self):
        return "map"
