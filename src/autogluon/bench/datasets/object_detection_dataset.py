import abc
import os

from autogluon.common.loaders import load_zip

from .constants import _OBJECT_DETECTION
from .utils import get_data_home_dir, get_repo_url


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
    @abc.abstractmethod
    def metric(self):
        pass

    @property
    def problem_type(self):
        return _OBJECT_DETECTION


class TinyMotorbike(BaseObjectDetectionDataset):
    _SOURCE = ("",)
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

    @property
    def metric(self):
        return "map"


__all__ = ["TinyMotorbike"]
