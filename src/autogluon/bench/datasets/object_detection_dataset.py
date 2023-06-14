import abc
import os

from autogluon.bench.utils.dataset_utils import get_data_home_dir, get_repo_url
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

# Add dataset class names here
__all__ = [
    "TinyMotorbike",
    "ApparelBloggerInfluencer",
    "ApparelFashionShow",
    "ApparelStreetwearDaily",
    "Chest10",
    "Cityscapes",
    "Clipart",
    "Comic",
    "DamagedVehicles",
    "Deepfruits",
    "Deeplesion",
    "Dota",
    "Duo",
    "Ena24",
    "F1",
    "Kitchen",
    "Kitti",
    "Lisa",
    "Mario",
    "Minneapple",
    "NflLogo",
    "Oktoberfest",
    "Pothole",
    "Rugrats",
    "Sixray",
    "Table",
    "Tt100k",
    "Uefa",
    "Utensils",
    "VehiclesTestCommercial",
    "Voc0712",
    "Widerface",
]


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


class AGDetBenchDataset(BaseObjectDetectionDataset):
    _SOURCE = ""
    _BENCHMARK_NAME = "AGDetBench"
    _registry_name = None

    def __init__(self, split="train", sha1sum=None):
        self._split = split
        self._sha1sum = sha1sum
        self._path = os.path.join(get_data_home_dir(), self._registry_name)

        self._INFO = {
            "data": {
                "url": self.data_url,
                "sha1sum": self._sha1sum,
            },
        }

        load_zip.unzip(self._INFO["data"]["url"], unzip_dir=self._path, sha1sum=self._INFO["data"]["sha1sum"])
        self._base_folder = os.path.join(self._path, self._registry_name)
        self._data_path = os.path.join(self._base_folder, "annotations", f"{self._split}.json")

    @property
    def data_url(self):
        return get_repo_url() + f"{self._BENCHMARK_NAME}/{self._registry_name}.zip"

    @property
    def base_folder(self):
        return self._base_folder

    @property
    def data(self):
        return self._data_path


class ApparelBloggerInfluencer(AGDetBenchDataset):
    _registry_name = "apparel_blogger_influencer"


class ApparelFashionShow(AGDetBenchDataset):
    _registry_name = "apparel_fashion_show"


class ApparelStreetwearDaily(AGDetBenchDataset):
    _registry_name = "apparel_streetwear_daily"


class Chest10(AGDetBenchDataset):
    _registry_name = "chest10"


class Cityscapes(AGDetBenchDataset):
    _registry_name = "cityscapes"


class Clipart(AGDetBenchDataset):
    _registry_name = "clipart"


class Comic(AGDetBenchDataset):
    _registry_name = "comic"


class DamagedVehicles(AGDetBenchDataset):
    _registry_name = "damaged_vehicles"


class Deepfruits(AGDetBenchDataset):
    _registry_name = "deepfruits"


class Deeplesion(AGDetBenchDataset):
    _registry_name = "deeplesion"


class Dota(AGDetBenchDataset):
    _registry_name = "dota"


class Duo(AGDetBenchDataset):
    _registry_name = "duo"


class Ena24(AGDetBenchDataset):
    _registry_name = "ena24"


class F1(AGDetBenchDataset):
    _registry_name = "f1"


class Kitchen(AGDetBenchDataset):
    _registry_name = "kitchen"


class Kitti(AGDetBenchDataset):
    _registry_name = "kitti"


class Lisa(AGDetBenchDataset):
    _registry_name = "lisa"


class Mario(AGDetBenchDataset):
    _registry_name = "mario"


class Minneapple(AGDetBenchDataset):
    _registry_name = "minneapple"


class NflLogo(AGDetBenchDataset):
    _registry_name = "nfl_logo"


class Oktoberfest(AGDetBenchDataset):
    _registry_name = "oktoberfest"


class Pothole(AGDetBenchDataset):
    _registry_name = "pothole"


class Rugrats(AGDetBenchDataset):
    _registry_name = "rugrats"


class Sixray(AGDetBenchDataset):
    _registry_name = "sixray"


class Table(AGDetBenchDataset):
    _registry_name = "table"


class Tt100k(AGDetBenchDataset):
    _registry_name = "tt100k"


class Uefa(AGDetBenchDataset):
    _registry_name = "uefa"


class Utensils(AGDetBenchDataset):
    _registry_name = "utensils"


class VehiclesTestCommercial(AGDetBenchDataset):
    _registry_name = "vehicles_test_commercial"


class Voc0712(AGDetBenchDataset):
    _registry_name = "voc0712"


class Widerface(AGDetBenchDataset):
    _registry_name = "widerface"
