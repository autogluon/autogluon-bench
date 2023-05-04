from .multimodal_dataset import *
from .object_detection_dataset import *
from .registry import Registry

multimodal_datasets = [Shopee, StanfordOnline, Flickr30k, SNLI, MitMovies,
           WomenClothingReview, MelBourneAirBnb, AEPricePrediction,
           IMDBGenrePrediction, JCPennyCategory, NewsPopularity,
           NewsChannel, TinyMotorbike]

multimodal_dataset_registry = Registry("multimodal_dataset")


for dataset in multimodal_datasets:
    multimodal_dataset_registry.register(dataset._registry_name, dataset)
