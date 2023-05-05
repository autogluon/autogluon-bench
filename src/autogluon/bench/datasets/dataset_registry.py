from . import multimodal_dataset, object_detection_dataset
from .registry import Registry

multimodal_dataset_registry = Registry("multimodal_dataset")

for data_classes in [multimodal_dataset, object_detection_dataset]:
    for class_name in data_classes.__all__:
        dataset_class = getattr(data_classes, class_name)
        multimodal_dataset_registry.register(dataset_class._registry_name, dataset_class)
