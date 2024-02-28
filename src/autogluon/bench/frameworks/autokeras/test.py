import argparse
import csv
import importlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional, Union
import autokeras as ak
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry
import pandas as pd

import tensorflow as tf


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _flatten_dict(data):
    flattened = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value))
        else:
            flattened[key] = value
    return flattened


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset that has been registered with multimodal_dataset_registry.",
    )
    parser.add_argument("--framework", type=str, help="Framework (and) branch/version.")
    parser.add_argument("--benchmark_dir", type=str, help="Directory to save benchmarking run.")
    parser.add_argument("--metrics_dir", type=str, help="Directory to save benchmarking metrics.")
    parser.add_argument("--constraint", type=str, default=None, help="AWS resources constraint setting.")
    parser.add_argument("--params", type=str, default=None, help="AWS resources constraint setting.")
    parser.add_argument(
        "--custom_dataloader", type=str, default=None, help="Custom dataloader to use in the benchmark."
    )
    parser.add_argument("--custom_metrics", type=str, default=None, help="Custom metrics to use in the benchmark.")

    args = parser.parse_args()
    return args


def load_dataset(dataset_name: str, custom_dataloader: dict = None):  # dataset name
    """Loads and preprocesses a dataset.

    Args:
        dataset_name (str): The name of the dataset to load.
        custom_dataloader (dict): A dictionary containing information about a custom dataloader to use. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test datasets.
    """
    splits = ["train", "val", "test"]
    data = {}
    if dataset_name in multimodal_dataset_registry.list_keys():
        logger.info(f"Loading dataset {dataset_name} from multimodal_dataset_registry")
        for split in splits:
            data[split] = multimodal_dataset_registry.create(dataset_name, split)
    elif custom_dataloader is not None:
        logger.info(f"Loading dataset {dataset_name} from custom dataloader {custom_dataloader}.")
        custom_dataloader_file = custom_dataloader.pop("dataloader_file")
        class_name = custom_dataloader.pop("class_name")
        spec = importlib.util.spec_from_file_location(class_name, custom_dataloader_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        custom_class = getattr(module, class_name)
        for split in splits:
            data[split] = custom_class(dataset_name=dataset_name, split=split, **custom_dataloader)
    else:
        raise ModuleNotFoundError(f"Dataset Loader for dataset {dataset_name} is not available.")

    return data.values()


def save_metrics(metrics_path: str, metrics: dict):
    """Saves evaluation metrics to a JSON file.

    Args:
        metrics_path (str): The path to the directory where the metrics should be saved.
        metrics: The evaluation metrics to save.

    Returns:
        None
    """
    if metrics is None:
        logger.warning("No metrics were created.")
        return

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    file = os.path.join(metrics_path, "results.csv")
    flat_metrics = _flatten_dict(metrics)
    field_names = flat_metrics.keys()

    with open(file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerow(flat_metrics)
    logger.info("Metrics saved to %s.", file)
    f.close()


def load_image(image_path, target_size=(224, 224)):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            #img = img.resize(target_size)
            return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)  # Placeholder for an invalid image

def create_zero_image(target_size=(224, 224)):
    # Create a zero (blank) image
    return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

def average_images(image_paths, target_size=(224, 224)):
    images = [load_image(path, target_size) for path in image_paths[:2]]  # Load the first two images
    # Calculate the average of the images
    average_img = np.mean(images, axis=0).astype(np.uint8)
    return average_img


def preprocess_data(features, image_columns, text_columns):
    # Process image data
    image_data = None
    if image_columns is not None and len(image_columns) > 0:
        image_data = []
        features.loc[:, image_columns[0]] = features[image_columns[0]].apply(lambda x: x.split(';')[0] if pd.notnull(x) else x)
        image_paths = features[image_columns[0]].values
        for path in image_paths:
            img = load_image(path)
            image_data.append(img)
    
        # Convert column image data to a NumPy array and normalize
        image_data = np.array(image_data)

    # Process text data
    text_data = None
    if text_columns is not None and len(text_columns) > 0:
        text_data = features.apply(lambda row: " ".join((str(row[col]) if row[col] is not None else "") for col in text_columns), axis=1) 
        text_data = text_data.to_numpy(dtype=str)
        print("Text data is: ", text_data)
    
    # Process tabular data
    tabular_data = None
    all_image_text_columns = image_columns or [] + text_columns or [] 
    tabular_columns = features.columns.difference(all_image_text_columns)
    print("tabular column is: ", tabular_columns) 
    if len(tabular_columns) > 0:
        tabular_data = features[tabular_columns].to_numpy()
        print(tabular_data[0])

    return image_data, tabular_data, text_data


def run(
    dataset_name: Union[str, dict],
    framework: str,
    benchmark_dir: str,
    metrics_dir: str,
    constraint: Optional[str] = None,
    params: Optional[dict] = None,
    custom_dataloader: Optional[dict] = None,
    custom_metrics: Optional[dict] = None,
):
    """Runs the AutoGluon multimodal benchmark on a given dataset.

    Args:
        dataset_name (Union[str, dict]): Dataset that has been registered with multimodal_dataset_registry.

                            To get a list of datasets:

                            from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry
                            multimodal_dataset_registry.list_keys()

        benchmark_dir (str): The path to the directory where benchmarking artifacts should be saved.
        constraint (str): The resource constraint used by benchmarking during AWS mode, default: None.
        params (str): The multimodal params, default: {}.
        custom_dataloader (dict): A dictionary containing information about a custom dataloader to use. Defaults to None.
                                To define a custom dataloader in the config file:

                                custom_dataloader:
                                    dataloader_file: path_to/dataloader.py   # relative path to WORKDIR
                                    class_name: DataLoaderClass
                                    dataset_config_file: path_to/dataset_config.yaml
                                    **kwargs (of DataLoaderClass)
        custom_metrics (dict): A dictionary containing information about a custom metrics to use. Defaults to None.
                                To define a custom metrics in the config file:

                                custom_metrics:
                                    metrics_path: path_to/metrics.py   # relative path to WORKDIR
                                    function_name: custom_metrics_function
                                    **kwargs (of autogluon.core.metrics.make_scorer)
    Returns:
        None
    """
    train_data, val_data, test_data = load_dataset(dataset_name=dataset_name, custom_dataloader=custom_dataloader)
    image_columns = train_data.image_columns
    text_columns = train_data.text_columns
    tabular_columns = list(set(train_data.data.columns) - set(image_columns) - set(text_columns) - set(train_data.columns_to_drop) - set(train_data.label_columns))
    feature_columns = tabular_columns + image_columns + text_columns
    print("Label column: ", train_data.label_columns, train_data.data[train_data.label_columns])

    features_train, labels_train = train_data.data[feature_columns], train_data.data[train_data.label_columns]
    if test_data.data is None:
        print("No test data found, splitting test data from train data")
        features_train, features_test, labels_train, labels_test = train_test_split(features_train, labels_train, test_size=0.2, random_state=42)
    else:
        features_test, labels_test = test_data.data[feature_columns], test_data.data[train_data.label_columns]

    features_val, labels_val = None, None 
    if val_data.data is not None:
        features_val, labels_val = val_data.data[feature_columns], val_data.data[train_data.label_columns]

    image_data_train, tabular_data_train, text_data_train = preprocess_data(features_train, image_columns, text_columns)
    image_data_test, tabular_data_test, text_data_test = preprocess_data(features_test, image_columns, text_columns)

    image_data_val, tabular_data_val, text_data_val = (None, None, None)
    
    if features_val is not None and labels_val is not None:
        image_data_val, tabular_data_val, text_data_val = preprocess_data(features_val, image_columns, text_columns)


    inputs = []
    if image_data_train is not None:
        print("has image_data")
        inputs.append(ak.ImageInput())
    if tabular_data_train is not None:
        print("has tabular_data")
        inputs.append(ak.StructuredDataInput())
    if text_data_train is not None:
        print("has text_data")
        inputs.append(ak.TextInput())
    
    import tensorflow as tf
    if train_data.problem_type == "regression":
        output_node = ak.RegressionHead(metrics=[tf.keras.metrics.RootMeanSquaredError()])
    elif train_data.problem_type in ["multiclass", "classification"]:
        output_node = ak.ClassificationHead(loss="categorical_crossentropy",metrics=[tf.keras.metrics.Accuracy()])
    elif train_data.problem_type == "binary":
        output_node = ak.ClassificationHead(loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(curve="ROC")])
    else:
        print("Warning: problem type unknown").

    # Combine the data into a list for the model
    train_data_list = [data for data in [image_data_train, tabular_data_train, text_data_train] if data is not None]

    # Combine the data into a list for the model
    test_data_list = [data for data in [image_data_test, tabular_data_test, text_data_test] if data is not None]


    auto_model = ak.AutoModel(
        inputs=inputs,
        outputs=output_node,
        overwrite=True,
    )

    utc_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    start_time = time.time()
    if features_val is not None and labels_val is not None:
        # Combine the data into a list for the model
        val_data_list = [data for data in [image_data_val, tabular_data_val, text_data_val] if data is not None]

        auto_model.fit(
            train_data_list,
            labels_train,
            validation_data=(val_data_list, labels_val),
        )
    else:
        auto_model.fit(
            train_data_list,
            labels_train,
        )
    end_time = time.time()
    training_duration = round(end_time - start_time, 1)

    start_time = time.time()
    metrics = auto_model.evaluate(test_data_list, labels_test)
    end_time = time.time()
    predict_duration = round(end_time - start_time, 1)

    metric_name = train_data.metric
    version = "master"
    metrics = {
        "id": "id/0",  # dummy id to make it align with amlb benchmark output
        "task": dataset_name,
        "framework": framework,
        "constraint": constraint,
        "version": version,
        "fold": 0,
        "type": train_data.problem_type,
        "result": metrics[1],
        "metric": metric_name,
        "utc": utc_time,
        "training_duration": training_duration,
        "predict_duration": predict_duration,
        "scores": metrics[1],
    }
    subdir = f"{framework}.{dataset_name}.{constraint}.local"
    save_metrics(os.path.join(metrics_dir, subdir, "scores"), metrics)
    

if __name__ == "__main__":
    args = get_args()
    if args.params is not None:
        args.params = json.loads(args.params)
    if args.custom_dataloader is not None:
        args.custom_dataloader = json.loads(args.custom_dataloader)
    if args.custom_metrics is not None:
        args.custom_metrics = json.loads(args.custom_metrics)

    run(
        dataset_name=args.dataset_name,
        framework=args.framework,
        benchmark_dir=args.benchmark_dir,
        metrics_dir=args.metrics_dir,
        constraint=args.constraint,
        params=args.params,
        custom_dataloader=args.custom_dataloader,
        custom_metrics=args.custom_metrics,
    )

