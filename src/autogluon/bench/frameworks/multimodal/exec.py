import argparse
import csv
import importlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional, Union

from autogluon.bench.datasets.constants import (
    _IMAGE_SIMILARITY,
    _IMAGE_TEXT_SIMILARITY,
    _OBJECT_DETECTION,
    _TEXT_SIMILARITY,
)
from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal import __version__ as ag_version

logger = logging.getLogger(__name__)


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

    args = parser.parse_args()
    return args


def load_dataset(dataset_name: str, custom_dataloader: dict = None):  # dataset name
    """Loads and preprocesses a dataset.

    Args:
        dataset_name (str): The name of the dataset to load.

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


def run(
    dataset_name: Union[str, dict],
    framework: str,
    benchmark_dir: str,
    metrics_dir: str,
    constraint: Optional[str] = None,
    params: Optional[dict] = {},
    custom_dataloader: Optional[dict] = None,
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
    Returns:
        None
    """
    train_data, val_data, test_data = load_dataset(dataset_name=dataset_name, custom_dataloader=custom_dataloader)
    try:
        label_column = train_data.label_columns[0]
    except (AttributeError, IndexError):  # Object Detection does not have label columns
        label_column = None

    predictor_args = {
        "label": label_column,
        "problem_type": train_data.problem_type,
        "presets": params.pop("presets", None),
        "path": os.path.join(benchmark_dir, "models"),
    }

    if train_data.problem_type == _IMAGE_SIMILARITY:
        predictor_args["query"] = train_data.image_columns[0]
        predictor_args["response"] = train_data.image_columns[1]
        predictor_args["match_label"] = train_data.match_label
    elif train_data.problem_type == _IMAGE_TEXT_SIMILARITY:
        predictor_args["query"] = train_data.text_columns[0]
        predictor_args["response"] = train_data.image_columns[0]
        predictor_args["eval_metric"] = train_data.metric
        del predictor_args["label"]
    elif train_data.problem_type == _TEXT_SIMILARITY:
        predictor_args["query"] = train_data.text_columns[0]
        predictor_args["response"] = train_data.text_columns[1]
        predictor_args["match_label"] = train_data.match_label
    elif train_data.problem_type == _OBJECT_DETECTION:
        predictor_args["sample_data_path"] = train_data.data
    predictor = MultiModalPredictor(**predictor_args)

    fit_args = {"train_data": train_data.data, "tuning_data": val_data.data, **params}

    utc_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    start_time = time.time()
    predictor.fit(**fit_args)
    end_time = time.time()
    training_duration = round(end_time - start_time, 1)

    evaluate_args = {
        "data": test_data.data,
        "label": label_column,
        "metrics": test_data.metric,
    }
    if test_data.problem_type == _IMAGE_TEXT_SIMILARITY:
        evaluate_args["query_data"] = test_data.data[test_data.text_columns[0]].unique().tolist()
        evaluate_args["response_data"] = test_data.data[test_data.image_columns[0]].unique().tolist()
        evaluate_args["cutoffs"] = [1, 5, 10]

    start_time = time.time()
    scores = predictor.evaluate(**evaluate_args)
    end_time = time.time()
    predict_duration = round(end_time - start_time, 1)

    if "#" in framework:
        framework, version = framework.split("#")
    else:
        framework, version = framework, ag_version

    metrics = {
        "id": "id/0",  # dummy id to make it align with amlb benchmark output
        "task": dataset_name,
        "framework": framework,
        "constraint": constraint,
        "version": version,
        "fold": 0,
        "type": predictor.problem_type,
        "result": scores[test_data.metric],
        "metric": test_data.metric,
        "utc": utc_time,
        "training_duration": training_duration,
        "predict_duration": predict_duration,
        "scores": scores,
    }
    subdir = f"{framework}.{dataset_name}.{constraint}.local"
    save_metrics(os.path.join(metrics_dir, subdir, "scores"), metrics)


if __name__ == "__main__":
    args = get_args()
    if args.params is not None:
        args.params = json.loads(args.params)
    if args.custom_dataloader is not None:
        args.custom_dataloader = json.loads(args.custom_dataloader)

    run(
        dataset_name=args.dataset_name,
        framework=args.framework,
        benchmark_dir=args.benchmark_dir,
        metrics_dir=args.metrics_dir,
        constraint=args.constraint,
        params=args.params,
        custom_dataloader=args.custom_dataloader,
    )
