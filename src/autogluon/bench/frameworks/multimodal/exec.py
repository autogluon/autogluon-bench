import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

from autogluon.bench.datasets.constants import (
    _IMAGE_SIMILARITY,
    _IMAGE_TEXT_SIMILARITY,
    _OBJECT_DETECTION,
    _TEXT_SIMILARITY,
)
from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry
from autogluon.bench.utils.general_utils import NumpyEncoder
from autogluon.multimodal import MultiModalPredictor

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset that has been registered with multimodal_dataset_registry.",
    )

    parser.add_argument("--benchmark_dir", type=str, help="Directory to save benchmarking run.")
    parser.add_argument("--metrics_dir", type=str, help="Directory to save benchmarking metrics.")
    parser.add_argument(
        "--time_limit", type=int, default=None, help="Time limit for the AutoGluon benchmark (in seconds)."
    )
    parser.add_argument("--presets", type=str, default=None, help="Preset configurations to use in the benchmark.")
    parser.add_argument("--hyperparameters", type=str, default=None, help="Hyperparameters to use in the benchmark.")

    args = parser.parse_args()
    return args


def load_dataset(
    dataset_name: str,  # dataset name
):
    """Loads and preprocesses a dataset.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test datasets.
    """
    train_data = multimodal_dataset_registry.create(dataset_name, "train")
    test_data = multimodal_dataset_registry.create(dataset_name, "test")

    return train_data, test_data


def save_metrics(metrics_path: str, metrics):
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
    file = os.path.join(metrics_path, "metrics.json")
    with open(file, "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
        logger.info("Metrics saved to %s.", metrics_path)
    f.close()


def run(
    dataset_name: str,
    benchmark_dir: str,
    metrics_dir: str,
    time_limit: Optional[int] = None,
    presets: Optional[str] = None,
    hyperparameters: Optional[dict] = None,
):
    """Runs the AutoGluon multimodal benchmark on a given dataset.

    Args:
        dataset_name (str): Dataset that has been registered with multimodal_dataset_registry.

                            To get a list of datasets:

                            from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry
                            multimodal_dataset_registry.list_keys()

        benchmark_dir (str): The path to the directory where benchmarking artifacts should be saved.
        time_limit (int): The maximum amount of time (in seconds) to spend training the predictor (default: 10).
        presets (str): The name of the AutoGluon preset to use (default: "None").
        hyperparameters (str): A JSON of hyperparameters to use for training (default: None).

    Returns:
        None
    """
    train_data, test_data = load_dataset(dataset_name=dataset_name)

    try:
        label_column = train_data.label_columns[0]
    except (AttributeError, IndexError):  # Object Detection does not have label columns
        label_column = None

    predictor_args = {
        "label": label_column,
        "problem_type": train_data.problem_type,
        "presets": presets,
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
        predictor_args["match_label"] = train_data.match_label
        del predictor_args["label"]
    elif train_data.problem_type == _TEXT_SIMILARITY:
        predictor_args["query"] = train_data.text_columns[0]
        predictor_args["response"] = train_data.text_columns[1]
        predictor_args["match_label"] = train_data.match_label
    elif train_data.problem_type == _OBJECT_DETECTION:
        predictor_args["sample_data_path"] = train_data.data
    predictor = MultiModalPredictor(**predictor_args)

    fit_args = {
        "train_data": train_data.data,
        "hyperparameters": hyperparameters,
        "time_limit": time_limit,
    }

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

    metrics = {
        "problem_type": predictor.problem_type,
        "utc_time": utc_time,
        "training_duration": training_duration,
        "predict_duration": predict_duration,
        "scores": scores,
    }
    save_metrics(metrics_dir, metrics)


if __name__ == "__main__":
    args = get_args()
    if args.hyperparameters is not None:
        args.hyperparameters = json.loads(args.hyperparameters)

    run(
        dataset_name=args.dataset_name,
        benchmark_dir=args.benchmark_dir,
        metrics_dir=args.metrics_dir,
        time_limit=args.time_limit,
        presets=args.presets,
        hyperparameters=args.hyperparameters,
    )
