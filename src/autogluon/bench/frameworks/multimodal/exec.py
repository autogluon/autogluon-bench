import argparse
import json
import logging
import os
from datetime import datetime
from typing import List

from sklearn.model_selection import train_test_split

from autogluon.core.utils.loaders import load_pd
from autogluon.multimodal import MultiModalPredictor

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        help="Can be one of: dataset name, local path, S3 path, AMLB task ID/name",
    )
    parser.add_argument("--benchmark_dir", type=str, help="Directory to save benchmarking run.")

    args = parser.parse_args()
    return args


def _convert_torchvision_dataset(dataset):
    from io import BytesIO

    import pandas as pd

    data = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        img_byte_arr = BytesIO()
        x.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        data.append((img_byte_arr, y))
    df = pd.DataFrame(data, columns=["image", "label"])
    return df


def load_dataset(
    data_path: str,  # can be dataset name or path to dataset
):
    """Loads and preprocesses a dataset.

    Args:
        data_path (str): The path or name of the dataset to load.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test datasets.
    """
    if data_path in ["MNIST"]:
        import torchvision.datasets as data

        train_data = data.MNIST("./data", train=True, download=True)
        test_data = data.MNIST("./data", train=False, download=True)
        train_data = _convert_torchvision_dataset(train_data)
        test_data = _convert_torchvision_dataset(test_data).sample(frac=0.1)
    else:
        df = load_pd.load(data_path)
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    return train_data, test_data


def save_metrics(metrics_path: str, metrics):
    """Saves evaluation metrics to a JSON file.

    Args:
        metrics_path (str): The path to the directory where the metrics should be saved.
        metrics: The evaluation metrics to save.

    Returns:
        None
    """
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    file = os.path.join(metrics_path, "metrics.json")
    with open(file, "w") as f:
        json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to %s.", metrics_path)
    f.close()


def run(
    data_path: str,
    benchmark_dir: str,
    problem_type: str = None,
    label: str = "label",
    presets: str = "best_quality",
    metrics: List[str] = ["acc"],
    time_limit: int = 10,
    hyperparameters: dict = None,
    # TODO: replace with config yaml
):
    """Runs the AutoGluon multimodal benchmark on a given dataset.

    Args:
        data_path (str): The path to the dataset to use for training and evaluation.
        metrics_dir (str): The path to the directory where the evaluation metrics should be saved.
        problem_type (str): The problem type of the dataset (default: None).
        label (str): The name of the label column in the dataset (default: "label").
        presets (str): The name of the AutoGluon preset to use (default: "best_quality").
        metrics (List[str]): The evaluation metrics to compute (default: ["acc"]).
        time_limit (int): The maximum amount of time (in seconds) to spend training the predictor (default: 10).
        hyperparameters (dict): A dictionary of hyperparameters to use for training (default: None).

    Returns:
        None
    """
    train_data, test_data = load_dataset(data_path=data_path)

    predictor = MultiModalPredictor(
        label=label,
        problem_type=problem_type,
        presets=presets,
        path=os.path.join(benchmark_dir, "models"),
    )
    predictor.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        time_limit=time_limit,
    )
    scores = predictor.evaluate(test_data)
    timestamp = datetime.now()
    metrics = {
        "problem_type": predictor.problem_type,
        "scores": scores,
        "timestamp": timestamp.strftime("%H:%M:%S"),
    }
    save_metrics(os.path.join(benchmark_dir, "results"), metrics)


if __name__ == "__main__":
    args = get_args()
    run(data_path=args.data_path, benchmark_dir=args.benchmark_dir)
