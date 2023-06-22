import os
import warnings
from pathlib import Path

import pandas as pd

_PATH_TO_DATA = str(Path(__file__).parent.parent.parent / "data" / "metadata")


def load_task_metadata(path="task_metadata.csv", path_relative_to_data=True, compute_aux_columns=True):
    if path_relative_to_data:
        path = _PATH_TO_DATA + os.sep + path
    task_metadata = pd.read_csv(path)
    if not compute_aux_columns:
        return task_metadata
    task_metadata["ClassRatio"] = task_metadata["MinorityClassSize"] / task_metadata["NumberOfInstances"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        task_metadata["problem_type"] = ""
        task_metadata["problem_type"][task_metadata["NumberOfClasses"] == 2] = "binary"
        task_metadata["problem_type"][task_metadata["NumberOfClasses"] > 2] = "multiclass"
        task_metadata["problem_type"][task_metadata["NumberOfClasses"] == 0] = "regression"
    task_metadata["NumberOfSymbolicFeatures"] = [
        num_sym if num_class == 0 else num_sym - 1
        for num_sym, num_class in zip(task_metadata["NumberOfSymbolicFeatures"], task_metadata["NumberOfClasses"])
    ]
    task_metadata["NumberOfNumericFeatures"] = [
        num_num if num_class != 0 else num_num - 1
        for num_num, num_class in zip(task_metadata["NumberOfNumericFeatures"], task_metadata["NumberOfClasses"])
    ]
    task_metadata["NumberOfFeatures"] -= 1
    return task_metadata
