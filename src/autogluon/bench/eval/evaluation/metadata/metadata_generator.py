from typing import List, Union

import openml
import pandas as pd

from autogluon.common.savers import save_pd


def generate_metadata(study: Union[int, List[int]]) -> pd.DataFrame:
    """
    Generates the OpenML metadata file for every task in a study.
    """
    tasks = openml.tasks.list_tasks(output_format="dataframe")
    if not isinstance(study, list):
        study = [study]
    suite_tasks = []
    for s in study:
        suite = openml.study.get_suite(s)
        suite_tasks += suite.tasks
    suite_tasks = list(set(suite_tasks))
    suite_tasks.sort()
    task_metadata = tasks[tasks["tid"].isin(set(suite_tasks))]
    return task_metadata


def generate_and_save_metadata(path: str, study: Union[int, List[int]]):
    task_metadata = generate_metadata(study=study)
    save_pd.save(path=path, df=task_metadata)


if __name__ == "__main__":
    # Generate the AutoMLBenchmark study metadata of 104 datasets
    path = "data/metadata/task_metadata.csv"
    generate_and_save_metadata(path=path, study=[269, 271])
