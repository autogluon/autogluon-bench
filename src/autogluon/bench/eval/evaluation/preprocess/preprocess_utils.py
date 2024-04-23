import numpy as np
import pandas as pd

from ..constants import *


def clean_result(result_df, folds_to_keep=None, remove_invalid=True):
    if folds_to_keep is None:
        folds_to_keep = sorted(list(result_df[FOLD].unique()))
    folds_required = len(folds_to_keep)
    result_df = result_df[result_df[FOLD].isin(folds_to_keep)]
    result_df = result_df[result_df[METRIC_ERROR].notnull()]

    if remove_invalid and folds_required > 1:
        results_fold_count_per_run = result_df[[FRAMEWORK, DATASET, FOLD]].groupby([FRAMEWORK, DATASET]).count()
        results_fold_count_per_run_filtered = results_fold_count_per_run[
            results_fold_count_per_run[FOLD] == folds_required
        ].reset_index()[[FRAMEWORK, DATASET]]
        results_clean_df = result_df.merge(results_fold_count_per_run_filtered, on=[FRAMEWORK, DATASET]).reset_index(
            drop=True
        )
    else:
        results_clean_df = result_df.reset_index(drop=True)
    return results_clean_df


def fill_missing_results_with_default(
    framework_nan_fill: str, frameworks_to_fill: list, results_raw: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill missing results with the result of `framework_nan_fill` framework.
    """
    assert framework_nan_fill is not None

    frameworks_valid = list(results_raw["framework"].unique())
    assert framework_nan_fill in frameworks_valid

    if frameworks_to_fill is None:
        frameworks_to_fill = [f for f in frameworks_valid if f != frameworks_valid]

    results_nan_fill = results_raw[results_raw["framework"] == framework_nan_fill]
    results_nan_fill = results_nan_fill[["dataset", "fold", "metric_error", "metric", "problem_type"]]
    results_nan_fill["time_train_s"] = np.nan
    results_nan_fill["time_infer_s"] = np.nan

    results_raw = results_raw[results_raw["framework"].isin(frameworks_to_fill)]
    return _fill_missing(results_nan_fill=results_nan_fill, results_raw=results_raw)


def fill_missing_results_with_worst(
    frameworks_to_consider: list, frameworks_to_fill: list, results_raw: pd.DataFrame
) -> pd.DataFrame:
    if frameworks_to_consider is None:
        frameworks_to_consider = list(results_raw["framework"].unique())
    if frameworks_to_fill is None:
        frameworks_to_fill = list(results_raw["framework"].unique())
    results_to_consider = results_raw[results_raw[FRAMEWORK].isin(frameworks_to_consider)]
    task_metric_problem_type = results_to_consider[["dataset", "fold", "metric", "problem_type"]].drop_duplicates()

    worst_result_per_task = (
        results_to_consider[[DATASET, FOLD, METRIC_ERROR]].groupby([DATASET, FOLD])[METRIC_ERROR].max()
    )
    worst_result_per_task = worst_result_per_task.to_frame().reset_index()

    results_nan_fill = pd.merge(task_metric_problem_type, worst_result_per_task)
    results_nan_fill["time_train_s"] = np.nan
    results_nan_fill["time_infer_s"] = np.nan
    assert len(results_nan_fill) == len(worst_result_per_task)

    results_raw = results_raw[results_raw["framework"].isin(frameworks_to_fill)]
    return _fill_missing(results_nan_fill=results_nan_fill, results_raw=results_raw)


def _fill_missing(results_nan_fill: pd.DataFrame, results_raw: pd.DataFrame) -> pd.DataFrame:
    frameworks = list(results_raw["framework"].unique())
    # datasets = results_nan_fill[['dataset', 'fold']].unique()
    results_nan_fill = results_nan_fill.set_index(["dataset", "fold"])
    results_nan_fills = []
    for f in frameworks:
        results_raw_f = results_raw[results_raw["framework"] == f]
        results_raw_f = results_raw_f.set_index(["dataset", "fold"])
        # results_raw_f['framework'] = f
        # results_nan_fill[results_nan_fill[]]
        a = results_nan_fill.index.difference(results_raw_f.index)
        results_nan_fill_f = results_nan_fill[results_nan_fill.index.isin(a)]
        results_nan_fill_f = results_nan_fill_f.reset_index()
        results_nan_fill_f["framework"] = f
        results_nan_fills.append(results_nan_fill_f)
    results_nan_fills = pd.concat(results_nan_fills, axis=0)
    results_raw = pd.concat([results_raw, results_nan_fills], axis=0)
    return results_raw


def convert_folds_into_separate_datasets(results_raw: pd.DataFrame, folds: list = None, fold_dummy=0) -> pd.DataFrame:
    """
    Converts results into appearing as if they are from only 1 fold.

    For example,
    Input:
        dataset='adult', fold=5
        dataset='adult', fold=6
    Output:
        dataset='adult_5', fold=0
        dataset='adult_6', fold=0

    :param folds:
    :param results_raw:
    :param fold_dummy:
    :return:
    """
    results_split_fold = results_raw.copy()
    if folds is not None:
        results_split_fold = results_split_fold[results_split_fold["fold"].isin(folds)]
    results_split_fold["dataset_og"] = results_split_fold["dataset"]
    results_split_fold["fold_og"] = results_split_fold["fold"]
    results_split_fold["dataset"] = results_split_fold["dataset"] + "_" + results_split_fold["fold"].astype(str)
    results_split_fold["fold"] = fold_dummy
    return results_split_fold


def assert_unique_dataset_tid_pairs(results_raw: pd.DataFrame):
    """
    Raises an assertion error if datset and tid columns do not have exact unique pairs,
    where each dataset is tied to exactly one tid and vice versa.

    Parameters
    ----------
    results_raw: pd.DataFrame
        A pandas DataFrame containing dataset and tid columns.

    """
    assert DATASET in results_raw.columns, f"{DATASET} must be a column in results_raw"
    assert "tid" in results_raw.columns, f"tid must be a column in results_raw when `use_tid_as_dataset_name=True`"
    results_unique_dataset_tid_pairs = results_raw[[DATASET, "tid"]].value_counts().reset_index(drop=False)
    unique_datasets = results_raw[DATASET].unique()
    unique_tids = results_raw["tid"].unique()
    if len(results_unique_dataset_tid_pairs) != len(unique_datasets) or len(results_unique_dataset_tid_pairs) != len(
        unique_tids
    ):
        dataset_counts = results_unique_dataset_tid_pairs[DATASET].value_counts()
        dataset_counts = dataset_counts[dataset_counts > 1].sort_values(ascending=False).to_frame()
        tid_counts = results_unique_dataset_tid_pairs["tid"].value_counts()
        tid_counts = tid_counts[tid_counts > 1].sort_values(ascending=False).to_frame()

        msg_dataset_counts = ""
        msg_tid_counts = ""
        if len(dataset_counts) > 0:
            msg_dataset_counts = f"\nDatasets appearing more than once:" f"\n{dataset_counts}"
        if len(tid_counts) > 0:
            msg_tid_counts = f"\nTIDs appearing more than once:" f"\n{tid_counts}"
        msg = (
            f"Error: Inconsistent pairings of dataset and tid column values! "
            f"Each dataset should pair with a unique tid. All below values should match:"
            f"\n\tUnique          Datasets: {len(unique_datasets)}"
            f"\n\tUnique              TIDs: {len(unique_tids)}"
            f"\n\tUnique Dataset/TID Pairs: {len(results_unique_dataset_tid_pairs)}"
            f"{msg_dataset_counts}"
            f"{msg_tid_counts}"
        )
        raise AssertionError(msg)
