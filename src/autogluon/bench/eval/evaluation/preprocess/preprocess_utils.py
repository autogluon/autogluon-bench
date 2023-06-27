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


def fill_missing_results_with_default(framework_nan_fill: str, frameworks_to_fill: list, results_raw: pd.DataFrame):
    """
    Fill missing results with the result of `framework_nan_fill` framework.
    """
    assert framework_nan_fill is not None
    results_nan_fill = results_raw[results_raw["framework"] == framework_nan_fill]
    results_nan_fill = results_nan_fill[["dataset", "fold", "metric_score", "metric_error", "problem_type"]]
    results_nan_fill["time_train_s"] = 3600
    results_nan_fill["time_infer_s"] = 1

    results_raw = results_raw[results_raw["framework"].isin(frameworks_to_fill)]

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
        print(f)
        print(results_nan_fill_f)
        print(a)
    results_nan_fills = pd.concat(results_nan_fills, axis=0)
    print(results_nan_fills)
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
