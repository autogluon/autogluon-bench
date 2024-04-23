from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..constants import *
from . import preprocess_utils


def _update_framework_name(name: str, parent: str, name_suffix_1: str, name_suffix_2: str):
    is_valid_parent = parent and isinstance(parent, str)
    if is_valid_parent:
        name_new = parent
    else:
        name_new = name
    if name_suffix_1:
        name_new = name_new + "_" + str(name_suffix_1)
    if name_suffix_2:
        name_new = name_new + "_" + str(name_suffix_2)
    if is_valid_parent:
        return name, name_new
    else:
        return name_new, parent


def _parent_rename(parent: str | None, name: str):
    if parent and isinstance(parent, str):
        rename = parent + "_" + name
    else:
        rename = name
    return rename


def preprocess_openml_input(
    df: pd.DataFrame,
    framework_suffix=None,
    framework_rename_dict=None,
    folds_to_keep=None,
    framework_suffix_column=None,
):
    raw_input = df.copy()
    assert len(raw_input.columns) == len(set(raw_input.columns))
    raw_input = _rename_openml_columns(raw_input)
    raw_input = raw_input[raw_input[FRAMEWORK].notnull()]
    if framework_rename_dict is not None:
        for key in framework_rename_dict.keys():
            raw_input[FRAMEWORK] = [
                framework_rename_dict[key] if framework[0] == key else framework[0]
                for framework in zip(raw_input[FRAMEWORK])
            ]

    if framework_suffix_column:
        framework_suffix_column_to_zip = [f for f in raw_input[framework_suffix_column]]
    else:
        framework_suffix_column_to_zip = [None] * len(raw_input)
    if "framework_parent" in raw_input.columns:
        framework_parent_to_zip = [f for f in raw_input["framework_parent"]]
    else:
        framework_parent_to_zip = [None] * len(raw_input)

    updated_names_and_parent = [
        _update_framework_name(
            name=name,
            parent=parent,
            name_suffix_1=name_suffix_1,
            name_suffix_2=framework_suffix,
        )
        for (name, parent, name_suffix_1) in zip(
            raw_input[FRAMEWORK], framework_parent_to_zip, framework_suffix_column_to_zip
        )
    ]

    updated_names = [name for name, parent in updated_names_and_parent]
    updated_parents = [parent for name, parent in updated_names_and_parent]

    raw_input[FRAMEWORK] = updated_names
    if "framework_parent" in raw_input.columns:
        raw_input["framework_parent"] = updated_parents

    if "framework_parent" in raw_input:
        # raw_input["framework_child"] = raw_input[FRAMEWORK]
        raw_input["framework_child"] = np.where(raw_input["framework_parent"].isnull(), np.nan, raw_input[FRAMEWORK])

    if "framework_parent" in raw_input:
        framework_parent_to_zip = [f for f in raw_input["framework_parent"]]
    else:
        framework_parent_to_zip = [None] * len(raw_input)

    raw_input[FRAMEWORK] = [
        _parent_rename(
            parent=parent,
            name=name,
        )
        for (parent, name) in zip(framework_parent_to_zip, raw_input[FRAMEWORK])
    ]

    # TODO: This is a hack and won't work for all metrics, metric_error should ideally be calculated prior to preprocessing
    metric_list = [
        "auc",
        "acc",
        "accuracy",
        "balacc",
        "map",
        "roc_auc",
        "r2",
        "coverage",
        "f1",
        "f1_macro",
        "f1_micro",
        "quadratic_kappa",
    ]
    if METRIC_ERROR not in raw_input:
        raw_input[METRIC_ERROR] = [
            1 - score if metric in metric_list else float(score) if metric == "rmse" else -score
            for score, metric in zip(raw_input[METRIC_SCORE], raw_input["metric"])
        ]

    if raw_input[METRIC_ERROR].min() < 0:
        eps = -1 / 1e8
        num_negative = len(raw_input[raw_input[METRIC_ERROR] < 0])

        if raw_input[METRIC_ERROR].min() < eps:
            raise AssertionError(
                f"METRIC_ERROR cannot be negative! There may be a bug. "
                f"Found min value: {raw_input[METRIC_ERROR].min()}. "
                f"{num_negative} rows had negative values!"
            )
        else:
            print(
                f"WARNING: min METRIC_ERROR was found to be negative, but was higher than epsilon {eps}! "
                f"({raw_input[METRIC_ERROR].min()}) {num_negative} rows had negative values! "
                f"Setting all negative values to 0."
            )
            raw_input.loc[raw_input[METRIC_ERROR] < 0, METRIC_ERROR] = 0
    assert raw_input[METRIC_ERROR].min() >= 0

    cleaned_input = preprocess_utils.clean_result(raw_input, folds_to_keep=folds_to_keep, remove_invalid=False)

    if "tid" not in cleaned_input:
        if "id" in cleaned_input:
            cleaned_input["tid"] = [
                int(part) for x in cleaned_input["id"] for part in re.split(r"[/_]", x)[-1:] if part.isdigit()
            ]

    if "tid" in cleaned_input:
        """
        Update tid to the new ones in case the runs are old

        Name                | oldtid | newtid

        KDDCup09-Upselling  | 360115 | 360975
        MIP-2016-regression | 359947 | 360945
        QSAR-TID-10980      |  14097 | 360933
        QSAR-TID-11         |  13854 | 360932
        """
        cleaned_input["tid"] = (
            cleaned_input["tid"]
            .map(
                {
                    360115: 360975,
                    359947: 360945,
                    14097: 360933,
                    13854: 360932,
                }
            )
            .fillna(cleaned_input["tid"])
        )

    required_output_columns = [
        DATASET,
        FOLD,
        FRAMEWORK,
        METRIC_ERROR,
        METRIC,
        PROBLEM_TYPE,
        TIME_TRAIN_S,
        TIME_INFER_S,
    ]
    actual_columns = list(cleaned_input.columns)
    missing_columns = [c for c in required_output_columns if c not in actual_columns]
    if missing_columns:
        raise AssertionError(f"Missing expected output columns: {missing_columns}")

    reordered_columns = required_output_columns + [c for c in actual_columns if c not in required_output_columns]
    cleaned_input = cleaned_input[reordered_columns]

    return cleaned_input


def _update_and_drop_column(df: pd.DataFrame, col_to_update: str, col_to_drop: str) -> pd.DataFrame:
    if col_to_drop in df:
        if col_to_update in df:
            df[col_to_update] = df[col_to_drop].fillna(df[col_to_update])
            df = df.drop(columns=[col_to_drop])
        else:
            df = df.rename(columns={col_to_drop: col_to_update})
    return df


def _rename_openml_columns(result_df):
    rename_order = [
        ("type", PROBLEM_TYPE),
        (TASK, DATASET),
        (RESULT, METRIC_SCORE),
        (DURATION, TIME_TRAIN_S),
        (PREDICT_DURATION, TIME_INFER_S),
        ("training_duration", TIME_TRAIN_S),
        ("score_test", METRIC_SCORE),
        ("pred_time_test", TIME_INFER_S),
    ]
    renamed_df = result_df.copy()
    for col_to_drop, col_to_update in rename_order:
        renamed_df = _update_and_drop_column(renamed_df, col_to_update=col_to_update, col_to_drop=col_to_drop)
    return renamed_df
