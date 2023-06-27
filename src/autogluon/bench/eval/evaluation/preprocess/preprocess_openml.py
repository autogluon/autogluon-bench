import pandas as pd

from ..constants import *
from . import preprocess_utils


def preprocess_openml_input(
    path, framework_suffix=None, framework_rename_dict=None, folds_to_keep=None, framework_suffix_column=None
):
    raw_input = pd.read_csv(path)
    raw_input = _rename_openml_columns(raw_input)
    raw_input = raw_input[raw_input[FRAMEWORK].notnull()]
    if framework_rename_dict is not None:
        for key in framework_rename_dict.keys():
            raw_input[FRAMEWORK] = [
                framework_rename_dict[key] if framework[0] == key else framework[0]
                for framework in zip(raw_input[FRAMEWORK])
            ]
    if framework_suffix_column is not None:
        if "framework_parent" in raw_input.columns:
            raw_input["framework_parent"] = raw_input["framework_parent"] + "_" + raw_input[framework_suffix_column]
        else:
            raw_input[FRAMEWORK] = raw_input[FRAMEWORK] + "_" + raw_input[framework_suffix_column]

    elif framework_suffix is not None:
        if "framework_parent" in raw_input.columns:
            raw_input["framework_parent"] = [
                framework[0] + framework_suffix for framework in zip(raw_input["framework_parent"])
            ]
        else:
            raw_input[FRAMEWORK] = [framework[0] + framework_suffix for framework in zip(raw_input[FRAMEWORK])]

    # TODO: This is a hack and won't work for all metrics, metric_error should ideally be calculated prior to preprocessing
    raw_input[METRIC_ERROR] = [
        1 - score if metric in ["auc", "acc", "balacc", "r2"] else -score
        for score, metric in zip(raw_input[METRIC_SCORE], raw_input["metric"])
    ]

    if raw_input[METRIC_ERROR].min() < 0:
        # TODO: update values below 0 to 0
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

    cleaned_input["tid"] = [int(x.split("/")[-1]) for x in cleaned_input["id"]]

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
    return cleaned_input


def _rename_openml_columns(result_df):
    renamed_df = result_df.rename(
        columns={
            "type": PROBLEM_TYPE,
            TASK: DATASET,
            FRAMEWORK: FRAMEWORK,
            RESULT: METRIC_SCORE,
            DURATION: TIME_TRAIN_S,
            PREDICT_DURATION: TIME_INFER_S,
        }
    )
    if "training_duration" in renamed_df.columns:
        renamed_df[TIME_TRAIN_S] = renamed_df["training_duration"]
        renamed_df = renamed_df.drop(columns=["training_duration"])
    if "score_test" in renamed_df.columns:
        print(
            f"score_test found in columns! Treating result as AG leaderboard format. score_test will be mapped to {METRIC_SCORE}, pred_time_test will be mapped to {TIME_INFER_S}."
        )
        renamed_df = renamed_df.rename(columns={"score_test": METRIC_SCORE})
        renamed_df = renamed_df.rename(columns={"pred_time_test": TIME_INFER_S})

    return renamed_df
