import math

import numpy as np
import pandas as pd

from .constants import *


def compare_frameworks(
    results_raw,
    frameworks=None,
    banned_datasets=None,
    folds_to_keep=None,
    filter_errors=True,
    verbose=True,
    columns_to_agg_extra=None,
    datasets=None,
):
    columns_to_agg = [DATASET, FRAMEWORK, PROBLEM_TYPE, TIME_TRAIN_S, METRIC_ERROR]
    if columns_to_agg_extra:
        columns_to_agg += columns_to_agg_extra
    if frameworks is None:
        frameworks = sorted(list(results_raw[FRAMEWORK].unique()))

    if filter_errors:  # FIXME: This should not be toggled, instead filter_errors should be passed to filter_results
        results = filter_results(
            results_raw=results_raw,
            valid_frameworks=frameworks,
            banned_datasets=banned_datasets,
            folds_to_keep=folds_to_keep,
        )
    else:
        results = results_raw.copy()

    results_agg = results[columns_to_agg].groupby([DATASET, FRAMEWORK, PROBLEM_TYPE]).mean().reset_index()

    worst_scores = results_agg.sort_values(METRIC_ERROR, ascending=False).drop_duplicates(DATASET)
    worst_scores = worst_scores[[DATASET, METRIC_ERROR]]
    worst_scores.columns = [DATASET, "WORST_ERROR"]
    best_scores = results_agg.sort_values(METRIC_ERROR, ascending=True).drop_duplicates(DATASET)
    best_scores = best_scores[[DATASET, METRIC_ERROR]]
    best_scores.columns = [DATASET, "BEST_ERROR"]

    results_agg = results_agg.merge(best_scores, on=DATASET)
    results_agg = results_agg.merge(worst_scores, on=DATASET)
    results_agg[BESTDIFF] = 1 - (results_agg["BEST_ERROR"] / results_agg[METRIC_ERROR])
    results_agg[LOSS_RESCALED] = (results_agg[METRIC_ERROR] - results_agg["BEST_ERROR"]) / (
        results_agg["WORST_ERROR"] - results_agg["BEST_ERROR"]
    )
    results_agg[BESTDIFF] = results_agg[BESTDIFF].fillna(0)
    results_agg[LOSS_RESCALED] = results_agg[LOSS_RESCALED].fillna(0)
    results_agg = results_agg.drop(["BEST_ERROR"], axis=1)
    results_agg = results_agg.drop(["WORST_ERROR"], axis=1)

    for time_attr in [TIME_TRAIN_S, TIME_INFER_S]:
        if time_attr in columns_to_agg:
            best_time_attr = "BEST_" + time_attr
            best_speed = (
                results_agg[[DATASET, time_attr]].sort_values(time_attr, ascending=True).drop_duplicates(DATASET)
            )
            best_speed.columns = [DATASET, best_time_attr]
            results_agg = results_agg.merge(best_speed, on=DATASET)
            results_agg[time_attr + "_rescaled"] = results_agg[time_attr] / results_agg[best_time_attr]
            results_agg = results_agg.drop([best_time_attr], axis=1)

    valid_tasks = list(results_agg[DATASET].unique())

    results_ranked, results_ranked_by_dataset = rank_result(results_agg)
    rank_1 = results_ranked_by_dataset[results_ranked_by_dataset[RANK] == 1]
    rank_1_count = rank_1[FRAMEWORK].value_counts()
    results_ranked["rank=1_count"] = rank_1_count
    results_ranked["rank=1_count"] = results_ranked["rank=1_count"].fillna(0).astype(int)

    rank_2 = results_ranked_by_dataset[(results_ranked_by_dataset[RANK] > 1) & (results_ranked_by_dataset[RANK] <= 2)]
    rank_2_count = rank_2[FRAMEWORK].value_counts()

    results_ranked["rank=2_count"] = rank_2_count
    results_ranked["rank=2_count"] = results_ranked["rank=2_count"].fillna(0).astype(int)

    rank_3 = results_ranked_by_dataset[(results_ranked_by_dataset[RANK] > 2) & (results_ranked_by_dataset[RANK] <= 3)]
    rank_3_count = rank_3[FRAMEWORK].value_counts()

    results_ranked["rank=3_count"] = rank_3_count
    results_ranked["rank=3_count"] = results_ranked["rank=3_count"].fillna(0).astype(int)

    rank_l3 = results_ranked_by_dataset[(results_ranked_by_dataset[RANK] > 3)]
    rank_l3_count = rank_l3[FRAMEWORK].value_counts()

    results_ranked["rank>3_count"] = rank_l3_count
    results_ranked["rank>3_count"] = results_ranked["rank>3_count"].fillna(0).astype(int)

    if datasets is None:
        datasets = sorted(list(results_ranked_by_dataset[DATASET].unique()))
    datasets_len = len(datasets)
    errors_list = []
    for framework in frameworks:
        results_framework = filter_results(
            results_raw=results_raw,
            valid_frameworks=[framework],
            banned_datasets=banned_datasets,
            folds_to_keep=folds_to_keep,
        )
        results_framework_agg = (
            results_framework[columns_to_agg].groupby([DATASET, FRAMEWORK, PROBLEM_TYPE]).mean().reset_index()
        )
        num_valid = len(results_framework_agg[results_framework_agg[FRAMEWORK] == framework])
        num_errors = datasets_len - num_valid
        errors_list.append(num_errors)
    errors_series = pd.Series(data=errors_list, index=frameworks)

    results_ranked["error_count"] = errors_series
    results_ranked["error_count"] = results_ranked["error_count"].fillna(0).astype(int)
    results_ranked = results_ranked.reset_index()

    if verbose:
        print("valid_tasks:", len(valid_tasks))
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(results_ranked)
        print()
    return results_ranked, results_ranked_by_dataset


def filter_results(results_raw, valid_frameworks, banned_datasets=None, folds_to_keep=None):
    results = results_raw.copy()
    if folds_to_keep is not None:
        results = results[results[FOLD].isin(folds_to_keep)]
    if banned_datasets is not None:
        results = results[~results[DATASET].isin(banned_datasets)]
    results = keep_only_valid_datasets(results, valid_models=valid_frameworks)

    return results


def keep_only_valid_datasets(result_df, valid_models):
    result_df = result_df[result_df[FRAMEWORK].isin(valid_models)]
    tasks = list(result_df[DATASET].unique())
    frameworks = list(result_df[FRAMEWORK].unique())

    fw_datasets = {}
    for framework in frameworks:
        fw_datasets[framework] = set(list(result_df[result_df[FRAMEWORK] == framework][DATASET].unique()))
    valid_tasks = []
    for task in tasks:
        skip_task = False
        for framework in frameworks:
            if task not in fw_datasets[framework]:
                skip_task = True
                break
        if not skip_task:
            valid_tasks.append(task)
    if len(valid_tasks) == 0:
        valid_result_df = pd.DataFrame(columns=result_df.columns)
    else:
        valid_tasks_set = set(valid_tasks)
        valid_result_df = result_df[result_df[DATASET].isin(valid_tasks_set)].reset_index(drop=True)

    return valid_result_df


def rank_result(result_df):
    datasets = list(result_df[DATASET].unique())
    result_df = result_df.copy()
    result_df[METRIC_ERROR] = [round(x[0], 5) for x in zip(result_df[METRIC_ERROR])]
    num_frameworks = len(result_df[FRAMEWORK].unique())
    dfs = []
    if num_frameworks == 1:
        sorted_df_full = result_df
        sorted_df_full[RANK] = 1
    else:
        for dataset in datasets:
            dataset_df = result_df[result_df[DATASET] == dataset]
            sorted_df = dataset_df.copy()
            sorted_df[RANK] = sorted_df[METRIC_ERROR].rank()
            dfs.append(sorted_df)
        sorted_df_full = pd.concat(dfs, ignore_index=True)
    model_ranks_df = sorted_df_full.groupby([FRAMEWORK]).mean(numeric_only=True).sort_values(by=RANK)
    return model_ranks_df, sorted_df_full


def graph_vs(results_df: pd.DataFrame, f1: str, f2: str, z_stats: pd.Series = None, datasets: list = None):
    results_df = results_df[results_df[FRAMEWORK].isin([f1, f2])][[FRAMEWORK, DATASET, PROBLEM_TYPE, METRIC_ERROR]]
    results_df = results_df.groupby([FRAMEWORK, DATASET, PROBLEM_TYPE]).mean(numeric_only=True).reset_index()
    problem_type_dict = results_df.set_index(DATASET)[PROBLEM_TYPE].to_dict()
    results_1 = results_df[results_df[FRAMEWORK] == f1]
    results_1 = results_1.set_index(DATASET)[METRIC_ERROR].to_dict()
    results_2 = results_df[results_df[FRAMEWORK] == f2]
    results_2 = results_2.set_index(DATASET)[METRIC_ERROR].to_dict()

    if datasets is None:
        datasets = list(set(results_1.keys()).intersection(results_2.keys()))
        if z_stats is not None:  # FIXME: need to filter missing folds still
            datasets = list(set(datasets).intersection(set(z_stats.index)))

    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig, ax = plt.subplots(dpi=300)  # Create a figure containing a single axes.
    ax.set(xlabel=f2, ylabel=f1, title="Dataset Error Rate Comparison")

    results_1_lst = []
    results_2_lst = []
    r_1_dict = dict()
    r_2_dict = dict()
    z_stat_lst = []

    win_1 = 0
    win_2 = 0
    tie = 0

    from collections import defaultdict

    problem_type_d = defaultdict(list)
    problem_type_lst = []
    for dataset in datasets:
        r_1 = results_1[dataset]
        r_2 = results_2[dataset]

        problem_type = problem_type_dict[dataset]
        problem_type_d[problem_type].append(dataset)

        if r_1 == r_2 or abs(z_stats[dataset]) < 2:
            tie += 1
        elif r_1 > r_2:
            win_2 += 1
        else:
            win_1 += 1
        z_stat_lst.append(-z_stats[dataset])
        max_r = max(r_1, r_2)
        if max_r > 1:
            r_1 = r_1 / max_r
            r_2 = r_2 / max_r

        results_1_lst.append(r_1)
        results_2_lst.append(r_2)
        r_1_dict[dataset] = r_1
        r_2_dict[dataset] = r_2

        # problem_type_lst.append(marker)
        # if z_stats is not None:
        #     ax.scatter(r_1, r_2, c=z_stats[dataset], label=dataset)
        # else:
        #     ax.scatter(r_1, r_2, label=dataset)
    ax.plot((0, 1))
    cm = plt.cm.get_cmap("RdYlBu")

    for pt in problem_type_d:
        if pt == "binary":
            marker = "o"
        elif pt == "multiclass":
            marker = "^"
        elif pt == "regression":
            marker = "*"
        else:
            raise AssertionError(f"Unknown problem type: {pt}")

        pt_datasets = problem_type_d[pt]
        r_1_lst = [r_1_dict[d] for d in pt_datasets]
        r_2_lst = [r_2_dict[d] for d in pt_datasets]
        z_stats_lst = [-z_stats[d] for d in pt_datasets]
        im = ax.scatter(
            r_2_lst,
            r_1_lst,
            c=z_stats_lst,
            cmap=cm,
            norm=colors.Normalize(vmin=-3, vmax=3, clip=True),
            edgecolor="k",
            linewidths=0.5,
            marker=marker,
            label=pt,
        )

    textstr = f"Win 1: {win_1}, Win 2: {win_2}, Tie: {tie}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.05, 0.75, textstr, transform=ax.transAxes, fontsize=8, verticalalignment="top", bbox=props)

    ax.grid()
    # ax.set_xlim(0, 0.02)
    # ax.set_ylim(0, 0.02)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    fig.colorbar(im, ax=ax)
    plt.legend()
    plt.show()


# TODO: USE T-STATISTIC?
def compute_stderr_z_stat(
    results_df: pd.DataFrame, f1: str, f2: str, folds: list = None, win_z_threshold: float = 1.96, verbose: bool = True
) -> pd.Series:
    """
    Compute z-scores for each dataset by comparing f1 and f2.

    COMPARE BY:

    FOR EACH FOLD 1-10

    A1 - B1 = C1
    A2 - B2 = C2
    ...

    stddev of C1 - C10
    Calc probability Mean >0

    Parameters
    ----------
    results_df : :class:`pd.DataFrame`
        The results DataFrame. Must contain `dataset`, `fold`, `framework`, and `metric_error` columns.
    f1 : str
        The name of the first framework to compare.
    f2 : str
        The name of the second framework to compare.
    folds : list, default None
        The list of folds to use in the calculation.
        If None, infers based on folds used in f1 and f2.
    win_z_threshold : float, default 1.96
        The z-score threshold for declaring a framework as winning a dataset.
        If the absolute z-score is less than this value, then the dataset is considered a tie.
        Only relevant if `verbose=True`.
    verbose : bool, default True
        Whether to print info such as wins, ties, and a preview of the output pandas Series.

    Returns
    -------
    Returns a pandas Series where the index is the dataset and the value is the z-statistic. Positive z-statistics favor f1.

    """
    if f1 == f2:
        raise AssertionError("f1 and f2 cannot be the same!")

    results_df = results_df[[FRAMEWORK, DATASET, FOLD, METRIC_ERROR]]
    results_df = results_df[results_df[FRAMEWORK].isin([f1, f2])]
    if folds is None:
        folds = list(results_df[FOLD].unique())
    num_folds = len(folds)
    if num_folds <= 1:
        raise AssertionError("Not enough folds to calculate stderr")
    results_df = results_df[results_df[FOLD].isin(folds)]

    mean_std_error_df = results_df.groupby([FRAMEWORK, DATASET])[METRIC_ERROR].agg(["mean", "std", "count"])
    mean_std_error_df = mean_std_error_df.reset_index()
    dataset_row_count = mean_std_error_df.groupby(DATASET)["count"].sum()
    valid_datasets = list(dataset_row_count[dataset_row_count == num_folds * 2].index)

    results1 = results_df[results_df[FRAMEWORK] == f1].set_index([DATASET, FOLD])
    results2 = results_df[results_df[FRAMEWORK] == f2].set_index([DATASET, FOLD])

    results_diff = results2[METRIC_ERROR] - results1[METRIC_ERROR]

    datasets_z_statistic = {}
    for dataset in valid_datasets:
        diff_per_fold = results_diff[results_diff.index.get_level_values(DATASET) == dataset]
        mean_diff = diff_per_fold.mean()
        std_diff = diff_per_fold.std()
        stderr_diff = std_diff / math.sqrt(len(diff_per_fold))
        if stderr_diff == 0:
            if mean_diff > 0:
                z_diff = 99
            elif mean_diff < 0:
                z_diff = -99
            else:
                z_diff = 0
        else:
            z_diff = mean_diff / stderr_diff

        datasets_z_statistic[dataset] = z_diff

    win1 = 0
    win2 = 0
    tie = 0

    z_stat_series = pd.Series(datasets_z_statistic)
    z_stat_series = z_stat_series.sort_values(ascending=False)

    for dataset in datasets_z_statistic:
        z_stat = datasets_z_statistic[dataset]
        if z_stat > win_z_threshold:
            win1 += 1
        elif z_stat < -win_z_threshold:
            win2 += 1
        else:
            tie += 1
    if verbose:
        print(z_stat_series)
        print(f"Win {win1} : {f1}")
        print(f"Win {win2} : {f2}")
        print(f"Tie {tie}")

    return z_stat_series


# TODO: document
def compute_stderr_z_stat_bulk(
    framework: str,
    frameworks_to_compare: list,
    results_raw: pd.DataFrame,
    folds: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compares framework with each framework in frameworks_to_compare, constructing an overall result DataFrame as output.

    :param framework:
    :param frameworks_to_compare:
    :param results_raw:
    :param folds:
    :param verbose:
    :return:
    """
    assert framework not in frameworks_to_compare
    z_stat_dict = {}
    for f2 in frameworks_to_compare:
        z_stat_series = compute_stderr_z_stat(results_raw, f1=framework, f2=f2, folds=folds, verbose=verbose)
        z_stat_dict[f2] = z_stat_series

    z_stat_df = pd.DataFrame(z_stat_dict)
    z_stat_df["min"] = z_stat_df.min(axis=1)
    z_stat_df = z_stat_df.sort_values(by=["min"], ascending=False)

    if verbose:
        print("Z-statistic comparisons:")
        with pd.option_context("display.max_columns", None, "display.width", 1000):
            print(z_stat_df)
    return z_stat_df


def compute_win_rate_per_dataset(
    f1: str,
    f2: str,
    results_raw: pd.DataFrame,
    folds: list = None,
    verbose: bool = True,
    epsilon=1e-5,
    epsilon_bestdiff=1e-4,
):
    if f1 == f2:
        raise AssertionError("f1 and f2 cannot be the same!")

    results_df = results_raw[[FRAMEWORK, DATASET, FOLD, METRIC_ERROR]]
    results_df = results_df[results_df[FRAMEWORK].isin([f1, f2])]
    if folds is None:
        folds = list(results_df[FOLD].unique())
    num_folds = len(folds)
    # if num_folds <= 1:
    #     raise AssertionError('Not enough folds to calculate stderr')
    results_df = results_df[results_df[FOLD].isin(folds)]

    dataset_row_count = results_df[DATASET].value_counts()
    valid_datasets = list(dataset_row_count.index)
    # valid_datasets = list(dataset_row_count[dataset_row_count == num_folds*2].index)

    out_dict = dict()

    for dataset in valid_datasets:
        bestdiff_dataset = []
        f1_wins = 0
        num_ties = 0
        valid_folds = 0
        dataset_df = results_df[results_df[DATASET] == dataset]
        for fold in folds:
            dataset_df_fold = dataset_df[dataset_df[FOLD] == fold]
            dataset_fold_dict = dataset_df_fold.set_index(FRAMEWORK)[METRIC_ERROR].to_dict()
            # print(dataset_fold_dict)
            if len(dataset_fold_dict.keys()) != 2:
                continue
            valid_folds += 1
            f1_metric_error = dataset_fold_dict[f1]
            f2_metric_error = dataset_fold_dict[f2]

            bestdiff = 0
            tie = False
            if f1_metric_error == f2_metric_error:
                tie = True
            elif epsilon is not None and abs(f1_metric_error - f2_metric_error) < epsilon:
                tie = True
                # print(f'within epsilon {dataset} {fold} | {f1_metric_error} | {f2_metric_error}')
            elif f1_metric_error < f2_metric_error:
                bestdiff = 1 - f1_metric_error / f2_metric_error
            else:
                bestdiff = -(1 - f2_metric_error / f1_metric_error)
            if not tie and epsilon_bestdiff is not None and abs(bestdiff) < epsilon_bestdiff:
                # print(f'bestdiff tie: {bestdiff} | {dataset} {fold} | {f1_metric_error} | {f2_metric_error}')
                tie = True
            if tie:
                f1_wins += 0.5
            elif bestdiff > 0:
                f1_wins += 1
            if tie:
                num_ties += 1
            bestdiff_dataset.append(bestdiff)
        if valid_folds == 0:
            continue
        tierate = num_ties / valid_folds
        bestdiff_dataset = np.mean(bestdiff_dataset)
        f1_winrate = f1_wins / valid_folds
        # print(f'{f1_wins}/{valid_folds} | {dataset}')
        out_dict[dataset] = dict(
            winrate=f1_winrate,
            bestdiff=bestdiff_dataset,
            tierate=tierate,
            wins=f1_wins,
            folds=valid_folds,
        )

    out_df = pd.DataFrame(out_dict).T
    out_df = out_df.sort_values(by=["winrate", "bestdiff", "folds", "tierate"], ascending=[False, False, False, True])
    print(f"winrate {f1} vs {f2}")
    print(out_df)
