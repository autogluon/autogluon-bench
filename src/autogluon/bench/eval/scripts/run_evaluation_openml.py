from __future__ import annotations

import argparse
from typing import Dict, List

import pandas as pd

from autogluon.bench.eval.evaluation import evaluate_results
from autogluon.bench.eval.evaluation.constants import TIME_INFER_S
from autogluon.bench.eval.evaluation.evaluate_utils import compute_stderr_z_stat, compute_stderr_z_stat_bulk, compute_win_rate_per_dataset, graph_vs
from autogluon.bench.eval.evaluation.benchmark_evaluator import BenchmarkEvaluator


# TODO: Rename to a more description function, or convert to a class
def run(
    *,
    frameworks_run: List[str],
    paths: List[str],
    output_suffix: str = 'ag_full_v5/1h8c',
    framework_nan_fill: str | None = None,
    problem_type: List[str] | str | None = None,
    folds_to_keep: List[int] | None = None,
    compute_z_score: bool = True,
    treat_folds_as_datasets: bool = False,
    banned_datasets: List[str] | None = None,
    infer_batch_size: int | None = None,
    clean_data: bool = True,
    use_tid_as_dataset_name: bool = True,
    filter_errors: bool = False,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]):
    """
    # TODO: Add description
    Parameters
    ----------
    frameworks_run : List[str]
        The list of frameworks to compare.
        These frameworks must be present in the "frameworks" column of the loaded input files listed in the `paths` arg.
    paths : List[str]
        The list of file paths to load the input data from.
        The resulting input DataFrame will be the concatenation of all files listed in `paths`.
        Filepaths can include files located in s3 assuming you have the proper read permissions.
        # TODO: Define each column that must exist in paths
        # TODO: Consider allowing non-preprocessed data that is preprocessed on-the-fly for convenience
        # TODO: Allow passing the raw results DataFrame directly rather than requiring always loading from paths.
    output_suffix : str
        The output suffix of the path to save the output files.
        # TODO: Expand this description, add more control over save locations, add option to avoid saving entirely
    framework_nan_fill : str, optional
        The framework used as the default result to fill missing values for the frameworks in `frameworks_run`.
        For example, if `framework_nan_fill='foo'`, for dataset A if `framework='bar'` has no result (or NaN result),
        it is replaced by the result of `'foo'` on dataset A.
        A common usage of this is to specify some simple baseline, such as a constant predictor or dummy model.
    problem_type : List[str] | str, optional, default None
        The list of problem types to filter results to. By default, all problem types are used, and no filtering occurs.
    folds_to_keep : List[int], optional
        The list of result folds to use. All others are dropped. By default folds 0-9 (10-fold) are used.
        More folds leads to greater statistical certainty.
    compute_z_score : bool, default True
        Whether to compute comparison z-scores. The framework being compared against is the first one listed in `frameworks_run`.
        Only valid when multiple folds are used and `treat_folds_as_datasets=False`.
    treat_folds_as_datasets : bool, default False
        If True, all dataset x fold results are treated as their own separate datasets,
        rather than averaging the fold results together.
    banned_datasets : List[str], optional
        If specified, the list of datasets are filtered from the datasets used in evaluation.
    infer_batch_size : int, optional
        If specified, will replace the `time_infer_s` column with the value in column `pred_time_test_with_transform_batch_size_{infer_batch_size}`
        If a given row does not have a value in the infer_batch_size column, the original `time_infer_s` value is used.
    clean_data : bool, default True
        If True, performs some general data cleanup to avoid inconsistencies in dataset naming compared to task ids.
        If True, `time_infer_s` is updated to be the time per row, rather than the overall time.
        Must be set to False if using results that don't contain tids or lack a task metadata file.
    use_tid_as_dataset_name : bool, default False
        If True, replaces dataset human-readable names with unique integer IDs associated with their OpenML task ID.
    filter_errors : bool, default False
        If True, any dataset missing a result from any framework in `frameworks_run` will be filtered from all results.
        This ensures a dense result evaluation, but may lead to many datasets being filtered due to sparse framework failures.
        If True, takes priority over `framework_nan_fill`.
    Returns
    -------
    results_ranked : pd.DataFrame
        A pandas DataFrame with 1 row per framework in `frameworks_run`.
        Includes a variety of analytics of overall performance aggregated across datasets.
        This result only contains datasets where all frameworks succeeded.
        This is the aggregated form of `results_ranked_by_dataset`.
        Default column names:
        ['framework', 'time_train_s', 'metric_error', 'time_infer_s', 'bestdiff', 'loss_rescaled',
        'time_train_s_rescaled', 'time_infer_s_rescaled',
        'rank', 'rank=1_count', 'rank=2_count', 'rank=3_count', 'rank>3_count', 'error_count']
        Column Definitions
            framework : The name of the framework
            time_train_s : The mean training time in seconds
            time_infer_s : The mean inference time in seconds (per row if clean_data=True)
            bestdiff : The mean percentage relative less errors the best framework on a given dataset has compared to this framework.
                If the framework is the best for a given dataset, then the bestdiff value on that dataset is 0.
                If the best framework gets 0.05 metric_error, and framework FOO gets 0.20 metric_error,
                    then bestdiff for this dataset is 0.75 for FOO because (0.20 - 0.05)/0.20 = 0.75
                    (aka FOO has 4x more error than the best framework, and the best framework has 75% less error than FOO)
                0 is the best, 1 is the worst.
            loss_rescaled : The mean rescaled version of `metric_error`, where the best framework is rescaled to 0 and the worst framework is rescaled to 1 for each dataset.
                0 is the best, 1 is the worst.
            *_rescaled : The mean rescaled version of the original column, where the fastest framework for each dataset is rescaled to 1.
            rank : mean rank
            rank=1_count : number of undisputed first-place (champion) datasets
            rank=2_count : number of second-place datasets (or 2-way tie for 1st place) (rank >1, <=2)
            rank=3_count : number of third-place datasets (rank >2, <=3)
            rank>3_count : number of datasets with placement worse than 3rd.
            error_count : number of failed datasets with no result.
    results_ranked_all : pd.DataFrame
        Identical to `results_ranked`, except it also includes datasets where a subset of frameworks failed.
    results_ranked_by_dataset : pd.DataFrame
        A pandas DataFrame with N rows per framework, where N is the number of datasets.
        This result contains evaluations on a per-dataset granularity.
        `results_ranked` is the aggregated form of this result.
        # TODO: Define each column
    results_ranked_by_dataset_all : pd.DataFrame
        Identical to `results_ranked_by_dataset`, except it also includes datasets where a subset of frameworks failed.
    results_pairs_merged_dict : Dict[str, pd.DataFrame]
        For each framework that was pair-wise compared with other frameworks, the key is the framework name,
        and the value is a pandas DataFrame of the comparison evaluation results.
        In general, only the first framework in `frameworks_run` is pair-wise compared.
        # TODO: Define each column
    """
    results_dir = 'data/results/'
    if folds_to_keep is None:
        folds_to_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    frameworks_compare_vs_all = []
    if len(frameworks_compare_vs_all) == 0:
        frameworks_compare_vs_all = [frameworks_run[0]]

    benchmark_evaluator = BenchmarkEvaluator(
        results_dir=results_dir,
        output_suffix=output_suffix,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        framework_nan_fill=framework_nan_fill,
        filter_errors=filter_errors,
    )

    results_raw = benchmark_evaluator.load_data(paths=paths,
                                                frameworks=frameworks_run,
                                                folds=folds_to_keep,
                                                problem_type=problem_type,
                                                banned_datasets=banned_datasets,
                                                infer_batch_size=infer_batch_size,
                                                treat_folds_as_datasets=treat_folds_as_datasets)

    folds_to_keep = sorted(results_raw['fold'].unique())

    if len(folds_to_keep) > 1:
        compute_win_rate_per_dataset(f1=frameworks_run[0], f2=frameworks_run[1], results_raw=results_raw, folds=folds_to_keep)
    if compute_z_score and len(folds_to_keep) > 1:
        z_stat_df = compute_stderr_z_stat_bulk(framework=frameworks_run[0], frameworks_to_compare=frameworks_run[1:], results_raw=results_raw)
        z_stat_series = compute_stderr_z_stat(results_raw, f1=frameworks_run[0], f2=frameworks_run[1], folds=folds_to_keep, verbose=False)
        graph_vs(results_df=results_raw, f1=frameworks_run[0], f2=frameworks_run[1], z_stats=z_stat_series)

    results_ranked, results_ranked_by_dataset, results_ranked_all, results_ranked_by_dataset_all, results_pairs_merged_dict = evaluate_results.evaluate(
        results_raw=results_raw,
        frameworks=frameworks_run,
        columns_to_agg_extra=[
            TIME_INFER_S,
        ],
        frameworks_compare_vs_all=frameworks_compare_vs_all,
        output_dir=benchmark_evaluator.results_dir_output,
    )

	return results_ranked, results_ranked_by_dataset, results_ranked_all, results_ranked_by_dataset_all, results_pairs_merged_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, help="Results Paths", required=True, nargs='+')
    parser.add_argument('--frameworks_run', type=str, help="Name of framework runs", required=True, nargs='+')
    parser.add_argument('--problem_types', type=str, help="Problem types to evaluate", choices=['binary', 'multiclass', 'regression'], default=['binary', 'multiclass', 'regression'], nargs="+")
    parser.add_argument('--folds_to_keep', type=int, help="Folds to keep for evaluation", nargs="*")
    parser.add_argument('--filter_errors', type=bool, help="Filter errors during evaluation", default=False)
    parser.add_argument('--banned_datasets', type=str, help="Datasets to skip", default=['car', 'kr-vs-kp', 'OnlineNewsPopularity'], nargs='+')

    args = parser.parse_args()

    run(
        paths=args.paths,
        frameworks_run=args.frameworks_run,
        output_suffix=f'autogluon-bench-text',
        framework_nan_fill='constantpredictor',
        problem_type=args.problem_types,
        treat_folds_as_datasets=False,
        infer_batch_size=None,
        filter_errors=args.filter_errors,
        use_tid_as_dataset_name=False,
        banned_datasets=args.banned_datasets,
        folds_to_keep=args.folds_to_keep,
        compute_z_score=False,
    )
