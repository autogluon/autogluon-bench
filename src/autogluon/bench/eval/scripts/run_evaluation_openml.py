from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd
import typer

from autogluon.bench.eval.evaluation import evaluate_results
from autogluon.bench.eval.evaluation.benchmark_evaluator import BenchmarkEvaluator
from autogluon.bench.eval.evaluation.constants import TIME_INFER_S
from autogluon.bench.eval.evaluation.evaluate_utils import (
    compute_stderr_z_stat,
    compute_stderr_z_stat_bulk,
    compute_win_rate_per_dataset,
    graph_vs,
)

app = typer.Typer()


@app.command()
def evaluate_amlb_results(
    results_dir: str = typer.Option("data/results/", help="Root directory of raw and prepared results"),
    results_dir_input: str = typer.Option(
        None,
        help="Directory of the results file '<file_prefix><constraint_str><benchmark_name_str>.csv' getting cleaned. Can be an S3 URI. If not provided, it defaults to '<results_dir>input/prepared/openml/'",
    ),
    results_dir_output: str = typer.Option(
        None,
        help="Output directory of evaluation files. If not provided, it defaults to '<results_dir>output/openml/<output_suffix>/'",
    ),
    paths: Optional[List[str]] = typer.Option(
        None,
        help="List of file paths under '<results_dir>input/prepared/openml/' or <results_dir_input> to load the input data from. Can also include files located in s3 assuming you have the proper read permissions. E.g. 'path1,path2,...",
    ),
    frameworks_run: Optional[List[str]] = typer.Option(
        None,
        help="List of framework to compare. These frameworks must be present in the 'framework' column of the loaded input files listed in the `paths` arg. E.g. 'framework1,framework2,...",
    ),
    output_suffix: str = typer.Option(
        "ag_eval",
        help="Output suffix of the path to save the output files, e.g. '<results_dir>output/openml/<output_suffix>/'.",
    ),
    framework_nan_fill: str = typer.Option(
        None,
        help="Framework used as the default result to fill missing values for the frameworks in `frameworks`. E.g., if `framework_nan_fill='foo'`, for dataset A of `framework='bar'` has no result (or NaN result), it is replaced by the result of `'foo'` on dataset A.",
    ),
    problem_types: Optional[List[str]] = typer.Option(
        None, help="List of problem types to filter results to. E.g. 'problem_type1,problem_type2,..."
    ),
    folds_to_keep: Optional[List[int]] = typer.Option(
        None, help="List of result folds to use. By default folds 0-9 (10-fold) are used."
    ),
    banned_datasets: Optional[List[str]] = typer.Option(
        None, help="List of datasets to skip during evaluation. E.g. 'dataset1,dataset2,..."
    ),
    infer_batch_size: int = typer.Option(
        None,
        help="If specified, will replace the `time_infer_s` column with the value in column `pred_time_test_with_transform_batch_size_{infer_batch_size}`. If a given row does not have a value in the infer_batch_size column, the original `time_infer_s` value is used.",
    ),
    compute_z_score: bool = typer.Option(
        True,
        help="Whether to compute comparison z-scores. The framework being compared against is the first one listed in `frameworks_run`. Only valid when multiple folds are used and `treat_folds_as_datasets=False`.",
    ),
    treat_folds_as_datasets: bool = typer.Option(False, help="If True, treat each fold as a separate dataset."),
    use_tid_as_dataset_name: bool = typer.Option(
        True,
        help="If True, replaces dataset human-readable names with unique integer IDs associated with their OpenML task ID.",
    ),
    filter_errors: bool = typer.Option(
        False,
        help="If True, any dataset missing a result from any framework in `frameworks` will be filtered from all results. Takes priority over `framework_nan_fill`.",
    ),
    clean_data: bool = typer.Option(
        True,
        help="If True, performs some general data cleanup to avoid inconsistencies in dataset naming compared to task ids.",
    ),
) -> Dict[str, pd.DataFrame]:
    """
    Generate evaluation results and saves the results to <results_dir>output/openml/<output_suffix>/evaluation_results.json

    Example:
    agbench evaluate-amlb-results --frameworks_run framework_1 --frameworks_run framework_2 --paths openml_ag_ag_bench_20230707T070230.csv --results-dir-input data/results/input/prepared/openml --no-clean-data
    """
    evaluate(
        frameworks_run=frameworks_run if frameworks_run else None,
        paths=paths if paths else None,
        results_dir=results_dir,
        results_dir_input=results_dir_input,
        results_dir_output=results_dir_output,
        output_suffix=output_suffix,
        framework_nan_fill=framework_nan_fill,
        problem_type=problem_types if problem_types else None,
        folds_to_keep=folds_to_keep if folds_to_keep else None,
        compute_z_score=compute_z_score,
        treat_folds_as_datasets=treat_folds_as_datasets,
        banned_datasets=banned_datasets if banned_datasets else None,
        infer_batch_size=infer_batch_size,
        clean_data=clean_data,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        filter_errors=filter_errors,
    )


# TODO: Rename to a more description function, or convert to a class
def evaluate(
    *,
    frameworks_run: List[str],
    paths: List[str],
    results_dir: str = "data/results/",
    results_dir_input: str = None,
    results_dir_output: str = None,
    output_suffix: str = "ag_full_v5/1h8c",
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
    if results_dir_input is None:
        results_dir_input = os.path.join(results_dir, "input/prepared/openml/")
    if results_dir_output is None:
        results_dir_output = os.path.join(results_dir, f"output/openml/{output_suffix}/")

    if folds_to_keep is None:
        folds_to_keep = [i for i in range(10)]

    frameworks_compare_vs_all = []
    if len(frameworks_compare_vs_all) == 0:
        frameworks_compare_vs_all = [frameworks_run[0]]

    benchmark_evaluator = BenchmarkEvaluator(
        results_dir_input=results_dir_input,
        results_dir_output=results_dir_output,
        output_suffix=output_suffix,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        framework_nan_fill=framework_nan_fill,
        filter_errors=filter_errors,
    )

    results_raw = benchmark_evaluator.load_data(
        paths=paths,
        frameworks=frameworks_run,
        folds=folds_to_keep,
        problem_type=problem_type,
        banned_datasets=banned_datasets,
        infer_batch_size=infer_batch_size,
        treat_folds_as_datasets=treat_folds_as_datasets,
        clean_data=clean_data,
    )

    folds_to_keep = sorted(results_raw["fold"].unique())

    if len(frameworks_run) > 1:
        compute_win_rate_per_dataset(
            f1=frameworks_run[0], f2=frameworks_run[1], results_raw=results_raw, folds=folds_to_keep
        )
    if compute_z_score and len(frameworks_run) > 1 and len(folds_to_keep) > 1:
        z_stat_df = compute_stderr_z_stat_bulk(
            framework=frameworks_run[0], frameworks_to_compare=frameworks_run[1:], results_raw=results_raw
        )
        z_stat_series = compute_stderr_z_stat(
            results_raw, f1=frameworks_run[0], f2=frameworks_run[1], folds=folds_to_keep, verbose=False
        )
        graph_vs(results_df=results_raw, f1=frameworks_run[0], f2=frameworks_run[1], z_stats=z_stat_series)

    (
        results_ranked,
        results_ranked_by_dataset,
        results_ranked_all,
        results_ranked_by_dataset_all,
        results_pairs_merged_dict,
    ) = evaluate_results.evaluate(
        results_raw=results_raw,
        frameworks=frameworks_run,
        columns_to_agg_extra=[
            TIME_INFER_S,
        ],
        frameworks_compare_vs_all=frameworks_compare_vs_all,
        output_dir=benchmark_evaluator.results_dir_output,
    )

    return (
        results_ranked,
        results_ranked_by_dataset,
        results_ranked_all,
        results_ranked_by_dataset_all,
        results_pairs_merged_dict,
    )


if __name__ == "__main__":
    app()
