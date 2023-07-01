import os
from typing import Dict

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
    results_input_dir: str = typer.Option(
        None,
        help="Directory of the results file '<file_prefix><constraint_str><benchmark_name_str>.csv' getting cleaned. Can be an S3 URI. If not provided, it defaults to '<results_dir>input/prepared/openml/'",
    ),
    results_output_dir: str = typer.Option(
        None,
        help="Output directory of evaluation files. If not provided, it defaults to '<results_dir>output/openml/<output_suffix>/'",
    ),
    paths: str = typer.Option(
        ...,
        help="List of file paths under '<results_dir>input/prepared/openml/' or <results_input_dir> to load the input data from. Can also include files located in s3 assuming you have the proper read permissions. E.g. 'path1,path2,...",
    ),
    frameworks: str = typer.Option(
        ...,
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
    problem_types: str = typer.Option(
        None, help="List of problem types to filter results to. E.g. 'problem_type1,problem_type2,..."
    ),
    folds_to_keep: str = typer.Option(
        None, help="List of result folds to use. By default folds 0-9 (10-fold) are used."
    ),
    banned_datasets: str = typer.Option(
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
) -> Dict[str, pd.DataFrame]:
    """
    Generate evaluation results and saves the results to <results_dir>output/openml/<output_suffix>/evaluation_results.json

    """

    if results_input_dir is None:
        results_input_dir = os.path.join(results_dir, "input/prepared/openml/")
    if results_output_dir is None:
        results_output_dir = os.path.join(results_dir, f"output/openml/{output_suffix}/")
    if paths is not None:
        paths = paths.split(",")
    if frameworks is not None:
        frameworks = frameworks.split(",")
    if problem_types is not None:
        problem_types = problem_types.split(",")
    if folds_to_keep is not None:
        folds_to_keep = [int(fold) for fold in folds_to_keep.split(",")]
    else:
        folds_to_keep = [i for i in range(10)]
    if banned_datasets is not None:
        banned_datasets = banned_datasets.split(",")

    frameworks_compare_vs_all = []
    if len(frameworks_compare_vs_all) == 0:
        frameworks_compare_vs_all = [frameworks[0]]

    benchmark_evaluator = BenchmarkEvaluator(
        results_input_dir=results_input_dir,
        results_output_dir=results_output_dir,
        output_suffix=output_suffix,
        use_tid_as_dataset_name=use_tid_as_dataset_name,
        framework_nan_fill=framework_nan_fill,
        filter_errors=filter_errors,
    )

    results_raw = benchmark_evaluator.load_data(
        paths=paths,
        frameworks=frameworks,
        folds=folds_to_keep,
        problem_type=problem_types,
        banned_datasets=banned_datasets,
        infer_batch_size=infer_batch_size,
        treat_folds_as_datasets=treat_folds_as_datasets,
    )

    folds_to_keep = sorted(results_raw["fold"].unique())

    if len(frameworks) > 1:
        compute_win_rate_per_dataset(f1=frameworks[0], f2=frameworks[1], results_raw=results_raw, folds=folds_to_keep)
    if compute_z_score and len(frameworks) > 1:
        z_stat_df = compute_stderr_z_stat_bulk(
            framework=frameworks[0], frameworks_to_compare=frameworks[1:], results_raw=results_raw
        )
        z_stat_series = compute_stderr_z_stat(
            results_raw, f1=frameworks[0], f2=frameworks[1], folds=folds_to_keep, verbose=False
        )
        graph_vs(results_df=results_raw, f1=frameworks[0], f2=frameworks[1], z_stats=z_stat_series)

    evaluate_results.evaluate(
        results_raw=results_raw,
        frameworks=frameworks,
        columns_to_agg_extra=[
            TIME_INFER_S,
        ],
        frameworks_compare_vs_all=frameworks_compare_vs_all,
        output_dir=benchmark_evaluator.results_dir_output,
    )
