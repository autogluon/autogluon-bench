import logging
import os

import pandas as pd
import typer

from autogluon.bench.eval.evaluation.constants import FRAMEWORK
from autogluon.bench.eval.evaluation.preprocess import preprocess_openml
from autogluon.common.savers import save_pd

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def clean_amlb_results(
    benchmark_name: str = typer.Option(
        ..., help="Benchmark name populated by benchmark run, in format <benchmark_name>_<timestamp>"
    ),
    results_dir: str = typer.Option("data/results/", help="Root directory of raw and prepared results."),
    results_input_dir: str = typer.Option(
        None,
        help="Directory of the results file '<file_prefix><constraint_str><benchmark_name_str>.csv' getting cleaned. Can be an S3 URI. If not provided, it defaults to '<results_dir>input/raw/'",
    ),
    results_output_dir: str = typer.Option(
        None,
        help="Output directory of cleaned file. Can be an S3 URI. If not provided, it defaults to '<results_dir>input/prepared/openml/'",
    ),
    file_prefix: str = typer.Option("results_automlbenchmark", help="File prefix of the input results files."),
    benchmark_name_in_input_path: bool = False,
    constraints: str = typer.Option(
        None,
        help="List of AMLB constraints, refer to https://github.com/openml/automlbenchmark/blob/master/resources/constraints.yaml",
    ),
    out_path_prefix: str = typer.Option("openml_ag_", help="Prefix of result file."),
    out_path_suffix: str = typer.Option("", help="suffix of result file."),
):
    """
    Cleans and aggregate results further with unified column names and adds benchmark name into framework column.

    Example:
        agbench clean-and-save-results --benchmark-name ag_tabular_20230629T140546 --results-input-dir s3://autogluon-benchmark-metrics/aggregated/tabular/ag_tabular_20230629T140546/ --benchmark-name-in-input-path
    """

    if results_input_dir is None:
        results_input_dir = os.path.join(results_dir, "input/raw/")
    if results_output_dir is None:
        results_output_dir = os.path.join(results_dir, "input/prepared/openml/")

    benchmark_name_str = f"_{benchmark_name}" if benchmark_name_in_input_path else ""
    if constraints is None:
        constraints = [None]
    else:
        constraints = constraints.split(",")

    results_list = []
    for constraint in constraints:
        constraint_str = f"_{constraint}"
        results = preprocess_openml.preprocess_openml_input(
            path=os.path.join(results_input_dir, f"{file_prefix}{constraint_str}{benchmark_name_str}.csv"),
            framework_suffix=constraint_str,
            framework_suffix_column="constraint",
        )
        results_list.append(results)

    results_raw = pd.concat(results_list, ignore_index=True, sort=True)

    if "framework_parent" in results_raw.columns:
        results_raw[FRAMEWORK] = results_raw["framework_parent"] + "_" + benchmark_name + "_" + results_raw[FRAMEWORK]
    else:
        results_raw[FRAMEWORK] = results_raw[FRAMEWORK] + "_" + benchmark_name

    save_path = results_output_dir + f"{out_path_prefix}{benchmark_name}{out_path_suffix}.csv"
    save_pd.save(path=save_path, df=results_raw)
    logger.info(f"Cleaned results are saved in file: {save_path}")
