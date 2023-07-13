import logging
import os
from typing import List, Optional

import pandas as pd
import typer
from typing_extensions import Annotated

from autogluon.bench.eval.evaluation.constants import FRAMEWORK
from autogluon.bench.eval.evaluation.preprocess import preprocess_openml
from autogluon.common.savers import save_pd

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def clean_amlb_results(
    benchmark_name: str = typer.Argument(
        help="Benchmark name populated by benchmark run, in format <benchmark_name>_<timestamp>"
    ),
    results_dir: str = typer.Option("data/results/", help="Root directory of raw and prepared results."),
    results_dir_input: str = typer.Option(
        None,
        help="Directory of the results file '<file_prefix><constraint_str><benchmark_name_str>.csv' getting cleaned. Can be an S3 URI. If not provided, it defaults to '<results_dir>input/raw/'",
    ),
    results_dir_output: str = typer.Option(
        None,
        help="Output directory of cleaned file. Can be an S3 URI. If not provided, it defaults to '<results_dir>input/prepared/openml/'",
    ),
    file_prefix: str = typer.Option("results_automlbenchmark", help="File prefix of the input results files."),
    benchmark_name_in_input_path: bool = False,
    constraints: Optional[List[str]] = typer.Option(
        None,
        help="List of AMLB constraints, refer to https://github.com/openml/automlbenchmark/blob/master/resources/constraints.yaml",
    ),
    out_path_prefix: str = typer.Option("openml_ag_", help="Prefix of result file."),
    out_path_suffix: str = typer.Option("", help="suffix of result file."),
    framework_suffix_column: str = typer.Option("constraint", help="Framework suffix column."),
):
    """
    Cleans and aggregate results further with unified column names and adds benchmark name into framework column.

    Example:
        agbench clean-and-save-results ag_tabular_20230629T140546 --results-dir-input s3://autogluon-benchmark-metrics/aggregated/tabular/ag_tabular_20230629T140546/ --benchmark-name-in-input-path --constraints constratint_1 --constraints constratint_2
    """
    clean_and_save_results(
        run_name=benchmark_name,
        results_dir=results_dir,
        results_dir_input=results_dir_input,
        results_dir_output=results_dir_output,
        file_prefix=file_prefix,
        run_name_in_input_path=benchmark_name_in_input_path,
        constraints=constraints if constraints else None,
        out_path_prefix=out_path_prefix,
        out_path_suffix=out_path_suffix,
        framework_suffix_column=framework_suffix_column,
    )


def clean_and_save_results(
    run_name,
    results_dir="data/results/",
    results_dir_input=None,
    results_dir_output=None,
    file_prefix="results_automlbenchmark",
    run_name_in_input_path=True,
    constraints=None,
    out_path_prefix="openml_ag_",
    out_path_suffix="",
    framework_suffix_column="constraint",
):
    if results_dir_input is None:
        results_dir_input = os.path.join(results_dir, "input/raw/")
    if results_dir_output is None:
        results_dir_output = os.path.join(results_dir, "input/prepared/openml/")
    run_name_str = f"_{run_name}" if run_name_in_input_path else ""

    results_list = []
    if constraints is None:
        constraints = [None]
    for constraint in constraints:
        constraint_str = f"_{constraint}" if constraint is not None else ""
        results = preprocess_openml.preprocess_openml_input(
            path=os.path.join(results_dir_input, f"{file_prefix}{constraint_str}{run_name_str}.csv"),
            framework_suffix=constraint_str,
            framework_suffix_column=framework_suffix_column,
        )
        results_list.append(results)

    results_raw = pd.concat(results_list, ignore_index=True, sort=True)

    if "framework_parent" in results_raw.columns:
        results_raw[FRAMEWORK] = results_raw["framework_parent"] + "_" + run_name + "_" + results_raw[FRAMEWORK]
    else:
        results_raw[FRAMEWORK] = results_raw[FRAMEWORK] + "_" + run_name

    save_path = os.path.join(results_dir_output, f"{out_path_prefix}{run_name}{out_path_suffix}.csv")
    save_pd.save(path=save_path, df=results_raw)
    logger.info(f"Cleaned results are saved in file: {save_path}")


if __name__ == "__main__":
    app()
