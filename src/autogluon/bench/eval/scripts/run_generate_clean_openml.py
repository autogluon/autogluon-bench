import argparse

import pandas as pd

from autogluon.bench.eval.evaluation.constants import FRAMEWORK
from autogluon.bench.eval.evaluation.preprocess import preprocess_openml
from autogluon.common.savers import save_pd


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
        results_dir_input = results_dir + "input/raw/"
    if results_dir_output is None:
        results_dir_output = results_dir + "input/prepared/openml/"
    run_name_str = f"_{run_name}" if run_name_in_input_path else ""

    results_list = []
    if constraints is None:
        constraints = [None]
    for constraint in constraints:
        constraint_str = f"_{constraint}" if constraint is not None else ""
        results = preprocess_openml.preprocess_openml_input(
            path=results_dir_input + f"{file_prefix}{constraint_str}{run_name_str}.csv",
            framework_suffix=constraint_str,
            framework_suffix_column=framework_suffix_column,
        )
        results_list.append(results)

    results_raw = pd.concat(results_list, ignore_index=True, sort=True)

    if "framework_parent" in results_raw.columns:
        results_raw[FRAMEWORK] = results_raw["framework_parent"] + "_" + run_name + "_" + results_raw[FRAMEWORK]
    else:
        results_raw[FRAMEWORK] = results_raw[FRAMEWORK] + "_" + run_name

    save_path = results_dir_output + f"{out_path_prefix}{run_name}{out_path_suffix}.csv"
    save_pd.save(path=save_path, df=results_raw)
    print(f"Saved file: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--run_name", type=str, help="Name of run", nargs="?")
    parser.add_argument("--file_prefix", type=str, help="Prefix of filename", nargs="?")
    parser.add_argument("--results_input_dir", type=str, help="Results input directory", nargs="?")
    parser.add_argument("--constraints", type=list, help="Time constraints", default=None, nargs="?")
    parser.add_argument("--run_name_in_input_path", type=str, help="Run name in input path", default=False, nargs="?")
    parser.add_argument("--out_path_suffix", type=str, help="Suffix added to output file name", default="", nargs="?")

    args = parser.parse_args()

    clean_and_save_results(
        args.run_name,
        file_prefix=args.file_prefix,
        results_dir_input=args.results_input_dir,
        constraints=args.constraints,
        run_name_in_input_path=args.run_name_in_input_path,
    )
