import argparse

from autogluon.bench.eval.evaluation import evaluate_results
from autogluon.bench.eval.evaluation.constants import TIME_INFER_S
from autogluon.bench.eval.evaluation.evaluate_utils import compute_stderr_z_stat, compute_stderr_z_stat_bulk, compute_win_rate_per_dataset, graph_vs
from autogluon.bench.eval.evaluation.benchmark_evaluator import BenchmarkEvaluator


def run(
    *,
    frameworks_run,
    paths,
    output_suffix='ag_full_v5/1h8c',
    framework_nan_fill=None,
    problem_type=None,
    folds_to_keep: list = None,
    compute_z_score=True,
    treat_folds_as_datasets=False,
    banned_datasets=None,
    infer_batch_size=None,
    use_tid_as_dataset_name=True,
    filter_errors=False,  # If True, all dataset errors will be filtered out
):
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
