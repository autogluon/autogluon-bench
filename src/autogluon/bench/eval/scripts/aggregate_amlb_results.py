import typer

from autogluon.bench.eval.aggregate.aggregate import aggregate

app = typer.Typer()


@app.command()
def aggregate_amlb_results(
    s3_bucket: str = typer.Argument(help="Name of the S3 bucket to which the aggregated results will be outputted."),
    module: str = typer.Argument(help="Can be one of ['tabular', 'timeseries', 'multimodal']."),
    benchmark_name: str = typer.Argument(
        help="Folder name of benchmark run in which all objects with path 'scores/results.csv' get aggregated."
    ),
    artifact: str = typer.Option(
        "results", help="What should be saved, one of ['results', 'learning_curves'], default='results'"
    ),
    constraint: str = typer.Option(
        None,
        help="Name of constraint used in benchmark, refer to https://github.com/openml/automlbenchmark/blob/master/resources/constraints.yaml. Not applicable when `module==multimodal`",
    ),
    include_infer_speed: bool = typer.Option(False, help="Include inference speed in aggregation."),
    mode: str = typer.Option("ray", help='Aggregation mode: "seq" or "ray".'),
):
    """
    Aggregates objects across an agbenchmark. Functionality depends on artifact specified:

    Params:
    -------
    s3_bucket: str
        Name of the relevant s3_bucket
    module: str
        The name of the relevant autogluon module: can be one of ['tabular', 'timeseries', 'multimodal']
    benchmark_name: str
        The name of the relevant benchmark that was run
    artifact: str
        The desired artifact to be aggregatedL can be one of ['results', 'learning_curves']
    constraint: str
        Name of constraint used in benchmark
    include_infer_speed: bool
        Include inference speed in aggregation.
    mode: str
        Can be one of ['seq', 'ray'].
        If seq, runs sequentially.
        If ray, utilizes parallelization.

    Artifact Outcomes: ['results', 'learning_curves']
        results:
            Finds "scores/results.csv" under s3://<s3_bucket>/<module>/<benchmark_name> recursively with the constraint if provided,
            and append all results into one file at s3://<s3_bucket>/aggregated/<module>/<benchmark_name>/results_automlbenchmark_<constraint>_<benchmark_name>.csv

            Example:
                agbench aggregate-amlb-results autogluon-benchmark-metrics tabular ag_tabular_20230629T140546 --constraint test

        learning_curves:
            Finds specified learning_curves.json files under s3://<s3_bucket>/<module>/<benchmark_name> recursively with the constraint if provided,
            and stores all artifacts in common directory at s3://<s3_bucket>/aggregated/<module>/<benchmark_name>/

            Example:
                agbench aggregate-amlb-results autogluon-benchmark-metrics tabular ag_bench_learning_curves_20240802T163522 --artifact learning_curves --constraint toy
    """

    aggregate(
        s3_bucket=s3_bucket,
        module=module,
        benchmark_name=benchmark_name,
        artifact=artifact,
        constraint=constraint,
        include_infer_speed=include_infer_speed,
        mode=mode,
    )


if __name__ == "__main__":
    app()
