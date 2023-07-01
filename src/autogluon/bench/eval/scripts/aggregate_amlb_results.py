import typer

from autogluon.bench.eval.aggregate.results import aggregate_results

app = typer.Typer()


@app.command()
def aggregate_amlb_results(
    s3_bucket: str = typer.Option(
        ..., help="Name of the S3 bucket to which the aggregated results will be outputted."
    ),
    module: str = typer.Option(..., help="Can be one of ['tabular', 'multimodal']."),
    benchmark_name: str = typer.Option(
        ..., help="Folder name of benchmark run in which all objects with path 'scores/results.csv' get aggregated."
    ),
    constraint: str = typer.Option(
        None,
        help="Name of constraint used in benchmark, refer to https://github.com/openml/automlbenchmark/blob/master/resources/constraints.yaml",
    ),
    include_infer_speed: bool = typer.Option(False, help="Include inference speed in aggregation."),
    mode: str = typer.Option("ray", help='Aggregation mode: "seq" or "ray".'),
):
    """
    Finds "scores/results.csv" under s3://<s3_bucket>/<module>/<benchmark_name> recursively with the constraint if provided,
    and append all results into one file at s3://<s3_bucket>/aggregated/<module>/<benchmark_name>/results_automlbenchmark_<constraint>_<benchmark_name>.csv

    Example:
        agbench aggregate-s3-results --s3-bucket autogluon-benchmark-metrics --module tabular --benchmark-name ag_tabular_20230629T140546
    """

    aggregate_results(
        s3_bucket=s3_bucket,
        s3_prefix=f"{module}/",
        version_name=benchmark_name,
        constraint=constraint,
        include_infer_speed=include_infer_speed,
        mode=mode,
    )
