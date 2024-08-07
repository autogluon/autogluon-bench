import logging

from autogluon.bench.eval.benchmark_context.output_suite_context import OutputSuiteContext
from autogluon.common.savers import save_pd

logger = logging.getLogger(__name__)


def aggregate(
        s3_bucket: str, 
        module: str, 
        benchmark_name: str, 
        artifact: str | None = "results", 
        constraint: str | None = None,
        include_infer_speed: bool = False ,
        mode: str = "ray"
    ) -> None:
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
    """
    result_path = f"{module}/{benchmark_name}"
    path_prefix = f"s3://{s3_bucket}/{result_path}/"
    contains = f".{constraint}." if constraint else None

    output_suite_context = OutputSuiteContext(
        path=path_prefix,
        contains=contains,
        include_infer_speed=include_infer_speed,
        mode=mode,
    )

    if artifact == "learning_curves":
        save_path = f"s3://{s3_bucket}/aggregated/{result_path}/{artifact}"
        artifact_path = output_suite_context.aggregate_learning_curves(save_path=save_path)
    else:
        aggregated_results_name = f"results_automlbenchmark_{constraint}_{benchmark_name}.csv"
        results_df = output_suite_context.aggregate_results()

        print(results_df)

        artifact_path = f"s3://{s3_bucket}/aggregated/{result_path}/{aggregated_results_name}"
        save_pd.save(path=artifact_path, df=results_df)

    logger.info(f"Aggregated output saved to {artifact_path}!")