import logging

from autogluon.bench.eval.benchmark_context.output_suite_context import OutputSuiteContext
from autogluon.common.savers import save_pd

logger = logging.getLogger(__name__)


def aggregate_results(s3_bucket, s3_prefix, version_name, constraint, include_infer_speed=False, mode="ray"):
    contains = f".{constraint}." if constraint else None
    result_path = f"{s3_prefix}{version_name}/"
    path_prefix = f"s3://{s3_bucket}/{result_path}"

    aggregated_results_name = f"results_automlbenchmark_{constraint}_{version_name}.csv"

    output_suite_context = OutputSuiteContext(
        path=path_prefix,
        contains=contains,
        include_infer_speed=include_infer_speed,
        mode=mode,
    )
    results_df = output_suite_context.aggregate_results()

    print(results_df)

    save_path = f"s3://{s3_bucket}/aggregated/{result_path}{aggregated_results_name}"
    save_pd.save(path=save_path, df=results_df)
    logger.info(f"Aggregated results saved to {save_path}!")
