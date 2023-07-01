import typer

from autogluon.bench.cloud.aws.stack_handler import destroy_stack
from autogluon.bench.eval.scripts.aggregate_amlb_results import aggregate_amlb_results
from autogluon.bench.eval.scripts.run_evaluation_openml import evaluate_amlb_results
from autogluon.bench.eval.scripts.run_generate_clean_openml import clean_amlb_results
from autogluon.bench.runbenchmark import get_job_status, run
from autogluon.bench.scripts.generate_cloud_configs import generate_cloud_config

app = typer.Typer()

app.command()(run)
app.command()(destroy_stack)
app.command()(get_job_status)
app.command()(generate_cloud_config)
app.command()(aggregate_amlb_results)
app.command()(clean_amlb_results)
app.command()(evaluate_amlb_results)

if __name__ == "__main__":
    app()
