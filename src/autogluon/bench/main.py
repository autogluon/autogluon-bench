import typer

from autogluon.bench.cloud.aws.stack_handler import destroy_stack
from autogluon.bench.runbenchmark import get_job_status, run
from autogluon.bench.scripts.generate_cloud_configs import generate_cloud_config

app = typer.Typer()

app.command()(run)
app.command()(destroy_stack)
app.command()(get_job_status)
app.command()(generate_cloud_config)

if __name__ == "__main__":
    app()
