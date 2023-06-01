import typer

from autogluon.bench.cloud.aws.stack_handler import destroy_stack
from autogluon.bench.runbenchmark import get_job_status, run

app = typer.Typer()

app.command()(run)
app.command()(destroy_stack)
app.command()(get_job_status)

if __name__ == "__main__":
    app()
