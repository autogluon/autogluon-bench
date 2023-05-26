import typer

from autogluon.bench.cloud.aws.stack_handler import destroy_stack
from autogluon.bench.runbenchmarks import run, get_job_status

app = typer.Typer()

app.command()(run)
app.command()(destroy_stack)
app.command()(get_job_status)

if __name__ == "__main__":
    app()
