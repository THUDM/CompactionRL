"""Harbor commands retained for CompactionRL datasets, trials and evaluation."""

from typer import Typer
from harbor.cli.datasets import datasets_app
from harbor.cli.jobs import jobs_app, start
from harbor.cli.tasks import tasks_app
from harbor.cli.trials import trials_app

app = Typer(no_args_is_help=True)
for name, child in [
    ("datasets", datasets_app),
    ("jobs", jobs_app),
    ("tasks", tasks_app),
    ("trials", trials_app),
]:
    app.add_typer(child, name=name)
app.command(name="run")(start)
if __name__ == "__main__":
    app()
