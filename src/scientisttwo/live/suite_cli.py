"""Model-free suite preparation; execution requires an explicit live opt-in."""

import json
from pathlib import Path

import typer

from .suite import SuiteSpec, check_suite, inspect_suite, make_suite, run_suite

app = typer.Typer(no_args_is_help=True, help="Frozen, sequential, aggregate-budgeted live suites.")


def emit(value):
    typer.echo(json.dumps(value, indent=2, default=str, allow_nan=False))


@app.command("create")
def create(
    output: Path,
    campaign: list[Path] = typer.Option(
        ..., help="Prepared LiveSpec manifest; repeat per task/model."
    ),
    max_calls: int = typer.Option(...),
    max_experiments: int = typer.Option(...),
    seconds: float = typer.Option(...),
    name: str = "live-suite",
    repetitions: int = typer.Option(1, min=1),
    token_stop_threshold: int | None = None,
    schedule_seed: int = 1729,
    order_policy: str = "interleaved",
):
    config = make_suite(
        campaign,
        name=name,
        repetitions=repetitions,
        wall_seconds=seconds,
        max_calls=max_calls,
        max_experiments=max_experiments,
        token_stop_threshold=token_stop_threshold,
        schedule_seed=schedule_seed, order_policy=order_policy,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(config.model_dump_json(indent=2) + "\n")
    emit(
        {
            "manifest": str(output.resolve()),
            "assignments": len(config.assignments),
            "live_model_calls": 0,
        }
    )


@app.command("check")
def check(manifest: Path):
    result = check_suite(SuiteSpec.model_validate_json(manifest.read_text()))
    emit(result)
    if not result["ready"]:
        raise typer.Exit(2)


@app.command("run")
def run(manifest: Path, output: Path, allow_live: bool = False, allow_literature_network: bool = False):
    result = run_suite(
        SuiteSpec.model_validate_json(manifest.read_text()), output, allow_live=allow_live,
        allow_literature_network=allow_literature_network
    )
    emit(result)
    if result["status"] != "complete":
        raise typer.Exit(2)


@app.command("resume")
def resume(output: Path, allow_live: bool = False, allow_literature_network: bool = False):
    if not allow_live:
        raise ValueError("Suite resume requires --allow-live; original budgets remain in force")
    config, _ = inspect_suite(output)
    result = run_suite(config, output, allow_live=allow_live, allow_literature_network=allow_literature_network)
    emit(result)
    if result["status"] != "complete":
        raise typer.Exit(2)


@app.command("inspect")
def inspect(output: Path):
    _, result = inspect_suite(output)
    emit(result)
