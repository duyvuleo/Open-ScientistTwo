"""Native benchmark CLI. No command enables a live agent or automatic dataset download."""

from collections import Counter
import json
from pathlib import Path
import typer

from ..store import Store, canonical
from .manifest import create_spec, preflight, preparation_plan
from .models import NativeSpec
from .registry import list_tasks
from .runner import NativeRunner
from ..testbed.cli import attach_commands

app = typer.Typer(
    no_args_is_help=True,
    help="Native benchmark execution and official graders; live agents disabled.",
)
attach_commands(app)


def emit(value):
    typer.echo(json.dumps(value, indent=2, default=str, allow_nan=False))


@app.command("list")
def list_command(benchmark: str, repo: Path, split: str | None = None):
    tasks = list_tasks(benchmark, repo, split)
    emit(
        {
            "benchmark": benchmark,
            "count": len(tasks),
            "tasks": [t.model_dump(exclude={"description"}) for t in tasks],
        }
    )


@app.command("prepare-plan")
def prepare_command(benchmark: str, task_id: str, repo: Path, data_root: Path | None = None):
    """Show upstream preparation instructions without accepting terms or downloading data."""
    if benchmark == "mlagentbench":
        from ..testbed.preparation import prepare_plan
        emit(prepare_plan(task_id, source=data_root or repo))
    else:
        emit(preparation_plan(benchmark, task_id, repo, data_root))


@app.command("fetch-leaderboard")
def fetch_leaderboard_command(repo: Path, task_id: str):
    """Explicitly fetch one small public LFS leaderboard and verify its pinned hash."""
    from .lfs import materialize_leaderboard
    from .registry import PINS, verify_repository

    verify_repository(repo, PINS["mle-bench"]["revision"])
    emit(materialize_leaderboard(repo.resolve(), PINS["mle-bench"]["revision"], task_id))


@app.command("create")
def create_command(
    benchmark: str,
    task_id: str,
    output: Path,
    repo: Path = typer.Option(...),
    data_root: Path | None = None,
    candidate: Path | None = None,
    command_json: str | None = None,
    mode: str = "baseline",
    provenance: str | None = None,
    image: str = "scientisttwo-native:0.1",
    grader_image: str | None = None,
    context: str | None = None,
    seconds: float = 1800,
    memory_mb: int = 2048,
    cpus: float = 1,
    gpus: str | None = None,
    submission: str = "submission.csv",
):
    command = json.loads(command_json) if command_json else None
    spec = create_spec(
        benchmark,
        task_id,
        repo,
        data_root=data_root,
        candidate=candidate,
        command=command,
        mode=mode,
        provenance=provenance,
        image=image,
        grader_image=grader_image,
        context=context,
        seconds=seconds,
        memory_mb=memory_mb,
        cpus=cpus,
        gpus=gpus,
        submission=submission,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(spec.model_dump_json(indent=2) + "\n")
    emit({"manifest": str(output.resolve()), "preflight": preflight(spec)})


@app.command("check")
def check_command(manifest: Path):
    checks = preflight(NativeSpec.model_validate_json(manifest.read_text()))
    emit(checks)
    if not checks["ready"]:
        raise typer.Exit(2)


def execute(manifest, output):
    from ..cli import locked

    spec = NativeSpec.model_validate_json(Path(manifest).read_text())
    output = Path(output)
    # Validate overlap before even creating the lock file in a caller-selected output tree.
    for path in (spec.repository, spec.public_dir, spec.private_dir, spec.candidate_dir):
        if path and (
            output.resolve().is_relative_to(path) or path.is_relative_to(output.resolve())
        ):
            raise ValueError("Run output must not overlap input trees")
    with locked(output):
        runner = NativeRunner(spec, output)
        try:
            return runner.run()
        finally:
            runner.close()


@app.command("run")
def run_command(manifest: Path, output: Path):
    result = execute(manifest, output)
    emit(result.model_dump())
    if result.status != "completed":
        raise typer.Exit(2)


@app.command("inspect")
def inspect_command(output: Path):
    if not (output / "campaign.sqlite").exists():
        raise ValueError("Native benchmark run not found")
    store = Store(output)
    try:
        emit(
            {
                "result": store.json(store.get("native_result_ref"))
                if store.get("native_result_ref")
                else None,
                "artifact_errors": store.verify(),
                "images": store.get("native_images"),
            }
        )
    finally:
        store.close()


@app.command("suite-run")
def suite_command(manifest_list: Path, output: Path):
    """Execute an explicit JSON list of manifests, preserving every assigned run in totals."""
    from ..cli import locked

    manifests = json.loads(manifest_list.read_text())
    if (
        not isinstance(manifests, list)
        or not manifests
        or any(not isinstance(p, str) for p in manifests)
    ):
        raise ValueError("Expected a nonempty JSON list of manifest paths")
    resolved = [(manifest_list.parent / Path(p)).resolve() for p in manifests]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate manifest assignment")
    assignment = [NativeSpec.model_validate_json(p.read_text()) for p in resolved]
    if len({s.name for s in assignment}) != len(assignment):
        raise ValueError("Native run names must be unique within a suite")
    with locked(output):
        frozen = output / "assignments.json"
        payload = [s.model_dump(mode="json") for s in assignment]
        if frozen.exists() and frozen.read_bytes() != canonical(payload):
            raise ValueError("Suite directory contains different frozen assignments")
        if not frozen.exists():
            frozen.write_bytes(canonical(payload))
        results = []
        for path, spec in zip(resolved, assignment):
            try:
                result = execute(path, output / spec.name).model_dump(mode="json")
            except Exception as error:
                result = {
                    "benchmark": spec.benchmark,
                    "task_id": spec.task_id,
                    "status": "integration_error",
                    "reason": str(error),
                }
            results.append(result)
            summary = {
                "assigned": len(assignment),
                "finished": len(results),
                "statuses": dict(Counter(r["status"] for r in results)),
                "agent_evaluation": False,
                "live_model_calls": 0,
                "results": results,
            }
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            emit(
                {
                    "finished": len(results),
                    "assigned": len(assignment),
                    "task": spec.task_id,
                    "status": result["status"],
                }
            )
        if any(r["status"] != "completed" for r in results):
            raise typer.Exit(2)
