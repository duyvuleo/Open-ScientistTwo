"""Offline default CLI; live execution is a separately authorized command group."""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import uuid

import typer

from .audit import audit_campaign
from .backends import FixtureBackend
from .controller import Controller, initialize
from .models import Profile, TaskSpec
from .runner import ContainerRunner, PrerequisiteError
from .store import IntegrityError, Store
from .tasks import load_task_adapter
from .benchmarks.cli import app as benchmark_app
from .live.cli import app as live_app
from .research.cli import app as workflow_app
from .integrations.paperorchestra.cli import app as paperorchestra_app
from .studies.cli import app as study_app
from .research.live_pilot_cli import app as live_pilot_app

app = typer.Typer(
    no_args_is_help=True, help="ScientistTwo workflow; offline by default, explicit live opt-in."
)


app.add_typer(benchmark_app, name="benchmark")
app.add_typer(live_app, name="live")
app.add_typer(workflow_app, name="workflow")
app.add_typer(paperorchestra_app, name="paperorchestra")
app.add_typer(study_app, name="study")
app.add_typer(live_pilot_app, name="live-pilot")


def emit(value):
    typer.echo(json.dumps(value, indent=2, default=str, allow_nan=False))


@contextmanager
def locked(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".campaign.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process owns this campaign") from error
        yield


def resolve_run(run_id: str, runs_dir: Path):
    path = Path(run_id)
    root = path if path.is_absolute() else runs_dir / path
    if not root.is_dir() or not (root / "campaign.sqlite").exists():
        raise ValueError(f"Campaign not found: {root}")
    return root.resolve()


def execute_legacy(store, task, profile, runtime, image, context=None):
    adapter = load_task_adapter(task.adapter, task)
    runner = ContainerRunner(store, profile, runtime, image, adapter=adapter, context=context)
    backend = FixtureBackend(adapter.fixture_response)
    state = Controller(store, task, profile, adapter, backend, runner).run()
    emit(
        {
            **state.model_dump(mode="json"),
            "directory": str(store.root),
            "report": str(store.root / "report.md") if store.get("report_ref") else None,
        }
    )
    if state.status == "blocked":
        raise typer.Exit(2)


def execute(store, task, profile, runtime, image, context=None):
    """New pilots use exactly the same engine as benchmark and live campaigns."""
    if store.get("live_pilot_options_ref"):
        raise ValueError("Use explicit live-pilot resume --allow-live for a live campaign")
    from .research.controller import ResearchController
    from .research.pilot_bridge import create_pilot_campaign
    from .research.fixtures import ResearchFixtures
    config, adapter, runner = create_pilot_campaign(store, task, profile, runtime, image, context)
    backend = ResearchFixtures(adapter.fixture_response)
    state = ResearchController(store, config, adapter, backend, runner, budget=runner.budget).run()
    emit({**state, "engine_version": config.engine_version, "directory": str(store.root),
          "report": str(store.root / "report.md")})
    if state["status"] != "complete" or state["outcome"] not in {"development_improvement", "no_improvement"}:
        raise typer.Exit(2)


@app.command()
def doctor(
    runtime: str = "docker", image: str = "scientisttwo-worker:0.1", context: str | None = None
):
    """Inspect prerequisites without model calls or starting services."""
    checks = {
        "python": platform.python_version(),
        "python_supported": sys.version_info[:2] == (3, 12),
        "architecture": platform.machine(),
        "runtime": runtime,
        "codex_installed": shutil.which("codex") is not None,
        "live_calls": False,
    }
    with tempfile.TemporaryDirectory(prefix="scientisttwo-doctor-") as temporary:
        store = Store(Path(temporary))
        try:
            runner = ContainerRunner(
                store,
                Profile(),
                runtime,
                image,
                adapter=load_task_adapter("classification"),
                context=context,
            )
            runner.preflight()
            checks.update(container="ready", image_id=runner.image_id)
        except (PrerequisiteError, ValueError) as error:
            checks.update(container="unavailable", reason=str(error))
        finally:
            store.close()
    emit(checks)
    if checks["container"] != "ready" or not checks["python_supported"]:
        raise typer.Exit(2)


@app.command()
def run(
    task: str = typer.Argument("pilot"),
    profile: str = "offline",
    runtime: str = "docker",
    image: str = "scientisttwo-worker:0.1",
    runs_dir: Path = Path("runs"),
    context: str | None = None,
    compiler_image: str | None = None,
    require_pdf: bool = False,
    writer_backend: str = typer.Option("canonical", help="Offline writer: canonical fixture or paperorchestra with scripted agent responses."),
    paperorchestra_root: Path | None = None,
    max_prompt_chars: int = typer.Option(200_000, min=1000,
        help="Frozen input-character ceiling for offline agent roles; this is not a model token limit."),
):
    """Start a pilot or a JSON TaskSpec using a registered task adapter."""
    spec = TaskSpec() if task == "pilot" else TaskSpec.model_validate_json(Path(task).read_text())
    settings = Profile.named(profile)
    root = runs_dir / (spec.name + "-" + uuid.uuid4().hex[:12])
    with locked(root):
        store = Store(root)
        try:
            initialize(store, spec, settings, load_task_adapter(spec.adapter, spec))
            store.set("runtime", {"runtime": runtime, "image": image, "context": context})
            store.set("pilot_workflow_options", {"compiler_image": compiler_image, "require_pdf": require_pdf,
                "writer_backend": writer_backend,
                "paperorchestra_root": str(paperorchestra_root.resolve()) if paperorchestra_root else None})
            store.set("pilot_controller_options", {"max_prompt_chars": max_prompt_chars})
            execute(store, spec, settings, runtime, image, context)
        finally:
            store.close()


@app.command()
def resume(run_id: str, runs_dir: Path = Path("runs")):
    """Resume using the original task, profile, runtime, and checkpoints."""
    root = resolve_run(run_id, runs_dir)
    with locked(root):
        store = Store(root)
        try:
            if store.get("live_pilot_options_ref"):
                raise ValueError("Use explicit live-pilot resume --allow-live for a live campaign")
            spec = TaskSpec.model_validate(store.json(store.get("task_ref")))
            profile = Profile.model_validate(store.json(store.get("profile_ref")))
            engine = execute if store.get("engine_version") == "research-v2" else execute_legacy
            engine(store, spec, profile, **store.get("runtime"))
        finally:
            store.close()


@app.command()
def inspect(run_id: str, runs_dir: Path = Path("runs")):
    """Display state, decisions, and execution resource accounting."""
    root = resolve_run(run_id, runs_dir)
    with locked(root):
        store = Store(root)
        try:
            backend = store.get("live_backend", {"synthetic": True})
            operations = ([dict(row) for row in store.db.execute("SELECT * FROM live_operations ORDER BY rowid")]
                if store.db.execute("SELECT 1 FROM sqlite_master WHERE name='live_operations'").fetchone() else [])
            emit(
                {
                    "state": store.get("live_state") if store.get("engine_version") == "research-v2" else store.get("state"),
                    "engine_version": store.get("engine_version", "legacy-pilot-v1"),
                    "live_calls": backend.get("synthetic") is False,
                    "backend": backend,
                    "live_model_calls": sum(row["kind"] == "model" for row in operations) if backend.get("synthetic") is False else 0,
                    "events": [
                        dict(row) for row in store.db.execute("SELECT * FROM events ORDER BY time")
                    ],
                    "jobs": [
                        dict(row)
                        for row in store.db.execute(
                            "SELECT key,status,reserved,charged FROM jobs ORDER BY started"
                        )
                    ],
                }
            )
        finally:
            store.close()


@app.command()
def audit(
    run_id: str, runs_dir: Path = Path("runs"), reexecute: bool = True, include_final: bool = False
):
    """Check artifacts and rerun the selected DEVELOPMENT experiment in its container."""
    root = resolve_run(run_id, runs_dir)
    with locked(root):
        store = Store(root)
        try:
            if store.get("engine_version") == "research-v2":
                from .research.audit import audit_research
                result = audit_research(root, reexecute=reexecute, include_final=include_final)
                emit(result)
                if result["status"] != "pass":
                    raise typer.Exit(2)
                return
            task = TaskSpec.model_validate(store.json(store.get("task_ref")))
            profile = Profile.model_validate(store.json(store.get("profile_ref")))
            adapter = load_task_adapter(task.adapter, task)
            runner = ContainerRunner(store, profile, **store.get("runtime"), adapter=adapter)
            result = audit_campaign(store, adapter, runner, reexecute, include_final=include_final)
            emit(result)
            if any(value == "fail" for value in result.values() if isinstance(value, str)):
                raise typer.Exit(1)
        finally:
            store.close()


@app.command("batch-create")
def batch_create(
    path: Path,
    suite: str = "cross-domain",
    workers: int = 2,
    context: str | None = None,
    image: str = "scientisttwo-worker:0.1",
):
    """Write a frozen 90-campaign manifest, or a separate three-task smoke manifest."""
    from .suite import make_suite, write_manifest

    if suite not in {"cross-domain", "smoke"}:
        raise ValueError("suite must be cross-domain or smoke")
    manifest = make_suite(smoke=suite == "smoke", workers=workers, context=context, image=image)
    write_manifest(path, manifest)
    emit(
        {
            "manifest": str(path.resolve()),
            "campaigns": len(manifest.campaigns),
            "tasks": len({c.task_id for c in manifest.campaigns}),
            "live_calls": False,
        }
    )


@app.command("batch-run")
def batch_run(
    manifest: Path, output: Path, max_campaigns: int | None = None, retry_errors: bool = False
):
    """Execute/resume isolated campaign processes and audit each result."""
    from .batch import run_batch

    result = run_batch(manifest, output, max_campaigns=max_campaigns, retry_errors=retry_errors)
    emit(
        {
            key: result[key]
            for key in (
                "assigned_campaigns",
                "unique_tasks",
                "completed_campaigns",
                "status_counts",
                "families",
                "experiment_jobs",
                "audit_jobs",
                "live_model_calls",
            )
        }
    )
    if any(
        row["status"] in {"error", "blocked"}
        or (row["status"] == "complete" and not row.get("audit_pass"))
        for row in result["results"]
    ):
        raise typer.Exit(1)
    if max_campaigns is None and result["completed_campaigns"] != result["assigned_campaigns"]:
        raise typer.Exit(1)


@app.command("batch-inspect")
def batch_inspect(output: Path):
    """Refresh the complete batch denominator and inspect current receipts."""
    from .batch import read_manifest
    from .batch_report import summarize

    with locked(output):
        result = summarize(output, read_manifest(output / "manifest.json"))
    emit(
        {
            key: result[key]
            for key in (
                "assigned_campaigns",
                "unique_tasks",
                "completed_campaigns",
                "status_counts",
                "families",
            )
        }
    )


def main():
    try:
        app()
    except (ValueError, IntegrityError, PrerequisiteError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
