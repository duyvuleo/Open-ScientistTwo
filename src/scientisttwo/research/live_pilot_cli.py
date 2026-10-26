"""Explicit live-pilot manifests and execution; ordinary pilots remain offline."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field
import typer

from ..backends import strict_json_loads
from ..live.adapters import implementation_identity
from ..models import Profile, Record, TaskSpec
from ..store import IntegrityError, Store, canonical, digest
from .live_pilot import LivePilotOptions, run_live_pilot, resume_live_pilot
from ..process import run_process


app = typer.Typer(no_args_is_help=True, help="Explicit bounded live agents on bundled CPU tasks.")


class LivePilotManifest(Record):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    schema_version: Literal["scientisttwo-live-pilot-v1"] = "scientisttwo-live-pilot-v1"
    task: TaskSpec
    profile: Profile
    options: LivePilotOptions
    runtime: Literal["docker", "podman"] = "docker"
    image: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    context: str | None = None
    implementation: dict[str, str]


def emit(value):
    typer.echo(json.dumps(value, indent=2, allow_nan=False, default=str))


@contextmanager
def locked_store(root):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Store(root)
        try:
            yield store
        finally:
            store.close()


@app.command("create")
def create(task: Path, options: Path, output: Path, image: str = typer.Option(...),
           runtime: str = "docker", context: str | None = None,
           experiment_seconds: float = typer.Option(60, min=1, max=120)):
    """Freeze a live configuration without constructing a model backend."""
    if output.exists():
        raise ValueError("Manifest exists; do not overwrite a frozen launch")
    spec = TaskSpec.model_validate(strict_json_loads(task.read_text()))
    live = LivePilotOptions.model_validate(strict_json_loads(options.read_text()))
    profile = Profile(campaign_seconds=live.campaign_seconds, experiment_seconds=experiment_seconds,
        limitation_rounds=live.limitation_rounds, seed_ideas=live.workflow.seed_ideas,
        search_rounds=live.search_rounds, engineering_rounds=live.engineering_retries,
        ablation_rounds=live.ablation_refinements, peer_rounds=live.review_rounds,
        meta_rounds=live.meta_refinements, success_target=live.success_target)
    manifest = LivePilotManifest(task=spec, profile=profile, options=live,
        runtime=runtime, image=image, context=context, implementation=implementation_identity())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(manifest.model_dump_json(indent=2) + "\n")
    emit({"status": "prepared_no_calls", "manifest": str(output.resolve()),
          "manifest_sha256": digest(canonical(manifest)), "live_model_calls": 0,
          "note": "Profile is an execution-limit contract; only explicit live-pilot run authorizes agents."})


@app.command("run")
def run(manifest: Path, output: Path, allow_live: bool = False, executable: str = "codex"):
    """Run a fresh pinned live pilot only with --allow-live."""
    if not allow_live:
        raise ValueError("No calls made: live-pilot run requires --allow-live")
    spec = LivePilotManifest.model_validate(strict_json_loads(manifest.read_text()))
    if spec.implementation != implementation_identity():
        raise IntegrityError("Implementation changed after live manifest freeze")
    with locked_store(output) as store:
        if store.get("task_ref"):
            raise ValueError("Campaign already initialized; use live-pilot resume")
        store.set("live_pilot_launch_ref", store.put_json(spec, "live-pilot-launch"))
        state = run_live_pilot(store, spec.task, spec.profile, options=spec.options,
            runtime=spec.runtime, image=spec.image, context=spec.context,
            allow_live=True, executable=executable)
        emit({"state": state, "directory": str(store.root),
              "model_accounting_ref": store.get("api_model_accounting_ref")
              or store.get("live_pilot_model_accounting")})
        if state.get("status") != "complete":
            raise typer.Exit(2)


@app.command("resume")
def resume(campaign: Path, allow_live: bool = False, executable: str = "codex"):
    """Reuse the exact saved options, deadlines and completions."""
    if not allow_live:
        raise ValueError("No calls made: live-pilot resume requires --allow-live")
    if not (campaign / "campaign.sqlite").is_file():
        raise ValueError("Existing campaign required")
    with locked_store(campaign) as store:
        state = resume_live_pilot(store, allow_live=True, executable=executable)
        emit({"state": state, "directory": str(store.root),
              "model_accounting_ref": store.get("api_model_accounting_ref")
              or store.get("live_pilot_model_accounting")})
        if state.get("status") != "complete":
            raise typer.Exit(2)


@app.command("check")
def check(manifest: Path):
    """Check the frozen configuration, local credentials and images without inference."""
    from ..live.cli import check_provider
    spec = LivePilotManifest.model_validate(strict_json_loads(manifest.read_text()))
    checks = check_provider(spec.options.provider)
    checks["implementation_matches"] = spec.implementation == implementation_identity()
    checks["live_model_calls"] = 0
    prefix = [spec.runtime] + (["--context", spec.context] if spec.context else [])
    images = {}
    wanted = {spec.image}
    if spec.options.workflow.compiler_image:
        wanted.add(spec.options.workflow.compiler_image)
    for image in wanted:
        try:
            result = run_process(prefix + ["image", "inspect", image, "--format", "{{.Id}}"], 15)
            images[image] = result.code == 0 and result.stdout.strip() == image
        except OSError:
            images[image] = False
    checks["images"] = images
    checks["writer_ready"] = True
    if spec.options.workflow.writer_backend == "paperorchestra":
        from ..integrations.paperorchestra.runtime import verify_upstream
        try:
            verify_upstream(Path(spec.options.workflow.paperorchestra_root))
        except (OSError, ValueError):
            checks["writer_ready"] = False
    checks["ready"] = bool(checks["provider_ready"] and checks["implementation_matches"]
                           and checks["writer_ready"] and all(images.values()))
    emit(checks)
    if not checks["ready"]:
        raise typer.Exit(2)
