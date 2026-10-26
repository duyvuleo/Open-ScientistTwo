"""Explicit opt-in CLI. Creating/checking a manifest never invokes a model."""

import json
from pathlib import Path
import shutil

import typer

from ..backends import CodexBackend
from ..process import run_process
from ..store import Store
from .adapters import implementation_identity, load_development
from .controller import LiveController
from .models import DevelopmentSpec, LiveSpec
from .runtime import Budget, DevelopmentRunner
from .workspace_task import task_spec, freeze_workspace, preflight
from .suite_cli import app as suite_app
from .workspace import validate_baseline_contract, live_readiness
from ..research.policy import profile_updates

app = typer.Typer(
    no_args_is_help=True, help="Explicitly budgeted Codex research on native benchmarks."
)
app.add_typer(suite_app, name="suite")


def emit(value):
    typer.echo(json.dumps(value, indent=2, default=str, allow_nan=False))


def default_development(native):
    if native.benchmark == "mlagentbench" and native.task_id == "vectorization":
        return DevelopmentSpec(epsilon=0.05, settings={"proposal_mode": "forward_body"})
    if (
        native.benchmark == "mle-bench"
        and native.task_id == "detecting-insults-in-social-commentary"
    ):
        return DevelopmentSpec(adapter="csv", metric="roc_auc")
    if native.benchmark == "mle-bench" and native.task_id == "spooky-author-identification":
        return DevelopmentSpec(
            adapter="csv",
            metric="log_loss",
            target="author",
            text_column="text",
            classes=["EAP", "HPL", "MWS"],
            prediction_columns=["EAP", "HPL", "MWS"],
        )
    raise ValueError(
        "Provide --development with a declared development adapter/protocol for this task"
    )


@app.command("create")
def create(
    native_manifest: Path,
    output: Path,
    model: str = typer.Option(...),
    max_calls: int = typer.Option(...),
    seconds: float = typer.Option(...),
    mode: str = "smoke",
    development: Path | None = None,
    max_experiments: int = 30,
    call_seconds: float = 120,
    token_stop_threshold: int | None = None,
    agent_output_retries: int = typer.Option(0, min=0, max=1, help="Bounded correction of completed outputs with invalid citation/seed contracts; uses the same budgets."),
    supplementary_seed_count: int = typer.Option(2, min=1, max=32, help="Maximum exact fresh seed count offered to each supplementary batch; requests are never silently shortened."),
    workflow_profile: str = "research-pilot",
    literature_network: bool = False,
    compiler_image: str | None = None,
    require_pdf: bool = False,
    template_directory: Path | None = None,
    writer_backend: str | None = typer.Option(None, help="Research defaults to paperorchestra; canonical is an explicit alternative."),
    paperorchestra_root: Path | None = typer.Option(None, help="Optional pinned-source override; the PaperOrchestra writer uses its bundled engine by default."),
    paperorchestra_swap_policy: str = typer.Option("deny", help="PDF containers: deny swap, or explicitly inherit a Podman worker's swap policy."),
):
    """Freeze live configuration without enabling execution or making model calls."""
    native = task_spec(json.loads(native_manifest.read_text()))
    config = LiveSpec(
        native=native,
        development=DevelopmentSpec.model_validate_json(development.read_text())
        if development
        else default_development(native),
        model=model,
        max_calls=max_calls,
        campaign_seconds=seconds,
        mode=mode,
        max_experiments=max_experiments,
        call_seconds=call_seconds,
        token_stop_threshold=token_stop_threshold,
        agent_output_retries=agent_output_retries,
        supplementary_seed_count=supplementary_seed_count,
        implementation=implementation_identity(),
    )
    config = config.model_copy(update=profile_updates(workflow_profile))
    config.workflow.network_literature = literature_network
    config.workflow.literature = {"providers": ["local", "arxiv", "crossref"], "max_requests": 64} if literature_network else {}
    config.workflow.compiler_image = compiler_image
    config.workflow.require_pdf = require_pdf
    config.workflow.template_directory = str(template_directory.resolve()) if template_directory else None
    config.workflow.writer_backend = writer_backend or ("paperorchestra" if mode == "research" else "canonical")
    config.workflow.paperorchestra_root = str(paperorchestra_root.resolve()) if paperorchestra_root else None
    config.workflow.paperorchestra_swap_policy = paperorchestra_swap_policy
    config.model_retries = 0
    config = LiveSpec.model_validate(config.model_dump(mode="json"))
    if config.workflow.writer_backend == "paperorchestra":
        from ..integrations.paperorchestra.runtime import verify_upstream
        verify_upstream(Path(config.workflow.paperorchestra_root))
    checks = preflight(native)
    if not checks["ready"]:
        raise ValueError(
            "Prepare native inputs before creating live manifest: " + "; ".join(checks["errors"])
        )
    adapter = load_development(native, config.development)
    validate_baseline_contract(adapter)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(config.model_dump_json(indent=2) + "\n")
    emit(
        {
            "manifest": str(output.resolve()),
            "mode": mode,
            "live_model_calls": 0,
            "execution_requires": "live run --allow-live",
        }
    )


@app.command("check")
def check(manifest: Path):
    """Check inputs, adapter, local images and CLI authentication without model calls."""
    config = LiveSpec.model_validate_json(manifest.read_text())
    checks = preflight(config.native)
    checks["implementation_matches"] = config.implementation == implementation_identity()
    adapter = load_development(config.native, config.development)
    validate_baseline_contract(adapter)
    checks["development_adapter"] = config.development.adapter
    checks["adapter_live_readiness"] = live_readiness(adapter)
    checks["codex_installed"] = shutil.which("codex") is not None
    checks["codex_authenticated"] = False
    if checks["codex_installed"]:
        # Do not print or persist the CLI's authentication status text.
        status = run_process(["codex", "login", "status"], 15)
        checks["codex_authenticated"] = status.code == 0 and not status.timed_out
    prefix = [config.native.runtime] + (
        ["--context", config.native.context] if config.native.context else []
    )
    images = {}
    wanted_images = {config.native.image, config.native.grader_image}
    if config.workflow.compiler_image:
        wanted_images.add(config.workflow.compiler_image)
    for image in wanted_images:
        try:
            result = run_process(prefix + ["image", "inspect", image, "--format", "{{.Id}}"], 15)
            images[image] = result.stdout.strip() if result.code == 0 else None
        except OSError:
            images[image] = None
    checks.update(images=images, model_access="unverified_without_inference", live_model_calls=0)
    checks["writer_ready"] = True
    if config.workflow.writer_backend == "paperorchestra":
        from ..integrations.paperorchestra.runtime import verify_upstream
        try:
            verify_upstream(Path(config.workflow.paperorchestra_root))
        except (OSError, ValueError):
            checks["writer_ready"] = False
    checks["ready"] = (
        checks["ready"]
        and checks["implementation_matches"]
        and checks["adapter_live_readiness"]["ready"]
        and checks["codex_authenticated"]
        and all(images.values())
        and checks["writer_ready"]
    )
    emit(checks)
    if not checks["ready"]:
        raise typer.Exit(2)


def execute(config, output, allow_live, allow_literature_network=False, *, executable="codex"):
    if not allow_live:
        raise ValueError(
            "Live model calls are disabled. Pass --allow-live to authorize this manifest's model and budgets."
        )
    output = output.resolve()
    for path in (
        config.native.repository,
        config.native.public_dir,
        config.native.private_dir,
        config.native.candidate_dir,
        getattr(config.native, "evaluator_dir", None),
    ):
        if path and (output.is_relative_to(path) or path.is_relative_to(output)):
            raise ValueError("Live output must not overlap benchmark inputs")
    from ..cli import locked

    with locked(output):
        store = Store(output)
        try:
            adapter = load_development(config.native, config.development)
            live_readiness(adapter, require=True)
            budget = Budget(store, config)
            runner = DevelopmentRunner(store, config, adapter, budget)
            if config.engine_version == "research-v2":
                from ..research.live_pilot import run_prepared_live
                state = run_prepared_live(store, config, adapter, runner, allow_live=True,
                    executable=executable, allow_network_literature=allow_literature_network)
            else:
                if executable != "codex":
                    raise ValueError("Selecting a Codex executable requires research-v2")
                backend = CodexBackend(
                    output / "agent", config.model, allow_live=True, timeout=config.call_seconds
                )
                state = LiveController(store, config, adapter, backend, runner, budget=budget,
                                       allow_network_literature=allow_literature_network).run()
            emit({"state": state, "report": str(output / "report.md")})
            if state["status"] != "complete" or state["outcome"] not in {
                "development_improvement",
                "no_improvement",
            }:
                raise typer.Exit(2)
        finally:
            store.close()


@app.command("run")
def run(manifest: Path, output: Path, allow_live: bool = False, allow_literature_network: bool = False,
        executable: str = "codex"):
    """Start/resume only with explicit live authorization and frozen budgets."""
    execute(LiveSpec.model_validate_json(manifest.read_text()), output, allow_live,
            allow_literature_network, executable=executable)


@app.command("workspace-freeze")
def workspace_freeze(definition: Path, output: Path):
    """Freeze a portable local task's code, data and protected final evaluator; no execution."""
    spec = freeze_workspace(json.loads(definition.read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(spec.model_dump_json(indent=2) + "\n")
    emit({"manifest": str(output.resolve()), "live_model_calls": 0, "task_id": spec.task_id})


@app.command("resume")
def resume(output: Path, allow_live: bool = False, allow_literature_network: bool = False,
           executable: str = "codex"):
    if not allow_live:
        raise ValueError("Resume requires --allow-live; original budget remains in force")
    if not (output / "campaign.sqlite").is_file():
        raise ValueError("Live campaign not found")
    store = Store(output)
    try:
        config = LiveSpec.model_validate(store.json(store.get("live_spec_ref")))
    finally:
        store.close()
    execute(config, output, allow_live, allow_literature_network, executable=executable)


@app.command("inspect")
def inspect(output: Path):
    if not (output / "campaign.sqlite").is_file():
        raise ValueError("Live campaign not found")
    store = Store(output)
    try:
        emit(
            {
                "state": store.get("live_state"),
                "best": store.get("live_best"),
                "summary": store.json(store.get("live_summary_ref"))
                if store.get("live_summary_ref")
                else None,
                "events": [dict(r) for r in store.db.execute("SELECT * FROM events ORDER BY time")],
                "artifact_errors": store.verify(),
            }
        )
    finally:
        store.close()


@app.command("audit")
def audit(output: Path, reexecute_evaluators: bool = False):
    """Verify stored evidence and recompute development metrics without model calls."""
    from ..research.audit import audit_research
    result = audit_research(output, reexecute=reexecute_evaluators)
    emit(result)
    if result["status"] != "pass":
        raise typer.Exit(2)
