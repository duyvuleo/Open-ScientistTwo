"""Preparation and evidence inspection for the separately named CIFAR study."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

import typer

from ..backends import strict_json_loads
from ..benchmarks.models import NativeResources
from ..live.adapters import implementation_identity
from ..live.models import LiveSpec
from ..store import Store
from .calibration import run_calibration, verify_calibration
from .cifar_adapter import CifarRecipeDevelopment, prepare_study, study_specs
from .cifar_recipe import calibration_plan

app = typer.Typer(no_args_is_help=True, help="Source-bound CIFAR study preparation; no model calls in these commands.")


def emit(value):
    typer.echo(json.dumps(value, indent=2, default=str, allow_nan=False))


@contextmanager
def locked_store(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Store(root)
        try:
            yield store
        finally:
            store.close()


@app.command("prepare")
def prepare(source: Path, output: Path):
    """Copy and verify existing official prepared CIFAR data into a new variant."""
    manifest = prepare_study(source, output)
    emit({"status": "prepared", "path": output.resolve(), "manifest": manifest,
          "calibration_plan": calibration_plan(), "runtime_validation": "unverified",
          "live_model_calls": 0})


@app.command("calibration-manifest")
def calibration_manifest(
    prepared: Path, output: Path,
    image: str = typer.Option(...), grader_image: str = typer.Option(...),
    gpu: str = typer.Option(..., help="Explicit allocated device selector, preferably device=GPU-UUID."),
    resources: Path = typer.Option(..., help="Frozen NativeResources JSON; GPU must match --gpu."),
    runtime: str = "docker", context: str | None = None,
    campaign_seconds: float = typer.Option(1800, min=1, max=7200),
):
    """Freeze a zero-model, development-only calibration configuration."""
    limits = NativeResources.model_validate(strict_json_loads(resources.read_text()))
    if limits.gpus != gpu:
        raise ValueError("Explicit GPU selection differs from the resource manifest")
    native, development = study_specs(prepared, image=image, grader_image=grader_image,
        resources=limits, runtime=runtime, context=context)
    config = LiveSpec(native=native, development=development, model="calibration-no-model",
        max_calls=1, campaign_seconds=campaign_seconds, max_experiments=2,
        model_retries=0, implementation=implementation_identity())
    with output.open("x") as stream:
        stream.write(config.model_dump_json(indent=2) + "\n")
    emit({"status": "manifest_frozen", "manifest": output.resolve(), "live_model_calls": 0,
          "final_access": False, "calibration_plan": calibration_plan()})


@app.command("calibrate")
def calibrate(manifest: Path, output: Path):
    """Run the bounded reference calibration through isolated real experiments."""
    config = LiveSpec.model_validate(strict_json_loads(manifest.read_text()))
    if config.model != "calibration-no-model" or config.max_experiments != 2:
        raise ValueError("Use a calibration-manifest with the fixed two-trial ceiling")
    adapter = CifarRecipeDevelopment(config.native, config.development)
    with locked_store(output) as store:
        result = run_calibration(store, config, adapter)
        emit(result.model_dump(mode="json"))
        if result.status != "pass":
            raise typer.Exit(2)


@app.command("verify-calibration")
def verify(manifest: Path, campaign: Path, receipt: str | None = None):
    """Verify actual immutable evidence; a synthetic trace cannot pass this gate."""
    config = LiveSpec.model_validate(strict_json_loads(manifest.read_text()))
    adapter = CifarRecipeDevelopment(config.native, config.development)
    # The imported context opens the database read-only; verification adds no operations.
    from ..integrations.paperorchestra.cli import readonly_campaign
    with readonly_campaign(campaign) as store:
        result = verify_calibration(store, receipt, config=config, adapter=adapter, require_actual=True)
        emit({"status": "verified", "receipt": result, "new_experiments": 0,
              "live_model_calls": 0})


@app.command("continuation", context_settings={"allow_extra_args": True,
    "ignore_unknown_options": True, "help_option_names": []},
    help="Plan/admit/run/audit a model-transport repair using the original pinned scientific runtime.")
def continuation(ctx: typer.Context):
    # A new interpreter is essential: importing the installed repaired package
    # into the pinned scientific process would invalidate its source identity.
    import subprocess
    import sys
    from .. import continuation_operator
    from ..store import file_digest
    operator = Path(continuation_operator.__file__).resolve()
    module = operator.with_name("continuation.py")
    args = list(ctx.args)
    if not any(arg == "--operator-module" or arg.startswith("--operator-module=") for arg in args):
        args.extend(["--operator-module", str(module), "--operator-sha256", file_digest(module)])
    result = subprocess.run([sys.executable, str(operator), *args], check=False)
    raise typer.Exit(result.returncode)


@app.command("continuation-relay", context_settings={"allow_extra_args": True,
    "ignore_unknown_options": True, "help_option_names": []},
    help="Inspect or explicitly serve a repaired relay bound to the original paid-call ledger.")
def continuation_relay(ctx: typer.Context):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "scientisttwo.continuation_relay_operator", *ctx.args], check=False)
    raise typer.Exit(result.returncode)
