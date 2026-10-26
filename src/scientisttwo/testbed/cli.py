"""Explicit preparation/validation commands; no live-backend construction."""

import json
from pathlib import Path

import typer

from .preparation import prepare
from .readiness import readiness
from .validation import validate


def _emit(value):
    typer.echo(json.dumps(value, indent=2, allow_nan=False, default=str))


def attach_commands(app):
    @app.command("export-live")
    def export_live_command(task_id: str, output: Path, model: str = typer.Option(...),
                            max_calls: int = typer.Option(...), campaign_seconds: float = typer.Option(...),
                            root: Path = typer.Option(Path(".testbed")), call_seconds: float = 300,
                            token_stop_threshold: int | None = None,
                            agent_output_retries: int = typer.Option(0, min=0, max=1, help="Allow one budgeted correction for invalid action/reference contracts; default off."),
                            supplementary_seed_count: int = typer.Option(2, min=1, max=32, help="Exact fresh seed allowance per supplementary batch; unsupported counts are refused."),
                            literature_network: bool = False, compiler_image: str | None = None,
                            require_pdf: bool = False):
        """Export ready frozen task inputs and explicit budgets; never invoke a live model."""
        from .export import export_live
        _emit(export_live(task_id, root, output, model=model, max_calls=max_calls,
                          campaign_seconds=campaign_seconds, call_seconds=call_seconds,
                          token_stop_threshold=token_stop_threshold,
                          agent_output_retries=agent_output_retries,
                          supplementary_seed_count=supplementary_seed_count,
                          literature_network=literature_network, compiler_image=compiler_image,
                          require_pdf=require_pdf))

    @app.command("register-workflow-contract")
    def register_workflow_command(validation_output: Path, root: Path = typer.Option(Path(".testbed"))):
        """Register verified real branch experiments, including explicitly labeled fixture data."""
        from .workflow_contracts import register_workflow_contract
        _emit(register_workflow_contract(root, validation_output))

    @app.command("generate-bundle")
    def generate_command(task_id: str, output: Path = typer.Option(...), image: str = typer.Option(...),
                         helpers: Path | None = None, assets: Path | None = None, source: Path | None = None,
                         runtime: str = "docker", context: str | None = None, gpus: str | None = None,
                         seconds: float = 3600, memory_mb: int = 8192, seed: int = 1729,
                         variant: str | None = None, terms_accepted: bool = False):
        """Generate CLRS or fixed 7B workloads in a restricted local container only."""
        from .generation import generate_bundle
        _emit(generate_bundle(task_id, output, image=image, helpers=helpers, assets=assets, source=source,
                              runtime=runtime, context=context, gpus=gpus, seconds=seconds,
                              memory_mb=memory_mb, seed=seed, variant=variant, terms_accepted=terms_accepted))

    @app.command("acquire")
    def acquire_command(task_id: str, output: Path = typer.Option(...), allow_downloads: bool = False,
                        terms_accepted: bool = False, download_seconds: float = 3600,
                        max_download_gb: float = 100, max_extracted_gb: float = 200,
                        include_images: bool = False):
        """Explicitly fetch a declared provider archive; retain source/member receipts."""
        from .providers import acquire
        _emit(acquire(task_id, output, allow_downloads=allow_downloads, terms_accepted=terms_accepted,
                      download_seconds=download_seconds, max_download_bytes=int(max_download_gb * 1024**3),
                      max_extracted_bytes=int(max_extracted_gb * 1024**3), include_images=include_images))

    @app.command("prepare")
    def prepare_command(
        task_id: str,
        source: Path = typer.Option(...),
        root: Path = typer.Option(Path(".testbed")),
        labels: Path | None = None,
        provenance: str = typer.Option(..., help="actual_data or synthetic_fixture; never inferred"),
        terms_accepted: bool = False,
        variant: str = "audited-v1",
        split_seed: int = 1729,
        derive_final: bool = False,
        source_format: str = "raw",
        source_receipt: Path | None = None,
    ):
        """Import explicitly obtained MLAgentBench data/assets and freeze task partitions."""
        value = prepare(task_id, source, root / task_id / "prepared", provenance=provenance,
                      labels=labels, terms_accepted=terms_accepted, variant=variant, split_seed=split_seed,
                      derive_final=derive_final, source_format=source_format,
                      source_receipt=source_receipt)
        _emit({key: value[key] for key in ("prepared", "manifest_sha256", "task_id", "recipe_version",
                                          "provenance", "dataset_identity", "source_format", "live_model_calls")})

    @app.command("readiness")
    def readiness_command(root: Path = typer.Option(Path(".testbed"))):
        """Report all 13 canonical slots; missing/failed tasks stay in the denominator."""
        _emit(readiness(root))

    @app.command("validate")
    def validate_command(
        task_id: str,
        root: Path = typer.Option(Path(".testbed")),
        image: str = typer.Option(...),
        grader_image: str | None = None,
        context: str | None = None,
        runtime: str = "docker",
        gpus: str | None = None,
        level: str = "smoke",
        seconds: float = 1800,
        campaign_seconds: float = 18000,
        max_prompt_chars: int = typer.Option(200_000, min=1000,
            help="Maximum serialized agent prompt characters; frozen for this validation run."),
        compiler_image: str | None = None,
        require_pdf: bool = False,
    ):
        """Run prepared baselines/controls with scripted agents and real isolated experiments."""
        output_name = "validation" if level == "full" else "validation-smoke"
        result = validate(root / task_id / "prepared", root / task_id / output_name, image=image,
                          grader_image=grader_image, context=context, level=level, seconds=seconds,
                          campaign_seconds=campaign_seconds, runtime=runtime, gpus=gpus,
                          max_prompt_chars=max_prompt_chars,
                          compiler_image=compiler_image, require_pdf=require_pdf)
        _emit(result)
        if not all(result["gates"].values()):
            raise typer.Exit(2)
