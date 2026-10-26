"""Explicit, optional frozen-experiment-to-manuscript integration commands."""

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import typer

from ...store import IntegrityError, Store, file_digest
from .evidence import (
    build_campaign_export_ledger, export_evidence_bundle, import_external_manuscript,
    verify_evidence_bundle,
)

app = typer.Typer(no_args_is_help=True, help="Official PaperOrchestra agents with frozen development evidence. Models require explicit opt-in.")


def _emit(value):
    typer.echo(json.dumps(value, indent=2, allow_nan=False, default=str))


@contextmanager
def readonly_campaign(path):
    root = Path(path).resolve()
    database = root / "campaign.sqlite"
    if not database.is_file() or database.is_symlink():
        raise ValueError("A campaign database is required")
    store = Store.__new__(Store)
    store.root = root
    store.db = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    store.db.row_factory = sqlite3.Row
    try:
        yield store
    finally:
        store.close()


def _load(path, maximum=5_000_000):
    from ...backends import strict_json_loads
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise IntegrityError("Missing, symlinked, or oversized integration record")
    return strict_json_loads(path.read_text())


def _files(root):
    items, size = {}, 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise IntegrityError("Symlinks are forbidden in an integration result")
        if path.is_dir():
            continue
        if not path.is_file():
            raise IntegrityError("Special files are forbidden in an integration result")
        name = path.relative_to(root).as_posix()
        if name == "integration-result.json":
            continue
        size += path.stat().st_size
        if len(items) >= 2000 or size > 200_000_000:
            raise IntegrityError("Integration output exceeds file or byte limits")
        items[name] = {"sha256": file_digest(path), "bytes": path.stat().st_size}
    return items


def audit_output(output):
    root = Path(output).resolve()
    report = _load(root / "integration-result.json")
    if report.get("schema_version") != "paperorchestra-integration-result-v1":
        raise IntegrityError("Unknown integration result schema")
    if report.get("files") != _files(root):
        raise IntegrityError("Integration artifacts changed, are missing, or contain extra files")
    return report


@app.command("export")
def export_command(
    campaign: Path = typer.Option(..., help="Existing campaign; opened read-only."),
    output: Path = typer.Option(..., help="New frozen materials directory."),
    title: str = typer.Option(...),
    hypothesis: str = typer.Option(...),
):
    """Export verified aggregate development evidence; never export final-test results."""
    from .runtime import UPSTREAM_REVISION
    try:
        with readonly_campaign(campaign) as store:
            ledger = build_campaign_export_ledger(store)
            manifest = export_evidence_bundle(store, output, ledger=ledger, title=title,
                                             hypothesis=hypothesis, upstream_revision=UPSTREAM_REVISION)
        _emit({"status": "exported", "materials": output.resolve(), "manifest": manifest,
               "model_calls": 0, "experiments": 0})
    except (ValueError, OSError, IntegrityError, sqlite3.Error) as error:
        _emit({"status": "failed", "error": str(error)})
        raise typer.Exit(2) from error


@app.command("run")
def run_command(
    materials: Path = typer.Option(...),
    upstream: Path | None = typer.Option(None, help="Optional pinned-source override; defaults to the bundled PaperOrchestra engine."),
    output: Path = typer.Option(..., help="New result directory, or a completed result to verify/reuse."),
    compiler_image: str = typer.Option(..., help="Already-built PDF-tool image sha256 digest."),
    context: str | None = typer.Option(None, help="Existing Docker context."),
    backend: str = typer.Option("fixture", help="fixture or codex; default makes zero live calls."),
    model: str | None = typer.Option(None, help="Required explicit model for codex."),
    allow_live: bool = typer.Option(False),
    max_calls: int = typer.Option(24, min=1, max=100),
    max_tokens: int = typer.Option(200_000, min=1, max=1_000_000),
    timeout_seconds: int = typer.Option(900, min=10, max=3600),
    reflections: int = typer.Option(1, min=1, max=3),
):
    """Outline, draft, internal review/refinement and compile from frozen evidence.

    This writing integration cannot launch research experiments or promote a
    method. A completed output is verified/reused; partial output requires a new
    destination and remains preserved. Token limits stop before the next call;
    the last completed call may cross the observed-token threshold.
    """
    from .runtime import RuntimeConfig, run_paperorchestra, verify_upstream, resolve_upstream_root
    from .pdf_tools import ContainerPdfTools
    from .fixtures import EvidenceFixtureTransport
    from .codex_transport import CodexTransport
    try:
        if backend not in {"fixture", "codex"}:
            raise ValueError("Backend must be fixture or codex")
        if backend == "codex" and (not allow_live or not model):
            raise ValueError("Codex requires --allow-live and an explicit --model")
        if backend == "fixture" and (allow_live or model):
            raise ValueError("Fixture runs cannot enable live calls or select a live model")
        verify_evidence_bundle(materials)
        upstream = resolve_upstream_root(upstream)
        source_hashes = verify_upstream(upstream)
        config = RuntimeConfig(model_name=model or "scripted-paperorchestra-example",
            max_reflections=reflections, max_model_calls=max_calls, max_tokens=max_tokens,
            timeout_seconds=timeout_seconds, allow_live=allow_live)
        binding = {"materials_manifest_sha256": file_digest(materials / "manifest.json"),
                   "backend": backend, "config": asdict(config),
                   "compiler_image": compiler_image, "context": context,
                   "upstream_source_hashes": source_hashes,
                   "integration_source_hashes": {
                       path.name: file_digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}}
        if output.exists():
            report = audit_output(output)
            if report["status"] != "success":
                raise ValueError("Prior output is incomplete; preserve it and choose a new directory")
            if report.get("request_binding") != binding:
                raise IntegrityError("Completed run belongs to different evidence, source, backend, or configuration")
            _emit({**report, "reused": True, "new_model_calls": 0, "new_experiments": 0})
            return
        pdf_tools = ContainerPdfTools(image_digest=compiler_image, context=context,
                                      budget_seconds=timeout_seconds, staging_root=output)
        transport = (CodexTransport(model=model, allow_live=True, pdf_tools=pdf_tools)
                     if backend == "codex" else EvidenceFixtureTransport(materials))
        # The runtime owns exclusive destination creation and partial receipts.
        result = run_paperorchestra(upstream_root=upstream, materials_dir=materials,
            output_dir=output, transport=transport, pdf_tools=pdf_tools, config=config,
            expected_manifest_sha256=binding["materials_manifest_sha256"])
        (output / "pdf-receipts.json").write_text(json.dumps(pdf_tools.receipts, indent=2, default=str) + "\n")
        imported = None
        if result.status == "success":
            # Bind the exported evidence before and after execution as well as
            # the writer's own snapshot; never import a replacement source tree.
            verify_evidence_bundle(materials)
            if file_digest(materials / "manifest.json") != binding["materials_manifest_sha256"]:
                raise IntegrityError("Materials changed during PaperOrchestra execution")
            frozen_materials = output / "raw_materials"
            verify_evidence_bundle(frozen_materials)
            if file_digest(frozen_materials / "manifest.json") != binding["materials_manifest_sha256"]:
                raise IntegrityError("Writer snapshot differs from admitted evidence")
            store = Store(output / "external-evidence")
            try:
                imported = import_external_manuscript(store, frozen_materials, result.final_tex)
            finally:
                store.close()
        status = ("failed_validation" if imported and imported.validation.get("status") == "fail"
                  else result.status)
        report = {"schema_version": "paperorchestra-integration-result-v1",
            "status": status, "runtime": asdict(result), "backend": backend,
            "request_binding": binding,
            "model": config.model_name, "synthetic_agents": transport.synthetic,
            "materials_manifest_sha256": file_digest(materials / "manifest.json"),
            "external_manuscript": imported.model_dump(mode="json") if imported else None,
            "scientisttwo_research_acceptance": "not_assessed",
            "experiments": 0, "files": _files(output)}
        (output / "integration-result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        _emit({"status": status, "output": output.resolve(), "backend": backend,
               "final_pdf": result.final_pdf, "synthetic_agents": transport.synthetic,
               "report": output / "integration-result.json", "error": result.error})
        if status != "success":
            raise typer.Exit(2)
    except (ValueError, OSError, IntegrityError) as error:
        _emit({"status": "failed", "error": str(error)})
        raise typer.Exit(2) from error


@app.command("inspect")
def inspect_command(output: Path):
    """Verify all sealed output files and show the result without new operations."""
    try:
        report = audit_output(output)
        _emit({key: value for key, value in report.items() if key != "files"})
    except (ValueError, OSError, IntegrityError) as error:
        _emit({"status": "failed", "error": str(error)})
        raise typer.Exit(2) from error


@app.command("audit")
def audit_command(output: Path):
    """Check sealed artifact integrity; this does not establish scientific truth."""
    try:
        report = audit_output(output)
        _emit({"integrity": "pass", "files_verified": len(report["files"]),
               "pipeline_status": report["status"], "scientific_validity": "unverified",
               "new_model_calls": 0, "new_experiments": 0})
    except (ValueError, OSError, IntegrityError) as error:
        _emit({"integrity": "fail", "error": str(error)})
        raise typer.Exit(2) from error
