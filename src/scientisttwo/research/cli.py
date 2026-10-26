"""Local acceptance planning and receipt assessment, with no execution capability."""

import json
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel, ValidationError

from ..live.adapters import implementation_identity
from .acceptance import (
    AcceptancePlan,
    AttemptRecord,
    TaskBinding,
    VerificationProof,
    assess_acceptance,
    prepare_acceptance_plan,
    validate_fresh_manifest,
)
from .policy import profile_updates


app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Model-free workflow profiles, bounded acceptance plans and immutable receipt checks. "
        "These commands do not authorize or start models, downloads, containers or cluster jobs. "
        "Example: scientisttwo workflow plan --output acceptance.json --bindings bindings.json; "
        "scientisttwo workflow check acceptance.json; "
        "scientisttwo workflow assess acceptance.json attempts.json."
    ),
)


def _json_default(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported output type: {type(value).__name__}")


def _emit(value):
    typer.echo(json.dumps(value, indent=2, allow_nan=False, default=_json_default))


def _error(operation, error):
    # Pydantic's normal exception string may include raw submitted values. The
    # JSON error reports contracts/locations but never copies input payloads.
    if isinstance(error, ValidationError):
        details = [
            {"location": list(item["loc"]), "type": item["type"], "message": item["msg"]}
            for item in error.errors(include_input=False, include_url=False)
        ]
    else:
        details = [str(error)]
    _emit({"status": "error", "operation": operation, "errors": details,
           "live_model_calls": 0, "execution_authorized": False})
    raise typer.Exit(2)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Input JSON contains duplicate object keys")
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError("Input JSON cannot contain NaN or infinity")


def _load(path: Path, expected: type) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"Input must be a bounded regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                       parse_constant=_nonfinite)
    if not isinstance(value, expected):
        name = "array" if expected is list else "object"
        raise ValueError(f"Input must contain a JSON {name}: {path}")
    return value


def _bindings(path: Path | None):
    return tuple(TaskBinding.model_validate(v) for v in _load(path, list)) if path else ()


def _coverage_blockers(plan):
    proof = plan.implementation_proof
    if proof is None:
        return ["No implementation or controlled-coverage verifier proof"]
    missing = {
        "implemented_roles": sorted(set(plan.required_roles) - set(proof.implemented_roles)),
        "controlled_roles": sorted(set(plan.required_roles) - set(proof.controlled_roles)),
        "controlled_branches": sorted(set(plan.required_branches) - set(proof.controlled_branches)),
    }
    return [f"Missing {key}: {', '.join(values)}" for key, values in missing.items() if values]


@app.command("profiles")
def profiles_command(
    name: str | None = typer.Argument(None, help="Optional offline-pilot, research-pilot or paper-methodology."),
):
    """Show frozen workflow policy defaults; never select a backend or run a campaign."""
    try:
        names = [name] if name else ["offline-pilot", "research-pilot", "paper-methodology"]
        values = {item: profile_updates(item) for item in names}
        _emit({"schema_version": 1, "profiles": values, "live_model_calls": 0,
               "execution_authorized": False,
               "note": "Workflow iteration limits are not resource estimates or permission to launch."})
    except (ValueError, OSError) as error:
        _error("profiles", error)


@app.command("plan")
def plan_command(
    output: Path = typer.Option(..., help="New JSON plan path; existing files are never overwritten."),
    registry: Path | None = typer.Option(None, help="Optional prepared testbed root; metadata reads only."),
    bindings: Path | None = typer.Option(None, help="JSON array of TaskBinding records; see workflow schema bindings."),
    implementation_proof: Path | None = typer.Option(None, help="VerificationProof JSON from the coverage verifier, with immutable evidence files."),
    prerequisite_proofs: Path | None = typer.Option(None, help="Optional JSON array of existing VerificationProof records."),
    writer_backend: str = typer.Option("paperorchestra", help="Writer expected by this new acceptance plan; canonical is available for legacy campaigns."),
):
    """Freeze a local acceptance schedule, retaining missing prerequisites as blockers.

    Example: scientisttwo workflow plan --output acceptance.json --registry .testbed
    --bindings bindings.json --implementation-proof coverage-proof.json
    """
    try:
        # Reject an existing destination before examining potentially costly input
        # manifests. Exclusive creation below also prevents a race-time overwrite.
        if output.exists() or output.is_symlink():
            raise ValueError(f"Refusing to overwrite existing output: {output}")
        proof = VerificationProof.model_validate(_load(implementation_proof, dict)) if implementation_proof else None
        prerequisites = tuple(VerificationProof.model_validate(v) for v in _load(prerequisite_proofs, list)) if prerequisite_proofs else ()
        plan = prepare_acceptance_plan(
            engine_identity=implementation_identity(), registry=registry, bindings=_bindings(bindings),
            implementation_proof=proof, prerequisite_proofs=prerequisites,
            writer_backend=writer_backend,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(plan.model_dump_json(indent=2) + "\n")
        _emit({"status": "planned", "plan": str(output.resolve()), "plan_sha256": plan.sha256,
               "blockers": list(dict.fromkeys([*plan.blockers, *_coverage_blockers(plan)])),
               "live_model_calls": 0, "execution_authorized": False,
               "release_ready": False,
               "note": "A saved budget is not launch approval or validation evidence."})
    except (ValueError, OSError, TypeError) as error:
        _error("plan", error)


@app.command("check")
def check_command(
    plan_path: Path = typer.Argument(..., help="Frozen AcceptancePlan JSON."),
    bindings: Path | None = typer.Option(None, help="Optional current TaskBinding JSON array to compare against the frozen plan."),
):
    """Check current implementation, data/protocol bindings and proof bytes without executing work."""
    try:
        plan = AcceptancePlan.model_validate(_load(plan_path, dict))
        current = _bindings(bindings) if bindings else None
        fresh = validate_fresh_manifest(plan, engine_identity=implementation_identity(), bindings=current)
        blockers = list(dict.fromkeys([*plan.blockers, *fresh, *_coverage_blockers(plan)]))
        _emit({"status": "blocked" if blockers else "consistent", "plan_sha256": plan.sha256,
               "manifest_fresh": not fresh, "blockers": blockers,
               "live_model_calls": 0, "execution_authorized": False, "release_ready": False,
               "note": "Manifest consistency does not establish runtime readiness or completed validation."})
        if blockers:
            raise typer.Exit(2)
    except (ValueError, OSError, TypeError) as error:
        _error("check", error)


@app.command("assess")
def assess_command(
    plan_path: Path = typer.Argument(..., help="Frozen AcceptancePlan JSON."),
    records_path: Path = typer.Argument(..., help="JSON array of AttemptRecord receipts produced by independent verifiers; free-form outcome statements are not accepted."),
    bindings: Path | None = typer.Option(None, help="Optional current TaskBinding JSON array."),
):
    """Assess immutable verifier receipts and retain failed/unresolved attempts in accounting.

    This reads receipts only. It does not replay experiments, authenticate receipt
    authorship, or establish the truth of manually supplied 'pass' assertions.
    """
    try:
        plan = AcceptancePlan.model_validate(_load(plan_path, dict))
        records = tuple(AttemptRecord.model_validate(v) for v in _load(records_path, list))
        report = assess_acceptance(plan, records)
        fresh = validate_fresh_manifest(plan, engine_identity=implementation_identity(),
                                        bindings=_bindings(bindings) if bindings else None)
        if fresh:
            report["errors"] = list(dict.fromkeys([*report["errors"], *fresh]))
            report["status"], report["release_ready"] = "blocked", False
        report.update({
            "assessment_kind": "immutable_verifier_receipt_consistency",
            "receipt_authorship": "not_authenticated_by_this_command",
            "independent_reexecution": False,
            "live_model_calls_by_this_command": 0,
            "execution_authorized": False,
            "note": "Gate assertions must originate from independent verifiers. This command checks their bindings and accounting; manually written pass fields do not establish validated scientific execution.",
        })
        _emit(report)
        if not report["release_ready"]:
            raise typer.Exit(2)
    except (ValueError, OSError, TypeError) as error:
        _error("assess", error)


@app.command("schema")
def schema_command(
    name: str = typer.Argument("plan", help="plan, bindings, proof, proofs or attempts."),
):
    """Print public JSON schemas; for example, scientisttwo workflow schema attempts."""
    classes = {"plan": AcceptancePlan, "bindings": TaskBinding, "proof": VerificationProof,
               "proofs": VerificationProof, "attempts": AttemptRecord}
    try:
        if name not in classes:
            raise ValueError("Unknown schema; choose plan, bindings, proof, proofs or attempts")
        schema = classes[name].model_json_schema()
        if name in {"bindings", "proofs", "attempts"}:
            # Definitions belong at the document root, where generated #/$defs
            # references resolve, rather than inside the array's items object.
            definitions = schema.pop("$defs", None)
            schema = {"type": "array", "items": schema}
            if definitions:
                schema["$defs"] = definitions
        _emit(schema)
    except ValueError as error:
        _error("schema", error)
