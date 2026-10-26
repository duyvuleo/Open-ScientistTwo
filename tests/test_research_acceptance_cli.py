"""Model-free CLI contracts; all passing receipts below are synthetic test traces."""

import hashlib
import json
import subprocess

import pytest
import typer
from typer.testing import CliRunner

from scientisttwo.research import cli
from scientisttwo.research.acceptance import (
    AttemptRecord,
    EngineIdentity,
    EvidenceFile,
    REQUIRED_BRANCHES,
    REQUIRED_LIVE_ROLES,
    REQUIRED_ROLES,
    TASKS,
    TaskBinding,
    VerificationProof,
    identity,
    prepare_acceptance_plan,
)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(cli, "implementation_identity", lambda: {"test-source.py": "a" * 64})

    def forbidden(*args, **kwargs):
        raise AssertionError("Acceptance commands must not launch a subprocess or model")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    root = typer.Typer()
    root.add_typer(cli.app, name="workflow")
    return root


@pytest.fixture
def bundle(tmp_path):
    evidence_path = tmp_path / "explicit-synthetic-unit-test.json"
    content = b'{"label":"synthetic receipt for CLI contract tests"}'
    evidence_path.write_bytes(content)
    evidence = EvidenceFile(path=str(evidence_path), sha256=hashlib.sha256(content).hexdigest())
    engine = EngineIdentity(implementation_sha256=identity({"test-source.py": "a" * 64}))
    proof = VerificationProof(engine=engine, status="pass", implemented_roles=REQUIRED_ROLES,
                              controlled_roles=REQUIRED_ROLES, controlled_branches=REQUIRED_BRANCHES,
                              evidence=(evidence,))
    bindings = tuple(TaskBinding(
        family=family, task_id=task, prepared_manifest_sha256="b" * 64,
        protocol_sha256="c" * 64, runtime_sha256="d" * 64,
        data_provenance="bundled_real_data" if family == "cpu_frames" else "actual_data",
        datasets=("iris", "wine", "digits") if family == "cpu_frames" else (task,),
    ) for family, task in TASKS.items())
    plan = prepare_acceptance_plan(engine_identity=engine, bindings=bindings, implementation_proof=proof)
    paths = {name: tmp_path / f"{name}.json" for name in ("plan", "bindings", "proof", "records")}
    paths["plan"].write_text(plan.model_dump_json())
    paths["proof"].write_text(proof.model_dump_json())
    paths["bindings"].write_text(json.dumps([v.model_dump(mode="json") for v in bindings]))
    records = []
    for slot in plan.slots:
        if not slot.mandatory or slot.conditional:
            continue
        binding = next(v for v in bindings if v.family == slot.family)
        records.append(AttemptRecord(
            slot_id=slot.id, attempt_id=slot.id + "/one", plan_sha256=plan.sha256,
            engine=engine, binding_sha256=identity(binding), scope=slot.scope, status="pass",
            model=slot.budget.model, data_provenance=binding.data_provenance,
            execution_origin="isolated_real", datasets=binding.datasets,
            model_calls=4 if slot.budget.model else 0, observed_tokens=100 if slot.budget.model else 0,
            elapsed_seconds=1, natural_live_roles=REQUIRED_LIVE_ROLES if slot.scope == "live_full" else (),
            real_experiments="pass", manuscript_pdf="pass", final_evaluation="pass",
            independent_audit="pass", completed_resume="pass", runtime_probe="pass",
            workflow_complete=True, scientific_outcome="supported_candidate", evidence=(evidence,),
        ))
    paths["records"].write_text(json.dumps([v.model_dump(mode="json") for v in records]))
    return plan, paths, evidence


def invoke(app, *args):
    result = CliRunner().invoke(app, ["workflow", *map(str, args)])
    return result, json.loads(result.output)


def test_profiles_are_model_free_and_have_paper_defaults(app):
    result, value = invoke(app, "profiles")
    assert result.exit_code == 0, result.output
    assert set(value["profiles"]) == {"offline-pilot", "research-pilot", "paper-methodology"}
    assert value["profiles"]["paper-methodology"]["workflow"]["seed_ideas"] == 5
    assert value["profiles"]["paper-methodology"]["limitation_rounds"] == 16
    assert not value["execution_authorized"] and value["live_model_calls"] == 0
    result, selected = invoke(app, "profiles", "research-pilot")
    assert result.exit_code == 0 and set(selected["profiles"]) == {"research-pilot"}


def test_invalid_profile_returns_structured_error(app):
    result, value = invoke(app, "profiles", "invented")
    assert result.exit_code == 2 and value["status"] == "error"


def test_plan_with_missing_inputs_still_saves_explicit_blockers(app, tmp_path):
    output = tmp_path / "new" / "acceptance.json"
    result, value = invoke(app, "plan", "--output", output)
    assert result.exit_code == 0, result.output
    assert value["status"] == "planned" and value["blockers"]
    assert not value["release_ready"] and not value["execution_authorized"]
    saved = json.loads(output.read_text())
    assert saved["blockers"] and not saved["automatic_execution"]


def test_plan_uses_current_identity_and_verified_input_contracts(app, bundle, tmp_path):
    original, paths, _ = bundle
    output = tmp_path / "new-plan.json"
    result, value = invoke(app, "plan", "--output", output, "--bindings", paths["bindings"],
                           "--implementation-proof", paths["proof"], "--writer-backend", "canonical")
    assert result.exit_code == 0, result.output
    assert not value["blockers"]
    assert value["plan_sha256"] == original.sha256
    assert json.loads(output.read_text())["engine"] == original.engine.model_dump()
    assert not value["release_ready"]  # Planning never manufactures run evidence.


def test_plan_refuses_existing_file_without_modifying_it(app, tmp_path):
    output = tmp_path / "protected.json"
    output.write_bytes(b"keep exact bytes")
    result, value = invoke(app, "plan", "--output", output)
    assert result.exit_code == 2 and value["status"] == "error"
    assert output.read_bytes() == b"keep exact bytes"


def test_plan_refuses_output_symlink(app, tmp_path):
    target = tmp_path / "target.json"
    target.write_text("unchanged")
    link = tmp_path / "output.json"
    link.symlink_to(target)
    result, value = invoke(app, "plan", "--output", link)
    assert result.exit_code == 2 and target.read_text() == "unchanged"
    assert value["status"] == "error"


def test_check_is_read_only_and_does_not_call_consistency_release_readiness(app, bundle):
    plan, paths, _ = bundle
    before = paths["plan"].read_bytes()
    result, value = invoke(app, "check", paths["plan"], "--bindings", paths["bindings"])
    assert result.exit_code == 0, result.output
    assert value["status"] == "consistent" and value["manifest_fresh"]
    assert value["plan_sha256"] == plan.sha256
    assert not value["release_ready"] and not value["execution_authorized"]
    assert paths["plan"].read_bytes() == before


def test_check_rejects_changed_current_implementation(app, bundle, monkeypatch):
    _, paths, _ = bundle
    monkeypatch.setattr(cli, "implementation_identity", lambda: {"new-source.py": "f" * 64})
    result, value = invoke(app, "check", paths["plan"])
    assert result.exit_code == 2 and not value["manifest_fresh"]
    assert any("different implementation" in item for item in value["blockers"])


def test_check_compares_current_bindings(app, bundle):
    _, paths, _ = bundle
    current = json.loads(paths["bindings"].read_text())
    current[0]["runtime_sha256"] = "f" * 64
    paths["bindings"].write_text(json.dumps(current))
    result, value = invoke(app, "check", paths["plan"], "--bindings", paths["bindings"])
    assert result.exit_code == 2
    assert any("different data/protocol/runtime" in item for item in value["blockers"])


def test_check_detects_missing_proof_coverage_without_running_tests(app, bundle):
    _, paths, _ = bundle
    value = json.loads(paths["plan"].read_text())
    value["implementation_proof"]["controlled_branches"] = []
    paths["plan"].write_text(json.dumps(value))
    result, value = invoke(app, "check", paths["plan"])
    assert result.exit_code == 2 and any("Missing controlled_branches" in v for v in value["blockers"])


def test_assess_outputs_receipt_consistency_with_explicit_verification_boundary(app, bundle):
    _, paths, _ = bundle
    result, value = invoke(app, "assess", paths["plan"], paths["records"])
    assert result.exit_code == 0, result.output
    assert value["release_ready"] and value["status"] == "pass"
    assert value["assessment_kind"] == "immutable_verifier_receipt_consistency"
    assert value["receipt_authorship"] == "not_authenticated_by_this_command"
    assert not value["independent_reexecution"] and not value["execution_authorized"]
    assert value["live_model_calls_by_this_command"] == 0
    assert "manually written pass fields" in value["note"]


def test_assess_rejects_plain_prose_and_raw_green_status(app, bundle):
    _, paths, _ = bundle
    paths["records"].write_text('[{"status":"pass","scientific_claim":"everything worked"}]')
    result, value = invoke(app, "assess", paths["plan"], paths["records"])
    assert result.exit_code == 2 and value["status"] == "error"
    assert "everything worked" not in result.output
    assert value["errors"]


def test_assess_rechecks_current_code_not_only_saved_hashes(app, bundle, monkeypatch):
    _, paths, _ = bundle
    monkeypatch.setattr(cli, "implementation_identity", lambda: {"new-source.py": "f" * 64})
    result, value = invoke(app, "assess", paths["plan"], paths["records"])
    assert result.exit_code == 2 and not value["release_ready"]
    assert any("different implementation" in item for item in value["errors"])


def test_assess_preserves_failed_attempts_and_does_not_schedule_retries(app, bundle):
    _, paths, _ = bundle
    records = json.loads(paths["records"].read_text())
    records[-1]["status"] = "fail"
    records[-1]["reason"] = "OOM"
    paths["records"].write_text(json.dumps(records))
    before = paths["records"].read_bytes()
    result, value = invoke(app, "assess", paths["plan"], paths["records"])
    assert result.exit_code == 2
    assert value["accounting"]["failed_attempts"] == 1
    assert value["accounting"]["attempts"] == 6
    assert not value["automatic_retry"]
    assert paths["records"].read_bytes() == before


def test_assess_modified_evidence_fails(app, bundle):
    _, paths, evidence = bundle
    from pathlib import Path
    Path(evidence.path).write_text("changed")
    result, value = invoke(app, "assess", paths["plan"], paths["records"])
    assert result.exit_code == 2 and not value["release_ready"]
    assert any("Evidence modified" in item for item in value["errors"])


@pytest.mark.parametrize("name", ["plan", "bindings", "proof", "proofs", "attempts"])
def test_public_schemas_have_resolvable_root_definitions(app, name):
    result, value = invoke(app, "schema", name)
    assert result.exit_code == 0, result.output
    assert value["type"] == ("array" if name in {"bindings", "proofs", "attempts"} else "object")

    def check_refs(node):
        if isinstance(node, dict):
            if "$ref" in node:
                assert node["$ref"].startswith("#/$defs/")
                assert node["$ref"].split("/")[-1] in value["$defs"]
            for child in node.values():
                check_refs(child)
        elif isinstance(node, list):
            for child in node:
                check_refs(child)

    check_refs(value)


def test_help_explains_launch_boundary_and_examples(app):
    result = CliRunner().invoke(app, ["workflow", "--help"])
    assert result.exit_code == 0
    assert "do not authorize" in result.output
    assert "workflow plan" in result.output
    result = CliRunner().invoke(app, ["workflow", "assess", "--help"])
    assert result.exit_code == 0 and "receipt" in result.output


@pytest.mark.parametrize("content", ['{"foo":1,"foo":2}', '{"score":NaN}', '[]'])
def test_ambiguous_or_nonobject_plan_json_is_rejected(app, tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    result, value = invoke(app, "check", path)
    assert result.exit_code == 2 and value["status"] == "error"


def test_plan_rejects_nonarray_bindings_without_writing_output(app, tmp_path):
    bindings = tmp_path / "not-a-list.json"
    bindings.write_text('{}')
    output = tmp_path / "new-plan.json"
    result, value = invoke(app, "plan", "--output", output, "--bindings", bindings)
    assert result.exit_code == 2 and value["status"] == "error"
    assert not output.exists()
