"""Synthetic metadata fixtures for live protocol binding; no model/runtime calls."""

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from scientisttwo.artifacts import ArtifactSpec
from scientisttwo.backends import CodexBackend
from scientisttwo.cli import app
from scientisttwo.live.models import LiveSpec
from scientisttwo.live.pipeline import EvaluationPipeline, EvaluationStage
from scientisttwo.store import Store
from scientisttwo.testbed.adapter import PreparedDevelopment
from scientisttwo.testbed import export, readiness, workflow_contracts
from scientisttwo.testbed.preparation import prepare
from scientisttwo.testbed.validation import create_validation_spec

EXECUTION = "sha256:" + "a" * 64
GRADING = "sha256:" + "b" * 64
INFERENCE = "sha256:" + "c" * 64
SCORING = "sha256:" + "d" * 64


def configured(tmp_path, monkeypatch, *, pipeline=False):
    source = tmp_path / "raw"
    source.mkdir()
    pd.DataFrame({"Id": range(100), "feature": range(100), "SalePrice": range(100)}).to_csv(source / "train.csv", index=False)
    root = tmp_path / "testbed"
    prepared = root / "house-price/prepared"
    prepare("house-price", source, prepared, provenance="synthetic_fixture")
    original = create_validation_spec(prepared, image="training:mutable", grader_image="grading:mutable")
    if pipeline:
        native = original.native
        native.output_artifacts = [ArtifactSpec(path="checkpoint", kind="directory", role="checkpoint")]
        native.evaluation_pipeline = EvaluationPipeline(
            inference=EvaluationStage(kind="inference", image="inference:mutable",
                                      source_dir=native.evaluator_dir, source_manifest=native.evaluator_manifest,
                                      command=["python", "evaluate.py"], outputs=[ArtifactSpec(path="predictions.npz")]),
            score=EvaluationStage(kind="score", image="scoring:mutable",
                                  source_dir=native.evaluator_dir, source_manifest=native.evaluator_manifest,
                                  command=["python", "evaluate.py"]),
            private_dir=native.private_dir, private_manifest=native.private_manifest,
        )
    validation = root / "house-price/validation"
    validation.mkdir()
    (validation / "validation-spec.json").write_text(original.model_dump_json())
    store = Store(validation / "campaign")
    try:
        store.set("live_spec_ref", store.put_json(original))
    finally:
        store.close()
    receipt = {"task_id": "house-price", "actual_task_validated": True,
               "images": {"execution": EXECUTION, "grading": GRADING}, "finals": {}, "final_operation_keys": {}}
    if pipeline:
        actual = original.native.evaluation_pipeline.model_dump(mode="json")
        actual["inference"]["image"], actual["score"]["image"] = INFERENCE, SCORING
        child = Store(validation / "campaign/final/baseline")
        try:
            report = child.put_json({"pipeline": actual})
        finally:
            child.close()
        receipt["finals"] = {"reference": {"official_report_ref": report}}
        receipt["final_operation_keys"] = {"reference": "final/baseline"}
    monkeypatch.setattr(export, "verify_validation", lambda *args: receipt)
    monkeypatch.setattr(readiness, "verify_validation", lambda *args: receipt)
    monkeypatch.setattr(workflow_contracts, "verified_workflow_contracts", lambda *args: {"cpu_frames": {"status": "pass"}})
    return root, validation, original, receipt


def test_live_export_pins_all_executed_image_ids_and_allows_agent_budgets(tmp_path, monkeypatch):
    root, validation, original, receipt = configured(tmp_path, monkeypatch, pipeline=True)
    output = tmp_path / "live.json"
    result = export.export_live("house-price", root, output, model="gpt-6.1-sol", max_calls=12,
                                campaign_seconds=1200, call_seconds=120, token_stop_threshold=10000)
    saved = LiveSpec.model_validate_json(output.read_text())
    assert saved.native.image == EXECUTION and saved.native.grader_image == GRADING
    assert saved.native.evaluation_pipeline.inference.image == INFERENCE
    assert saved.native.evaluation_pipeline.score.image == SCORING
    assert saved.model == "gpt-6.1-sol" and saved.max_calls == 12 and saved.campaign_seconds == 1200
    assert saved.development.settings["validation_output"] == str(validation)
    assert saved.development.seeds == original.development.seeds
    assert result["live_model_calls"] == 0 and result["live_execution_enabled"] is False
    assert PreparedDevelopment(saved.native, saved.development).live_readiness()["ready"]


@pytest.mark.parametrize("change", ["original_tag", "new_tag", "digest", "resources", "data", "evaluation", "stage_image"])
def test_live_hook_rejects_native_protocol_drift(tmp_path, monkeypatch, change):
    _, validation, _, receipt = configured(tmp_path, monkeypatch, pipeline=True)
    proposed = export.pinned_validation_config(validation, receipt)
    if change == "original_tag":
        proposed.native.image = "training:mutable"
    elif change == "new_tag":
        proposed.native.image = "training:other"
    elif change == "digest":
        proposed.native.image = "sha256:" + "e" * 64
    elif change == "resources":
        proposed.native.resources.seconds += 1
    elif change == "data":
        proposed.native.public_manifest = {}
    elif change == "evaluation":
        proposed.native.evaluation_pipeline.score.command = ["python", "different.py"]
    else:
        proposed.native.evaluation_pipeline.inference.image = "sha256:" + "f" * 64
    result = PreparedDevelopment(proposed.native, proposed.development).live_readiness()
    assert not result["ready"]
    assert any("validated protocol" in error for error in result["errors"])


@pytest.mark.parametrize("change", ["seeds", "epsilon", "settings"])
def test_live_hook_rejects_development_protocol_drift(tmp_path, monkeypatch, change):
    _, validation, _, receipt = configured(tmp_path, monkeypatch)
    proposed = export.pinned_validation_config(validation, receipt)
    if change == "seeds":
        proposed.development.seeds = [7, 8, 9]
    elif change == "epsilon":
        proposed.development.epsilon += 0.1
    else:
        proposed.development.settings["unvalidated_override"] = True
    result = PreparedDevelopment(proposed.native, proposed.development).live_readiness()
    assert not result["ready"]
    assert any("development evaluation" in error for error in result["errors"])


def test_modified_convenience_spec_cannot_redefine_prior_validation(tmp_path, monkeypatch):
    _, validation, original, receipt = configured(tmp_path, monkeypatch)
    changed = original.model_dump(mode="json")
    changed["native"]["resources"]["memory_mb"] += 256
    (validation / "validation-spec.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="executed configuration"):
        export.pinned_validation_config(validation, receipt)


def test_missing_pipeline_image_evidence_cannot_be_inferred_from_tags(tmp_path, monkeypatch):
    _, validation, _, receipt = configured(tmp_path, monkeypatch, pipeline=True)
    receipt["finals"] = {}
    with pytest.raises(ValueError, match="absent or inconsistent"):
        export.pinned_validation_config(validation, receipt)


def forbid_live_backend(monkeypatch):
    constructed = []

    def forbidden(self, *args, **kwargs):
        constructed.append((args, kwargs))
        raise AssertionError("Manifest creation/export must not construct a live backend")

    monkeypatch.setattr(CodexBackend, "__init__", forbidden)
    return constructed


@pytest.mark.parametrize("option,expected", [(None, 0), (1, 1)])
def test_live_export_correction_policy_defaults_off_and_requires_explicit_one(tmp_path, monkeypatch, option, expected):
    root, _, original, _ = configured(tmp_path, monkeypatch)
    backend_calls = forbid_live_backend(monkeypatch)
    output = tmp_path / "live-policy.json"
    policy = {} if option is None else {"agent_output_retries": option}
    result = export.export_live("house-price", root, output, model="gpt-6-astra",
                                max_calls=12, campaign_seconds=1200, **policy)
    saved = LiveSpec.model_validate_json(output.read_text())
    assert saved.agent_output_retries == expected
    assert saved.implementation == original.implementation  # No silent upgrade of prior evidence.
    assert saved.max_calls == 12 and saved.campaign_seconds == 1200
    assert result["live_model_calls"] == 0 and result["live_execution_enabled"] is False
    assert not backend_calls


def test_benchmark_export_live_cli_forwards_correction_option_without_a_backend(tmp_path, monkeypatch):
    root, _, _, _ = configured(tmp_path, monkeypatch)
    backend_calls = forbid_live_backend(monkeypatch)
    output = tmp_path / "cli-live.json"
    exported, original_export = [], export.export_live

    def observed_export(*args, **kwargs):
        exported.append((args, kwargs.copy()))
        return original_export(*args, **kwargs)

    monkeypatch.setattr(export, "export_live", observed_export)
    result = CliRunner().invoke(app, ["benchmark", "export-live", "house-price", str(output),
                                     "--root", str(root), "--model", "gpt-6-astra",
                                     "--max-calls", "12", "--campaign-seconds", "1200",
                                     "--agent-output-retries", "1"])
    assert result.exit_code == 0, result.output
    assert len(exported) == 1 and exported[0][1]["agent_output_retries"] == 1
    assert LiveSpec.model_validate_json(output.read_text()).agent_output_retries == 1
    assert json.loads(result.stdout)["live_model_calls"] == 0
    assert not backend_calls


def test_benchmark_export_live_cli_rejects_more_than_one_before_export(tmp_path, monkeypatch):
    exported = []
    backend_calls = forbid_live_backend(monkeypatch)

    def forbidden_export(*args, **kwargs):
        exported.append((args, kwargs))
        raise AssertionError("CLI must reject the invalid option before export")

    monkeypatch.setattr(export, "export_live", forbidden_export)
    output = tmp_path / "must-not-exist.json"
    result = CliRunner().invoke(app, ["benchmark", "export-live", "house-price", str(output),
                                     "--root", str(tmp_path / "unprepared"), "--model", "gpt-6-astra",
                                     "--max-calls", "12", "--campaign-seconds", "1200",
                                     "--agent-output-retries", "2"])
    assert result.exit_code == 2
    assert not exported and not backend_calls and not output.exists()
    assert "agent-output-retries" in result.output


@pytest.mark.parametrize("option,expected", [(None, 0), (1, 1)])
def test_live_create_cli_persists_explicit_correction_policy_without_backend(tmp_path, monkeypatch, option, expected):
    from scientisttwo.live import cli as live_cli

    _, _, original, _ = configured(tmp_path, monkeypatch)
    native_file, development_file = tmp_path / "native.json", tmp_path / "development.json"
    native_file.write_text(original.native.model_dump_json())
    development_file.write_text(original.development.model_dump_json())
    backend_calls = forbid_live_backend(monkeypatch)
    checked = []

    def preflight(native):
        checked.append(native)
        return {"ready": True, "errors": [], "synthetic_test": True}

    monkeypatch.setattr(live_cli, "preflight", preflight)
    monkeypatch.setattr(live_cli, "load_development", PreparedDevelopment)
    output = tmp_path / "created-live.json"
    args = ["live", "create", str(native_file), str(output), "--development", str(development_file),
            "--model", "gpt-6-astra", "--max-calls", "12", "--seconds", "1200", "--mode", "research",
            "--writer-backend", "canonical"]
    if option is not None:
        args += ["--agent-output-retries", str(option)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    saved = LiveSpec.model_validate_json(output.read_text())
    assert saved.agent_output_retries == expected
    assert saved.model == "gpt-6-astra" and saved.max_calls == 12
    assert saved.campaign_seconds == 1200 and saved.mode == "research"
    assert len(checked) == 1 and checked[0].model_dump() == original.native.model_dump()
    assert json.loads(result.stdout)["live_model_calls"] == 0
    assert not backend_calls
