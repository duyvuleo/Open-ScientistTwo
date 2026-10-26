"""Synthetic adapter seams; no actual CIFAR data, CUDA runs or model calls."""

import json
from types import SimpleNamespace

import pytest

from scientisttwo.benchmarks.files import tree_manifest
from scientisttwo.benchmarks.models import NativeResources
from scientisttwo.live.models import DevelopmentSpec
from scientisttwo.studies import cifar_adapter
from scientisttwo.studies.cifar_adapter import CifarRecipeDevelopment, prepare_study, study_specs, verify_study
from scientisttwo.studies.cifar_recipe import VARIANT, baseline_candidate, recipe_candidate

SNAPSHOT = "artifacts/" + "a" * 64 + ".json"


def synthetic_import_fixture(tmp_path, monkeypatch):
    """Mock only the prior actual-data verification boundary, with obvious tiny text fixtures."""
    source = tmp_path / "original-prepared"
    for phase in ("screen", "full", "final"):
        for name in ("public", "private"):
            directory = source / "phases" / phase / name
            directory.mkdir(parents=True)
            (directory / "synthetic-fixture.txt").write_text(
                "SYNTHETIC_FINAL_LABEL_SENTINEL" if phase == "final" else "synthetic " + phase + name)
    original = {
        "task_id": "cifar10", "provenance": "actual_data",
        "dataset_identity": {"status": "verified", "kind": "official-published-archive-and-member-hashes",
                             "training_rows": 50_000, "final_rows": 10_000},
        "split_metadata": {"original_training_rows": 50_000, "original_final_rows": 10_000,
                           "normalization": "float32 NCHW: uint8 / 127.5 - 1"},
        "files": tree_manifest(source),
    }
    validation = {"status": "pass", "counts": {
        phase: {"training": rows, "evaluation": 10_000}
        for phase, rows in {"screen": 20_000, "full": 40_000, "final": 50_000}.items()
    }}
    destination = tmp_path / "study"
    with monkeypatch.context() as patch:
        patch.setattr(cifar_adapter, "verify_prepared", lambda path: original)
        patch.setattr(cifar_adapter, "get_family", lambda name: SimpleNamespace(
            validate_prepared=lambda root, metadata: validation))
        manifest = prepare_study(source, destination)
    return source, destination, original, manifest


def adapter(root):
    native = SimpleNamespace(image="sha256:" + "1" * 64, grader_image="sha256:" + "2" * 64,
                             resources=NativeResources(gpus="device=0"))
    config = DevelopmentSpec(adapter="cifar_recipe", metric="accuracy", direction="maximize",
                             seeds=[0, 1, 2], settings={"prepared_root": str(root)})
    result = CifarRecipeDevelopment(native, config)
    result.bind_source(SNAPSHOT)
    return result


def test_new_study_copies_data_and_does_not_inherit_execution_readiness(tmp_path, monkeypatch):
    source, root, original, manifest = synthetic_import_fixture(tmp_path, monkeypatch)
    assert tree_manifest(source) == original["files"]
    assert verify_study(root) == manifest
    assert manifest["variant"] == VARIANT
    assert manifest["runtime_validation"] == manifest["execution_validation"] == "unverified"
    assert all(not path.is_symlink() for path in root.rglob("*"))
    assert not adapter(root).live_readiness()["ready"]
    with pytest.raises(ValueError, match="new and disjoint"):
        prepare_study(source, root)


def test_study_refuses_unverified_or_incomplete_data():
    with pytest.raises(ValueError, match="independently verified"):
        cifar_adapter._check_official_identity({"task_id": "cifar10", "provenance": "synthetic_fixture"})
    with pytest.raises(ValueError, match="fixed 20000/40000/50000"):
        cifar_adapter._check_counts({"status": "pass", "counts": {"full": {"training": 100, "evaluation": 10}}})


def test_frozen_parent_partition_changes_are_detected(tmp_path, monkeypatch):
    _, root, _, _ = synthetic_import_fixture(tmp_path, monkeypatch)
    (root / "phases/full/public/synthetic-fixture.txt").write_text("modified")
    with pytest.raises(ValueError, match="changed after freezing"):
        verify_study(root)


def test_adapter_exposes_only_fixed_model_recipe_and_development_data(tmp_path, monkeypatch):
    _, root, _, _ = synthetic_import_fixture(tmp_path, monkeypatch)
    integration = adapter(root)
    context = integration.context()
    assert context["editable_files"] == ["training_config.json"]
    assert "ResNet18" in context["study_contract"]["architecture"]
    assert "weights=None" in context["study_contract"]["fixed_sources"]["model.py"]
    assert "SYNTHETIC_FINAL_LABEL_SENTINEL" not in json.dumps(context)
    prepared = integration.prepare("full", tmp_path / "development")
    assert prepared["recipe"] == integration.recipe.identity
    assert integration.prepare("full", tmp_path / "development") == prepared
    with pytest.raises(ValueError, match="Final evidence"):
        integration.prepare("final", tmp_path / "forbidden")
    integration.validate(baseline_candidate(SNAPSHOT))
    integration.validate(recipe_candidate(SNAPSHOT, {"learning_rate": 0.03}, description="Fixture change"))
    assert any("studies/cifar_adapter.py" in name for name in integration.source_files())


def test_native_study_spec_uses_isolated_final_evaluation_and_matching_development_seeds(tmp_path, monkeypatch):
    _, root, _, _ = synthetic_import_fixture(tmp_path, monkeypatch)
    native, development = study_specs(root, image="sha256:" + "1" * 64,
                                      grader_image="sha256:" + "2" * 64,
                                      resources=NativeResources(gpus="device=0"))
    assert native.public_dir == root / "phases/final/public"
    assert native.evaluation_pipeline.private_dir == root / "phases/final/private"
    assert "/data/private" not in " ".join(native.evaluation_pipeline.inference.command)
    assert "/candidate/model.py" not in native.evaluation_pipeline.score.command
    assert native.evaluation_pipeline.score.resources.gpus is None
    assert development.adapter == "cifar_recipe" and development.seeds == [0, 1, 2]
    assert native.candidate_manifest["model.py"] == tree_manifest(root / "reference")["model.py"]


def test_diagnostics_cannot_consume_final_partition(tmp_path, monkeypatch):
    _, root, _, _ = synthetic_import_fixture(tmp_path, monkeypatch)
    integration = adapter(root)
    prepared = integration.prepare("full", tmp_path / "development")
    contract = integration.diagnostics_contract("classification", "full", prepared=prepared)
    assert contract["contract"]["format"] == "class_scores_npz"
    with pytest.raises(ValueError, match="non-final"):
        integration.diagnostics_contract("classification", "final")
    prepared["private"] = str(root / "phases/final/private")
    with pytest.raises(ValueError, match="differ"):
        integration.diagnostics_contract("classification", "full", prepared=prepared)


def test_calibrated_reference_rebinds_only_configuration_to_current_snapshot(tmp_path, monkeypatch):
    _, root, _, _ = synthetic_import_fixture(tmp_path, monkeypatch)
    integration = adapter(root)
    integration.config.settings["calibration_root"] = str(tmp_path / "synthetic-calibration")
    monkeypatch.setattr(integration, "_calibration", lambda: {
        "selected_overrides": {"learning_rate": 0.03},
        "selected_candidate": {"base_snapshot_ref": "artifacts/" + "b" * 64 + ".json"},
    })
    candidate = integration.baseline()
    assert candidate.base_snapshot_ref == SNAPSHOT
    assert json.loads(candidate.files[0].content) == {"learning_rate": 0.03}
    integration.validate(candidate)


def test_calibration_paths_and_receipts_do_not_create_evidence_on_readiness_check(tmp_path, monkeypatch):
    _, root, _, _ = synthetic_import_fixture(tmp_path, monkeypatch)
    integration = adapter(root)
    absent = tmp_path / "absent-calibration"
    integration.config.settings["calibration_root"] = str(absent)
    result = integration.live_readiness()
    assert not result["ready"] and not absent.exists()
    assert "existing campaign" in result["errors"][0]
