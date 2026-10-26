import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scientisttwo.testbed.adapter import PreparedDevelopment
from scientisttwo.testbed.catalog import FEEDBACK_TARGETS
from scientisttwo.testbed.preparation import prepare, prepare_plan, verify_prepared
from scientisttwo.testbed.readiness import readiness
from scientisttwo.testbed.validation import create_validation_spec


def raw_data(tmp_path, task="house-price", count=100):
    root = tmp_path / "source"
    root.mkdir()
    if task == "house-price":
        data = {"Id": [f"{i:04}" for i in range(count)], "LotArea": np.arange(count) + 100,
                "Neighborhood": ["A", "B"] * (count // 2), "SalePrice": np.arange(count) * 100 + 10000}
    elif task == "spaceship-titanic":
        data = {"PassengerId": [f"{i:04}_01" for i in range(count)],
                "HomePlanet": ["Earth", "Mars"] * (count // 2),
                "Age": np.arange(count) % 70, "Transported": [True, False] * (count // 2)}
    else:
        data = {"text_id": [f"essay-{i}" for i in range(count)],
                "full_text": [f"Essay number {i} with unique content." for i in range(count)],
                **{name: np.arange(count) % 4 + 1 for name in FEEDBACK_TARGETS}}
    pd.DataFrame(data).to_csv(root / "train.csv", index=False)
    return root


@pytest.mark.parametrize("task", ["house-price", "spaceship-titanic", "feedback"])
def test_real_preparation_deterministic_disjoint_and_protected(tmp_path, task):
    source = raw_data(tmp_path, task)
    output = tmp_path / "out" / task / "prepared"
    prepared = prepare(task, source, output, provenance="synthetic_fixture")
    manifest = verify_prepared(output)
    ids = manifest["split_metadata"]["ids"]
    if task == "feedback":
        assert (len(ids["train"]), len(ids["development"]), len(ids["final"])) == (78, 20, 2)
        assert manifest["split_metadata"]["policy"].startswith("upstream-seed42-shuffle-98/2")
    else:
        assert (len(ids["train"]), len(ids["development"]), len(ids["final"])) == (64, 16, 20)
        assert manifest["split_metadata"]["policy"].startswith("upstream-row-order-80/20")
    assert not set(ids["train"]) & set(ids["development"])
    assert not set(ids["train"]) & set(ids["final"])
    assert set(ids["screening_train"]) < set(ids["train"])
    again = prepare(task, source, tmp_path / "another", provenance="synthetic_fixture")
    assert prepared["manifest_sha256"] == again["manifest_sha256"]
    spec = create_validation_spec(output, image="local:test", level="full")
    adapter = PreparedDevelopment(spec.native, spec.development)
    adapter.validate(adapter.baseline())
    adapter.validate(adapter.control())
    stage = adapter.prepare("subset", tmp_path / "phase")
    evaluation = pd.read_csv(Path(stage["public"]) / "evaluation.csv")
    assert not set(adapter.recipe.targets) & set(evaluation)
    assert "labels" not in json.dumps(adapter.context())
    with pytest.raises(ValueError, match="Final evidence"):
        adapter.prepare("final", tmp_path / "bad")
    status = readiness(tmp_path / "out")
    assert status["assigned"] == 13 and status["ready_count"] == 0
    assert not status["full_audited_testbed_ready"]


def test_final_ids_preserved_and_labels_not_exposed(tmp_path):
    source = raw_data(tmp_path)
    pd.DataFrame({"Id": ["9000", "9001"], "LotArea": [300, 500], "Neighborhood": ["A", "B"]}).to_csv(source / "test.csv", index=False)
    labels = tmp_path / "answer.csv"
    pd.DataFrame({"Id": ["9001", "9000"], "SalePrice": [20000, 22000]}).to_csv(labels, index=False)
    output = tmp_path / "prepared"
    prepare("house-price", source, output, labels=labels, provenance="synthetic_fixture")
    final = pd.read_csv(output / "phases/final/public/evaluation.csv", dtype={"Id": str})
    assert final.Id.tolist() == ["9000", "9001"]
    assert "SalePrice" not in final
    assert "preserved-supplied-final" in verify_prepared(output)["split_metadata"]["policy"]


def test_unlabeled_competition_test_is_not_the_upstream_final_and_actual_requires_terms(tmp_path):
    source = raw_data(tmp_path)
    with pytest.raises(ValueError, match="terms"):
        prepare("house-price", source, tmp_path / "out", provenance="actual_data")
    pd.DataFrame({"Id": [9999]}).to_csv(source / "test.csv", index=False)
    value = prepare("house-price", source, tmp_path / "out", provenance="synthetic_fixture")
    assert "9999" not in value["split_metadata"]["ids"]["final"]
    assert value["split_metadata"]["ids"]["final"] == [f"{i:04}" for i in range(80, 100)]


def test_integrity_no_overwrite_symlinks_or_credentials(tmp_path):
    source = raw_data(tmp_path)
    output = tmp_path / "prepared"
    prepare("house-price", source, output, provenance="synthetic_fixture")
    with pytest.raises(ValueError, match="new"):
        prepare("house-price", source, output, provenance="synthetic_fixture")
    (output / "reference/train.py").write_text("print('altered')")
    with pytest.raises(ValueError, match="changed"):
        verify_prepared(output)
    (source / "kaggle.json").write_text("{}")
    with pytest.raises(ValueError, match="Credential"):
        prepare("house-price", source, tmp_path / "other", provenance="synthetic_fixture")


def test_parkinsons_patient_consistency_across_all_tables(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    clinical = pd.DataFrame([{"visit_id": f"{patient}_{month}", "patient_id": str(patient),
                             "visit_month": month, **{f"updrs_{i}": float(patient + i) for i in range(1, 5)}}
                            for patient in range(20) for month in (0, 6, 12, 24)])
    clinical.to_csv(source / "train_clinical_data.csv", index=False)
    for name in ("train_proteins.csv", "train_peptides.csv"):
        clinical[["patient_id", "visit_id", "visit_month"]].assign(abundance=1).to_csv(source / name, index=False)
    task = "amp-parkinsons-disease-progression-prediction"
    output = tmp_path / "prepared"
    prepare(task, source, output, provenance="synthetic_fixture")
    manifest = verify_prepared(output)
    for phase in ("screen", "full", "final"):
        public = output / "phases" / phase / "public"
        train = pd.read_csv(public / "train.csv", dtype={"patient_id": str})
        evaluation = pd.read_csv(public / "evaluation.csv", dtype={"patient_id": str})
        assert not set(train.patient_id) & set(evaluation.patient_id)
        for name in ("train_proteins.csv", "train_peptides.csv"):
            assert set(pd.read_csv(public / name, dtype={"patient_id": str}).patient_id) <= set(train.patient_id)
        assert "rating" not in evaluation
    assert manifest["split_metadata"]["patient_groups"]


def test_plan_reports_access_and_does_not_download():
    plan = prepare_plan("house-price")
    assert plan["live_calls"] is False
    assert "No automatic downloads" in plan["download_policy"]
    assert plan["recipe"]["access_requirements"]


def test_missing_every_task_still_reports_all_13(tmp_path):
    report = readiness(tmp_path)
    assert report["canonical_tasks"] == 13 and len(report["tasks"]) == 13
    assert all(not item["ready"] and item["blockers"] for item in report["tasks"])


def test_vectorization_workloads_use_held_out_final_seed(tmp_path, monkeypatch):
    source = tmp_path / "upstream"
    env = source / "MLAgentBench/benchmarks/vectorization/env"
    env.mkdir(parents=True)
    (env / "train.py").write_text("import numpy as np\nclass Conv2DLayer:\n    def forward(self, features_batch):\n        return features_batch\n")
    # Tests exercise the preparation contract; production independently verifies
    # the actual pinned Git checkout before importing the source.
    monkeypatch.setattr("scientisttwo.benchmarks.registry.verify_repository", lambda *args: None)
    output = tmp_path / "prepared"
    prepare("vectorization", source, output, provenance="synthetic_fixture")
    manifest = verify_prepared(output)
    assert manifest["split_metadata"]["final_seed"] != manifest["split_metadata"]["split_seed"]
    assert (output / "phases/full/public/inputs.npz").read_bytes() != (output / "phases/final/public/inputs.npz").read_bytes()
    config = create_validation_spec(output, image="test:numpy", level="full")
    adapter = PreparedDevelopment(config.native, config.development)
    assert config.development.metric == "latency_ratio"
    assert callable(adapter.evaluate_pair) and callable(adapter.evaluate_final)
    adapter.validate(adapter.control())
    prepared = adapter.prepare("final", tmp_path / "final-data")
    assert len(prepared["cases"]) == 3


def test_explicit_derived_final_does_not_silently_use_competition_test(tmp_path):
    source = raw_data(tmp_path)
    pd.DataFrame({"Id": ["test-only"], "LotArea": [1]}).to_csv(source / "test.csv", index=False)
    result = prepare("house-price", source, tmp_path / "prepared", provenance="synthetic_fixture", derive_final=True)
    assert "test-only" not in result["split_metadata"]["ids"]["final"]
    assert result["split_metadata"]["policy"].startswith("adapted-deterministic")
