"""Synthetic receipts prove presentation provenance, not actual GPU execution."""

import json
from pathlib import Path

import pytest

from scientisttwo.artifacts import ArtifactSpec, freeze_bundle, load_bundle
from scientisttwo.live.models import Candidate, Observation, SourceFile
from scientisttwo.live.source_tree import candidate_snapshot, freeze_source_tree
from scientisttwo.research.execution_context import build_execution_context, _worker_defaults
from scientisttwo.store import IntegrityError, Store, canonical, digest


def context_fixture(tmp_path, *, overrides=None, metadata=True):
    store = Store(tmp_path / "campaign")
    source = tmp_path / "source"
    source.mkdir()
    original = Path(__file__).parents[1] / "src/scientisttwo/task_families/templates"
    (source / "family_worker.py").write_text((original / "family_worker.py").read_text())
    (source / "train.py").write_text("command = ['python', 'family_worker.py', '--task', 'cifar10']\n")
    (source / "model.py").write_text("# Synthetic model; no execution\n")
    (source / "transforms.py").write_text("# Synthetic transform; no execution\n")
    (source / "training.json").write_text(json.dumps({"epochs": 5, "max_epochs": 10, "max_steps": 1000, "batch_size": 128, "seed": 0}))
    (source / "training_config.json").write_text("{}")
    base = freeze_source_tree(store, source, ["model.py", "transforms.py", "training_config.json"])
    candidate = Candidate(base_snapshot_ref=base, files=[SourceFile(path="training_config.json", content=json.dumps(overrides or {}))],
                          command=["python", "train.py"], description="Synthetic context fixture")
    tree = candidate_snapshot(store, candidate)
    tree_ref, code_ref = store.put_json(tree), store.put_json(candidate)
    protocol = {"phase": "full", "seeds": [4]}
    receipt = store.put_json({"stage": "dev-" + digest(b"trial4")[:24], "job_ref": store.put_json({"seed": 4}),
                             "resources": {"cpus": 8, "memory_mb": 65536}, "image_id": "sha256:synthetic",
                             "returncode": 0, "timed_out": False, "seconds": 12})
    checkpoint = tmp_path / "output/checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.bin").write_bytes(b"MUST_NOT_READ_MODEL_BINARY")
    (checkpoint / "config.json").write_text(json.dumps({"epochs": 5, "batch_size": 128, **(overrides or {}), "seed": 4}))
    if metadata:
        (checkpoint / "training.json").write_text(json.dumps({"task": "cifar10", "seed": 4,
            "steps": 1000, "epochs_started": 4, "training_seconds": 10.1, "initialization": "declared_reference"}))
    bundle = freeze_bundle(store, checkpoint.parent, [ArtifactSpec(path="checkpoint", kind="directory", role="checkpoint")])
    result = Observation(id="trial", phase="full", status="success", metric="accuracy", value=.7, values=[.7],
                         protocol_hash=digest(canonical(protocol)), protocol_ref=store.put_json(protocol), code_ref=code_ref,
                         source_tree_ref=tree_ref, log_refs=[receipt], bundle_refs=[bundle], prediction_refs=[store.put_json({"synthetic": True})])
    store.cache("development/trial", {}, result)
    store.cache("materialization/development/trial", {}, {"source_tree_ref": tree_ref, "code_ref": code_ref})
    return store, result


def test_empty_override_reports_verified_source_defaults_and_realized_epochs_started(tmp_path):
    store, result = context_fixture(tmp_path)
    try:
        item = build_execution_context(store, [result])["observations"][0]
        execution = item["executions"][0]
        assert item["candidate_overrides"] == {}
        assert execution["configured"]["epochs"] == 5
        assert execution["configured"]["optimizer"] == "sgd"
        assert execution["configured"]["learning_rate"] == .1
        assert execution["configured"]["momentum"] == .9
        assert execution["configured"]["weight_decay"] == 0
        assert execution["configured"]["seed"] == 4
        assert execution["realized"]["epochs_started"] == 4
        assert "epochs_completed" in execution["unknown_realization"]
        assert execution["resource_caps"]["memory_mb"] == 65536
        assert execution["initialization_source_hash"] and execution["augmentation_source_hash"]
        assert "unverified" == item["source"]["method_isolation"]
    finally:
        store.close()


def test_candidate_overrides_are_visible_separately_from_equal_permitted_caps(tmp_path):
    store, result = context_fixture(tmp_path, overrides={"epochs": 8, "optimizer": "adamw", "weight_decay": .001})
    try:
        item = build_execution_context(store, [result])["observations"][0]
        assert item["base_configuration"]["epochs"] == 5
        assert item["candidate_overrides"]["epochs"] == 8
        assert item["executions"][0]["configured"]["epochs"] == 8
        assert item["executions"][0]["configured"]["optimizer"] == "adamw"
    finally:
        store.close()


def test_missing_metadata_is_unknown_and_does_not_invent_realization(tmp_path):
    store, result = context_fixture(tmp_path, metadata=False)
    try:
        execution = build_execution_context(store, [result])["observations"][0]["executions"][0]
        assert execution["realized"] == {}
        assert execution["configured"]["optimizer"] == "sgd"
        assert "steps" in execution["unknown_realization"]
    finally:
        store.close()


def test_context_does_not_read_binary_labels_logs_or_expose_paths_and_does_not_mutate(tmp_path, monkeypatch):
    store, result = context_fixture(tmp_path)
    try:
        before = list(store.db.iterdump())
        original = store.read
        def read(ref):
            assert not ref.endswith((".bin", ".log", ".npy", ".npz"))
            return original(ref)
        monkeypatch.setattr(store, "read", read)
        final = result.model_copy(update={"id": "FINAL_SECRET", "phase": "final", "source_tree_ref": "protected/missing"})
        text = canonical(build_execution_context(store, [result, final])).decode()
        assert "FINAL_SECRET" not in text and str(tmp_path) not in text
        assert list(store.db.iterdump()) == before
    finally:
        store.close()


@pytest.mark.parametrize("damage", ["offered", "protocol", "source", "receipt", "metadata"])
def test_modified_evidence_fails_instead_of_becoming_unknown(tmp_path, damage):
    store, result = context_fixture(tmp_path)
    try:
        if damage == "offered":
            result.reason = "tampered"
        else:
            ref = {"protocol": result.protocol_ref, "source": result.source_tree_ref, "receipt": result.log_refs[0]}.get(damage)
            if damage == "metadata":
                ref = load_bundle(store, result.bundle_refs[0], verify=False).files["checkpoint/training.json"].ref
            (store.root / ref).write_text("{}")
        with pytest.raises(IntegrityError):
            build_execution_context(store, [result])
    finally:
        store.close()


def test_literal_default_resolution_does_not_execute_calls_and_omits_ambiguity():
    text = '''
def train():
    dangerous = config.get("learning_rate", __import__("os").system("exit 99"))
    one = config.get("dropout", .1)
    two = config.get("dropout", .2)
    optimizer = config.get("optimizer", "sgd" if task == "cifar10" else "adam")
'''
    assert _worker_defaults(text, "cifar10") == {"optimizer": "sgd"}
    assert _worker_defaults(text, "imdb") == {"optimizer": "adam"}
