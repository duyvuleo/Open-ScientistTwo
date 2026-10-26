import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scientisttwo.batch import run_batch, validate_manifest
from scientisttwo.batch_report import summarize, atomic_json, read_receipt
from scientisttwo.store import Store, IntegrityError
from scientisttwo.suite import make_suite, write_manifest, BatchManifest


def seal_receipt(root, entry, status="complete", outcome="no_improvement"):
    directory = Path(root) / "campaigns" / entry.id
    store = Store(directory)
    result = {
        "status": status,
        "outcome": outcome,
        "audit_pass": status == "complete",
        "experiment_jobs": 2,
        "audit_jobs": 1,
        "observed_split_hash": entry.split_manifest_hash,
        "execution_seconds": 1,
        "live_model_calls": 0,
    }
    store.set("batch_entry", entry.model_dump(mode="json"))
    store.set("batch_result_ref", store.put_json(result))
    atomic_json(directory / "batch-result.json", result)
    store.close()


def test_suite_has_30_tasks_90_campaigns_and_frozen_repetition_partitions():
    suite = make_suite()
    assert len(suite.campaigns) == 90
    assert len({c.task_id for c in suite.campaigns}) == 30
    for task_id in {c.task_id for c in suite.campaigns}:
        entries = [c for c in suite.campaigns if c.task_id == task_id]
        assert len({c.task_version for c in entries}) == 1
        assert len({c.split_manifest_hash for c in entries}) == 1
        assert len({tuple(c.task.seeds) for c in entries}) == 3
    validate_manifest(suite)


def test_manifest_detects_mutation_and_duplicate_assignment():
    suite = make_suite(smoke=True)
    payload = suite.model_dump(mode="json")
    payload["campaigns"].append(payload["campaigns"][0])
    with pytest.raises(ValidationError, match="unique"):
        BatchManifest.model_validate(payload)
    bad = suite.model_copy(update={"implementation": {"changed": True}})
    with pytest.raises(IntegrityError, match="Implementation changed"):
        validate_manifest(bad)


def test_summary_keeps_all_assigned_tasks_and_detects_receipt_tampering(tmp_path):
    suite = make_suite(smoke=True)
    seal_receipt(tmp_path, suite.campaigns[0])
    seal_receipt(tmp_path, suite.campaigns[1], status="error", outcome="error")
    summary = summarize(tmp_path, suite)
    assert summary["assigned_campaigns"] == 3 and summary["completed_campaigns"] == 1
    assert summary["status_counts"] == {"complete": 1, "error": 1, "pending": 1}
    path = tmp_path / "campaigns" / suite.campaigns[0].id / "batch-result.json"
    payload = json.loads(path.read_text())
    payload["outcome"] = "candidate_supported"
    path.write_text(json.dumps(payload))
    assert read_receipt(tmp_path, suite.campaigns[0])["status"] == "error"


def test_queue_resumes_without_duplicate_completed_campaigns(tmp_path, monkeypatch):
    suite = make_suite(smoke=True)
    manifest = tmp_path / "manifest-source.json"
    write_manifest(manifest, suite)
    root = tmp_path / "batch"
    launched = []

    def preflight(self):
        self.image_id = "sha256:fake-for-unit-test"

    monkeypatch.setattr("scientisttwo.batch.ContainerRunner.preflight", preflight)

    class Process:
        def __init__(self, argv, **kwargs):
            launched.append(argv[-1])
            entry = next(c for c in suite.campaigns if c.id == argv[-1])
            seal_receipt(root, entry)
            self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr("scientisttwo.batch.subprocess.Popen", Process)
    assert run_batch(manifest, root, max_campaigns=1)["completed_campaigns"] == 1
    assert run_batch(manifest, root)["completed_campaigns"] == 3
    assert run_batch(manifest, root)["completed_campaigns"] == 3
    assert len(launched) == len(set(launched)) == 3


def test_global_deadline_preserves_not_started_denominator(tmp_path, monkeypatch):
    suite = make_suite(smoke=True).model_copy(update={"wall_seconds": 1e-9})
    manifest = tmp_path / "m.json"
    write_manifest(manifest, suite)

    def preflight(self):
        self.image_id = "sha256:fake"

    monkeypatch.setattr("scientisttwo.batch.ContainerRunner.preflight", preflight)

    def forbidden(*args, **kwargs):
        pytest.fail("No worker may start after the batch deadline")

    monkeypatch.setattr("scientisttwo.batch.subprocess.Popen", forbidden)
    summary = run_batch(manifest, tmp_path / "batch")
    assert summary["completed_campaigns"] == 0
    assert summary["status_counts"] == {"not_started_budget_exhausted": 3}
