"""Pilot batch receipt presentation with explicit synthetic scientific fixtures."""
from copy import deepcopy

import pytest

from scientisttwo.live.evidence_context import build_evidence_context
from scientisttwo.models import Profile, TaskSpec
from scientisttwo.research.pilot_bridge import create_pilot_campaign
from scientisttwo.store import Store, IntegrityError, canonical

from test_research_pilot_bridge import SyntheticContainer


def evidence(tmp_path, *, phase="full", status="success"):
    store = Store(tmp_path)
    worker = SyntheticContainer(store)
    worker.status = status
    _, adapter, runner = create_pilot_campaign(store, TaskSpec(), Profile(), container_runner=worker)
    result = runner.execute("candidate/" + phase, adapter.baseline(), phase)
    return store, worker, result


def replace_observation(store, result):
    ref = store.put_json(result)
    with store.db:
        store.db.execute("UPDATE calls SET output_ref=? WHERE key=?", (ref, "development/" + result.id))
        store.db.execute("UPDATE live_operations SET result_ref=? WHERE key=?", (ref, "experiment/" + result.id))


def test_one_batch_completion_is_truthfully_shared_across_all_seed_cells(tmp_path, monkeypatch):
    store, worker, result = evidence(tmp_path)
    original = store.json(result.legacy_result_ref)
    before = list(store.db.iterdump())
    read = store.read

    def forbid_raw_payloads(ref):
        assert ref not in {original["observations_ref"], original["logs_ref"]}
        return read(ref)

    monkeypatch.setattr(store, "read", forbid_raw_payloads)
    package = build_evidence_context(store, [result])
    executions = package["observations"][0]["training_executions"]
    assert len(executions) == 3 and [row["seed"] for row in executions] == [0, 1, 2]
    assert len({row["execution_id"] for row in executions}) == 1
    assert len({row["receipt_ref"] for row in executions}) == 1
    assert all(row["shared_batch_receipt"] and not row["separate_container_per_seed"] for row in executions)
    assert all(row["metric_grid_complete"] for row in executions)
    assert all(row["status"] == "synthetic_test_recorded" and row["measurement_origin"] == "synthetic_test" for row in executions)
    assert all("not established" in row["statistical_independence"] for row in executions)
    assert "independent execution binding" not in canonical(package).decode()
    assert "No training receipt" not in canonical(package).decode()
    assert str(tmp_path) not in canonical(package).decode()
    assert list(store.db.iterdump()) == before
    assert len(worker.executions) == 1


def test_legacy_batch_without_new_snapshot_retains_truthful_ledger_binding(tmp_path):
    store, _, result = evidence(tmp_path, phase="subset")
    receipt = store.json(result.log_refs[0])
    receipt.pop("legacy_completion_ref")
    result.log_refs = [store.put_json(receipt)]
    replace_observation(store, result)
    execution = build_evidence_context(store, [result])["observations"][0]["training_executions"][0]
    assert execution["kind"] == "pilot_batch" and execution["completion_ref"] is None
    assert execution["grid_seeds"] == [0] and execution["shared_batch_receipt"]
    assert execution["status"] == "synthetic_test_recorded"


@pytest.mark.parametrize("damage", ["job_status", "job_result", "job_request", "job_charge", "shared_request",
    "receipt_seed", "receipt_source", "receipt_job", "frozen_completion", "prediction", "duplicate_receipt"])
def test_broken_batch_bindings_are_integrity_failures_not_silent_missing_receipts(tmp_path, damage):
    store, _, result = evidence(tmp_path)
    receipt = store.json(result.log_refs[0])
    legacy_key = receipt["legacy_job_key"]
    if damage.startswith("job_"):
        field, value = {"job_status": ("status", "failed"), "job_result": ("result_ref", None),
                        "job_request": ("request_hash", "not-a-hash"), "job_charge": ("charged", 99)}[damage]
        store.db.execute("UPDATE jobs SET " + field + "=? WHERE key=?", (value, legacy_key))
        store.db.commit()
    elif damage == "shared_request":
        store.db.execute("UPDATE live_operations SET request_hash=? WHERE key=?", ("0" * 64, "experiment/" + result.id))
        store.db.commit()
    else:
        if damage == "receipt_seed":
            receipt["seeds"] = [7, 8, 9]
        elif damage == "receipt_source":
            receipt["source_tree_ref"] = result.code_ref
        elif damage == "receipt_job":
            receipt["legacy_job_key"] += "/different"
        elif damage == "frozen_completion":
            completion = deepcopy(store.json(receipt["legacy_completion_ref"]))
            completion["worker_source_sha256"] = "0" * 64
            receipt["legacy_completion_ref"] = store.put_json(completion)
        elif damage == "prediction":
            selection = store.json(result.prediction_refs[0])
            result.prediction_refs[0] = store.put_json({**selection, "seed": 99})
        result.log_refs = [store.put_json(receipt)]
        if damage == "duplicate_receipt":
            result.log_refs *= 2
        replace_observation(store, result)
    with pytest.raises(IntegrityError):
        build_evidence_context(store, [result])


@pytest.mark.parametrize("status", ["execution_failed", "timeout"])
def test_failed_batch_does_not_gain_successful_scientific_evidence(tmp_path, status):
    store, _, result = evidence(tmp_path, status=status)
    presented = build_evidence_context(store, [result])["observations"][0]
    assert presented["status"] == status and presented["per_seed_metrics"] == []
    assert all(row["measurement_origin"] == "synthetic_test" for row in presented["training_executions"])
    assert not any(row["status"] == "recorded_success" for row in presented["training_executions"])


def test_duplicate_metric_cell_cannot_be_presented_as_a_complete_grid(tmp_path):
    store, _, result = evidence(tmp_path)
    original = store.json(result.legacy_result_ref)
    original["rows"].append(deepcopy(original["rows"][0]))
    result.legacy_result_ref = store.put_json(original)
    result.metric_rows = original["rows"]
    replace_observation(store, result)
    with pytest.raises(IntegrityError, match="Invalid authoritative"):
        build_evidence_context(store, [result])


def test_aggregate_tampering_cannot_override_bound_per_cell_scores(tmp_path):
    store, _, result = evidence(tmp_path)
    result.metrics = {key: .25 for key in result.metrics}
    result.metric_values = {key: [.25] * 3 for key in result.metric_values}
    result.value, result.values = .25, [.25] * 3
    replace_observation(store, result)
    with pytest.raises(IntegrityError, match="aggregate differs"):
        build_evidence_context(store, [result])
