import pytest

from scientisttwo.store import Store
from scientisttwo.testbed.readiness import _verify_audit_proof


def test_requested_flag_without_actual_replay_cannot_pass(tmp_path):
    store = Store(tmp_path)
    try:
        with pytest.raises(ValueError, match="did not complete"):
            _verify_audit_proof(store, {"container_evaluator_reexecution": "requested"}, {})
        with pytest.raises(ValueError, match="at least one"):
            _verify_audit_proof(store, {"complete": True}, {})
        with pytest.raises(ValueError, match="every required"):
            _verify_audit_proof(store, {"complete": True}, {"required_container_replays": ["final/reference"]})
    finally:
        store.close()


def test_structural_replay_proof_binds_source_grade_and_execution(tmp_path):
    # Explicit synthetic proof-contract test; no runtime or benchmark readiness claim.
    store = Store(tmp_path)
    metrics = {"mae": 1.0}
    grade = store.put_json({"valid_submission": True, "metrics": metrics})
    execution = store.put_json({"returncode": 0, "timed_out": False,
                                "log_ref": store.put(b"SYNTHETIC proof fixture", "log"),
                                "job_ref": store.put_json({"synthetic": True})})
    proof = {"verification_kind": "container_evaluator_replay", "status": "pass",
             "id": "final/reference", "source_grade_ref": "original-grade", "metrics": metrics,
             "execution_refs": [execution], "replayed_grade_ref": grade}
    saved = {"complete": True, "container_replays": [{"id": proof["id"], "metrics": metrics,
                                                       "evidence_ref": store.put_json(proof)}]}
    current = {"required_container_replays": [proof["id"]],
               "replay_sources": {proof["id"]: {"grade_ref": "original-grade", "metrics": metrics}}}
    try:
        _verify_audit_proof(store, saved, current)
        altered = {**proof, "source_grade_ref": "unrelated-grade"}
        saved["container_replays"][0]["evidence_ref"] = store.put_json(altered)
        with pytest.raises(ValueError, match="differs"):
            _verify_audit_proof(store, saved, current)
    finally:
        store.close()
