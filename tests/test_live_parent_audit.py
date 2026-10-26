"""Synthetic independent parent-patch audit, with no model or container calls."""

from copy import deepcopy

import pytest

from scientisttwo.live.audit import audit_live
from scientisttwo.live.models import Candidate
from scientisttwo.live.parent_audit import audit_parent_edits
from scientisttwo.live.parent_edits import ParentChangePlan
from scientisttwo.store import IntegrityError, canonical, digest
from test_live import candidate as synthetic_candidate, idea, setup
from scientisttwo.models import CriticDecision
from test_scientific_evidence_integration import evidence_cycle as staged_evidence_fixture

evidence_cycle = staged_evidence_fixture


def refine(controller, best):
    result, improved = controller.refine("meta/audited", best, {"evidence": []})
    assert result == best and not improved  # Scripted equal scores preserve incumbent.
    row = controller.store.db.execute(
        "SELECT output_ref FROM calls WHERE key='parent-edit/meta/audited/implement'").fetchone()
    return controller.store.json(row[0])


def replace_proof(store, changed):
    ref = store.put_json(changed, "synthetic-corrupted-proof")
    with store.db:
        store.db.execute("UPDATE calls SET output_ref=? WHERE key='parent-edit/meta/audited/implement'", (ref,))


def test_parent_audit_replays_accepted_patch_and_binds_actual_child_source_read_only(evidence_cycle):
    controller, store, _, _, best = evidence_cycle
    proof = refine(controller, best)
    before = list(store.db.iterdump())
    checks = audit_parent_edits(store, controller.adapter)
    assert list(store.db.iterdump()) == before  # All patch replay occurs in scratch storage.
    assert len(checks) == 1
    check = checks[0]
    assert check["status"] == "pass" and check["executed_child"] == "recorded"
    assert check["source_fidelity"] == "replayed_exact_parent_patch"
    assert check["parent_binding"] == "bound_to_executed_parent_source"
    assert check["observations"][0]["id"] == "meta/audited"
    assert check["observations"][0]["lineage"] == "parent_recorded"
    assert check["semantic_intent"] == "unverified"
    child = Candidate.model_validate(store.json(proof["candidate_ref"]))
    assert "SYNTHETIC_REVISION" in child.files[1].content
    assert child.files[0] == Candidate.model_validate(best["candidate"]).files[0]
    # The public audit hook must perform the same source proof check.
    result = audit_live(store.root)
    assert result["status"] == "pass", result
    assert any(item.get("parent_edit") == "meta/audited/implement" for item in result["checks"])
    assert result["method_code_consistency"] == "unverified"


@pytest.mark.parametrize("field", ["parent_source_ref", "candidate_source_ref", "response_ref", "declared_paths", "source_hashes"])
def test_new_registered_proof_tampering_is_rejected_even_if_artifact_hashes_pass(evidence_cycle, field):
    controller, store, _, _, best = evidence_cycle
    proof = refine(controller, best)
    changed = deepcopy(proof)
    if field == "parent_source_ref":
        changed[field] = proof["candidate_source_ref"]
    elif field == "candidate_source_ref":
        changed[field] = proof["parent_source_ref"]
    elif field == "response_ref":
        changed[field] = proof["candidate_ref"]
    elif field == "declared_paths":
        changed[field] = ["train.py"]
    else:
        changed[field]["train.py"]["after"] = "0" * 64
    replace_proof(store, changed)
    assert store.verify() == []  # Content-addressed storage alone is insufficient.
    with pytest.raises(IntegrityError):
        audit_parent_edits(store, controller.adapter)
    assert audit_live(store.root)["status"] == "fail"


def test_forged_child_candidate_hash_cannot_override_replayed_patch(evidence_cycle):
    controller, store, _, _, best = evidence_cycle
    proof = refine(controller, best)
    wrong = Candidate.model_validate(store.json(proof["candidate_ref"]))
    wrong.files[0].content += "\n# UNDECLARED_INITIALIZATION_CHANGE\n"
    proof["candidate_ref"] = store.put_json(wrong)
    proof["candidate_proposal_sha256"] = digest(canonical(wrong))
    replace_proof(store, proof)
    with pytest.raises(IntegrityError, match="recorded child candidate"):
        audit_parent_edits(store, controller.adapter)


def test_accepted_correction_uses_actual_corrected_response_and_request(evidence_cycle):
    controller, store, backend, _, best = evidence_cycle
    controller.config.agent_output_retries = 1
    original = backend.responder

    def respond(role, inputs):
        response = original(role, inputs)
        if role == "implement" and "parent" in inputs and "output_correction" not in inputs:
            response["parent_source_ref"] = "artifacts/" + "0" * 64 + ".json"
        return response

    backend.responder = respond
    proof = refine(controller, best)
    assert proof["response_key"] == "meta/audited/implement/output-correction/1"
    assert proof["request_ref"] == store.get("request/" + proof["response_key"])
    assert store.json(proof["request_ref"])["inputs"]["output_correction"]["attempt"] == 1
    assert audit_parent_edits(store, controller.adapter)[0]["status"] == "pass"
    proof["response_key"] = "meta/audited/implement"
    replace_proof(store, proof)
    with pytest.raises(IntegrityError, match="accepted agent response"):
        audit_parent_edits(store, controller.adapter)


def test_proposed_patch_is_not_presented_as_executed(evidence_cycle):
    controller, store, _, _, best = evidence_cycle
    controller.implement_change("only-proposed/implement", "implement", idea("proposal"), best,
        ParentChangePlan(changed_files=["custom.py"], reason="Synthetic proposed edit"))
    result = audit_parent_edits(store, controller.adapter)
    assert len(result) == 1 and result[0]["status"] == "pass"
    assert result[0]["executed_child"] == "proposed_not_executed"
    assert result[0]["observations"] == []


def test_child_trial_requires_the_declared_parent_lineage(evidence_cycle):
    controller, store, _, _, best = evidence_cycle
    refine(controller, best)
    row = store.db.execute("SELECT output_ref FROM calls WHERE key='live-trial/meta/audited'").fetchone()
    trial = store.json(row[0])
    trial["parents"] = []
    ref = store.put_json(trial)
    with store.db:
        store.db.execute("UPDATE calls SET output_ref=? WHERE key='live-trial/meta/audited'", (ref,))
    with pytest.raises(IntegrityError, match="parent lineage"):
        audit_parent_edits(store, controller.adapter)


def test_absent_parent_proofs_make_no_source_fidelity_claim(evidence_cycle):
    controller, store, _, _, _ = evidence_cycle
    assert audit_parent_edits(store, controller.adapter) == []


def test_materialization_without_completion_is_unknown_not_unexecuted(evidence_cycle):
    controller, store, _, _, best = evidence_cycle
    refine(controller, best)
    # Isolated fixture simulates loss of the completion checkpoint; audit must
    # neither rerun it nor describe the candidate as definitely not executed.
    with store.db:
        store.db.execute("DELETE FROM calls WHERE key='development/meta/audited'")
    before = list(store.db.iterdump())
    result = audit_parent_edits(store, controller.adapter)[0]
    assert list(store.db.iterdump()) == before
    assert result["executed_child"] == "completion_unknown"
    assert result["observations"][0]["status"] == "completion_unknown"


def test_synthetic_observation_without_source_receipt_remains_unverified(tmp_path):
    controller, store, _, runner = setup(tmp_path)
    try:
        store.set("live_state", {"stage": "search", "status": "running"})
        parent = synthetic_candidate(8)
        observation = runner.execute("parent", parent)
        best = controller.record("parent", idea("parent"), parent, observation,
                                 CriticDecision(decision="Good", reason="Synthetic parent"))
        controller.set_best(best)
        controller.refine("revision", best, {"evidence": []})
        check = audit_parent_edits(store, controller.adapter)[0]
        assert check["source_fidelity"] == "replayed_exact_parent_patch"
        assert check["parent_binding"].endswith("executed_source_unverified")
        assert check["executed_child"] == "source_binding_unverified"
        assert check["observations"][0]["source_binding"] == "unverified"
    finally:
        store.close()
