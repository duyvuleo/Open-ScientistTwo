"""Resolved delivery budget contracts with saved synthetic responses only."""
from copy import deepcopy
import time

import pytest

from scientisttwo import continuation as core
from scientisttwo import continuation_delivery as delivery
from scientisttwo import continuation_delivery_gate as gate
from scientisttwo.live.runtime import AmbiguousCall, LiveBudgetExhausted
from scientisttwo.store import IntegrityError, canonical, digest

from test_continuation import allowance
from test_continuation_delivery import saved_response


def resolved_campaign(tmp_path, *, calls=40, tokens=700000):
    owner, original_failed, ask, proof = saved_response(tmp_path)
    store = owner.store
    owner.config.token_stop_threshold = 1_500_000
    owner.config.max_calls = 120
    store.set("live_spec_ref", store.put_json(owner.config))
    plan_ref = core.prepare_plan(store, original_failed, "resolved-retry")
    admission_ref = core.admit_continuation(store, plan_ref,
        allowance(plan_ref, max_new_calls=calls, token_stop_threshold=tokens))
    original = core.install_continuation(owner, plan_ref, admission_ref)
    request = store.json(store.get("request/" + ask))
    original.reserve(original_failed, "model", request)
    original.finish(original_failed, "failed", {"error": "Codex invocation timed out; its process group was cancelled"}, 300)
    retry = store.json(plan_ref)["replacement_operation"]
    # Synthetic local completion occurred after the continuation admission and
    # before its unchanged request deadline. No model was actually invoked.
    now = store.json(admission_ref)["admitted_at"]
    queued = proof["claim"]["request"]
    queued.update(created_at=now + .001, deadline=now + 100)
    proof["claim"]["claimed_at"] = now + .002
    proof["dispatch"].update(started_at=now + .003, completed_at=now + .004)
    request_hash = digest(canonical(queued))
    proof["claim"]["request_sha256"] = proof["dispatch"]["request_sha256"] = proof["response"]["request_sha256"] = request_hash
    delivery_plan = delivery.plan_reconciliation(store, retry, ask, proof)
    receipt = delivery.reconcile_response(store, delivery_plan)
    return owner, original_failed, retry, ask, request, plan_ref, admission_ref, receipt


def test_only_valid_reconciliation_unlocks_next_request_without_repeating_outline_or_training(tmp_path):
    owner, failed, retry, ask, request, plan, admission, receipt = resolved_campaign(tmp_path)
    store = owner.store
    historical = [dict(row) for row in store.db.execute("SELECT * FROM live_operations")]
    experiments = [row for row in historical if row["kind"] == "experiment"]
    with pytest.raises((AmbiguousCall, IntegrityError)):
        owner.budget.reserve("model/next-writer/0", "model", {"role": "paperorchestra_model"})
    budget = gate.install_delivery_gate(owner, plan, admission, receipt)
    assert budget is owner.runner.budget
    assert budget.reserve(failed, "model", request)["status"] == "success"
    assert budget.reserve(retry, "model", request)["status"] == "success"
    assert [dict(row) for row in store.db.execute("SELECT * FROM live_operations")] == historical
    assert budget.reserve("experiment/baseline/full", "experiment", {"synthetic": True})["status"] == "success"
    budget.reserve("model/next-writer/0", "model", {"role": "paperorchestra_model"})
    budget.finish("model/next-writer/0", "success", {"synthetic": True}, 2, 100)
    report = gate.audit_delivery_gate(store, budget.binding_ref)
    assert report["integrity"] == "pass", report
    assert report["new_model_operations"] == 3 and report["administrative_model_operations"] == 1
    assert report["new_observed_tokens"] == 32478 and report["new_experiments"] == 0
    assert not report["original_total_token_ceiling_verified"]
    assert [dict(row) for row in store.db.execute("SELECT * FROM live_operations WHERE kind='experiment'")] == experiments
    assert store.db.execute("SELECT status FROM live_operations WHERE key=?", (failed,)).fetchone()[0] == "failed"
    assert store.db.execute("SELECT status FROM live_operations WHERE key=?", (retry,)).fetchone()[0] == "failed"
    assert gate.install_delivery_gate(owner, plan, admission, receipt) is budget


def test_expired_original_deadline_blocks_new_work_even_with_a_valid_receipt(tmp_path, monkeypatch):
    owner, failed, _, _, request, plan, admission, receipt = resolved_campaign(tmp_path)
    deadline = owner.store.get("live_deadline")
    budget = gate.install_delivery_gate(owner, plan, admission, receipt)
    monkeypatch.setattr(time, "time", lambda: deadline + 100)
    assert budget.reserve(failed, "model", request)["status"] == "success"  # Cached, not dispatched.
    with pytest.raises(LiveBudgetExhausted, match="wall-clock"):
        budget.reserve("model/after-deadline", "model", {})
    assert owner.store.get("live_deadline") == deadline


@pytest.mark.parametrize("calls,tokens", [(2, 700000), (40, 32378)])
def test_original_segment_limits_cannot_reset_after_reconciliation(tmp_path, calls, tokens):
    owner, _, _, _, _, plan, admission, receipt = resolved_campaign(tmp_path, calls=calls, tokens=tokens)
    budget = gate.install_delivery_gate(owner, plan, admission, receipt)
    with pytest.raises(LiveBudgetExhausted, match="segment ceiling"):
        budget.reserve("model/new", "model", {})


@pytest.mark.parametrize("mutation", ["receipt_ref", "proof", "source", "config", "binding"])
def test_changed_binding_or_source_rejects_before_new_dispatch(tmp_path, monkeypatch, mutation):
    owner, _, _, _, _, plan, admission, receipt = resolved_campaign(tmp_path)
    store = owner.store
    if mutation == "receipt_ref":
        with pytest.raises(IntegrityError):
            gate.install_delivery_gate(owner, plan, admission, store.put_json({"wrong": "receipt"}))
        return
    budget = gate.install_delivery_gate(owner, plan, admission, receipt)
    prior = len(budget.summary())
    if mutation == "proof":
        delivery_plan = store.json(store.json(receipt)["plan_ref"])
        (store.root / delivery_plan["proof_ref"]).write_text("changed")
    elif mutation == "source":
        monkeypatch.setattr(gate, "_module_hashes", lambda: {"modified": True})
    elif mutation == "config":
        owner.config.max_calls += 1
    else:
        store.set("delivery-gate/resolved-retry", store.put_json({"wrong": "binding"}))
    with pytest.raises(IntegrityError):
        budget.reserve("model/new", "model", {})
    assert len(budget.summary()) == prior


@pytest.mark.parametrize("status", ["failed", "running"])
def test_other_failed_or_running_model_is_never_covered_by_the_exception(tmp_path, status):
    owner, _, _, _, _, plan, admission, receipt = resolved_campaign(tmp_path)
    budget = gate.install_delivery_gate(owner, plan, admission, receipt)
    budget.reserve("model/another", "model", {})
    if status == "failed":
        budget.finish("model/another", "failed", {"error": "Another unknown result"}, 2, 0)
    for key, kind in [("model/later", "model"), ("experiment/later", "experiment"), ("probe/later", "probe")]:
        with pytest.raises(AmbiguousCall):
            budget.reserve(key, kind, {})
    assert gate.audit_delivery_gate(owner.store, budget.binding_ref)["integrity"] == "fail"


def test_final_activation_retains_both_arm_grant_and_zero_cost_aggregate_contract(tmp_path):
    owner, _, _, _, _, plan, admission, receipt = resolved_campaign(tmp_path)
    store = owner.store
    budget = gate.install_delivery_gate(owner, plan, admission, receipt)
    grant = store.put_json({"status": "BOTH_ARMS_FROZEN", "synthetic": True})
    deadline = time.time() + 60
    with pytest.raises(IntegrityError, match="both-arms"):
        budget.activate_final_grant(grant, deadline)
    store.set("study_final_release_ref", grant)
    activation = budget.activate_final_grant(grant, deadline)
    audit_key = "delivery-gate/final-audit/" + digest(canonical([budget.binding_ref, grant]))
    initial_audit = store.get(audit_key)
    budget.reserve("final/selected", "experiment", {"execution_type": "aggregation"})
    assert budget.activate_final_grant(grant, deadline) == activation
    budget.reserve("final/selected-seed-0", "experiment", {"synthetic": True})
    with pytest.raises(IntegrityError, match="audit must pass"):
        budget.activate_final_grant(grant, deadline)
    budget.finish("final/selected-seed-0", "completed", {"synthetic": True}, 1)
    assert budget.activate_final_grant(grant, deadline) == activation
    assert store.get(audit_key) == initial_audit
    budget.finish("final/selected", "completed", {"synthetic": True}, 0)
    assert gate.audit_delivery_gate(store, budget.binding_ref)["integrity"] == "pass"
    with pytest.raises(IntegrityError):
        budget.reserve("model/after-final-release", "model", {})


def test_gate_does_not_resolve_a_different_failed_call(tmp_path):
    owner, _, retry, _, _, plan, admission, receipt = resolved_campaign(tmp_path)
    other = deepcopy(owner.store.json(receipt))
    other["failed_operation"] = retry + "/different"
    with pytest.raises(IntegrityError):
        gate.install_delivery_gate(owner, plan, admission, owner.store.put_json(other))
