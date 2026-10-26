"""Explicit synthetic continuation contracts; no live model or remote execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sqlite3
import time

import pytest

from scientisttwo import continuation as module
from scientisttwo.backends import BackendError
from scientisttwo.live.adapters import implementation_identity
from scientisttwo.live.models import LiveSpec, DevelopmentSpec
from scientisttwo.live.runtime import LiveBudgetExhausted, AmbiguousCall
from scientisttwo.store import Store, IntegrityError, canonical, digest
from scientisttwo.studies.budget import StudyBudget


def small_campaign(tmp_path):
    from test_staged_evaluation import make_staged_task
    from scientisttwo import backends
    native = make_staged_task(tmp_path / "task")
    spec = LiveSpec(native=native, development=DevelopmentSpec(metric="accuracy", direction="maximize"),
                    model="synthetic-model", max_calls=10, max_experiments=10,
                    campaign_seconds=300, token_stop_threshold=1000,
                    implementation=implementation_identity())
    store = Store(tmp_path / "campaign")
    budget = StudyBudget(store, spec, max_experiment_seconds=100,
                         max_final_seconds=100, max_final_wall_seconds=100)
    store.set("live_spec_ref", store.put_json(spec))
    store.set("live_backend", {"name": "fixture", "synthetic": True, "model": spec.model})
    store.set("live_sources_ref", store.put_json({"backends.py": store.put_file(Path(backends.__file__))}))
    store.set("live_deadline", time.time() + 300)
    store.set("live_state", {"stage": "draft", "status": "blocked", "reason": "Synthetic transport failure"})
    budget.reserve("experiment/baseline/full", "experiment", {"synthetic": True})
    budget.finish("experiment/baseline/full", "success", {"synthetic": True}, 7)
    request = {"role": "paperorchestra_model", "inputs": {"prompt": "Synthetic outline"}, "model": spec.model}
    store.set("request/method/0/draft/0/manuscript/upstream/0001", store.put_json(request))
    operation = "model/method/0/draft/0/manuscript/upstream/0001/0"
    budget.reserve(operation, "model", request)
    budget.finish(operation, "failed", {"error": "Synthetic unknown token usage"}, 3)
    controller = SimpleNamespace(store=store, config=spec, budget=budget, runner=SimpleNamespace(budget=budget))
    return controller, operation, request


def allowance(ref, **changes):
    return {"schema_version": "scientisttwo-continuation-allowance-v1", "plan_ref": ref,
            "max_new_calls": 3, "token_stop_threshold": 100,
            "usage_disposition": "acknowledge_unknown_with_segment_limit",
            "authorization_ref": "synthetic-test-authorization", **changes}


def install(controller, operation):
    plan_ref = module.prepare_plan(controller.store, operation, "test-1")
    admitted = module.admit_continuation(controller.store, plan_ref, allowance(plan_ref))
    budget = module.install_continuation(controller, plan_ref, admitted)
    return plan_ref, admitted, budget


def test_prepare_is_no_admission_and_snapshot_history_stays_immutable(tmp_path):
    owner, operation, _ = small_campaign(tmp_path)
    store = owner.store
    before = module._rows(store.db)
    plan_ref = module.prepare_plan(store, operation, "test-1")
    plan = store.json(plan_ref)
    assert plan["historical_rows"] == before
    assert not plan["live_admission"] and not plan["original_total_token_ceiling_verified"]
    snapshot = store.root / plan["snapshot"]["path"]
    connection = sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    assert module._rows(connection) == before
    connection.close()
    assert module.prepare_plan(store, operation, "test-1") == plan_ref
    assert module.audit_continuation(store, plan_ref)["integrity"] == "pass"
    with pytest.raises(ValueError, match="explicit"):
        module.admit_continuation(store, plan_ref, {})
    assert not store.get("continuation/admission/test-1")


@pytest.mark.parametrize("change", ["running", "final", "wrong_role", "success", "wrong_stage"])
def test_preparation_rejects_ineligible_parent(tmp_path, change):
    owner, operation, _ = small_campaign(tmp_path)
    store = owner.store
    if change == "running":
        owner.budget.reserve("model/unknown", "model", {})
    elif change == "final":
        store.set("live_frozen", {"selected": "synthetic"})
    elif change == "wrong_role":
        store.set("request/method/0/draft/0/manuscript/upstream/0001", store.put_json({"role": "critic"}))
    elif change == "success":
        store.db.execute("UPDATE live_operations SET status='success' WHERE key=?", (operation,))
        store.db.commit()
    else:
        store.set("live_state", {"stage": "search"})
    with pytest.raises(IntegrityError):
        module.prepare_plan(store, operation, "test-1")


def test_only_failed_writer_is_remapped_and_all_prior_experiments_reuse(tmp_path):
    owner, operation, request = small_campaign(tmp_path)
    original = dict(owner.store.db.execute("SELECT * FROM live_operations WHERE key=?", (operation,)).fetchone())
    plan_ref, admission_ref, budget = install(owner, operation)
    assert budget is owner.runner.budget is owner.budget
    assert budget.reserve("experiment/baseline/full", "experiment", {"synthetic": True})["status"] == "success"
    for key, kind in [("experiment/new", "experiment"), ("model/new", "model"), ("probe/new", "probe")]:
        with pytest.raises(IntegrityError, match="exact writer"):
            budget.reserve(key, kind, {})
    with pytest.raises(IntegrityError, match="exact failed"):
        budget.reserve(operation, "model", {"different": True})
    assert budget.reserve(operation, "model", request) is None
    budget.finish(operation, "success", {"synthetic": "writer-complete"}, 2, 12)
    assert dict(owner.store.db.execute("SELECT * FROM live_operations WHERE key=?", (operation,)).fetchone()) == original
    assert budget.reserve(operation, "model", request)["status"] == "success"
    budget.reserve("experiment/review-new", "experiment", {"new_seeds": [3]})
    budget.finish("experiment/review-new", "success", {"synthetic": True}, 5)
    result = module.audit_continuation(owner.store, plan_ref, admission_ref)
    assert result["integrity"] == "pass" and result["new_experiments"] == 1
    assert result["new_model_calls"] == 1 and result["new_observed_tokens"] == 12
    assert result["original_total_token_ceiling_verified"] is False


@pytest.mark.parametrize("field,value", [("max_new_calls", 0), ("max_new_calls", True),
    ("max_new_calls", 10), ("token_stop_threshold", 1001),
    ("usage_disposition", "assume_zero"), ("authorization_ref", "")])
def test_allowance_cannot_infer_zero_cost_reset_limits_or_extend_deadline(tmp_path, field, value):
    owner, operation, _ = small_campaign(tmp_path)
    plan_ref = module.prepare_plan(owner.store, operation, "test-1")
    with pytest.raises((ValueError, LiveBudgetExhausted)):
        module.admit_continuation(owner.store, plan_ref, allowance(plan_ref, **{field: value}))


def test_segment_limit_and_original_deadline_are_both_enforced(tmp_path, monkeypatch):
    owner, operation, request = small_campaign(tmp_path)
    plan_ref, admission_ref, budget = install(owner, operation)
    original_deadline = owner.store.get("live_deadline")
    budget.reserve(operation, "model", request)
    budget.finish(operation, "success", {"synthetic": True}, 2, 101)  # Observed threshold may cross once.
    with pytest.raises(LiveBudgetExhausted, match="segment"):
        budget.reserve("model/next", "model", {})
    assert module.audit_continuation(owner.store, plan_ref, admission_ref)["integrity"] == "pass"
    monkeypatch.setattr(time, "time", lambda: original_deadline + 1)
    with pytest.raises(LiveBudgetExhausted, match="wall-clock"):
        budget.remaining()
    assert owner.store.get("live_deadline") == original_deadline


def test_new_failed_model_halts_fresh_operations_and_cannot_pass_audit(tmp_path):
    owner, operation, request = small_campaign(tmp_path)
    plan_ref, admission_ref, budget = install(owner, operation)
    budget.reserve(operation, "model", request)
    budget.finish(operation, "success", {"synthetic": True}, 2, 12)
    budget.reserve("model/next", "model", {})
    budget.finish("model/next", "failed", {"error": "Synthetic unknown usage"}, 1, 0)
    for key, kind in [("model/later", "model"), ("experiment/later", "experiment"), ("probe/later", "probe")]:
        with pytest.raises(AmbiguousCall, match="model failure"):
            budget.reserve(key, kind, {})
    report = module.audit_continuation(owner.store, plan_ref, admission_ref)
    assert report["integrity"] == "fail" and "failed model" in report["error"]


def test_repeated_final_activation_keeps_initial_audit_and_deadline(tmp_path):
    owner, operation, request = small_campaign(tmp_path)
    plan_ref, admission_ref, budget = install(owner, operation)
    budget.reserve(operation, "model", request)
    budget.finish(operation, "success", {"synthetic": True}, 2, 12)
    grant = owner.store.put_json({"synthetic": "BOTH_ARMS_FROZEN"})
    owner.store.set("study_final_release_ref", grant)
    deadline = time.time() + 60
    first = budget.activate_final_grant(grant, deadline)
    key = "continuation/final-audit/test-1/" + digest(grant.encode())
    audit_ref = owner.store.get(key)
    budget.reserve("final/selected", "experiment", {"execution_type": "aggregation"})
    assert budget.activate_final_grant(grant, deadline) == first
    budget.reserve("final/selected-seed-0", "experiment", {"synthetic": True})
    with pytest.raises(IntegrityError, match="integrity"):
        budget.activate_final_grant(grant, deadline)  # A physical seed may not be unresolved.
    budget.finish("final/selected-seed-0", "completed", {"synthetic": True}, 4)
    assert budget.activate_final_grant(grant, deadline) == first
    assert owner.store.get(key) == audit_ref
    assert owner.store.get("study_final_budget_activation")["deadline"] == deadline
    assert module.audit_continuation(owner.store, plan_ref, admission_ref)["integrity"] == "fail"
    budget.finish("final/selected", "completed", {"synthetic": True}, 0)
    assert module.audit_continuation(owner.store, plan_ref, admission_ref)["integrity"] == "pass"


def live_dispatch_proof(owner, operation, request):
    """Synthetic receipt bytes for cryptographic contract tests; never dispatched."""
    from scientisttwo.backends import StructuredOutputCodec, STRUCTURED_WIRE_CONTRACT_KEY
    from scientisttwo.studies.media_relay import INSTRUCTIONS, response_envelope, _result
    store = owner.store
    schema = {"type": "object", "properties": {"output": {"type": "string"}},
              "required": ["output"], "additionalProperties": False}
    request = {**request, "schema": schema}
    ask = operation.removeprefix("model/").removesuffix("/0")
    store.set("request/" + ask, store.put_json(request))
    store.db.execute("UPDATE live_operations SET request_hash=? WHERE key=?", (digest(canonical(request)), operation))
    store.db.commit()
    store.set("live_backend", {"name": "codex", "synthetic": False, "model": owner.config.model})
    codec = StructuredOutputCodec(schema)
    queued = {"schema_version": 2, "request_id": "a" * 32, "policy_sha256": "b" * 64,
              "run_id": "synthetic-proof-only", "model": owner.config.model,
              "role": "paperorchestra_model", "created_at": time.time() - 10,
              "deadline": time.time() + 100, "timeout": 120, "media": [],
              "source_sha256": "c" * 64, "schema": codec.wire_schema,
              "prompt": json.dumps({"role": "paperorchestra_model", "instructions": INSTRUCTIONS,
                "prompt_version": "scientisttwo-v1", "inputs": {**request["inputs"],
                  STRUCTURED_WIRE_CONTRACT_KEY: {"codec_identity": codec.identity, "version": codec.version}}})}
    request_hash = digest(canonical(queued))
    response = response_envelope(queued, _result(125), None)
    return {"claim": {"request": queued, "request_sha256": request_hash},
            "dispatch": {"request_sha256": request_hash, "synthetic": False, "status": "completed",
                         "model": owner.config.model, "usage_status": "unknown", "returncode": 125,
                         "source_sha256": queued["source_sha256"],
                         "relay_sha256": owner.config.implementation["studies/media_relay.py"]},
            "response": response}


def test_live_plan_requires_exact_failed_dispatch_proof_and_persists_only_hashes(tmp_path):
    owner, operation, request = small_campaign(tmp_path)
    proof = live_dispatch_proof(owner, operation, request)
    with pytest.raises(IntegrityError, match="local failed-dispatch"):
        module.prepare_plan(owner.store, operation, "proof-required")
    ref = module.prepare_plan(owner.store, operation, "proof-valid", failed_dispatch_proof=proof)
    record = owner.store.json(ref)["failed_dispatch_proof"]
    assert record["status"] == "verified" and record["observed_tokens"] is None
    assert record["claim_sha256"] == digest(canonical(proof["claim"]))
    assert "prompt" not in record and "Synthetic outline" not in json.dumps(record)
    assert module.prepare_plan(owner.store, operation, "proof-valid", failed_dispatch_proof=proof) == ref


@pytest.mark.parametrize("mutation", ["inputs", "schema", "request_hash", "relay", "response", "model"])
def test_live_receipt_binding_rejects_swapped_requests_and_receipts(tmp_path, mutation):
    owner, operation, request = small_campaign(tmp_path)
    proof = live_dispatch_proof(owner, operation, request)
    queued = proof["claim"]["request"]
    if mutation == "inputs":
        prompt = json.loads(queued["prompt"])
        prompt["inputs"]["prompt"] = "different research evidence"
        queued["prompt"] = json.dumps(prompt)
    elif mutation == "schema":
        queued["schema"] = {"type": "object"}
    elif mutation == "request_hash":
        proof["dispatch"]["request_sha256"] = "0" * 64
    elif mutation == "relay":
        proof["dispatch"]["relay_sha256"] = "0" * 64
    elif mutation == "response":
        proof["response"]["policy_sha256"] = "0" * 64
    else:
        proof["dispatch"]["model"] = "different-model"
    with pytest.raises((IntegrityError, ValueError)):
        module.verify_failed_dispatch_proof(owner.store, operation, proof)


@pytest.mark.parametrize("target", ["operation", "call", "event", "artifact", "deadline", "snapshot", "source"])
def test_changes_to_bound_history_are_detected(tmp_path, target, monkeypatch):
    owner, operation, _ = small_campaign(tmp_path)
    store = owner.store
    store.cache("example", {}, {"synthetic": True})
    store.event("example", "draft", "Synthetic")
    plan_ref, admission_ref, _ = install(owner, operation)
    if target == "operation":
        store.db.execute("UPDATE live_operations SET tokens=1 WHERE key=?", (operation,))
    elif target == "call":
        store.db.execute("DELETE FROM calls WHERE key='example'")
    elif target == "event":
        store.db.execute("UPDATE events SET reason='changed' WHERE key='example'")
    elif target == "artifact":
        ref = store.get("live_sources_ref")
        (store.root / ref).write_text("changed")
    elif target == "deadline":
        store.set("live_deadline", store.get("live_deadline") + 100)
    elif target == "snapshot":
        path = store.root / store.json(plan_ref)["snapshot"]["path"]
        path.chmod(0o600)
        path.write_bytes(b"changed")
    else:
        monkeypatch.setattr(module, "implementation_identity", lambda: {"changed": "source"})
    store.db.commit()
    assert module.audit_continuation(store, plan_ref, admission_ref)["integrity"] == "fail"


def test_real_controller_replays_through_failed_writer_without_repeating_training(tmp_path):
    from test_paperorchestra_research_workflow import paper_campaign
    from scientisttwo.research.controller import ResearchController
    controller, store, runner, agents, execution = paper_campaign(tmp_path, writing_refinements=0)
    controller.config.token_stop_threshold = 1_500_000
    original_invoke = agents.invoke
    failed = False

    def fail_first_writer(role, inputs, schema):
        nonlocal failed
        if role == "paperorchestra_model" and not failed:
            failed = True
            raise BackendError("Synthetic completed CLI response has unknown usage")
        return original_invoke(role, inputs, schema)

    agents.invoke = fail_first_writer
    try:
        state = controller.run()
        assert state["status"] == "blocked" and state["stage"] == "draft"
        operation = store.db.execute("SELECT key FROM live_operations WHERE kind='model' AND status='failed'").fetchone()[0]
        experiments = [dict(row) for row in store.db.execute("SELECT * FROM live_operations WHERE kind='experiment'")]
        stages = deepcopy(execution.scientific.stages)
        plan_ref = module.prepare_plan(store, operation, "writer-recovery")
        admission_ref = module.admit_continuation(store, plan_ref,
            allowance(plan_ref, max_new_calls=60, token_stop_threshold=100000))
        agents.invoke = original_invoke
        resumed = ResearchController(store, controller.config, controller.adapter, agents, runner, budget=controller.budget)
        module.install_continuation(resumed, plan_ref, admission_ref)
        result = resumed.run()
        assert result["status"] == "complete", result
        # Scientific search, candidate selection and ablations are cached exactly.
        assert [dict(row) for row in store.db.execute("SELECT * FROM live_operations WHERE kind='experiment' AND key NOT LIKE 'final/%'")] == experiments
        training_before = [item for item in stages if item == "train"]
        training_after = [item for item in execution.scientific.stages if item == "train"]
        assert training_before == training_after
        report = module.audit_continuation(store, plan_ref, admission_ref)
        assert report["integrity"] == "pass", report
        assert report["new_model_calls"] > 0 and not report["original_total_token_ceiling_verified"]
        assert dict(store.db.execute("SELECT * FROM live_operations WHERE key=?", (operation,)).fetchone())["status"] == "failed"
        before = resumed.budget.summary(), len(agents.seen), len(execution.writer.calls)
        assert resumed.run() == result
        assert before == (resumed.budget.summary(), len(agents.seen), len(execution.writer.calls))
    finally:
        store.close()
