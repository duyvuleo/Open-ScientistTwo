"""Synthetic saved-response recovery. No CLI, model, network or training runs."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import pytest

from scientisttwo import continuation_delivery as delivery
from scientisttwo.backends import BackendError, CodexBackend
from scientisttwo.integrations.paperorchestra.controller_writer import PaperOrchestraResponse
from scientisttwo.process import ProcessResult
from scientisttwo.store import IntegrityError, canonical, digest
from scientisttwo.studies.media_relay import response_envelope

from test_continuation import small_campaign


def saved_response(tmp_path):
    owner, operation, _ = small_campaign(tmp_path)
    store = owner.store
    ask = operation.removeprefix("model/").removesuffix("/0")
    request = {"role": "paperorchestra_model", "inputs": {"prompt": "Synthetic outline task"},
               "schema": PaperOrchestraResponse.model_json_schema(), "model": owner.config.model}
    store.set("request/" + ask, store.put_json(request))
    failed_result = store.put_json({"error": "Codex invocation timed out; its process group was cancelled"})
    store.db.execute("UPDATE live_operations SET request_hash=?,result_ref=? WHERE key=?",
                     (digest(canonical(request)), failed_result, operation))
    store.db.commit()
    recorded = {}
    output = canonical({"output": '{"plotting_plan":[],"intro_related_work_plan":{},"section_plan":[]}'}).decode()
    stdout = "\n".join([json.dumps({"type": "thread.started"}),
                        json.dumps({"type": "item.completed", "item": {"type": "agent_message"}}),
                        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 32000, "output_tokens": 378}})]) + "\n"
    process = ProcessResult(0, stdout, "", 1, stdout_bytes=len(stdout.encode()))

    def fake_executor(argv, timeout, *, stdin):
        directory = Path(argv[argv.index("--cd") + 1])
        recorded.update(prompt=stdin, schema=json.loads((directory / "response-schema.json").read_text()))
        (directory / "response.json").write_text(output)
        return process

    backend = CodexBackend(store.root, request["model"], allow_live=True, executor=fake_executor)
    backend.invoke(request["role"], request["inputs"], PaperOrchestraResponse)
    now = time.time()
    queued = {"schema_version": 2, "request_id": "a" * 32, "policy_sha256": "b" * 64,
              "run_id": "synthetic-offline-proof", "model": request["model"], "role": request["role"],
              "created_at": now - 5, "deadline": now + 100, "timeout": 120, "media": [],
              "source_sha256": "c" * 64, **recorded}
    request_hash = digest(canonical(queued))
    proof = {"claim": {"request": queued, "request_sha256": request_hash, "claimed_at": now - 4},
             "dispatch": {"request_sha256": request_hash, "usage_status": "known", "observed_tokens": 32378,
                          "synthetic": False, "status": "completed", "returncode": 0, "timed_out": False,
                          "model": request["model"], "source_sha256": queued["source_sha256"],
                          "relay_sha256": owner.config.implementation["studies/media_relay.py"],
                          "started_at": now - 4, "completed_at": now - 2},
             "response": response_envelope(queued, asdict(process), output)}
    return owner, operation, ask, proof


def test_known_late_response_reconciles_once_without_reissuing_model_or_training(tmp_path, monkeypatch):
    from scientisttwo.research.audit import _agent
    owner, operation, ask, proof = saved_response(tmp_path)
    store = owner.store
    before = [dict(row) for row in store.db.execute("SELECT * FROM live_operations")]
    original_failed = dict(store.db.execute("SELECT * FROM live_operations WHERE key=?", (operation,)).fetchone())

    def forbidden(*args, **kwargs):
        raise AssertionError("Reconciliation cannot launch any process")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    ref = delivery.plan_reconciliation(store, operation, ask, proof)
    assert [dict(row) for row in store.db.execute("SELECT * FROM live_operations")] == before
    assert not store.db.execute("SELECT 1 FROM calls WHERE key=?", ("live-agent/" + ask,)).fetchone()
    # Administrative recovery is allowed after the original deadline because
    # the retained completion, rather than this verification, occurred in time.
    monkeypatch.setattr(time, "time", lambda: store.get("live_deadline") + 100)
    receipt_ref = delivery.reconcile_response(store, ref)
    assert dict(store.db.execute("SELECT * FROM live_operations WHERE key=?", (operation,)).fetchone()) == original_failed
    rows = [dict(row) for row in store.db.execute("SELECT * FROM live_operations")]
    assert len(rows) == len(before) + 1
    assert rows[-1]["kind"] == "model" and rows[-1]["status"] == "success"
    assert rows[-1]["tokens"] == 32378 and rows[-1]["seconds"] == 0
    assert rows[-1]["key"].startswith("model/" + ask + "/delivery-reconciliation/")
    request, payload = _agent(store, ask)
    assert request["role"] == "paperorchestra_model" and json.loads(payload["output"])["section_plan"] == []
    audit = delivery.audit_reconciliation(store, receipt_ref)
    assert audit["integrity"] == "pass" and audit["paid_calls"] == audit["training_runs"] == 0
    assert audit["known_tokens"] == 32378 and audit["continuation_ready"] is False
    assert audit["original_unknown_cost"] == "unchanged_unknown"
    assert delivery.reconcile_response(store, ref) == receipt_ref
    assert delivery.plan_reconciliation(store, operation, ask, proof) == ref
    assert [dict(row) for row in store.db.execute("SELECT * FROM live_operations")] == rows


@pytest.mark.parametrize("mutation", ["late", "usage", "schema", "prompt", "hash", "timeout", "truncated", "unknown", "wrong_model", "invalid_output", "malformed_events"])
def test_bad_late_response_is_rejected_before_any_cache_or_charge(tmp_path, mutation):
    owner, operation, ask, proof = saved_response(tmp_path)
    store = owner.store
    prior = [dict(row) for row in store.db.execute("SELECT * FROM live_operations")]
    queued, response = proof["claim"]["request"], proof["response"]
    if mutation == "late":
        proof["dispatch"]["completed_at"] = queued["deadline"] + 1
    elif mutation == "usage":
        proof["dispatch"]["observed_tokens"] += 1
    elif mutation == "schema":
        queued["schema"] = {"type": "string"}
    elif mutation == "prompt":
        prompt = json.loads(queued["prompt"])
        prompt["inputs"]["prompt"] = "Different experiment evidence"
        queued["prompt"] = canonical(prompt).decode()
    elif mutation == "hash":
        response["response_sha256"] = "0" * 64
    elif mutation == "timeout":
        response["process"]["timed_out"] = True
    elif mutation == "truncated":
        response["process"]["stdout_truncated"] = True
    elif mutation == "unknown":
        proof["dispatch"]["usage_status"] = "unknown"
    elif mutation == "wrong_model":
        proof["dispatch"]["model"] = "another-model"
    elif mutation == "invalid_output":
        response["response"] = '{"output":false}'
        response["response_sha256"] = digest(response["response"].encode())
    else:
        response["process"]["stdout"] += "malformed-event\n"
    if mutation in {"schema", "prompt"}:
        changed_hash = digest(canonical(queued))
        proof["claim"]["request_sha256"] = proof["dispatch"]["request_sha256"] = response["request_sha256"] = changed_hash
    with pytest.raises((IntegrityError, BackendError, ValueError)):
        delivery.plan_reconciliation(store, operation, ask, proof)
    assert [dict(row) for row in store.db.execute("SELECT * FROM live_operations")] == prior
    assert not store.db.execute("SELECT 1 FROM calls WHERE key=?", ("live-agent/" + ask,)).fetchone()


@pytest.mark.parametrize("mutation", ["failed_row", "source", "registered_artifact", "new_operation", "final_release"])
def test_reconciliation_rechecks_history_before_attachment(tmp_path, monkeypatch, mutation):
    owner, operation, ask, proof = saved_response(tmp_path)
    store = owner.store
    ref = delivery.plan_reconciliation(store, operation, ask, proof)
    if mutation == "failed_row":
        store.db.execute("UPDATE live_operations SET tokens=1 WHERE key=?", (operation,))
        store.db.commit()
    elif mutation == "source":
        monkeypatch.setattr(delivery, "implementation_identity", lambda: {"modified": True})
    elif mutation == "registered_artifact":
        artifact = store.get("request/" + ask)
        (store.root / artifact).write_text("modified")
    elif mutation == "new_operation":
        owner.budget.reserve("model/unrelated", "model", {})
        owner.budget.finish("model/unrelated", "success", {"synthetic": True}, 1, 1)
    else:
        store.set("study_final_release_ref", "synthetic-final-grant")
    with pytest.raises(IntegrityError):
        delivery.reconcile_response(store, ref)
    assert not store.db.execute("SELECT 1 FROM calls WHERE key=?", ("live-agent/" + ask,)).fetchone()


def test_delivery_plan_does_not_modify_old_receipts_or_proofs(tmp_path):
    owner, operation, ask, proof = saved_response(tmp_path)
    original = deepcopy(proof)
    ref = delivery.plan_reconciliation(owner.store, operation, ask, proof)
    receipt = delivery.reconcile_response(owner.store, ref)
    assert proof == original
    plan = owner.store.json(ref)
    assert owner.store.json(plan["proof_ref"]) == original
    assert owner.store.json(receipt)["continuation_ready"] is False


def test_audit_rejects_duplicate_success_charge_for_the_same_request(tmp_path):
    owner, operation, ask, proof = saved_response(tmp_path)
    store = owner.store
    ref = delivery.plan_reconciliation(store, operation, ask, proof)
    receipt = delivery.reconcile_response(store, ref)
    row = dict(store.db.execute("SELECT * FROM live_operations WHERE key=?",
                               (store.json(ref)["success_operation"],)).fetchone())
    row["key"] = "model/" + ask + "/unapproved-duplicate/0"
    store.db.execute("INSERT INTO live_operations(key,kind,request_hash,status,seconds,tokens,result_ref) VALUES(:key,:kind,:request_hash,:status,:seconds,:tokens,:result_ref)", row)
    store.db.commit()
    report = delivery.audit_reconciliation(store, receipt)
    assert report["integrity"] == "fail" and "duplicate" in report["error"]
