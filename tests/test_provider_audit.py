"""Synthetic provider completions test accounting, never real model behavior."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scientisttwo.backends import BackendError
from scientisttwo.live.runtime import Budget, LiveBudgetExhausted
from scientisttwo.models import Record
from scientisttwo.process import ProcessResult
from scientisttwo.providers.audit import inspect_api_ledger, verify_api_accounting
from scientisttwo.providers.backend import APIBackend
from scientisttwo.providers.config import ModelProvider
from scientisttwo.research.controller import ResearchController
from scientisttwo.research.minimal_live_acceptance import _model_accounting, _roles, capture_resume_snapshot
from scientisttwo.store import IntegrityError, Store, canonical, digest

from test_native_live_execution import native_config


class Answer(Record):
    answer: str


def outcome(*, output='{"answer":"Synthetic provider result"}', usage=None, status="completed",
            usage_known=True, request_sent=True):
    usage = {"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 3} if usage is None and usage_known else usage
    return {"version": "provider-http-v1", "status": status, "output_text": output,
        "usage": usage, "usage_known": usage_known, "request_sent": request_sent,
        "http_status": 200 if request_sent else None, "reported_model": "mock-resolved-model",
        "response_id": "mock-response", "response_sha256": "d" * 64,
        "error_code": None if status == "completed" else "mock_failure"}


@pytest.fixture
def api_campaign(tmp_path, request, monkeypatch):
    kind = getattr(request, "param", "openai_compatible")
    monkeypatch.setenv("SCIENTISTTWO_TEST_AUDIT_KEY", "synthetic-audit-credential")
    config = native_config(tmp_path, model="custom-test-model",
        provider=ModelProvider(kind=kind,
            base_url="http://localhost:8000/v1" if kind == "openai_compatible" else None,
            api_key_env="SCIENTISTTWO_TEST_AUDIT_KEY" if kind != "openai_compatible" else None,
            max_output_tokens=64))
    store = Store(tmp_path / "campaign")
    store.set("live_spec_ref", store.put_json(config))
    store.set("live_backend", {"backend": kind, "synthetic": False, "model": config.model})
    budget = Budget(store, config)
    budget.remaining()
    sent = []
    response = {"value": outcome()}

    def fake_executor(argv, timeout, **kwargs):
        sent.append(json.loads(kwargs["stdin"]))
        if isinstance(response["value"], BaseException):
            raise response["value"]
        return ProcessResult(0, canonical(response["value"]).decode(), "", .001)

    backend = APIBackend(store, config, allow_live=True, process_executor=fake_executor)
    controller = ResearchController(store, config, SimpleNamespace(context=lambda: {"synthetic_fixture": True}),
                                    backend, SimpleNamespace(), budget=budget)
    value = SimpleNamespace(store=store, config=config, backend=backend,
                            controller=controller, response=response, sent=sent)
    yield value
    store.close()


def save(item):
    item.store.set("api_model_accounting_ref", item.store.put_json(item.backend.accounting(), "api-accounting"))


def invoke(item, key="first"):
    return item.controller.ask(key, "critic", {"fixture": "Synthetic HTTP response accounting only"}, Answer)


def record(item, index=0):
    records = inspect_api_ledger(item.store)["records"]
    return records[index]


def replace_receipt(item, value, index=0):
    current = record(item, index)
    path = item.store.root / "api-model-ledger/calls" / current["id"] / "receipt.json"
    path.write_bytes(canonical(value))
    item.store.set("api_receipt/" + current["id"], item.store.put_json(value, "deliberately-modified-test-receipt"))


def test_api_success_usage_roles_and_completed_cache_reconcile(api_campaign):
    item = api_campaign
    assert invoke(item).answer == "Synthetic provider result"
    save(item)
    audited = verify_api_accounting(item.store, item.config)
    assert audited["status"] == "pass" and audited["calls"] == 1
    assert audited["observed_tokens"] == audited["controller_accounted_tokens"] == 12
    assert audited["known_tokens_not_in_controller_total"] == 0
    assert _model_accounting(item.store, item.config) == audited
    assert _roles(item.store, item.config) == {"critic": 1}
    assert all(b"synthetic-audit-credential" not in path.read_bytes()
               for path in item.store.root.rglob("*") if path.is_file())
    before = capture_resume_snapshot(item.store)
    invoke(item)
    assert capture_resume_snapshot(item.store) == before
    assert len(item.sent) == 1
    assert before["model_accounting_ref"] == item.store.get("api_model_accounting_ref")
    assert any(name.endswith("receipt.json") for name in before["model_ledger_files"])


def test_known_schema_failure_keeps_cost_and_allows_new_declared_invocation(api_campaign):
    item = api_campaign
    item.response["value"] = outcome(output='{"answer":4}', usage={"input_tokens": 5, "output_tokens": 2})
    with pytest.raises(BackendError):
        invoke(item)
    item.response["value"] = outcome()
    invoke(item, "second")
    save(item)
    result = verify_api_accounting(item.store, item.config)
    assert result["status"] == "pass" and result["observed_tokens"] == 19
    assert result["controller_accounted_tokens"] == 19
    assert result["calls"] == 2 and len(item.sent) == 2


@pytest.mark.parametrize("api_campaign", ["openai", "anthropic"], indirect=True)
def test_commercial_provider_identity_is_bound_without_a_codex_assumption(api_campaign):
    item = api_campaign
    invoke(item)
    save(item)
    report = verify_api_accounting(item.store, item.config)
    assert report["provider"] == item.config.provider.kind
    assert report["model"] == "custom-test-model"
    assert _roles(item.store, item.config) == {"critic": 1}
    assert all(b"synthetic-audit-credential" not in path.read_bytes()
               for path in item.store.root.rglob("*") if path.is_file())
    different = item.config.model_copy(update={"provider": ModelProvider(
        kind="openai_compatible", base_url="http://localhost:8000/v1", max_output_tokens=64)})
    with pytest.raises(IntegrityError, match="provider"):
        verify_api_accounting(item.store, different)


def test_model_and_response_metadata_allow_versioned_at_identifiers(api_campaign):
    item = api_campaign
    item.response["value"].update(reported_model="custom-model@revision", response_id="response@revision")
    invoke(item)
    save(item)
    assert verify_api_accounting(item.store, item.config)["status"] == "pass"
    receipt = record(item)["receipt"]
    assert receipt["reported_model"] == "custom-model@revision"
    assert receipt["response_id"] == "response@revision"


def test_accepted_payload_must_equal_decoded_provider_output(api_campaign):
    item = api_campaign
    invoke(item)
    save(item)
    rec = record(item)
    receipt = deepcopy(rec["receipt"])
    result = item.store.json(receipt["result_ref"])
    result["payload"]["answer"] = "A different answer with an otherwise valid schema"
    receipt["result_ref"] = item.store.put_json(result, "deliberately-modified-test-result")
    replace_receipt(item, receipt)
    with pytest.raises(IntegrityError, match="differs from its decoded provider output"):
        inspect_api_ledger(item.store)


def test_unknown_usage_is_unverified_never_zero_or_automatically_retried(api_campaign):
    item = api_campaign
    item.response["value"] = outcome(usage_known=False)
    with pytest.raises(BackendError):
        invoke(item)
    save(item)
    result = verify_api_accounting(item.store, item.config, require_complete=False)
    assert result["status"] == "unverified" and result["unknown_calls"] == 1
    inspected = inspect_api_ledger(item.store)
    assert not inspected["usage_complete"]
    assert inspected["records"][0]["receipt"]["observed_tokens"] is None
    with pytest.raises(IntegrityError, match="unknown"):
        verify_api_accounting(item.store, item.config)
    with pytest.raises(BackendError):
        invoke(item)
    assert len(item.sent) == 1


def test_interruption_retains_pending_claim_and_unknown_controller_operation(api_campaign):
    item = api_campaign
    item.response["value"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        invoke(item)
    save(item)
    result = verify_api_accounting(item.store, item.config, require_complete=False)
    assert result["status"] == "unverified" and result["unknown_calls"] == 1
    assert result["unknown_controller_operations"] == ["model/first/0"]
    assert inspect_api_ledger(item.store)["completed_receipts"] == 0


def test_last_call_may_cross_threshold_but_no_later_dispatch_is_admitted(api_campaign):
    item = api_campaign
    item.response["value"] = outcome(usage={"input_tokens": 90, "output_tokens": 12})
    invoke(item)
    save(item)
    assert verify_api_accounting(item.store, item.config)["observed_tokens"] == 102
    with pytest.raises(LiveBudgetExhausted, match="token"):
        invoke(item, "must-not-dispatch")
    assert len(item.sent) == 1


@pytest.mark.parametrize("damage", ["token_total", "result_ref", "request_binding", "unknown_to_zero",
                                    "cached_tokens", "missing_receipt", "summary", "controller_charge", "policy"])
def test_fabricated_or_incomplete_accounting_cannot_pass(api_campaign, damage):
    item = api_campaign
    invoke(item)
    save(item)
    rec = record(item)
    if damage == "missing_receipt":
        (item.store.root / "api-model-ledger/calls" / rec["id"] / "receipt.json").unlink()
    elif damage == "summary":
        summary = item.backend.accounting()
        summary["known_observed_tokens"] = 0
        item.store.set("api_model_accounting_ref", item.store.put_json(summary))
    elif damage == "controller_charge":
        with item.store.db:
            item.store.db.execute("UPDATE live_operations SET tokens=0 WHERE kind='model'")
    elif damage == "policy":
        path = item.store.root / "api-model-ledger/policy.json"
        policy = json.loads(path.read_text())
        policy["max_calls"] += 1
        path.write_bytes(canonical(policy))
    else:
        receipt = deepcopy(rec["receipt"])
        if damage == "token_total":
            receipt["observed_tokens"] += 1
        elif damage == "result_ref":
            receipt["result_ref"] = item.store.put_json({"fabricated": True})
        elif damage == "request_binding":
            receipt["controller_request_sha256"] = "a" * 64
        elif damage == "unknown_to_zero":
            receipt.update(usage_known=False, usage=None, observed_tokens=0)
        elif damage == "cached_tokens":
            receipt["usage"]["cached_input_tokens"] = 999
        replace_receipt(item, receipt)
    with pytest.raises((IntegrityError, ValueError)):
        verify_api_accounting(item.store, item.config)


def test_claim_sequence_cannot_admit_after_unknown_or_exhaustion(api_campaign):
    item = api_campaign
    item.response["value"] = outcome(usage_known=False)
    with pytest.raises(BackendError):
        invoke(item)
    first = record(item)
    forged = deepcopy(first["claim"])
    forged.update(id="f" * 32, prior_calls=1, prior_observed_tokens=0)
    directory = item.store.root / "api-model-ledger/calls" / forged["id"]
    directory.mkdir()
    (directory / "claim.json").write_bytes(canonical(forged))
    with pytest.raises(IntegrityError, match="followed an unknown"):
        inspect_api_ledger(item.store)


def test_receipt_is_bound_to_exact_registered_payload_and_controller_request(api_campaign):
    item = api_campaign
    invoke(item)
    save(item)
    rec = record(item)
    request = deepcopy(rec["request"])
    request["controller_request"]["inputs"] = {"unrelated": "request"}
    changed = deepcopy(rec["claim"])
    changed["request_ref"] = item.store.put_json(request)
    changed["request_sha256"] = digest(canonical(request))
    (item.store.root / "api-model-ledger/calls" / rec["id"] / "claim.json").write_bytes(canonical(changed))
    with pytest.raises(IntegrityError, match="controller input"):
        verify_api_accounting(item.store, item.config)


def test_api_audit_never_adopts_codex_inherited_accounting(api_campaign):
    item = api_campaign
    invoke(item)
    save(item)
    item.store.set("inherited_model_accounting_ref", item.store.put_json({"synthetic": True}))
    with pytest.raises(IntegrityError, match="Codex"):
        verify_api_accounting(item.store, item.config)


@pytest.mark.parametrize("unknown", [False, True])
def test_research_audit_includes_provider_accounting_without_dispatch(api_campaign, monkeypatch, unknown):
    item = api_campaign
    from scientisttwo.research import audit as research_audit
    from scientisttwo.live import audit as execution_audit
    if unknown:
        item.response["value"] = outcome(usage_known=False)
        with pytest.raises(BackendError):
            invoke(item)
    else:
        invoke(item)
    save(item)
    monkeypatch.setattr(research_audit, "implementation_identity", lambda: item.config.implementation)
    monkeypatch.setattr(research_audit, "load_development",
                        lambda *a: SimpleNamespace(context=lambda: {"editable_files": ["train.py"]}))
    monkeypatch.setattr(execution_audit, "audit_live", lambda *a, **kw: {"status": "pass", "errors": []})
    before = len(item.sent)
    audited = research_audit.audit_research(item.store.root)
    assert audited["status"] == ("unverified" if unknown else "pass"), audited
    assert audited["checks"]["model_accounting"]["status"] == ("unverified" if unknown else "pass")
    assert audited["live_model_calls"] == 0 and len(item.sent) == before


def test_malformed_agent_result_does_not_echo_input_values(api_campaign):
    item = api_campaign
    invoke(item)
    save(item)
    rec = record(item)
    receipt = deepcopy(rec["receipt"])
    sensitive = "synthetic-sensitive-value-not-for-error-output"
    receipt["result_ref"] = item.store.put_json({"payload": sensitive})
    replace_receipt(item, receipt)
    with pytest.raises(IntegrityError) as error:
        inspect_api_ledger(item.store)
    assert sensitive not in str(error.value)
