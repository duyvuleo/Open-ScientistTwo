"""Synthetic API transport receipts; no external providers or live LLM calls."""
from contextlib import closing
import json
import os
from types import SimpleNamespace
import time

import pytest
from pydantic import field_validator

from scientisttwo.process import ProcessResult
from scientisttwo.models import Record
from scientisttwo.live.adapters import implementation_identity
from scientisttwo.providers.backend import APIBackend, APIBackendError
from scientisttwo.providers.config import ModelProvider
from scientisttwo.providers.audit import inspect_api_ledger
from scientisttwo.store import Store, canonical, digest, IntegrityError


class Answer(Record):
    answer: str


def config(**updates):
    values = dict(provider=ModelProvider(kind="openai_compatible", base_url="http://localhost:8000/v1",
                                        max_output_tokens=128), model="local/test-model",
                  model_retries=0, token_stop_threshold=100, max_calls=4,
                  call_seconds=5, engine_version="research-v2", implementation=implementation_identity())
    values.update(updates)
    return SimpleNamespace(**values)


def response(text='{"answer":"synthetic response"}', *, usage=None, status="completed"):
    usage = {"input_tokens": 10, "output_tokens": 2} if usage is None else usage
    return {"version": "provider-http-v1", "status": status, "output_text": text,
            "usage": usage, "usage_known": True, "request_sent": True, "http_status": 200,
            "reported_model": "test-model-revision", "response_id": "fixture-response",
            "response_sha256": digest(text.encode()), "error_code": None}


def setup(tmp_path, payload=None, **updates):
    store = Store(tmp_path / "campaign")
    store.set("live_deadline", time.time() + 120)
    calls = []
    def execute(argv, timeout, **kwargs):
        calls.append({"argv": argv, "timeout": timeout, **kwargs})
        return ProcessResult(0, canonical(payload or response()).decode(), "", .01)
    backend = APIBackend(store, config(**updates), allow_live=True, process_executor=execute)
    return store, backend, calls


def test_successful_call_preserves_exact_request_and_known_usage(tmp_path):
    store, backend, calls = setup(tmp_path)
    with closing(store):
        result = backend.invoke("critic", {"evidence": "synthetic"}, Answer)
        assert result.payload == {"answer": "synthetic response"}
        assert result.backend == "openai_compatible" and result.model == "local/test-model"
        assert result.usage["input_tokens"] == 10
        summary = backend.accounting()
        assert summary["claimed_calls"] == summary["completed_receipts"] == 1
        assert summary["known_observed_tokens"] == 12 and summary["usage_complete"]
        records = inspect_api_ledger(store)["records"]
        assert len(records) == 1
        assert len(calls) == 1 and calls[0]["argv"][-1] == "scientisttwo.providers.worker"
        wire = json.loads(calls[0]["stdin"])
        assert wire["model"] == "local/test-model" and wire["wire_schema"]["additionalProperties"] is False
        assert wire["timeout_seconds"] <= 5
        assert not (store.root / "model-ledger").exists()


def test_credential_is_env_only_and_other_credentials_are_not_forwarded(tmp_path, monkeypatch):
    key = "fixture" + "-credential-one"
    monkeypatch.setenv("SCIENTISTTWO_TEST_KEY", key)
    monkeypatch.setenv("UNRELATED_TEST_CREDENTIAL", "must-not-forward")
    provider = ModelProvider(kind="openai", api_key_env="SCIENTISTTWO_TEST_KEY", max_output_tokens=100)
    store, backend, calls = setup(tmp_path, provider=provider)
    with closing(store):
        backend.invoke("critic", {}, Answer)
        assert calls[0]["env"]["SCIENTISTTWO_TEST_KEY"] == key
        assert "UNRELATED_TEST_CREDENTIAL" not in calls[0]["env"]
        assert key not in calls[0]["stdin"] and all(key not in item for item in calls[0]["argv"])
        assert all(key.encode() not in path.read_bytes() for path in store.root.rglob("*") if path.is_file())


@pytest.mark.parametrize("escaped", [False, True])
def test_request_credential_is_rejected_before_artifact_persistence(tmp_path, monkeypatch, escaped):
    secret = "synthetic" + "-only-credential"
    monkeypatch.setenv("SCIENTISTTWO_TEST_KEY", secret)
    provider = ModelProvider(kind="openai", max_output_tokens=100, api_key_env="SCIENTISTTWO_TEST_KEY")
    store, backend, calls = setup(tmp_path, provider=provider)
    with closing(store):
        value = ('"' + ''.join('\\u%04x' % ord(char) for char in secret) + '"') if escaped else secret
        before = {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
        with pytest.raises(APIBackendError, match="credential_in_request_body") as error:
            backend.invoke("critic", {"accidental_input": value}, Answer)
        assert error.value.usage_known and sum(error.value.usage.values()) == 0
        assert not calls
        assert before == {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage", ["usage_missing", "worker_timeout", "worker_exception", "truncated_capture"])
def test_unknown_usage_blocks_every_later_dispatch(tmp_path, damage):
    store, backend, calls = setup(tmp_path)
    with closing(store):
        def failed(argv, timeout, **kwargs):
            calls.append({})
            if damage == "worker_exception":
                raise OSError("Untrusted exception must not be persisted")
            if damage == "usage_missing":
                value = response()
                value.update(usage=None, usage_known=False)
                return ProcessResult(0, canonical(value).decode(), "", .01)
            return ProcessResult(0, "", "", .01, timed_out=damage == "worker_timeout",
                                 stdout_truncated=damage == "truncated_capture")
        backend.executor = failed
        with pytest.raises(APIBackendError) as error:
            backend.invoke("critic", {}, Answer)
        assert not error.value.usage_known
        assert backend.accounting()["unknown_calls"]
        with pytest.raises(APIBackendError, match="prior_usage_or_completion_unknown"):
            backend.invoke("critic", {}, Answer)
        assert len(calls) == 1
        assert backend.accounting()["known_observed_tokens"] == 0  # Known portion only; incomplete stays explicit.
        assert not backend.accounting()["usage_complete"]


def test_invalid_output_retains_usage_and_consumes_token_budget(tmp_path):
    store, backend, calls = setup(tmp_path, payload=response('{"wrong":"field"}'), token_stop_threshold=12)
    with closing(store):
        with pytest.raises(APIBackendError, match="structured_output_invalid") as error:
            backend.invoke("critic", {}, Answer)
        assert error.value.usage_known and error.value.usage["input_tokens"] == 10
        assert backend.accounting()["known_observed_tokens"] == 12
        with pytest.raises(APIBackendError, match="model_budget_exhausted"):
            backend.invoke("critic", {}, Answer)
        assert len(calls) == 1


def test_transforming_schema_is_rejected_with_known_cost(tmp_path):
    class TransformingAnswer(Answer):
        @field_validator("answer")
        @classmethod
        def transform(cls, value):
            return value.upper()

    store, backend, calls = setup(tmp_path)
    with closing(store):
        with pytest.raises(APIBackendError, match="structured_output_invalid") as error:
            backend.invoke("critic", {}, TransformingAnswer)
        assert error.value.usage_known
        assert backend.accounting()["known_observed_tokens"] == 12
        assert len(calls) == 1


def test_source_change_blocks_dispatch_after_policy_freeze(tmp_path, monkeypatch):
    store, backend, calls = setup(tmp_path)
    with closing(store):
        monkeypatch.setattr("scientisttwo.providers.backend.provider_sources", lambda: {})
        with pytest.raises(IntegrityError, match="source changed"):
            backend.invoke("critic", {}, Answer)
        assert not calls
        assert not list(backend.calls.iterdir())


def test_claim_without_completion_blocks_resume(tmp_path):
    store, backend, calls = setup(tmp_path)
    with closing(store):
        backend.invoke("critic", {}, Answer)
        receipt = next(backend.calls.glob("*/receipt.json"))
        identifier = receipt.parent.name
        receipt.unlink()  # Explicit crash-state fixture; no actual campaign is modified.
        store.db.execute("DELETE FROM metadata WHERE key=?", ("api_receipt/" + identifier,))
        store.db.commit()
        resumed = APIBackend(store, backend.config, allow_live=True, process_executor=backend.executor)
        with pytest.raises(APIBackendError, match="prior_usage_or_completion_unknown"):
            resumed.invoke("critic", {}, Answer)
        assert len(calls) == 1


def test_frozen_policy_cannot_change_endpoint_model_or_limits(tmp_path):
    store, backend, _ = setup(tmp_path)
    with closing(store):
        for change in ({"model": "other"}, {"max_calls": 5}, {"token_stop_threshold": 200}):
            with pytest.raises(IntegrityError, match="immutable"):
                APIBackend(store, config(**change), allow_live=True)


def test_no_opt_in_does_not_create_a_model_ledger(tmp_path):
    with closing(Store(tmp_path)) as store:
        with pytest.raises(ValueError, match="authorization"):
            APIBackend(store, config(), allow_live=False)
        assert not (tmp_path / "api-model-ledger").exists()


def test_model_identifier_cannot_persist_selected_credential(tmp_path, monkeypatch):
    secret = "synthetic" + "-model-credential"
    monkeypatch.setenv("SCIENTISTTWO_TEST_KEY", secret)
    provider = ModelProvider(kind="openai", max_output_tokens=100, api_key_env="SCIENTISTTWO_TEST_KEY")
    with closing(Store(tmp_path)) as store:
        store.set("live_deadline", time.time() + 120)
        with pytest.raises(ValueError):
            APIBackend(store, config(provider=provider, model=secret), allow_live=True)
        assert not (tmp_path / "api-model-ledger").exists()
        assert all(secret.encode() not in p.read_bytes() for p in store.root.rglob("*") if p.is_file())


def test_parent_defense_does_not_persist_credential_echo(tmp_path, monkeypatch):
    secret = "test" + "-credential-echo"
    monkeypatch.setenv("SCIENTISTTWO_TEST_KEY", secret)
    provider = ModelProvider(kind="openai", max_output_tokens=100, api_key_env="SCIENTISTTWO_TEST_KEY")
    store, backend, _ = setup(tmp_path, payload=response(json.dumps({"answer": secret})), provider=provider)
    with closing(store):
        with pytest.raises(APIBackendError, match="credential_echo_detected"):
            backend.invoke("critic", {}, Answer)
        assert backend.accounting()["known_observed_tokens"] == 12
        for path in store.root.rglob("*"):
            if path.is_file():
                assert secret.encode() not in path.read_bytes()


def test_images_are_bound_to_bytes_and_rejected_without_declared_vision(tmp_path):
    store, backend, calls = setup(tmp_path)
    with closing(store):
        with pytest.raises(ValueError, match="image support"):
            with backend.attach_images([]):
                pass
        provider = ModelProvider(kind="openai_compatible", base_url="http://localhost:8000/v1",
                                 max_output_tokens=100, vision=True)
        other = Store(tmp_path / "vision")
        with closing(other):
            other.set("live_deadline", time.time() + 120)
            camera = APIBackend(other, config(provider=provider), allow_live=True, process_executor=backend.executor)
            image = tmp_path / "page.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-image-header")
            with camera.attach_images([{"path": image, "mime_type": "image/png", "sha256": digest(image.read_bytes())}]):
                camera.invoke("paperorchestra_model", {}, Answer)
            assert json.loads(calls[-1]["stdin"])["images"][0]["mime_type"] == "image/png"
            assert camera._images == []


def test_ledger_permissions_are_private(tmp_path):
    store, backend, _ = setup(tmp_path)
    with closing(store):
        backend.invoke("critic", {}, Answer)
        for path in [backend.root, backend.calls, *backend.calls.iterdir(), backend.root / "policy.json"]:
            assert os.stat(path).st_mode & 0o077 == 0
