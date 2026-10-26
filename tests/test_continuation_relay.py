"""Continuation runs a real fake CLI and never contacts a model service."""

from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import pytest

from scientisttwo import continuation_relay as continuation
from scientisttwo.process import ProcessResult
from scientisttwo.studies import media_relay as relay


def _request(policy, request_id, role="paperorchestra_model", **changes):
    now = time.time()
    request = {"schema_version": 2, "request_id": request_id,
        "policy_sha256": policy.identity, "run_id": policy.run_id, "model": policy.model,
        "role": role, "created_at": now, "deadline": min(now + 30, policy.expires_at),
        "timeout": 30, "source_sha256": policy.source_sha256, "media": [],
        "prompt": json.dumps({"role": role, "prompt_version": "scientisttwo-v1",
            "instructions": relay.INSTRUCTIONS, "inputs": {"prompt": "Frozen outline evidence"}}),
        "schema": {"type": "object", "properties": {"output": {"type": "string"}},
                   "required": ["output"], "additionalProperties": False}}
    request.update(changes)
    return request


@pytest.fixture
def fixture(tmp_path, request):
    executable = tmp_path / "fake-codex"
    calls = tmp_path / "executed-calls.txt"
    executable.write_text("#!" + sys.executable + "\n" + f'''
import json, pathlib, sys
request = json.load(sys.stdin)
with pathlib.Path({str(calls)!r}).open('a') as stream:
    stream.write(request['role'] + '\\n')
text = 'Executed outline with literal ' + chr(0x2028) + ' Unicode separator'
pathlib.Path(sys.argv[sys.argv.index('--output-last-message') + 1]).write_text(json.dumps({{'output': text}}))
print(json.dumps({{'type': 'item.completed', 'item': {{'type': 'agent_message', 'text': text}}}}, ensure_ascii=False))
print(json.dumps({{'type': 'turn.completed', 'usage': {{'input_tokens': 7, 'output_tokens': 5}}}}))
''')
    executable.chmod(0o700)
    policy = relay.RelayPolicy(run_id="continuation-fixture", expires_at=time.time() + 120,
        model="gpt-6-astra", allowed_models=("gpt-6-astra",), source_sha256="1" * 64,
        relay_sha256=relay.sha(Path(relay.__file__).read_bytes()), max_calls=5,
        token_stop_threshold=getattr(request, "param", {}).get("threshold", 100), call_seconds=30)
    parent = tmp_path / "parent"
    service = relay.LocalService(parent, policy, executable=executable, synthetic=True)
    service.serve(_request(policy, "1" * 32, role="critic"))
    service.executor = lambda *args, **kwargs: ProcessResult(17, "incomplete-json", "", .1)
    failed = _request(policy, "2" * 32)
    service.serve(failed)
    # Bind the historical protocol to a distinct old relay hash, as in the
    # deployed repair. The prepared parent fixture is then frozen for tests.
    historical_policy = replace(policy, relay_sha256="a" * 64)
    pfile = parent / "policy.json"
    pfile.write_bytes(relay.canonical(asdict(historical_policy)))
    for entry in (parent / "claims").iterdir():
        claim = relay.read_json(entry / "claim.json", relay.REQUEST_LIMIT + 4096)
        claim["request"]["policy_sha256"] = historical_policy.identity
        claim["request_sha256"] = relay.sha(relay.canonical(claim["request"]))
        (entry / "claim.json").write_bytes(relay.canonical(claim))
        dispatch = relay.read_json(entry / "dispatch.json", 16384)
        dispatch.update(request_sha256=claim["request_sha256"], relay_sha256="a" * 64)
        (entry / "dispatch.json").write_bytes(relay.canonical(dispatch))
        response = relay.read_json(entry / "response.json", relay.RESPONSE_LIMIT)
        response.update(policy_sha256=historical_policy.identity, request_sha256=claim["request_sha256"])
        (entry / "response.json").write_bytes(relay.canonical(response))
    manifest = continuation.parent_ledger_manifest(parent)
    auth = continuation.ContinuationRelayAuthorization(
        authorization_id="fixture-repair", authorized_state_path=str(tmp_path / "continued"),
        authorized_parent_state_path=str(parent), controller_plan_ref="artifacts/" + "3" * 64 + ".json",
        controller_admission_ref="artifacts/" + "4" * 64 + ".json",
        parent_manifest_sha256=relay.sha(relay.canonical(manifest)),
        failed_request_id="2" * 32, repaired_sources=continuation.repaired_source_hashes(),
        max_new_calls=3, new_token_stop_threshold=min(60, policy.token_stop_threshold), deadline=policy.expires_at,
        unknown_cost_acknowledged=True)
    return {"parent": parent, "manifest": manifest, "policy": historical_policy,
            "authorization": auth, "executable": executable, "calls": calls,
            "state": tmp_path / "continued"}


def _service(fixture, **kwargs):
    values = {key: fixture[key] for key in ("policy", "executable", "authorization")}
    values.update(parent_state=fixture["parent"], synthetic=True)
    values.update(kwargs)
    return continuation.ContinuationLocalService(fixture["state"], **values)


def test_continuation_preserves_wire_policy_parent_and_exact_writer_first(fixture):
    service = _service(fixture)
    policy = fixture["policy"]
    assert asdict(service.policy) == asdict(policy)
    assert service.policy.identity == policy.identity
    with pytest.raises(relay.RelayError, match="exact failed writer"):
        service.serve(_request(policy, "3" * 32, role="critic"))
    first = _request(policy, "3" * 32)
    response = service.serve(first)
    assert response["process"]["code"] == 0
    assert "\u2028" in json.loads(response["response"])["output"]
    assert service.serve(first) == response
    service.serve(_request(policy, "4" * 32, role="critic"))
    assert fixture["calls"].read_text().splitlines() == ["critic", "paperorchestra_model", "critic"]
    assert continuation.parent_ledger_manifest(fixture["parent"]) == fixture["manifest"]
    proof = relay.read_json(service.state / "continuation-proof.json", 2 * 1024 * 1024)
    assert proof["parent_calls"] == 2 and proof["parent_known_tokens"] == 12
    assert proof["parent_unknown_usage_calls"] == 1
    assert proof["original_total_tokens_verified"] is False
    receipt = relay.read_json(service.continuation_receipts / ("3" * 32 + ".json"), 16384)
    assert receipt["repaired_sources"] == continuation.repaired_source_hashes()
    assert receipt["parent_unknown_usage_calls"] == 1
    # Resume with the same immutable proof reuses committed responses.
    assert _service(fixture).serve(first) == response


def test_parent_ids_never_redispatched_even_for_exact_failed_writer(fixture):
    service = _service(fixture)
    for old_id in ("1" * 32, "2" * 32):
        with pytest.raises(relay.UnknownDispatch, match="Parent request IDs"):
            service.serve(_request(fixture["policy"], old_id))
    assert fixture["calls"].read_text().splitlines() == ["critic"]


@pytest.mark.parametrize("field", ["prompt", "schema", "media"])
def test_writer_retry_identity_cannot_be_changed(fixture, field):
    service = _service(fixture)
    request = _request(fixture["policy"], "3" * 32)
    if field == "prompt":
        prompt = json.loads(request[field])
        prompt["inputs"]["prompt"] += " tampered"
        request[field] = json.dumps(prompt)
    elif field == "schema":
        request[field]["title"] = "different"
    else:
        # An invalid attachment must fail request admission even before retry matching.
        request[field] = [{"mime_type": "text/plain"}]
    with pytest.raises(relay.RelayError):
        service.serve(request)
    assert fixture["calls"].read_text().splitlines() == ["critic"]


@pytest.mark.parametrize("change,match", [
    ({"unknown_cost_acknowledged": False}, "unknown cost"),
    ({"max_new_calls": 4}, "call ceiling"),
    ({"new_token_stop_threshold": 101}, "original threshold"),
    ({"parent_manifest_sha256": "f" * 64}, "parent ledger"),
    ({"failed_request_id": "1" * 32}, "failed PaperOrchestra"),
])
def test_constructor_rejects_unapproved_budget_or_parent_changes(fixture, change, match):
    auth = replace(fixture["authorization"], **change)
    with pytest.raises(relay.RelayError, match=match):
        _service(fixture, authorization=auth)


def test_constructor_requires_explicit_live_permission(fixture):
    with pytest.raises(relay.RelayError, match="allow_live"):
        _service(fixture, synthetic=False)
    assert not fixture["state"].exists()


def test_parent_partial_completion_is_not_acknowledgeable(fixture):
    (fixture["parent"] / "claims" / ("2" * 32) / "response.json").unlink()
    with pytest.raises(relay.UnknownDispatch, match="incomplete"):
        _service(fixture)


def test_active_parent_dispatch_prevents_continuation(fixture):
    fd = os.open(fixture["parent"] / "service.lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(relay.UnknownDispatch, match="still dispatching"):
            _service(fixture)
    finally:
        os.close(fd)


def test_parent_drift_blocks_calls_after_authorization(fixture):
    service = _service(fixture)
    path = fixture["parent"] / "claims" / ("2" * 32) / "dispatch.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(relay.RelayError, match="Parent ledger changed"):
        service.serve(_request(fixture["policy"], "3" * 32))
    assert fixture["calls"].read_text().splitlines() == ["critic"]


@pytest.mark.parametrize("fixture,scope", [({}, "new"), ({"threshold": 24}, "combined"),
                                           ({}, "calls")], indirect=["fixture"])
def test_new_and_parent_budget_usage_are_carried_forward(fixture, scope):
    auth = fixture["authorization"]
    if scope == "new":
        auth = replace(auth, new_token_stop_threshold=12)
    elif scope == "calls":
        auth = replace(auth, max_new_calls=1)
    service = _service(fixture, authorization=auth)
    service.serve(_request(fixture["policy"], "3" * 32))
    with pytest.raises(relay.RelayError, match="threshold"):
        service.serve(_request(fixture["policy"], "4" * 32, role="critic"))


def test_new_unknown_usage_stops_future_calls_but_cached_response_is_readable(fixture):
    service = _service(fixture, executor=lambda *args, **kwargs: ProcessResult(17, "bad-json", "", .1))
    request = _request(fixture["policy"], "3" * 32)
    response = service.serve(request)
    assert response["process"]["code"] == 125
    assert service.serve(request) == response
    with pytest.raises(relay.UnknownDispatch, match="New continuation token usage"):
        service.serve(_request(fixture["policy"], "4" * 32))
    receipt = relay.read_json(service.claims / ("3" * 32) / "dispatch.json", 16384)
    assert receipt["observed_tokens"] is None and receipt["usage_status"] == "unknown"


def test_interrupted_new_dispatch_never_repeats(fixture):
    invoked = []

    def crash(*args, **kwargs):
        invoked.append(True)
        raise RuntimeError("Fixture interrupted after possible dispatch")

    service = _service(fixture, executor=crash)
    request = _request(fixture["policy"], "3" * 32)
    with pytest.raises(RuntimeError):
        service.serve(request)
    for retry in (request, _request(fixture["policy"], "4" * 32)):
        with pytest.raises(relay.UnknownDispatch):
            service.serve(retry)
    assert invoked == [True]


def test_new_receipt_or_response_tampering_blocks_replay(fixture):
    service = _service(fixture)
    request = _request(fixture["policy"], "3" * 32)
    service.serve(request)
    receipt = service.continuation_receipts / ("3" * 32 + ".json")
    value = relay.read_json(receipt, 16384)
    value["response_sha256"] = "f" * 64
    receipt.write_bytes(relay.canonical(value))
    with pytest.raises(relay.RelayError, match="evidence changed"):
        service.serve(request)


def test_deadline_is_bounded_and_does_not_modify_wire_policy(fixture):
    seen = []

    def execute(argv, timeout, **kwargs):
        seen.append(timeout)
        return ProcessResult(17, "", "", 0)

    auth = replace(fixture["authorization"], deadline=time.time() + 10)
    service = _service(fixture, authorization=auth, executor=execute)
    service.serve(_request(fixture["policy"], "3" * 32))
    assert 0 < seen[0] <= 8
    assert service.policy.expires_at == fixture["policy"].expires_at
    with pytest.raises(relay.RelayError, match="original expiry"):
        _service(fixture, authorization=replace(auth, deadline=fixture["policy"].expires_at + 1))


def test_new_source_authorization_cannot_be_mutated_or_implicitly_bound(fixture):
    service = _service(fixture)
    service.authorization.repaired_sources["backends.py"] = "f" * 64
    with pytest.raises(relay.RelayError, match="authorization changed"):
        service.serve(_request(fixture["policy"], "3" * 32))
    with pytest.raises(relay.RelayError, match="explicitly bound"):
        continuation.VerifiedRepairPolicy(**asdict(fixture["policy"])).verify_relay_source()


def test_authorization_roundtrip_requires_all_fields(fixture):
    auth = fixture["authorization"]
    assert continuation.ContinuationRelayAuthorization.from_mapping(asdict(auth)) == auth
    value = asdict(auth)
    del value["unknown_cost_acknowledged"]
    with pytest.raises(relay.RelayError, match="every exact field"):
        continuation.ContinuationRelayAuthorization.from_mapping(value)


def test_historical_zero_with_unknown_status_is_never_known_cost(fixture):
    path = fixture["parent"] / "claims" / ("2" * 32) / "dispatch.json"
    receipt = relay.read_json(path, 16384)
    receipt["observed_tokens"] = 0  # Old deployed relay recorded this sentinel.
    path.write_bytes(relay.canonical(receipt))
    manifest = continuation.parent_ledger_manifest(fixture["parent"])
    auth = replace(fixture["authorization"], parent_manifest_sha256=relay.sha(relay.canonical(manifest)))
    service = _service(fixture, authorization=auth)
    assert service.parent_known_tokens == 12 and service.parent_unknown_calls == 1
    service.serve(_request(fixture["policy"], "3" * 32))
    proof = relay.read_json(service.continuation_receipts / ("3" * 32 + ".json"), 16384)
    assert proof["combined_total_tokens_verified"] is False
    assert relay.read_json(path, 16384)["usage_status"] == "unknown"


def test_changed_continuation_proof_blocks_dispatch(fixture):
    service = _service(fixture)
    path = service.state / "continuation-proof.json"
    proof = relay.read_json(path, 2 * 1024 * 1024)
    proof["parent_unknown_usage_calls"] = 0
    path.write_bytes(relay.canonical(proof))
    with pytest.raises(relay.RelayError, match="proof changed"):
        service.serve(_request(fixture["policy"], "3" * 32))


def test_missing_continuation_receipt_preserves_ambiguous_completion(fixture):
    service = _service(fixture)
    request = _request(fixture["policy"], "3" * 32)
    service.serve(request)
    (service.continuation_receipts / ("3" * 32 + ".json")).unlink()
    for attempted in (request, _request(fixture["policy"], "4" * 32)):
        with pytest.raises(relay.UnknownDispatch, match="receipt is unknown"):
            service.serve(attempted)
    assert fixture["calls"].read_text().splitlines() == ["critic", "paperorchestra_model"]


def test_parent_symlink_or_extra_claim_artifact_is_rejected(fixture):
    entry = fixture["parent"] / "claims" / ("2" * 32)
    (entry / "uncommitted.partial").write_text("partial")
    with pytest.raises(relay.UnknownDispatch, match="unexpected evidence"):
        _service(fixture)


def test_expired_continuation_cannot_start(fixture):
    auth = replace(fixture["authorization"], deadline=time.time() - 1)
    with pytest.raises(relay.RelayError, match="must be live"):
        _service(fixture, authorization=auth)


def test_parent_and_continuation_directories_cannot_overlap(fixture):
    fixture["state"] = fixture["parent"] / "child"
    fixture["authorization"] = replace(fixture["authorization"], authorized_state_path=str(fixture["state"]))
    with pytest.raises(relay.RelayError, match="separate directories"):
        _service(fixture)
    assert not fixture["state"].exists()


@pytest.mark.parametrize("scope", ["state", "parent"])
def test_authorization_is_bound_to_one_state_and_parent_path(fixture, scope):
    if scope == "state":
        fixture["state"] = fixture["state"].with_name("unauthorized-second-state")
    else:
        fixture["authorization"] = replace(fixture["authorization"],
            authorized_parent_state_path=str(fixture["parent"].with_name("different-parent")))
    with pytest.raises(relay.RelayError, match="state path differs"):
        _service(fixture)


@pytest.mark.parametrize("field,value", [
    ("authorized_state_path", "relative"), ("authorized_parent_state_path", "relative"),
    ("controller_plan_ref", "unbound"), ("controller_admission_ref", "unbound"),
])
def test_required_controller_and_path_bindings_are_validated(fixture, field, value):
    with pytest.raises(relay.RelayError, match="Invalid explicit continuation"):
        replace(fixture["authorization"], **{field: value})
