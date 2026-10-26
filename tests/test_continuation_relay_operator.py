"""Bind controller authorization to the local relay; every record is synthetic."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scientisttwo import continuation_relay_operator as operator
from scientisttwo.studies import media_relay as relay
from test_continuation_relay import fixture as make_fixture


@pytest.fixture
def admitted(tmp_path):
    data = make_fixture.__wrapped__(tmp_path, SimpleNamespace(param={}))
    auth = data["authorization"]
    entry = data["parent"] / "claims" / auth.failed_request_id
    claim = relay.read_json(entry / "claim.json", relay.REQUEST_LIMIT + 4096)
    proof = {name + "_sha256": relay.sha(relay.canonical(relay.read_json(entry / (name + ".json"),
        relay.REQUEST_LIMIT + 4096))) for name in ("claim", "dispatch", "response")}
    proof.update(status="verified", request_sha256=claim["request_sha256"])
    plan = {"schema_version": "scientisttwo-writer-continuation-v1",
            "continuation_id": auth.authorization_id, "failed_dispatch_proof": proof,
            "protected_metadata": {"live_deadline": auth.deadline}}
    plan_ref = "artifacts/" + relay.sha(relay.canonical(plan)) + ".json"
    admission = {"schema_version": "scientisttwo-continuation-admission-v1", "plan_ref": plan_ref,
        "allowance": {"plan_ref": plan_ref, "usage_disposition": "acknowledge_unknown_with_segment_limit",
                      "max_new_calls": auth.max_new_calls, "token_stop_threshold": auth.new_token_stop_threshold,
                      "authorization_ref": "synthetic-test-only"}}
    admission_ref = "artifacts/" + relay.sha(relay.canonical(admission)) + ".json"
    auth = replace(auth, controller_plan_ref=plan_ref, controller_admission_ref=admission_ref)
    return data, auth, plan, admission


def test_controller_artifact_and_relay_proof_binding(admitted, tmp_path):
    data, auth, plan, admission = admitted
    operator.bind_controller(auth, plan, admission, data["parent"])
    path = tmp_path / "plan.json"
    relay.publish_new(path, plan)
    assert operator.bound_artifact(path, auth.controller_plan_ref) == plan
    with pytest.raises(ValueError, match="immutable reference"):
        operator.bound_artifact(path, auth.controller_admission_ref)


@pytest.mark.parametrize("change", [
    {"max_new_calls": 4}, {"new_token_stop_threshold": 61},
    {"unknown_cost_acknowledged": False}, {"deadline": 1e20},
])
def test_relay_cannot_expand_controller_allowance(admitted, change):
    data, auth, plan, admission = admitted
    with pytest.raises(ValueError, match="allowance"):
        operator.bind_controller(replace(auth, **change), plan, admission, data["parent"])


def test_controller_must_bind_exact_failed_local_claim(admitted):
    data, auth, plan, admission = admitted
    plan["failed_dispatch_proof"]["claim_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="failed local dispatch"):
        operator.bind_controller(auth, plan, admission, data["parent"])


def test_serve_without_allow_live_never_constructs_model_service(admitted, monkeypatch):
    data, _, _, _ = admitted
    monkeypatch.setattr(operator.continuation, "ContinuationLocalService",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Must not construct backend")))
    with pytest.raises(ValueError, match="--allow-live"):
        operator.run(SimpleNamespace(parent_state=data["parent"], mode="serve", allow_live=False))
