"""Explicit local service for an admitted continuation; inspect is model-free."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import time

from scientisttwo import continuation_relay as continuation
from scientisttwo.studies import media_relay as relay


def bound_artifact(path, ref):
    value = relay.read_json(Path(path), 16 * 1024 * 1024, private=False)
    if ref != "artifacts/" + relay.sha(relay.canonical(value)) + ".json":
        raise ValueError("Controller artifact differs from its immutable reference")
    return value


def bind_controller(authorization, plan, admission, parent_state):
    """Tie local dispatch limits to the worker's independently admitted segment."""
    if (authorization.controller_plan_ref != admission.get("plan_ref")
            or plan.get("continuation_id") != authorization.authorization_id
            or plan.get("schema_version") != "scientisttwo-writer-continuation-v1"
            or admission.get("schema_version") != "scientisttwo-continuation-admission-v1"):
        raise ValueError("Controller and relay continuation identities differ")
    allowance = admission["allowance"]
    if (allowance.get("plan_ref") != authorization.controller_plan_ref
            or allowance.get("usage_disposition") != "acknowledge_unknown_with_segment_limit"
            or not allowance.get("authorization_ref")
            or not authorization.unknown_cost_acknowledged
            or authorization.max_new_calls > allowance["max_new_calls"]
            or authorization.new_token_stop_threshold > allowance["token_stop_threshold"]
            or authorization.deadline > plan["protected_metadata"]["live_deadline"]):
        raise ValueError("Relay allowance exceeds or differs from controller admission")
    failed = Path(parent_state) / "claims" / authorization.failed_request_id
    claim = relay.read_json(failed / "claim.json", relay.REQUEST_LIMIT + 4096)
    dispatch = relay.read_json(failed / "dispatch.json", 16384)
    response = relay.read_json(failed / "response.json", relay.RESPONSE_LIMIT)
    proof = plan.get("failed_dispatch_proof") or {}
    expected = {"claim_sha256": relay.sha(relay.canonical(claim)),
                "dispatch_sha256": relay.sha(relay.canonical(dispatch)),
                "response_sha256": relay.sha(relay.canonical(response)),
                "request_sha256": claim["request_sha256"]}
    if proof.get("status") != "verified" or any(proof.get(k) != v for k, v in expected.items()):
        raise ValueError("Worker continuation does not bind this failed local dispatch")


def run(args):
    parent = Path(args.parent_state).resolve()
    manifest = continuation.parent_ledger_manifest(parent)
    if args.mode == "inspect":
        if args.proof_output:
            if not args.failed_request_id:
                raise ValueError("Export requires an exact failed request ID")
            import re
            if not re.fullmatch(r"[a-f0-9]{32}", args.failed_request_id):
                raise ValueError("Invalid failed request ID")
            failed = parent / "claims" / args.failed_request_id
            value = {name: relay.read_json(failed / (name + ".json"), limit)
                     for name, limit in (("claim", relay.REQUEST_LIMIT + 4096),
                                         ("dispatch", 16384), ("response", relay.RESPONSE_LIMIT))}
            if value["claim"]["request"]["role"] != "paperorchestra_model" or value["dispatch"]["usage_status"] != "unknown":
                raise ValueError("Only the failed unknown-cost writer proof may be exported")
            relay.publish_new(args.proof_output, value)
        return {"status": "inspected_no_calls", "parent_manifest_sha256": relay.sha(relay.canonical(manifest)),
                "repaired_sources": continuation.repaired_source_hashes(),
                "new_model_calls": 0, "proof_exported": bool(args.proof_output)}
    if not args.allow_live:
        raise ValueError("No live calls: relay continuation requires --allow-live")
    if not all((args.authorization, args.controller_plan, args.controller_admission,
                args.state, args.policy, args.transport, args.transport_sha256,
                args.host, args.queue, args.codex)):
        raise ValueError("Serve requires the complete admitted controller/relay/transport bindings")
    authorization = continuation.ContinuationRelayAuthorization.from_mapping(
        relay.read_json(args.authorization, 16384))
    plan = bound_artifact(args.controller_plan, authorization.controller_plan_ref)
    admission = bound_artifact(args.controller_admission, authorization.controller_admission_ref)
    bind_controller(authorization, plan, admission, parent)
    policy = relay.RelayPolicy.from_mapping(relay.read_json(args.policy, 4096))
    service = continuation.ContinuationLocalService(args.state, policy, parent_state=parent,
        authorization=authorization, executable=args.codex, allow_live=True)
    if (args.transport.is_symlink() or not args.transport.is_file()
            or relay.sha(args.transport.read_bytes()) != args.transport_sha256):
        raise ValueError("Transport operator differs from its pinned source")
    spec = importlib.util.spec_from_file_location("continuation_queue_transport", args.transport)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    # The original fixed SSH queue transport accepts this service by injection.
    # It performs no model selection or backend construction itself.
    print(json.dumps({"event": "continuation_relay_started", "authorization": asdict(authorization),
                      "parent_usage": "known tokens plus unresolved original call"}), flush=True)
    while time.time() < authorization.deadline:
        if not transport.serve_once(service, args.host, args.queue):
            time.sleep(2)
    return {"status": "continuation_deadline_reached"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["inspect", "serve"])
    parser.add_argument("--parent-state", type=Path, required=True)
    for name in ("authorization", "controller-plan", "controller-admission", "state", "policy",
                 "transport", "codex", "proof-output"):
        parser.add_argument("--" + name, type=Path)
    for name in ("host", "queue", "transport-sha256", "failed-request-id"):
        parser.add_argument("--" + name)
    parser.add_argument("--allow-live", action="store_true")
    try:
        print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "blocked", "error_type": type(error).__name__, "reason": str(error)}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
