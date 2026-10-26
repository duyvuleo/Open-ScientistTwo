"""Offline reconciliation of a completed response missed by the controller.

This administrative module never invokes a CLI, model, experiment or transport.
It preserves the failed delivery operation and adds a separately identified
successful completion. It does NOT authorize further continuation: the original
ContinuationBudget deliberately continues to reject the new failed operation
until a separately reviewed resolution-aware gate is implemented.
"""
from pathlib import Path
import math
import re
import time

from .backends import CodexBackend, StructuredOutputCodec, strict_json_loads
from .continuation import _rows, PROTECTED_METADATA, FINAL_MARKERS
from .integrations.paperorchestra.controller_writer import PaperOrchestraResponse
from .live.adapters import implementation_identity
from .models import AgentResult
from .process import ProcessResult
from .store import IntegrityError, canonical, digest, file_digest
from .studies.media_relay import validate_response


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _entry(store, failed_operation, ask_key):
    if (not isinstance(ask_key, str) or not ask_key or len(ask_key) > 1000
            or not failed_operation.startswith("model/" + ask_key + "/")):
        raise IntegrityError("The failed operation must belong to the exact original ask prefix")
    failed = store.db.execute("SELECT * FROM live_operations WHERE key=?", (failed_operation,)).fetchone()
    if (failed is None or failed["kind"] != "model" or failed["status"] != "failed"
            or not failed["result_ref"] or failed["tokens"] != 0):
        raise IntegrityError("An uncharged, committed failed model delivery is required")
    error = store.json(failed["result_ref"])
    if not isinstance(error.get("error"), str) or "timed out" not in error["error"]:
        raise IntegrityError("The controller did not record a delivery timeout")
    request_ref = store.get("request/" + ask_key)
    if not request_ref:
        raise IntegrityError("Original controller request is missing")
    request = store.json(request_ref)
    if (request.get("role") != "paperorchestra_model"
            or digest(canonical(request)) != failed["request_hash"]
            or canonical(request.get("schema")) != canonical(PaperOrchestraResponse.model_json_schema())):
        raise IntegrityError("Only the exact original PaperOrchestra response contract is reconcilable")
    return dict(failed), request_ref, request


def _validated_response(store, failed_operation, ask_key, proof):
    failed, request_ref, request = _entry(store, failed_operation, ask_key)
    spec_ref = store.get("live_spec_ref")
    spec = store.json(spec_ref)
    if spec.get("implementation") != implementation_identity():
        raise IntegrityError("Reconciliation must use the original pinned CodexBackend")
    if not isinstance(proof, dict) or set(proof) != {"claim", "dispatch", "response"}:
        raise IntegrityError("Claim, dispatch and response evidence are all required")
    if len(canonical(proof)) > 90 * 1024**2:
        raise IntegrityError("Reconciliation proof exceeds its byte bound")
    try:
        claim, dispatch, response = (proof[key] for key in ("claim", "dispatch", "response"))
        queued = claim["request"]
        queued_hash = digest(canonical(queued))
        if (claim["request_sha256"] != queued_hash or dispatch["request_sha256"] != queued_hash
                or response["request_sha256"] != queued_hash
                or dispatch["usage_status"] != "known" or dispatch["synthetic"] is not False
                or dispatch["status"] != "completed" or dispatch["returncode"] != 0
                or dispatch["timed_out"] is not False or queued["schema_version"] != 2
                or queued["role"] != request["role"] or queued["model"] != request["model"]
                or dispatch["model"] != request["model"]
                or dispatch["source_sha256"] != queued["source_sha256"]
                or dispatch["relay_sha256"] != spec["implementation"]["studies/media_relay.py"]
                or not re.fullmatch(r"[a-f0-9]{64}", queued["source_sha256"])):
            raise IntegrityError("Successful relay provenance does not match the failed controller request")
        timestamps = [queued["created_at"], claim["claimed_at"], dispatch["started_at"],
                      dispatch["completed_at"], queued["deadline"], store.get("live_deadline")]
        if not all(_finite(value) for value in timestamps) or timestamps != sorted(timestamps):
            raise IntegrityError("The response was not demonstrably completed before its original deadline")
        if not _finite(queued["timeout"]) or not 0 < queued["timeout"] <= 600:
            raise IntegrityError("Unbounded original model call")
        if queued["deadline"] > queued["created_at"] + queued["timeout"] + 1e-6:
            raise IntegrityError("Original request deadline exceeds its call limit")
        process, output = validate_response(response, queued)
        if (output is None or process["code"] or process["timed_out"] or process["limit_exceeded"]
                or process["stdout_truncated"] or process["stderr_truncated"]):
            raise IntegrityError("Only an intact successful relay response can be reconciled")
        codec = StructuredOutputCodec(request["schema"])
        if canonical(codec.wire_schema) != canonical(queued["schema"]):
            raise IntegrityError("Saved relay schema differs from the controller schema")
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise IntegrityError("Invalid or unbound completed relay proof") from error

    replay_calls = []

    def replay(argv, timeout, *, stdin=None):
        # This is the only executor supplied to CodexBackend. No process exists.
        if stdin != queued["prompt"]:
            raise IntegrityError("Replay backend prompt differs from the originally dispatched bytes")
        directory = Path(argv[argv.index("--cd") + 1])
        schema = strict_json_loads((directory / "response-schema.json").read_text())
        if canonical(schema) != canonical(queued["schema"]) or argv[argv.index("--model") + 1] != queued["model"]:
            raise IntegrityError("Replay invocation differs from original model/schema")
        (directory / "response.json").write_text(output)
        replay_calls.append(True)
        return ProcessResult(**process)

    backend = CodexBackend(store.root, request["model"], allow_live=True,
                           executable="offline-replay-no-process", executor=replay)
    result = backend.invoke(request["role"], request["inputs"], PaperOrchestraResponse)
    if len(replay_calls) != 1:
        raise IntegrityError("Offline backend replay did not validate exactly one response")
    usage = result.usage
    if any(type(usage.get(key)) is not int or usage[key] < 0 for key in ("input_tokens", "output_tokens")):
        raise IntegrityError("Replayed response has no valid completed usage")
    tokens = usage["input_tokens"] + usage["output_tokens"]
    if type(dispatch.get("observed_tokens")) is not int or dispatch["observed_tokens"] != tokens:
        raise IntegrityError("Relay usage and replayed completed-turn usage differ")
    return {"failed_record": failed, "request_ref": request_ref, "request": request,
            "queued_request_sha256": queued_hash, "provider_request_id": queued["request_id"],
            "result": result, "events": backend.events, "tokens": tokens,
            "completed_at": dispatch["completed_at"], "deadline": queued["deadline"],
            "backend_sha256": file_digest(Path(__import__(CodexBackend.__module__, fromlist=["__file__"]).__file__))}


def _verify_plan(store, plan_ref):
    plan = store.json(plan_ref)
    if (plan.get("schema_version") != "scientisttwo-delivery-reconciliation-v1"
            or plan["store_root"] != str(store.root)
            or plan["module_sha256"] != file_digest(Path(__file__))
            or store.get("delivery/plan/" + plan["queued_request_sha256"]) != plan_ref):
        raise IntegrityError("Delivery plan/module/store binding changed")
    if {key: store.get(key) for key in PROTECTED_METADATA} != plan["protected_metadata"]:
        raise IntegrityError("Original execution configuration or deadline changed")
    if implementation_identity() != plan["implementation"]:
        raise IntegrityError("Pinned backend/source changed after reconciliation planning")
    current = _rows(store.db)
    for table, records in plan["historical_rows"].items():
        if any(current.get(table, {}).get(key) != value for key, value in records.items()):
            raise IntegrityError("Historical record changed during delivery reconciliation: " + table)
    if store.verify():
        raise IntegrityError("Registered evidence changed")
    return plan


def plan_reconciliation(store, failed_operation, ask_key, proof):
    """Validate saved output offline and freeze an administrative reconciliation plan."""
    validated = _validated_response(store, failed_operation, ask_key, proof)
    queued_hash = validated["queued_request_sha256"]
    if previous := store.get("delivery/plan/" + queued_hash):
        plan = _verify_plan(store, previous)
        if (plan["failed_operation"] != failed_operation or plan["ask_key"] != ask_key
                or plan["proof_sha256"] != digest(canonical(proof))):
            raise IntegrityError("Reconciliation inputs changed")
        return previous
    if any(store.get(key) for key in FINAL_MARKERS):
        raise IntegrityError("Reconciliation cannot attach writing after selection freeze or final access")
    if store.db.execute("SELECT 1 FROM live_operations WHERE status='running'").fetchone():
        raise IntegrityError("Active operations block administrative reconciliation")
    if store.db.execute("SELECT 1 FROM calls WHERE key=?", ("live-agent/" + ask_key,)).fetchone():
        raise IntegrityError("The original request already has an attached response")
    if any(row["status"] == "success" for row in store.db.execute(
            "SELECT * FROM live_operations WHERE kind='model'") if row["key"].startswith("model/" + ask_key + "/")):
        raise IntegrityError("The original request already has a successful model completion")
    historical = _rows(store.db)
    result_ref = store.put_json(validated["result"], "offline-validated-model-response")
    events_ref = store.put_json(validated["events"], "offline-replayed-model-events")
    proof_ref = store.put_json(proof, "late-response-proof")
    plan = {"schema_version": "scientisttwo-delivery-reconciliation-v1", "store_root": str(store.root),
            "module_sha256": file_digest(Path(__file__)), "backend_sha256": validated["backend_sha256"],
            "implementation": implementation_identity(), "created_at": time.time(),
            "protected_metadata": {key: store.get(key) for key in PROTECTED_METADATA},
            "historical_rows": historical, "ask_key": ask_key, "failed_operation": failed_operation,
            "failed_record": validated["failed_record"], "request_ref": validated["request_ref"],
            "queued_request_sha256": queued_hash, "provider_request_id": validated["provider_request_id"],
            "proof_ref": proof_ref, "proof_sha256": digest(canonical(proof)),
            "result_ref": result_ref, "events_ref": events_ref, "known_tokens": validated["tokens"],
            "completed_at": validated["completed_at"], "request_deadline": validated["deadline"],
            "success_operation": "model/" + ask_key + "/delivery-reconciliation/" + queued_hash + "/0",
            "paid_calls": 0, "training_runs": 0, "continuation_ready": False,
            "original_unknown_cost": "unchanged_unknown"}
    ref = store.put_json(plan, "delivery-reconciliation-plan")
    store.set("delivery/plan/" + queued_hash, ref)
    return ref


def reconcile_response(store, plan_ref):
    """Atomically attach a validated cached result; never erase the timed-out row."""
    plan = _verify_plan(store, plan_ref)
    key = "delivery/receipt/" + plan["queued_request_sha256"]
    if previous := store.get(key):
        report = audit_reconciliation(store, previous)
        if report["integrity"] != "pass":
            raise IntegrityError("Existing reconciliation failed integrity verification")
        return previous
    proof = store.json(plan["proof_ref"])
    validated = _validated_response(store, plan["failed_operation"], plan["ask_key"], proof)
    if (canonical(validated["result"]) != canonical(store.json(plan["result_ref"]))
            or validated["tokens"] != plan["known_tokens"]):
        raise IntegrityError("Offline validation changed after planning")
    if any(store.get(name) for name in FINAL_MARKERS):
        raise IntegrityError("Final access prevents writing-response reconciliation")
    receipt = {"schema_version": "scientisttwo-delivery-receipt-v1", "plan_ref": plan_ref,
               "failed_operation": plan["failed_operation"], "success_operation": plan["success_operation"],
               "result_ref": plan["result_ref"], "known_tokens": plan["known_tokens"],
               "paid_calls": 0, "training_runs": 0, "charged_seconds": 0,
               "accounting": "Known completed-call tokens attached once; timeout wall-time stays in failed operation",
               "original_unknown_cost": "unchanged_unknown", "continuation_ready": False}
    ref = store.put_json(receipt, "late-response-reconciliation-receipt")
    store.db.execute("BEGIN IMMEDIATE")
    try:
        # No operation may slip in between review and administrative attachment.
        if _rows(store.db)["live_operations"] != plan["historical_rows"]["live_operations"]:
            raise IntegrityError("Operations changed after reconciliation planning")
        if store.db.execute("SELECT 1 FROM calls WHERE key=?", ("live-agent/" + plan["ask_key"],)).fetchone():
            raise IntegrityError("Response was attached concurrently")
        store.db.execute("INSERT INTO live_operations(key,kind,request_hash,status,seconds,tokens,result_ref) VALUES(?,?,?,?,?,?,?)",
                         (plan["success_operation"], "model", plan["failed_record"]["request_hash"],
                          "success", 0, plan["known_tokens"], plan["result_ref"]))
        store.db.execute("INSERT INTO calls(key,input_hash,output_ref) VALUES(?,?,?)",
                         ("live-agent/" + plan["ask_key"], plan["failed_record"]["request_hash"], plan["result_ref"]))
        store.db.execute("INSERT INTO metadata(key,value) VALUES(?,?)", (key, canonical(ref).decode()))
        store.db.execute("INSERT INTO metadata(key,value) VALUES(?,?)",
                         ("events/" + plan["success_operation"], canonical(plan["events_ref"]).decode()))
        store.db.execute("INSERT INTO events(key,time,stage,reason) VALUES(?,?,?,?)", (
            "delivery/reconciled/" + plan["queued_request_sha256"], time.time(), "draft",
            "Attached the previously completed response offline; failed delivery preserved; no new paid call"))
        store.db.commit()
    except BaseException:
        store.db.rollback()
        raise
    return ref


def audit_reconciliation(store, receipt_ref):
    """Validate only the delivery repair. This never grants workflow/final readiness."""
    report = {"integrity": "fail", "paid_calls": 0, "training_runs": 0,
              "continuation_ready": False, "original_unknown_cost": "unchanged_unknown"}
    try:
        receipt = store.json(receipt_ref)
        plan = _verify_plan(store, receipt["plan_ref"])
        if (receipt.get("schema_version") != "scientisttwo-delivery-receipt-v1"
                or store.get("delivery/receipt/" + plan["queued_request_sha256"]) != receipt_ref
                or receipt["result_ref"] != plan["result_ref"] or receipt["known_tokens"] != plan["known_tokens"]
                or receipt["failed_operation"] != plan["failed_operation"]
                or receipt["success_operation"] != plan["success_operation"]
                or receipt["paid_calls"] != 0 or receipt["training_runs"] != 0
                or receipt["charged_seconds"] != 0 or receipt["continuation_ready"] is not False):
            raise IntegrityError("Reconciliation receipt binding changed")
        row = store.db.execute("SELECT * FROM live_operations WHERE key=?", (plan["success_operation"],)).fetchone()
        cached = store.db.execute("SELECT * FROM calls WHERE key=?", ("live-agent/" + plan["ask_key"],)).fetchone()
        if (row is None or cached is None or row["status"] != "success" or row["kind"] != "model"
                or row["seconds"] != 0 or row["tokens"] != plan["known_tokens"]
                or row["request_hash"] != plan["failed_record"]["request_hash"]
                or row["result_ref"] != plan["result_ref"] or cached["output_ref"] != row["result_ref"]
                or cached["input_hash"] != row["request_hash"]):
            raise IntegrityError("Reconciled operation/cache/accounting binding changed")
        response = AgentResult.model_validate(store.json(row["result_ref"]))
        if response.usage["input_tokens"] + response.usage["output_tokens"] != row["tokens"]:
            raise IntegrityError("Reconciled usage differs from validated response")
        completions = [dict(item) for item in store.db.execute("SELECT * FROM live_operations WHERE kind='model' AND status='success'")
                       if item["key"].startswith("model/" + plan["ask_key"] + "/")]
        if len(completions) != 1 or completions[0]["key"] != plan["success_operation"]:
            raise IntegrityError("The original model request has duplicate successful completions")
        report.update(integrity="pass", known_tokens=row["tokens"], success_operation=plan["success_operation"])
    except (IntegrityError, ValueError, OSError, KeyError, TypeError) as error:
        report["error"] = str(error)
    return report
