"""Explicit append-only continuation of a failed writer transport operation.

An operator may load this module from outside an unchanged pinned package using
the name ``scientisttwo.continuation``. It never changes scientific source,
configuration, historical evidence, deadlines, or the original failed operation.
An acknowledged unknown cost remains unknown, even when integrity checks pass.
"""
from pathlib import Path
import math
import re
import sqlite3
import time

from .live.adapters import implementation_identity
from .live.runtime import AmbiguousCall, LiveBudgetExhausted
from .store import IntegrityError, canonical, digest, file_digest


TABLE_KEYS = {"live_operations": "key", "calls": "key", "events": "key", "artifacts": "path",
              "study_compute_operations": "key"}
PROTECTED_METADATA = (
    "live_spec_ref", "live_sources_ref", "adapter_sources_ref", "live_adapter_identity",
    "live_backend", "engine_version", "authorized_source_snapshot", "study_plan_ref",
    "live_deadline", "study_compute_budget", "live_images", "runtime_hardware",
)
FINAL_MARKERS = ("live_frozen", "study_final_grant_ref", "study_final_release_ref",
                 "study_final_budget_activation", "study_research_freeze_ref")
ALLOWANCE_KEYS = {"schema_version", "plan_ref", "max_new_calls", "token_stop_threshold",
                  "usage_disposition", "authorization_ref"}


def _rows(connection):
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    result = {}
    for table, key in TABLE_KEYS.items():
        if table not in tables:
            if table == "study_compute_operations":
                continue
            raise IntegrityError("Continuation requires the campaign evidence tables")
        result[table] = {row[key]: digest(canonical(dict(row)))
                         for row in connection.execute("SELECT * FROM " + table)}
    return result


def _append_metadata(store, key, value):
    old = store.get(key)
    if old is not None and old != value:
        raise IntegrityError("Continuation metadata already binds different evidence: " + key)
    if old is None:
        store.set(key, value)


def _source_identity(store, expected):
    if not expected or implementation_identity() != expected:
        raise IntegrityError("Pinned scientific implementation changed")
    index = store.get("live_sources_ref")
    if not index:
        raise IntegrityError("Continuation requires the recorded implementation source closure")
    from . import backends
    package = Path(backends.__file__).resolve().parent
    for name, ref in store.json(index).items():
        relative = Path(name)
        source = package / relative
        if (relative.is_absolute() or ".." in relative.parts or source.is_symlink()
                or not source.is_file() or digest(store.read(ref)) != file_digest(source)):
            raise IntegrityError("Pinned implementation closure differs: " + name)


def _plan(store, plan_ref):
    plan = store.json(plan_ref)
    if (plan.get("schema_version") != "scientisttwo-writer-continuation-v1"
            or plan.get("store_root") != str(store.root)
            or plan.get("module_sha256") != file_digest(Path(__file__))
            or store.get("continuation/plan/" + plan.get("continuation_id", "")) != plan_ref):
        raise IntegrityError("Continuation plan/module/store identity changed")
    return plan


def _verify_parent(store, plan, *, deep=False):
    _source_identity(store, plan["implementation"])
    if {key: store.get(key) for key in PROTECTED_METADATA} != plan["protected_metadata"]:
        raise IntegrityError("Original specification, source, protocol, budget or deadline changed")
    current = _rows(store.db)
    for table, rows in plan["historical_rows"].items():
        if any(current.get(table, {}).get(key) != value for key, value in rows.items()):
            raise IntegrityError("Historical evidence rows changed: " + table)
    snapshot = store.root / plan["snapshot"]["path"]
    if (snapshot.is_symlink() or not snapshot.is_file()
            or file_digest(snapshot) != plan["snapshot"]["sha256"]):
        raise IntegrityError("Immutable pre-continuation SQLite snapshot changed")
    if deep and store.verify():
        raise IntegrityError("Registered campaign artifact content changed")


def _eligible(store, failed_operation):
    if store.get("live_state", {}).get("stage") != "draft":
        raise IntegrityError("Writer continuation requires an existing draft-stage checkpoint")
    if any(store.get(key) for key in FINAL_MARKERS):
        raise IntegrityError("Writer continuation cannot follow a selection freeze or final grant")
    if store.db.execute("SELECT 1 FROM live_operations WHERE status='running' OR key LIKE 'final/%'").fetchone():
        raise IntegrityError("Running or protected-final operations block continuation")
    match = re.fullmatch(r"model/(.+)/0", failed_operation)
    if not match:
        raise ValueError("Continuation accepts one exact first-attempt model operation")
    row = store.db.execute("SELECT * FROM live_operations WHERE key=?", (failed_operation,)).fetchone()
    if row is None or row["kind"] != "model" or row["status"] != "failed" or not row["result_ref"]:
        raise IntegrityError("Continuation requires a committed failed model operation")
    request_ref = store.get("request/" + match[1])
    if not request_ref:
        raise IntegrityError("Failed writer request provenance is missing")
    request = store.json(request_ref)
    if request.get("role") != "paperorchestra_model" or digest(canonical(request)) != row["request_hash"]:
        raise IntegrityError("Failed operation is not bound to the PaperOrchestra request")
    if store.db.execute("SELECT 1 FROM calls WHERE key=?", ("live-agent/" + match[1],)).fetchone():
        raise IntegrityError("Failed writer already has a successful cached response")
    store.read(row["result_ref"])
    return dict(row), request_ref


def verify_failed_dispatch_proof(store, failed_operation, proof):
    """Bind a private local relay receipt to this exact controller request.

    Return only fixed fields and hashes; the proof's prompts and diagnostics are
    not copied into the continuation plan.
    """
    from .backends import StructuredOutputCodec, STRUCTURED_WIRE_CONTRACT_KEY, strict_json_loads
    from .studies.media_relay import INSTRUCTIONS, RelayError, validate_response
    if not isinstance(proof, dict) or set(proof) != {"claim", "dispatch", "response"}:
        raise IntegrityError("Failed-dispatch proof requires claim, dispatch and response records")
    match = re.fullmatch(r"model/(.+)/0", failed_operation)
    row = store.db.execute("SELECT * FROM live_operations WHERE key=?", (failed_operation,)).fetchone()
    if (not match or row is None or row["kind"] != "model"
            or row["status"] != "failed" or not row["result_ref"]):
        raise IntegrityError("Failed-dispatch proof has no committed failed controller operation")
    request_ref = store.get("request/" + match[1])
    request = store.json(request_ref)
    if request.get("role") != "paperorchestra_model" or digest(canonical(request)) != row["request_hash"]:
        raise IntegrityError("Failed-dispatch request differs from its committed controller operation")
    try:
        claim, dispatch, response = (proof[key] for key in ("claim", "dispatch", "response"))
        queued = claim["request"]
        request_hash = digest(canonical(queued))
        if (claim["request_sha256"] != request_hash or dispatch["request_sha256"] != request_hash
                or response["request_sha256"] != request_hash
                or dispatch["synthetic"] is not False or dispatch["status"] != "completed"
                or dispatch["usage_status"] != "unknown"
                or dispatch["model"] != request["model"] or queued["model"] != request["model"]
                or queued["role"] != request["role"] or queued["schema_version"] != 2
                or dispatch["source_sha256"] != queued["source_sha256"]
                or not re.fullmatch(r"[a-f0-9]{64}", queued["source_sha256"])):
            raise IntegrityError("Failed relay provenance differs from the controller request")
        spec = store.json(store.get("live_spec_ref"))
        if dispatch["relay_sha256"] != spec["implementation"]["studies/media_relay.py"]:
            raise IntegrityError("Failed dispatch used a different pinned relay implementation")
        process, output = validate_response(response, queued)
        if output is not None or not process["code"] or dispatch["returncode"] != process["code"]:
            raise IntegrityError("Continuation requires a failed response without an admitted output")
        prompt = strict_json_loads(queued["prompt"])
        codec = StructuredOutputCodec(request["schema"])
        inputs = dict(prompt["inputs"])
        wire = inputs.pop(STRUCTURED_WIRE_CONTRACT_KEY)
        if (set(prompt) != {"inputs", "instructions", "prompt_version", "role"}
                or prompt["instructions"] != INSTRUCTIONS or prompt["prompt_version"] != "scientisttwo-v1"
                or prompt["role"] != request["role"] or canonical(inputs) != canonical(request["inputs"])
                or wire["codec_identity"] != codec.identity or wire["version"] != codec.version
                or canonical(queued["schema"]) != canonical(codec.wire_schema)):
            raise IntegrityError("Failed relay prompt/schema is not the exact failed controller request")
    except (KeyError, TypeError, ValueError, RelayError) as error:
        raise IntegrityError("Malformed or unbound failed-dispatch proof") from error
    return {"status": "verified", "claim_sha256": digest(canonical(claim)),
            "dispatch_sha256": digest(canonical(dispatch)), "response_sha256": digest(canonical(response)),
            "request_sha256": request_hash, "controller_request_ref": request_ref,
            "source_sha256": queued["source_sha256"], "relay_sha256": dispatch["relay_sha256"],
            "usage_status": "unknown", "observed_tokens": None}


def prepare_plan(store, failed_operation, continuation_id, *, failed_dispatch_proof=None):
    """Snapshot and bind evidence. This prepares no backend and grants no live calls."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", continuation_id):
        raise ValueError("A bounded simple continuation ID is required")
    proof = (verify_failed_dispatch_proof(store, failed_operation, failed_dispatch_proof)
             if failed_dispatch_proof is not None else None)
    if not store.get("live_backend", {}).get("synthetic", False) and proof is None:
        raise IntegrityError("Live continuation planning requires the bound local failed-dispatch proof")
    key = "continuation/plan/" + continuation_id
    if ref := store.get(key):
        plan = _plan(store, ref)
        if plan["failed_operation"] != failed_operation or plan["failed_dispatch_proof"] != proof:
            raise IntegrityError("Continuation preparation inputs changed")
        _verify_parent(store, plan, deep=True)
        return ref
    failed, request_ref = _eligible(store, failed_operation)
    spec_ref = store.get("live_spec_ref")
    if not spec_ref:
        raise IntegrityError("An immutable campaign specification is required")
    specification = store.json(spec_ref)
    implementation = specification.get("implementation")
    _source_identity(store, implementation)
    if store.verify():
        raise IntegrityError("Campaign artifacts failed integrity verification")
    deadline = store.get("live_deadline")
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise IntegrityError("A committed finite campaign deadline is required")
    parent = store.root / ".continuations"
    if parent.is_symlink():
        raise IntegrityError("Continuation snapshot directory cannot be a symlink")
    parent.mkdir(exist_ok=True, mode=0o700)
    directory = parent / continuation_id
    directory.mkdir(mode=0o700)  # Never overwrite an interrupted preparation.
    snapshot = directory / "snapshot.sqlite"
    with snapshot.open("xb"):
        pass
    snapshot.chmod(0o600)
    connection = sqlite3.connect(snapshot)
    connection.row_factory = sqlite3.Row
    try:
        store.db.backup(connection)
        historical = _rows(connection)
    finally:
        connection.close()
    snapshot.chmod(0o400)
    if _rows(store.db) != historical:
        raise IntegrityError("Campaign changed during continuation snapshot")
    plan = {
        "schema_version": "scientisttwo-writer-continuation-v1", "continuation_id": continuation_id,
        "store_root": str(store.root), "created_at": time.time(),
        "module_sha256": file_digest(Path(__file__)), "implementation": implementation,
        "protected_metadata": {key: store.get(key) for key in PROTECTED_METADATA},
        "historical_rows": historical,
        "snapshot": {"path": snapshot.relative_to(store.root).as_posix(), "sha256": file_digest(snapshot)},
        "failed_operation": failed_operation, "failed_record": failed,
        "request_ref": request_ref, "failed_dispatch_proof": proof,
        "replacement_operation": failed_operation[:-2] + "/continuation/" + continuation_id + "/0",
        "usage_status": "unknown", "live_admission": False,
        "original_total_token_ceiling_verified": False,
    }
    ref = store.put_json(plan, "writer-continuation-plan")
    _append_metadata(store, key, ref)
    return ref


def admit_continuation(store, plan_ref, allowance):
    """Accept an explicit bounded segment allowance; never infer user approval.

    The caller is responsible for obtaining the referenced authorization. This
    initial implementation deliberately does not accept estimated/reconciled
    historical token usage. The original unknown amount remains visible.
    """
    plan = _plan(store, plan_ref)
    _verify_parent(store, plan, deep=True)
    if not isinstance(allowance, dict) or set(allowance) != ALLOWANCE_KEYS:
        raise ValueError("An explicit complete continuation allowance is required")
    if (allowance["schema_version"] != "scientisttwo-continuation-allowance-v1"
            or allowance["plan_ref"] != plan_ref
            or allowance["usage_disposition"] != "acknowledge_unknown_with_segment_limit"
            or not isinstance(allowance["authorization_ref"], str)
            or not 0 < len(allowance["authorization_ref"]) <= 1000):
        raise ValueError("Unknown usage needs an explicit bounded-segment acknowledgment")
    for key in ("max_new_calls", "token_stop_threshold"):
        if type(allowance[key]) is not int or allowance[key] <= 0:
            raise ValueError("Continuation model ceilings must be explicit positive integers")
    if previous := store.get("continuation/admission/" + plan["continuation_id"]):
        saved = store.json(previous)
        if saved["allowance"] != allowance:
            raise IntegrityError("Continuation allowance cannot change on resume")
        return previous
    _eligible(store, plan["failed_operation"])
    if _rows(store.db)["live_operations"] != plan["historical_rows"]["live_operations"]:
        raise IntegrityError("Operations were added after planning without continuation admission")
    spec = store.json(plan["protected_metadata"]["live_spec_ref"])
    used_calls, used_tokens = store.db.execute(
        "SELECT count(*),coalesce(sum(tokens),0) FROM live_operations WHERE kind='model'").fetchone()
    if (allowance["max_new_calls"] > spec["max_calls"] - used_calls
            or spec.get("token_stop_threshold") is None
            or allowance["token_stop_threshold"] > spec["token_stop_threshold"] - used_tokens):
        raise LiveBudgetExhausted("Segment allowance exceeds the remaining known original ceilings")
    if time.time() >= store.get("live_deadline"):
        raise LiveBudgetExhausted("Original campaign deadline expired; continuation cannot extend it")
    admission = {"schema_version": "scientisttwo-continuation-admission-v1", "plan_ref": plan_ref,
                 "allowance": allowance, "admitted_at": time.time(),
                 "original_total_token_ceiling_verified": False,
                 "original_unknown_operation": plan["failed_operation"],
                 "scope": "New segment limits only; original total token consumption remains unknown"}
    ref = store.put_json(admission, "writer-continuation-admission")
    _append_metadata(store, "continuation/admission/" + plan["continuation_id"], ref)
    return ref


def _admission(store, plan_ref, admission_ref):
    plan = _plan(store, plan_ref)
    value = store.json(admission_ref)
    if (value.get("schema_version") != "scientisttwo-continuation-admission-v1"
            or value.get("plan_ref") != plan_ref
            or store.get("continuation/admission/" + plan["continuation_id"]) != admission_ref):
        raise IntegrityError("Continuation admission is absent or belongs to another plan")
    allowance = value.get("allowance")
    if (not isinstance(allowance, dict) or set(allowance) != ALLOWANCE_KEYS
            or allowance["schema_version"] != "scientisttwo-continuation-allowance-v1"
            or allowance["plan_ref"] != plan_ref
            or allowance["usage_disposition"] != "acknowledge_unknown_with_segment_limit"
            or not isinstance(allowance["authorization_ref"], str)
            or not 0 < len(allowance["authorization_ref"]) <= 1000
            or any(type(allowance[key]) is not int or allowance[key] <= 0
                   for key in ("max_new_calls", "token_stop_threshold"))):
        raise IntegrityError("Saved continuation allowance is invalid")
    spec = store.json(plan["protected_metadata"]["live_spec_ref"])
    original = [dict(row) for row in store.db.execute("SELECT * FROM live_operations WHERE kind='model'")
                if row["key"] in plan["historical_rows"]["live_operations"]]
    if (allowance["max_new_calls"] > spec["max_calls"] - len(original)
            or spec.get("token_stop_threshold") is None
            or allowance["token_stop_threshold"] > spec["token_stop_threshold"] - sum(row["tokens"] for row in original)):
        raise IntegrityError("Saved allowance exceeds original remaining known ceilings")
    return plan, value


def _pending_final_aggregate(store, row):
    if not (row["status"] == "running" and row["result_ref"] is None
            and row["key"] in {"final/baseline", "final/selected"}
            and row["kind"] == "experiment" and row["seconds"] == row["tokens"] == 0
            and store.get("study_final_release_ref")):
        return False
    marker = store.db.execute("SELECT bucket,aggregation FROM study_compute_operations WHERE key=?",
                              (row["key"],)).fetchone()
    return bool(marker and marker["aggregation"] == 1 and marker["bucket"] == row["key"])


class ContinuationBudget:
    """Preserve old operations; remap exactly one failed writer operation."""
    def __init__(self, delegate, plan_ref, admission_ref):
        self.delegate, self.store, self.config = delegate, delegate.store, delegate.config
        self.plan_ref, self.admission_ref = plan_ref, admission_ref
        self.plan, self.admission = _admission(self.store, plan_ref, admission_ref)
        _verify_parent(self.store, self.plan, deep=True)
        if canonical(self.config) != canonical(self.store.json(self.plan["protected_metadata"]["live_spec_ref"])):
            raise IntegrityError("Continuation budget configuration differs from the original specification")
        self.store.db.execute("""CREATE TABLE IF NOT EXISTS continuation_operations(
            key TEXT PRIMARY KEY, continuation_id TEXT NOT NULL, kind TEXT NOT NULL,
            request_hash TEXT NOT NULL, after_writer_completion INTEGER NOT NULL)""")
        self.store.db.commit()

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def _mapped(self, key):
        return self.plan["replacement_operation"] if key == self.plan["failed_operation"] else key

    def _verify(self):
        _verify_parent(self.store, self.plan)
        if canonical(self.config) != canonical(self.store.json(self.plan["protected_metadata"]["live_spec_ref"])):
            raise IntegrityError("Continuation configuration cannot change")

    def _new_rows(self):
        historical = self.plan["historical_rows"]["live_operations"]
        return [dict(row) for row in self.store.db.execute("SELECT * FROM live_operations") if row["key"] not in historical]

    def _writer_completed(self):
        row = self.store.db.execute("SELECT status,result_ref FROM live_operations WHERE key=?",
                                    (self.plan["replacement_operation"],)).fetchone()
        return bool(row and row["status"] == "success" and row["result_ref"])

    def reserve(self, key, kind, request):
        self._verify()
        mapped = self._mapped(key)
        if key == self.plan["failed_operation"] and (kind != "model" or digest(canonical(request)) != self.plan["failed_record"]["request_hash"]):
            raise IntegrityError("Replacement must use the exact failed writer request")
        prior = self.store.db.execute("SELECT * FROM live_operations WHERE key=?", (mapped,)).fetchone()
        if prior is not None:
            if mapped not in self.plan["historical_rows"]["live_operations"]:
                marker = self.store.db.execute("SELECT * FROM continuation_operations WHERE key=?", (mapped,)).fetchone()
                if not marker or marker["continuation_id"] != self.plan["continuation_id"]:
                    raise IntegrityError("New operation lacks continuation admission provenance")
            return self.delegate.reserve(mapped, kind, request)
        completed = self._writer_completed()
        if not completed and mapped != self.plan["replacement_operation"]:
            raise IntegrityError("Only the exact writer retry may start before it completes")
        rows = self._new_rows()
        if any(row["status"] == "running" and not _pending_final_aggregate(self.store, row) for row in rows):
            raise AmbiguousCall("Continuation has an unresolved operation; no further dispatch")
        if any(row["kind"] == "model" and row["status"] != "success" for row in rows):
            raise AmbiguousCall("A new continuation model failure requires separate review; no further dispatch")
        if kind == "model":
            models = [row for row in self._new_rows() if row["kind"] == "model"]
            limits = self.admission["allowance"]
            if (len(models) >= limits["max_new_calls"]
                    or sum(row["tokens"] for row in models) >= limits["token_stop_threshold"]):
                raise LiveBudgetExhausted("Continuation model-call or observed-token segment ceiling reached")
        result = self.delegate.reserve(mapped, kind, request)
        with self.store.db:
            self.store.db.execute("INSERT INTO continuation_operations VALUES(?,?,?,?,?)", (
                mapped, self.plan["continuation_id"], kind, digest(canonical(request)), int(completed)))
        return result

    def finish(self, key, status, result, seconds, tokens=0):
        self._verify()
        mapped = self._mapped(key)
        if mapped in self.plan["historical_rows"]["live_operations"]:
            raise IntegrityError("Continuation cannot rewrite a historical completion")
        marker = self.store.db.execute("SELECT * FROM continuation_operations WHERE key=?", (mapped,)).fetchone()
        if marker is None or marker["continuation_id"] != self.plan["continuation_id"]:
            raise IntegrityError("Operation was not admitted by this continuation")
        self.delegate.finish(mapped, status, result, seconds, tokens)

    def activate_final_grant(self, grant_ref, deadline):
        report = audit_continuation(self.store, self.plan_ref, self.admission_ref,
                                    _allow_final_aggregates=True)
        if report["integrity"] != "pass" or not self._writer_completed():
            raise IntegrityError("Continuation integrity must pass before protected final activation")
        key = "continuation/final-audit/" + self.plan["continuation_id"] + "/" + digest(grant_ref.encode())
        if prior := self.store.get(key):
            saved = self.store.json(prior)
            if (saved.get("integrity") != "pass" or saved.get("plan_ref") != self.plan_ref
                    or saved.get("admission_ref") != self.admission_ref):
                raise IntegrityError("The recorded final-admission audit changed")
        else:
            _append_metadata(self.store, key, self.store.put_json(report, "continuation-final-admission-audit"))
        return self.delegate.activate_final_grant(grant_ref, deadline)


def install_continuation(controller, plan_ref, admission_ref):
    """Attach the budget only; ordinary controller replay reuses historical work."""
    if isinstance(controller.budget, ContinuationBudget):
        if (controller.budget.plan_ref, controller.budget.admission_ref) != (plan_ref, admission_ref):
            raise IntegrityError("Controller already has a different continuation")
        return controller.budget
    if controller.runner.budget is not controller.budget:
        raise IntegrityError("Controller and experiment runner must share one budget")
    budget = ContinuationBudget(controller.budget, plan_ref, admission_ref)
    controller.budget = controller.runner.budget = budget
    return budget


def audit_continuation(store, plan_ref, admission_ref=None, *, _allow_final_aggregates=False):
    """Check immutable history and new admissions without changing evidence."""
    report = {"integrity": "fail", "original_total_token_ceiling_verified": False,
              "historical_usage_status": "unknown", "repeated_historical_experiments": 0}
    try:
        plan = _plan(store, plan_ref)
        _verify_parent(store, plan, deep=True)
        parent = plan["historical_rows"]["live_operations"]
        added = [dict(row) for row in store.db.execute("SELECT * FROM live_operations ORDER BY rowid") if row["key"] not in parent]
        if admission_ref is None:
            if added:
                raise IntegrityError("New operations exist without continuation admission")
            return {**report, "integrity": "pass", "admitted": False, "new_operations": 0}
        _, admission = _admission(store, plan_ref, admission_ref)
        markers = {row["key"]: dict(row) for row in store.db.execute("SELECT * FROM continuation_operations")}
        pending_aggregates = []
        for row in added:
            marker = markers.get(row["key"])
            if (not marker or marker["continuation_id"] != plan["continuation_id"]
                    or marker["kind"] != row["kind"] or marker["request_hash"] != row["request_hash"]):
                raise IntegrityError("New operation lacks an exact continuation admission")
            if not marker["after_writer_completion"] and row["key"] != plan["replacement_operation"]:
                raise IntegrityError("An operation preceded completion of the writer replacement")
            if row["status"] == "running" or not row["result_ref"]:
                # The seed verifier runs inside a declared zero-cost aggregate.
                # This is a known coordination record, never a physical process.
                if not _allow_final_aggregates or not _pending_final_aggregate(store, row):
                    raise IntegrityError("Continuation contains an unresolved completion")
                pending_aggregates.append(row["key"])
            if row["kind"] == "model" and row["status"] != "success":
                raise IntegrityError("New failed model usage is unresolved; continuation cannot pass")
        models = [row for row in added if row["kind"] == "model"]
        if len(models) > admission["allowance"]["max_new_calls"]:
            raise IntegrityError("Continuation exceeded its model-call ceiling")
        cumulative = 0
        spec = store.json(plan["protected_metadata"]["live_spec_ref"])
        original_models = [dict(row) for row in store.db.execute("SELECT * FROM live_operations WHERE kind='model'")
                           if row["key"] in parent]
        original_tokens = sum(row["tokens"] for row in original_models)
        if len(models) + len(original_models) > spec["max_calls"]:
            raise IntegrityError("Continuation exceeded the combined original call ceiling")
        for row in models:
            if cumulative >= admission["allowance"]["token_stop_threshold"]:
                raise IntegrityError("Continuation admitted a call after its observed-token stop threshold")
            if original_tokens + cumulative >= spec["token_stop_threshold"]:
                raise IntegrityError("Continuation admitted a call after the original known-token threshold")
            cumulative += row["tokens"]
        return {**report, "integrity": "pass", "admitted": True,
                "plan_ref": plan_ref, "admission_ref": admission_ref, "new_operations": len(added),
                "new_model_calls": len(models), "new_observed_tokens": sum(row["tokens"] for row in models),
                "new_experiments": sum(row["kind"] == "experiment" for row in added),
                "pending_final_aggregations": pending_aggregates,
                "unknown_historical_operation": plan["failed_operation"],
                "accounting_scope": "New segment measured; original unknown call remains unresolved"}
    except (IntegrityError, ValueError, OSError, KeyError, sqlite3.Error) as error:
        report["error"] = str(error)
        return report
