"""Read-only API accounting checks; no credentials, clients or requests are loaded.

Provider-reported token counts are accounting evidence, not independent billing
verification. Unknown or interrupted outcomes never become zero-cost successes.
"""
import math
from pathlib import Path
import re

from ..backends import StructuredOutputCodec, strict_json_loads
from ..models import AgentResult
from ..store import IntegrityError, canonical, digest
from .config import ModelProvider


POLICY_VERSION = "scientisttwo-api-policy-v1"
CLAIM_VERSION = "scientisttwo-api-claim-v1"
RECEIPT_VERSION = "scientisttwo-api-receipt-v1"
PROMPT_VERSION = "scientisttwo-api-v1+json-map-wire-v1"
_USAGE_FIELDS = {"input_tokens", "output_tokens", "cached_input_tokens",
                 "cache_creation_input_tokens", "cache_read_input_tokens", "reasoning_tokens"}
_HEX = re.compile(r"[a-f0-9]{64}")
_ID = re.compile(r"[a-f0-9]{32}")


def _require(condition, message):
    if not condition:
        raise IntegrityError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _load(path, maximum=4 * 1024**2):
    path = Path(path)
    _require(not path.is_symlink() and path.is_file() and path.stat().st_size <= maximum,
             "API ledger file is missing, oversized or unsafe")
    try:
        value = strict_json_loads(path.read_text())
    except (ValueError, UnicodeError):
        raise IntegrityError("API ledger file is incomplete or malformed") from None
    _require(isinstance(value, dict), "API ledger record must be an object")
    return value


def _artifact(store, ref, *, text=False):
    _require(isinstance(ref, str) and re.fullmatch(r"artifacts/[a-f0-9]{64}\.[A-Za-z0-9_-]{1,16}", ref),
             "API ledger artifact reference is invalid")
    row = store.db.execute("SELECT size FROM artifacts WHERE path=?", (ref,)).fetchone()
    _require(row is not None and row[0] <= 64 * 1024**2, "API ledger artifact is missing or oversized")
    raw = store.read(ref)
    try:
        return raw.decode("utf-8") if text else strict_json_loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise IntegrityError("API ledger artifact is not valid structured text") from None


def _usage(value):
    _require(isinstance(value, dict) and {"input_tokens", "output_tokens"} <= set(value)
             and set(value) <= _USAGE_FIELDS
             and all(type(v) is int and v >= 0 for v in value.values()),
             "API receipt has missing or invalid normalized usage")
    _require(value.get("cached_input_tokens", 0) <= value["input_tokens"]
             and value.get("cache_creation_input_tokens", 0) + value.get("cache_read_input_tokens", 0) <= value["input_tokens"]
             and value.get("reasoning_tokens", 0) <= value["output_tokens"],
             "API usage detail exceeds its normalized token totals")
    return value["input_tokens"] + value["output_tokens"]


def _agent_result(value):
    try:
        return AgentResult.model_validate(value)
    except (ValueError, TypeError):
        raise IntegrityError("API agent-result record is invalid") from None


def _policy(store, value):
    keys = {"schema_version", "run_id", "model", "provider", "implementation_sha256", "provider_sources",
            "max_calls", "token_stop_threshold", "call_seconds", "deadline"}
    _require(set(value) == keys and value["schema_version"] == POLICY_VERSION,
             "Unsupported API accounting policy")
    _require(value["run_id"] == digest(str(store.root / "api-model-ledger").encode())
             and isinstance(value["model"], str) and bool(value["model"])
             and isinstance(value["implementation_sha256"], str)
             and _HEX.fullmatch(value["implementation_sha256"])
             and type(value["max_calls"]) is int and value["max_calls"] > 0
             and type(value["token_stop_threshold"]) is int and value["token_stop_threshold"] > 0
             and _finite(value["call_seconds"]) and value["call_seconds"] > 0
             and _finite(value["deadline"]), "Malformed API accounting limits or source identity")
    try:
        provider = ModelProvider.model_validate(value["provider"])
    except ValueError:
        raise IntegrityError("Malformed API provider policy") from None
    _require(provider.kind != "codex" and provider.model_dump(mode="json") == value["provider"],
             "API ledger provider is not its canonical declared configuration")
    _require(isinstance(value["provider_sources"], dict) and bool(value["provider_sources"])
        and all(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_]+\.py", name)
                and isinstance(sha, str) and _HEX.fullmatch(sha)
                for name, sha in value["provider_sources"].items()),
        "API provider source closure is missing or invalid")
    return provider


def _request(store, claim, policy):
    descriptor = _artifact(store, claim["request_ref"])
    _require(isinstance(descriptor, dict) and set(descriptor) == {
        "controller_request", "provider", "wire_schema", "codec_identity", "images", "timeout_seconds"}
        and digest(canonical(descriptor)) == claim["request_sha256"],
        "API claim does not bind its complete request descriptor")
    request = descriptor["controller_request"]
    _require(isinstance(request, dict) and set(request) == {"role", "inputs", "schema", "model", "prompt_version"}
        and isinstance(request["role"], str) and bool(request["role"])
        and isinstance(request["inputs"], dict) and isinstance(request["schema"], dict)
        and request["model"] == policy["model"]
        and request["prompt_version"] == "research-v2-complete-workflow"
        and digest(canonical(request)) == claim["controller_request_sha256"]
        and descriptor["provider"] == policy["provider"], "API request differs from its frozen model or controller input")
    timeout = descriptor["timeout_seconds"]
    _require(_finite(timeout) and 0 < timeout <= policy["call_seconds"]
        and claim["deadline"] <= min(claim["created_at"] + timeout, policy["deadline"]) + 1e-6,
        "API request exceeded its frozen call deadline")
    if descriptor["wire_schema"] is not None:
        try:
            codec = StructuredOutputCodec(request["schema"])
        except (ValueError, TypeError):
            raise IntegrityError("API request has an invalid structured-output codec") from None
        _require(codec.wire_schema == descriptor["wire_schema"] and codec.identity == descriptor["codec_identity"],
                 "API wire schema or codec identity differs from its domain schema")
    else:
        _require(descriptor["codec_identity"] is None, "API codec identity has no wire schema")
    images = descriptor["images"]
    _require(isinstance(images, list) and len(images) <= 32, "API image inventory is invalid")
    _require(not images or policy["provider"].get("vision") is True,
             "API images were sent without a declared vision capability")
    total_image_bytes = 0
    for image in images:
        _require(isinstance(image, dict) and set(image) == {"mime_type", "sha256", "artifact_ref", "bytes"}
            and image["mime_type"] in {"image/png", "image/jpeg"}
            and isinstance(image["artifact_ref"], str)
            and re.fullmatch(r"artifacts/[a-f0-9]{64}\.(png|jpg)", image["artifact_ref"])
            and type(image["bytes"]) is int and 0 < image["bytes"] <= 8 * 1024**2,
            "API image descriptor is invalid")
        raw = store.read(image["artifact_ref"])
        total_image_bytes += len(raw)
        _require(len(raw) == image["bytes"] and digest(raw) == image["sha256"]
            and total_image_bytes <= 20 * 1024**2
            and (image["mime_type"] == "image/png" and raw.startswith(b"\x89PNG\r\n\x1a\n")
                 or image["mime_type"] == "image/jpeg" and raw.startswith(b"\xff\xd8\xff")),
                 "API image payload differs from its frozen identity")
    return descriptor


def inspect_api_ledger(store, expected_policy=None, require_complete=False):
    """Reconcile immutable API claims/completions without requiring controller commit.

    The backend may use this while a controller completion is still being saved.
    Campaign audits additionally call verify_api_accounting below.
    """
    root = store.root / "api-model-ledger"
    _require(not root.is_symlink() and root.is_dir(), "API model ledger is missing or unsafe")
    _require(set(p.name for p in root.iterdir()) <= {"policy.json", "calls", "ledger.lock"},
             "API ledger contains undeclared files")
    _require(not any(p.is_symlink() for p in root.iterdir()), "API ledger contains a symlink")
    policy = _load(root / "policy.json")
    provider = _policy(store, policy)
    if expected_policy is not None:
        _require(canonical(policy) == canonical(expected_policy), "API policy changed from the frozen campaign")
    calls = root / "calls"
    _require(not calls.is_symlink() and calls.is_dir(), "API claim directory is missing or unsafe")
    records = []
    for directory in calls.iterdir():
        _require(_ID.fullmatch(directory.name) and not directory.is_symlink() and directory.is_dir(),
                 "API call directory is invalid")
        _require(set(p.name for p in directory.iterdir()) <= {"claim.json", "receipt.json"},
                 "API call contains undeclared files")
        claim = _load(directory / "claim.json")
        _require(set(claim) == {"schema_version", "id", "policy_sha256", "request_ref", "request_sha256",
            "controller_request_sha256", "created_at", "deadline", "prior_calls", "prior_observed_tokens"}
            and claim["schema_version"] == CLAIM_VERSION and claim["id"] == directory.name
            and claim["policy_sha256"] == digest(canonical(policy))
            and _finite(claim["created_at"]) and _finite(claim["deadline"])
            and claim["created_at"] < claim["deadline"] <= policy["deadline"]
            and type(claim["prior_calls"]) is int and claim["prior_calls"] >= 0
            and type(claim["prior_observed_tokens"]) is int and claim["prior_observed_tokens"] >= 0,
            "API claim differs from its policy, time or sequence")
        descriptor = _request(store, claim, policy)
        receipt = None
        receipt_ref = store.get("api_receipt/" + directory.name)
        if (directory / "receipt.json").exists():
            receipt = _load(directory / "receipt.json")
            keys = {"schema_version", "id", "claim_sha256", "policy_sha256", "controller_request_sha256",
                    "status", "worker_status", "usage_known", "usage", "observed_tokens", "request_sent",
                    "http_status", "reported_model", "response_id", "response_sha256", "error_code",
                    "output_ref", "result_ref", "completed_at"}
            _require(set(receipt) == keys and receipt["schema_version"] == RECEIPT_VERSION
                and receipt["id"] == directory.name and receipt["claim_sha256"] == digest(canonical(claim))
                and receipt["policy_sha256"] == claim["policy_sha256"]
                and receipt["controller_request_sha256"] == claim["controller_request_sha256"]
                and receipt["status"] in {"success", "failed", "unknown"}
                and isinstance(receipt["worker_status"], str)
                and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", receipt["worker_status"])
                and type(receipt["usage_known"]) is bool
                and (receipt["request_sent"] is None or type(receipt["request_sent"]) is bool)
                and _finite(receipt["completed_at"]) and receipt["completed_at"] >= claim["created_at"],
                "API completion does not bind its original claim")
            _require(receipt_ref is not None and _artifact(store, receipt_ref) == receipt,
                     "API completion is not bound to its registered artifact")
            _require(receipt["http_status"] is None or type(receipt["http_status"]) is int
                and 100 <= receipt["http_status"] <= 599, "API HTTP status is invalid")
            for name in ("reported_model", "response_id"):
                value = receipt[name]
                _require(value is None or isinstance(value, str) and len(value) <= 256
                    and re.fullmatch(r"[A-Za-z0-9_.:/+@-]+", value), "API response metadata is invalid")
            _require(receipt["response_sha256"] is None or isinstance(receipt["response_sha256"], str)
                and _HEX.fullmatch(receipt["response_sha256"]), "API response digest is invalid")
            _require(receipt["error_code"] is None or isinstance(receipt["error_code"], str)
                and re.fullmatch(r"[a-z_]{1,64}", receipt["error_code"]), "API failure label is invalid")
            if receipt["usage_known"]:
                tokens = _usage(receipt["usage"])
                _require(type(receipt["observed_tokens"]) is int and receipt["observed_tokens"] == tokens
                    and receipt["status"] != "unknown", "API token total differs from normalized usage")
            else:
                _require(receipt["usage"] is None and receipt["observed_tokens"] is None
                    and receipt["status"] == "unknown" and receipt["result_ref"] is None,
                    "Unknown API usage was converted to a known result")
            if receipt["request_sent"] is False:
                _require(receipt["worker_status"] == "not_sent" and receipt["status"] == "failed"
                    and receipt["usage_known"] and receipt["observed_tokens"] == 0,
                    "An unsent API request has inconsistent accounting")
            if descriptor["wire_schema"] is None:
                _require(receipt["request_sent"] is False, "API request was sent without a valid schema")
            if receipt["output_ref"] is not None:
                _artifact(store, receipt["output_ref"], text=True)
            if receipt["status"] == "success":
                _require(receipt["worker_status"] == "completed" and receipt["request_sent"] is True
                    and receipt["usage_known"] and receipt["error_code"] is None
                    and receipt["output_ref"] is not None, "API success lacks a completed response and known usage")
                result = _agent_result(_artifact(store, receipt["result_ref"]))
                _require(result.synthetic is False and result.backend == provider.kind
                    and result.model == policy["model"] and result.prompt_version == PROMPT_VERSION
                    and result.usage == {**receipt["usage"], "api_call_id": directory.name},
                    "API agent result differs from the provider receipt")
                try:
                    decoded = StructuredOutputCodec(descriptor["controller_request"]["schema"]).decode_json(
                        _artifact(store, receipt["output_ref"], text=True))
                except ValueError:
                    raise IntegrityError("Successful API result has malformed structured output") from None
                _require(result.payload == decoded,
                         "Accepted API payload differs from its decoded provider output")
            else:
                _require(receipt["result_ref"] is None, "Failed API response has an accepted agent result")
        elif receipt_ref is not None:
            raise IntegrityError("API receipt artifact exists without its immutable completion file")
        records.append({"id": directory.name, "claim": claim, "receipt": receipt,
                        "receipt_ref": receipt_ref, "request": descriptor})
    records.sort(key=lambda item: item["claim"]["prior_calls"])
    tokens, unknown = 0, []
    for index, record in enumerate(records):
        claim, receipt = record["claim"], record["receipt"]
        _require(not unknown, "An API call followed an unknown or uncommitted outcome")
        _require(claim["prior_calls"] == index and claim["prior_observed_tokens"] == tokens
            and index < policy["max_calls"] and tokens < policy["token_stop_threshold"],
            "API dispatch sequence or admission counters exceed the frozen policy")
        if receipt is None or not receipt["usage_known"]:
            unknown.append(record["id"])
        else:
            tokens += receipt["observed_tokens"]
    if require_complete:
        _require(not unknown, "API completion or token usage is unknown")
    return {"source": "durable-api-model-ledger", "ledger": str(root), "policy": policy,
        "provider": provider.kind, "model": policy["model"], "claimed_calls": len(records),
        "completed_receipts": sum(item["receipt"] is not None for item in records),
        "unknown_calls": unknown, "known_observed_tokens": tokens, "usage_complete": not unknown,
        "records": records}


def verify_api_accounting(store, config, require_complete=True):
    """Verify provider accounting against the saved summary and controller ledger."""
    _require(not store.get("inherited_model_accounting_ref"),
             "Codex source-epoch accounting cannot be adopted by an API campaign")
    _require(store.get("live_backend") == {"backend": config.provider.kind, "model": config.model, "synthetic": False},
             "API campaign backend provenance differs from its frozen provider")
    root = store.root / "api-model-ledger"
    provider_sources = {name.removeprefix("providers/"): value
                        for name, value in config.implementation.items() if name.startswith("providers/")}
    _require(bool(provider_sources), "Campaign has no frozen API provider source closure")
    expected = {"schema_version": POLICY_VERSION, "run_id": digest(str(root).encode()),
        "model": config.model, "provider": config.provider.model_dump(mode="json"),
        "provider_sources": provider_sources,
        "implementation_sha256": digest(canonical(config.implementation)), "max_calls": config.max_calls,
        "token_stop_threshold": config.token_stop_threshold, "call_seconds": config.call_seconds,
        "deadline": store.get("live_deadline")}
    inspected = inspect_api_ledger(store, expected_policy=expected)
    records = {item["id"]: item for item in inspected["records"]}
    summary = {k: v for k, v in inspected.items() if k != "records"}
    summary_ref = store.get("api_model_accounting_ref")
    reasons = []
    if summary_ref is None:
        reasons.append("API accounting summary is missing")
    else:
        saved = _artifact(store, summary_ref)
        _require(saved == summary, "API accounting artifact does not reconcile with immutable receipts")
    mapped, unknown_operations = set(), []
    operations = list(store.db.execute("SELECT * FROM live_operations WHERE kind='model' ORDER BY rowid"))
    for operation in operations:
        _require(type(operation["tokens"]) is int and operation["tokens"] >= 0,
                 "API controller token charge is invalid")
        if operation["status"] == "running" or operation["result_ref"] is None:
            unknown_operations.append(operation["key"])
            continue
        result = _artifact(store, operation["result_ref"])
        if operation["status"] == "success":
            result = _agent_result(result)
            identifier = result.usage.get("api_call_id")
        else:
            _require(operation["status"] == "failed" and isinstance(result, dict),
                     "Unsupported API controller operation status")
            identifier = result.get("api_call_id")
        if identifier is None:
            _require(operation["status"] == "failed" and operation["tokens"] == 0,
                     "API controller success or charged failure has no dispatch identity")
            unknown_operations.append(operation["key"])
            continue
        _require(isinstance(identifier, str) and identifier in records and identifier not in mapped,
                 "API controller operation has a missing or duplicated dispatch")
        record, mapped_receipt = records[identifier], records[identifier]["receipt"]
        _require(mapped_receipt is not None and operation["request_hash"] == record["claim"]["controller_request_sha256"],
                 "API controller operation differs from its dispatch request")
        _require(operation["tokens"] == (mapped_receipt["observed_tokens"] if mapped_receipt["usage_known"] else 0),
                 "API controller token charge differs from its receipt")
        if operation["status"] == "success":
            _require(mapped_receipt["status"] == "success" and operation["result_ref"] == mapped_receipt["result_ref"],
                     "API controller result differs from its accepted provider result")
        else:
            _require(mapped_receipt["status"] != "success"
                and result.get("api_receipt_ref") == record["receipt_ref"]
                and result.get("usage_known") is mapped_receipt["usage_known"]
                and result.get("usage") == mapped_receipt["usage"],
                "API failed-call usage or receipt binding is inconsistent")
        mapped.add(identifier)
    unmatched = set(records) - mapped
    for identifier in unmatched:
        _require(any(op["key"] in unknown_operations
            and op["request_hash"] == records[identifier]["claim"]["controller_request_sha256"] for op in operations),
            "API dispatch has no controller reservation")
    if inspected["unknown_calls"]:
        reasons.append("Provider completion or usage remains unknown")
    if unknown_operations:
        reasons.append("Controller has uncommitted or unbound API outcomes")
    if require_complete:
        _require(not reasons, "; ".join(reasons))
    return {"status": "unverified" if reasons else "pass", "calls": inspected["claimed_calls"],
        "observed_tokens": inspected["known_observed_tokens"], "unknown_calls": len(inspected["unknown_calls"]),
        "unknown_controller_operations": unknown_operations,
        "controller_attempts": len(operations), "controller_accounted_tokens": sum(row["tokens"] for row in operations),
        "known_tokens_not_in_controller_total": inspected["known_observed_tokens"] - sum(row["tokens"] for row in operations),
        "authoritative_token_source": "durable-api-model-ledger", "provider": config.provider.kind,
        "model": config.model, "accounting_ref": summary_ref, "reasons": reasons,
        "scope": "Provider-reported usage and exact local receipts; independent billing and model quality are unverified"}
