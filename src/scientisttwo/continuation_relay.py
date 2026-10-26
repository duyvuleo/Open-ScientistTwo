"""Explicitly authorized relay repair over a closed, immutable parent ledger.

The pinned remote controller keeps its original wire policy. This local adapter
records the repaired implementation separately, carries parent accounting, and
admits the exact failed writer request first. It does not reconcile unknown cost,
edit old responses, or grant controller retries. No transport is provided here.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import fcntl
import os
from pathlib import Path
import re
import stat
import time

from . import backends, process
from .studies import media_relay as relay


def repaired_source_hashes():
    """Exact closure of the local repair, separate from the old wire identity."""
    return {name: relay.sha(Path(path).read_bytes()) for name, path in {
        "continuation_relay.py": __file__, "backends.py": backends.__file__,
        "process.py": process.__file__, "studies/media_relay.py": relay.__file__,
    }.items()}


def _existing_private(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise relay.RelayError("Parent ledger must be an existing absolute private directory")
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise relay.RelayError("Parent ledger must be private to the current account")
    return path.resolve()


def _read_bound(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or info.st_size > limit):
            raise relay.RelayError("Parent ledger file is not private, regular and bounded")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise relay.RelayError("Parent ledger file exceeds its bound")
        return backends.strict_json_loads(raw.decode()), {"bytes": len(raw), "sha256": relay.sha(raw)}
    finally:
        os.close(fd)


@contextmanager
def _parent_lock(parent):
    """Exclude an original dispatcher while inspecting or using its ledger."""
    fd = os.open(parent / "service.lock", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise relay.RelayError("Parent service lock is not a private regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise relay.UnknownDispatch("Parent relay is still dispatching") from error
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _parent_snapshot(parent):
    policy, binding = _read_bound(parent / "policy.json", 4096)
    policy = relay.RelayPolicy.from_mapping(policy)
    manifest, records = {"policy.json": binding}, {}
    claims = _existing_private(parent / "claims")
    entries = sorted(claims.iterdir())
    if not entries or len(entries) > policy.max_calls:
        raise relay.RelayError("Parent call ledger is empty or exceeds its frozen allowance")
    for entry in entries:
        if not re.fullmatch(r"[a-f0-9]{32}", entry.name):
            raise relay.RelayError("Unexpected parent claim directory")
        _existing_private(entry)
        if {p.name for p in entry.iterdir()} != {"claim.json", "dispatch.json", "response.json"}:
            raise relay.UnknownDispatch("Parent dispatch has incomplete or unexpected evidence")
        values = {}
        for name, limit in (("claim.json", relay.REQUEST_LIMIT + 4096),
                            ("dispatch.json", 16384), ("response.json", relay.RESPONSE_LIMIT)):
            values[name], manifest["claims/" + entry.name + "/" + name] = _read_bound(entry / name, limit)
        claim, receipt, response = (values[name] for name in ("claim.json", "dispatch.json", "response.json"))
        if not all(isinstance(value, dict) for value in (claim, receipt, response)):
            raise relay.RelayError("Parent evidence records must be JSON objects")
        request = claim.get("request")
        request_hash = relay.validate_request(request, policy)
        relay.validate_response(response, request)
        if (request["request_id"] != entry.name or claim.get("request_sha256") != request_hash
                or receipt.get("request_sha256") != request_hash
                or receipt.get("status") != "completed"
                or receipt.get("source_sha256") != policy.source_sha256
                or receipt.get("relay_sha256") != policy.relay_sha256
                or receipt.get("model") != policy.model
                or receipt.get("returncode") != response["process"]["code"]
                or receipt.get("timed_out") != response["process"]["timed_out"]):
            raise relay.RelayError("Parent dispatch evidence has inconsistent identities or completion")
        usage = receipt.get("usage_status")
        tokens = receipt.get("observed_tokens")
        if usage == "known":
            if type(tokens) is not int or tokens < 0:
                raise relay.RelayError("Parent known token accounting is invalid")
        elif usage == "unknown":
            # Legacy receipts wrote zero. The explicit status governs; zero is
            # never promoted to a measured token count by this continuation.
            if tokens not in (None, 0) or type(tokens) is bool:
                raise relay.RelayError("Parent unknown token accounting is inconsistent")
        else:
            raise relay.UnknownDispatch("Parent token accounting has no explicit disposition")
        records[entry.name] = {"request": request, "receipt": receipt, "response": response}
    return policy, manifest, records


def parent_ledger_manifest(parent_state):
    """Read-only inspection used to bind an explicit authorization before launch."""
    parent = _existing_private(parent_state)
    with _parent_lock(parent):
        _, manifest, _ = _parent_snapshot(parent)
    return manifest


@dataclass(frozen=True)
class ContinuationRelayAuthorization:
    authorization_id: str
    authorized_state_path: str
    authorized_parent_state_path: str
    controller_plan_ref: str
    controller_admission_ref: str
    parent_manifest_sha256: str
    failed_request_id: str
    repaired_sources: dict[str, str]
    max_new_calls: int
    new_token_stop_threshold: int
    deadline: float
    unknown_cost_acknowledged: bool

    def __post_init__(self):
        if (not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,95}", self.authorization_id)
                or any(not isinstance(path, str) or not Path(path).is_absolute()
                       or ".." in Path(path).parts
                       for path in (self.authorized_state_path, self.authorized_parent_state_path))
                or any(not isinstance(ref, str) or not re.fullmatch(r"artifacts/[a-f0-9]{64}\.json", ref)
                       for ref in (self.controller_plan_ref, self.controller_admission_ref))
                or not re.fullmatch(r"[a-f0-9]{64}", self.parent_manifest_sha256)
                or not re.fullmatch(r"[a-f0-9]{32}", self.failed_request_id)
                or type(self.max_new_calls) is not int or not 1 <= self.max_new_calls <= 1000
                or type(self.new_token_stop_threshold) is not int
                or not 1 <= self.new_token_stop_threshold <= 10_000_000
                or not relay._finite(self.deadline)
                or type(self.unknown_cost_acknowledged) is not bool
                or self.repaired_sources != repaired_source_hashes()):
            raise relay.RelayError("Invalid explicit continuation authorization or repaired source identity")

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != {field.name for field in fields(cls)}:
            raise relay.RelayError("Continuation authorization must include every exact field")
        return cls(**value)


class VerifiedRepairPolicy(relay.RelayPolicy):
    """Original dataclass fields/identity with an explicitly bound local repair."""

    @classmethod
    def bind(cls, original, expected_sources):
        if expected_sources != repaired_source_hashes():
            raise relay.RelayError("Repaired relay source differs from authorization")
        result = cls(**asdict(original))
        object.__setattr__(result, "_repaired_sources", dict(expected_sources))
        result.verify_relay_source()
        return result

    def verify_relay_source(self):
        if getattr(self, "_repaired_sources", None) != repaired_source_hashes():
            raise relay.RelayError("Repaired relay source changed or was not explicitly bound")


def _same_writer_request(left, right):
    return all(relay.canonical(left[key]) == relay.canonical(right[key])
               for key in ("role", "prompt", "schema", "media", "model", "source_sha256"))


class ContinuationLocalService(relay.LocalService):
    """Append a bounded continuation; never rewrite or redispatch parent claims."""

    def __init__(self, state, policy, *, parent_state, authorization, executable,
                 executor=None, allow_live=False, synthetic=False):
        if not allow_live and not synthetic:
            raise relay.RelayError("Live continuation dispatch requires explicit allow_live=True")
        if not isinstance(authorization, ContinuationRelayAuthorization):
            raise relay.RelayError("Continuation requires a validated authorization record")
        self.authorization = authorization
        self.authorization_sha256 = relay.sha(relay.canonical(asdict(authorization)))
        self.parent_state = _existing_private(parent_state)
        state = Path(state).absolute()
        if (state.resolve() != Path(authorization.authorized_state_path).resolve()
                or self.parent_state != Path(authorization.authorized_parent_state_path).resolve()):
            raise relay.RelayError("Continuation or parent state path differs from authorization")
        if (state.resolve() == self.parent_state or state.resolve().is_relative_to(self.parent_state)
                or self.parent_state.is_relative_to(state.resolve())):
            raise relay.RelayError("Continuation and parent states must be separate directories")
        with _parent_lock(self.parent_state):
            parent_policy, self.parent_manifest, self.parent_records = _parent_snapshot(self.parent_state)
        if (relay.canonical(asdict(parent_policy)) != relay.canonical(asdict(policy))
                or relay.sha(relay.canonical(self.parent_manifest)) != authorization.parent_manifest_sha256):
            raise relay.RelayError("Original wire policy or parent ledger differs from authorization")
        if authorization.deadline > policy.expires_at or authorization.deadline <= time.time():
            raise relay.RelayError("Continuation deadline must be live and within the original expiry")
        failed = self.parent_records.get(authorization.failed_request_id)
        if (failed is None or failed["request"]["role"] != "paperorchestra_model"
                or failed["response"]["process"]["code"] == 0
                or failed["response"]["response"] is not None):
            raise relay.RelayError("Continuation must bind the failed PaperOrchestra call")
        self.failed_request = failed["request"]
        self.parent_known_tokens = sum(row["receipt"]["observed_tokens"]
            for row in self.parent_records.values() if row["receipt"]["usage_status"] == "known")
        self.parent_unknown_calls = sum(row["receipt"]["usage_status"] == "unknown"
                                       for row in self.parent_records.values())
        if self.parent_unknown_calls and not authorization.unknown_cost_acknowledged:
            raise relay.UnknownDispatch("Original unknown cost requires explicit acknowledgment")
        if len(self.parent_records) + authorization.max_new_calls > policy.max_calls:
            raise relay.RelayError("Continuation allowance would exceed the original call ceiling")
        if authorization.new_token_stop_threshold > policy.token_stop_threshold:
            raise relay.RelayError("Continuation token threshold exceeds the original threshold")
        repaired = VerifiedRepairPolicy.bind(policy, authorization.repaired_sources)
        original_executor = executor or process.run_process

        def bounded_executor(argv, timeout, **kwargs):
            remaining = authorization.deadline - time.time() - 2
            if remaining <= 0:
                return process.ProcessResult(124, "", "Continuation deadline expired", 0,
                                             timed_out=True)
            return original_executor(argv, min(timeout, remaining), **kwargs)

        super().__init__(state, repaired, executable=executable, executor=bounded_executor,
                         allow_live=allow_live, synthetic=synthetic)
        self.continuation_receipts = relay.private_directory(self.state / "continuation-dispatches")
        proof = {"schema_version": "scientisttwo-relay-continuation-v1",
                 "authorization": asdict(authorization), "parent_state": str(self.parent_state),
                 "parent_manifest": self.parent_manifest, "wire_policy_sha256": policy.identity,
                 "parent_calls": len(self.parent_records), "parent_known_tokens": self.parent_known_tokens,
                 "parent_unknown_usage_calls": self.parent_unknown_calls,
                 "original_total_tokens_verified": self.parent_unknown_calls == 0,
                 "token_threshold_kind": "observed_after_call_not_hard_cap",
                 "synthetic": synthetic}
        proof_path = self.state / "continuation-proof.json"
        if proof_path.exists():
            if relay.read_json(proof_path, 2 * 1024 * 1024) != proof:
                raise relay.RelayError("Continuation authorization or parent proof changed")
        else:
            if list(self.claims.iterdir()):
                raise relay.RelayError("Existing continuation claims lack their admission proof")
            relay.publish_new(proof_path, proof)
        self.proof_sha256 = relay.sha(relay.canonical(proof))

    def serve(self, request):
        if relay.sha(relay.canonical(asdict(self.authorization))) != self.authorization_sha256:
            raise relay.RelayError("Continuation authorization changed after initialization")
        saved_proof = relay.read_json(self.state / "continuation-proof.json", 2 * 1024 * 1024)
        if relay.sha(relay.canonical(saved_proof)) != self.proof_sha256:
            raise relay.RelayError("Continuation proof changed after initialization")
        with _parent_lock(self.parent_state):
            _, manifest, _ = _parent_snapshot(self.parent_state)
            if manifest != self.parent_manifest:
                raise relay.RelayError("Parent ledger changed after continuation authorization")
            return super().serve(request)

    def _new_record(self, entry):
        if (not entry.is_dir() or entry.is_symlink()
                or {path.name for path in entry.iterdir()} != {"claim.json", "dispatch.json", "response.json"}
                or not (entry / "response.json").is_file()
                or not (self.continuation_receipts / (entry.name + ".json")).is_file()):
            raise relay.UnknownDispatch("Continuation completion or admission receipt is unknown")
        claim = relay.read_json(entry / "claim.json", relay.REQUEST_LIMIT + 4096)
        receipt = relay.read_json(entry / "dispatch.json", 16384)
        response = relay.read_json(entry / "response.json", relay.RESPONSE_LIMIT)
        local = relay.read_json(self.continuation_receipts / (entry.name + ".json"), 16384)
        request_hash = relay.validate_request(claim["request"], self.policy)
        relay.validate_response(response, claim["request"])
        if (claim.get("request_sha256") != request_hash or receipt.get("request_sha256") != request_hash
                or claim["request"]["request_id"] != entry.name
                or receipt.get("status") != "completed"
                or receipt.get("source_sha256") != self.policy.source_sha256
                or receipt.get("relay_sha256") != self.policy.relay_sha256
                or receipt.get("model") != self.policy.model
                or receipt.get("returncode") != response["process"]["code"]
                or receipt.get("timed_out") != response["process"]["timed_out"]
                or local.get("request_sha256") != request_hash
                or local.get("continuation_proof_sha256") != self.proof_sha256
                or local.get("dispatch_sha256") != relay.sha(relay.canonical(receipt))
                or local.get("response_sha256") != relay.sha(relay.canonical(response))
                or local.get("repaired_sources") != self.authorization.repaired_sources):
            raise relay.RelayError("Continuation dispatch evidence changed")
        return claim, receipt, response

    def _new_accounting(self):
        tokens, repaired_success = 0, False
        entries = sorted(self.claims.iterdir())
        for entry in entries:
            claim, receipt, response = self._new_record(entry)
            if receipt.get("usage_status") != "known":
                raise relay.UnknownDispatch("New continuation token usage is unknown; no further calls")
            value = receipt.get("observed_tokens")
            if type(value) is not int or value < 0:
                raise relay.RelayError("Continuation token accounting is invalid")
            tokens += value
            if (response["process"]["code"] == 0 and response["response"] is not None
                    and _same_writer_request(claim["request"], self.failed_request)):
                repaired_success = True
        return len(entries), tokens, repaired_success

    def _serve_locked(self, request, request_hash):
        if request["request_id"] in self.parent_records:
            raise relay.UnknownDispatch("Parent request IDs cannot be redispatched by a continuation")
        # Cached responses remain readable even after deadline or unknown usage;
        # they do not dispatch again. Require their separate provenance receipt.
        entry = self.claims / request["request_id"]
        if entry.exists():
            self._new_record(entry)
            return super()._serve_locked(request, request_hash)
        calls, tokens, repaired_success = self._new_accounting()
        if (calls >= self.authorization.max_new_calls
                or len(self.parent_records) + calls >= self.policy.max_calls
                or tokens >= self.authorization.new_token_stop_threshold
                or self.parent_known_tokens + tokens >= self.policy.token_stop_threshold):
            raise relay.RelayError("Combined continuation call or observed-token threshold reached")
        if time.time() >= self.authorization.deadline:
            raise relay.RelayError("Continuation deadline expired")
        if not repaired_success and not _same_writer_request(request, self.failed_request):
            raise relay.RelayError("The exact failed writer request must succeed before other calls")
        response = super()._serve_locked(request, request_hash)
        dispatch_path = entry / "dispatch.json"
        if dispatch_path.exists():
            dispatch = relay.read_json(dispatch_path, 16384)
            receipt = {"schema_version": "scientisttwo-relay-continuation-dispatch-v1",
                       "continuation_proof_sha256": self.proof_sha256,
                       "request_sha256": request_hash,
                       "dispatch_sha256": relay.sha(relay.canonical(dispatch)),
                       "response_sha256": relay.sha(relay.canonical(response)),
                       "repaired_sources": self.authorization.repaired_sources,
                       "parent_calls": len(self.parent_records),
                       "parent_known_tokens": self.parent_known_tokens,
                       "prior_new_calls": calls, "prior_new_known_tokens": tokens,
                       "parent_unknown_usage_calls": self.parent_unknown_calls,
                       "original_total_tokens_verified": self.parent_unknown_calls == 0,
                       "combined_total_tokens_verified": self.parent_unknown_calls == 0
                           and dispatch.get("usage_status") == "known",
                       "failed_request_id": self.authorization.failed_request_id}
            relay.publish_new(self.continuation_receipts / (entry.name + ".json"), receipt)
        return response
