"""Explicit live pilots using the existing controller and isolated CPU runner.

The legacy Profile remains an experiment-resource contract. Only this separate
entrypoint authorizes live agents. Its direct local executor reuses the durable
model ledger without a filesystem queue, SSH, or a shared-filesystem handoff.
"""
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import sqlite3
import time
from typing import Literal
import uuid

from pydantic import Field, model_validator

from ..backends import CodexBackend
from ..live.adapters import implementation_identity
from ..live.models import Source, LiveSpec
from ..models import Profile, Record, TaskSpec
from ..process import ProcessResult, run_process
from ..store import IntegrityError, canonical, digest, file_digest
from ..studies import media_relay as relay
from .literature import LiteratureConfig, SourceRecord
from .policy import ResearchPolicy


INHERITED_ACCOUNTING_VERSION = "scientisttwo-inherited-model-accounting-v1"


def _validated_dispatches(items, policy):
    """Validate immutable dispatch triplets under their actual source epoch."""
    records, identifiers = [], set()
    for claim, receipt, response in items:
        request = claim["request"]
        identity = relay.validate_request(request, policy)
        process, _ = relay.validate_response(response, request)
        identifier = request["request_id"]
        if identifier in identifiers:
            raise IntegrityError("Duplicate inherited/current model dispatch identity")
        identifiers.add(identifier)
        if (claim.get("request_sha256") != identity or receipt.get("request_sha256") != identity
                or receipt.get("status") != "completed" or receipt.get("usage_status") != "known"
                or receipt.get("synthetic") is not False or receipt.get("model") != policy.model
                or type(receipt.get("observed_tokens")) is not int or receipt["observed_tokens"] < 0
                or receipt.get("returncode") != process["code"]
                or receipt.get("source_sha256", policy.source_sha256) != policy.source_sha256
                or receipt.get("relay_sha256", policy.relay_sha256) != policy.relay_sha256
                or not relay._finite(receipt.get("started_at"))
                or receipt["started_at"] > request["deadline"]):
            raise IntegrityError("Model dispatch provenance/completion/usage is unknown or inconsistent")
        # Successful raw CLI completions preserve their sanitized usage events.
        # Known failed CLI costs remain receipt-backed, never silently zeroed.
        if process["code"] == 0 and relay.usage_tokens(process["stdout"]) != receipt["observed_tokens"]:
            raise IntegrityError("Model dispatch usage differs from its completed response")
        records.append(receipt)
    records.sort(key=lambda row: row["started_at"])
    tokens = 0
    for receipt in records:
        if tokens >= policy.token_stop_threshold:
            raise IntegrityError("A model call started after its source epoch token threshold")
        if receipt.get("prior_observed_tokens", tokens) != tokens:
            raise IntegrityError("Model dispatch prior usage does not reconcile")
        tokens += receipt["observed_tokens"]
    if len(records) > policy.max_calls:
        raise IntegrityError("Model dispatch count exceeds its source epoch policy")
    return {"calls": len(records), "tokens": tokens, "request_ids": sorted(identifiers)}


def validate_inherited_model_accounting(store, config, deadline=None):
    """Verify a single closed parent epoch without reading its live directory.

    Parent receipts retain the parent source/policy identities. The child owns
    only new dispatches; copied controller operations are never charged again.
    """
    ref = store.get("inherited_model_accounting_ref")
    if ref is None:
        return {"manifest_ref": None, "calls": 0, "tokens": 0, "request_ids": [], "artifact_hashes": {}}
    manifest = store.json(ref)
    expected = {"schema_version", "parent_root", "parent_spec_ref", "parent_policy_ref", "claims",
                "parent_deadline", "parent_summary_ref", "parent_operations_ref", "parent_snapshot_ref",
                "source_epoch", "includes_prior_epochs"}
    if (set(manifest) != expected or manifest["schema_version"] != INHERITED_ACCOUNTING_VERSION
            or manifest["source_epoch"] != "parent" or manifest["includes_prior_epochs"] is not False
            or not isinstance(manifest["parent_root"], str) or not Path(manifest["parent_root"]).is_absolute()
            or not isinstance(manifest["claims"], list) or not 1 <= len(manifest["claims"]) <= 1000):
        raise IntegrityError("Unsupported or incomplete inherited model accounting manifest")
    artifact_refs = {ref, *[manifest[name] for name in (
        "parent_spec_ref", "parent_policy_ref", "parent_summary_ref", "parent_operations_ref", "parent_snapshot_ref")]}
    parent = LiveSpec.model_validate(store.json(manifest["parent_spec_ref"]))
    policy = relay.RelayPolicy.from_mapping(store.json(manifest["parent_policy_ref"]))
    deadline = store.get("live_deadline") if deadline is None else deadline
    same_limits = ("model", "max_calls", "token_stop_threshold", "call_seconds", "campaign_seconds", "max_experiments")
    if (any(getattr(parent, key) != getattr(config, key) for key in same_limits)
            or manifest["parent_deadline"] != deadline or policy.expires_at != deadline
            or policy.model != parent.model or policy.allowed_models != (parent.model,)
            or policy.max_calls != parent.max_calls or policy.token_stop_threshold != parent.token_stop_threshold
            or policy.call_seconds != parent.call_seconds
            or policy.source_sha256 != digest(canonical(parent.implementation))
            or policy.relay_sha256 != parent.implementation.get("studies/media_relay.py")):
        raise IntegrityError("Inherited policy differs from its frozen parent source or carried global limits")
    items = []
    for entry in manifest["claims"]:
        if not isinstance(entry, dict) or set(entry) != {"claim_ref", "dispatch_ref", "response_ref"}:
            raise IntegrityError("Incomplete inherited dispatch artifact triplet")
        artifact_refs.update(entry.values())
        items.append(tuple(store.json(entry[key]) for key in ("claim_ref", "dispatch_ref", "response_ref")))
    totals = _validated_dispatches(items, policy)
    summary = store.json(manifest["parent_summary_ref"])
    if (summary.get("ledger") != str(Path(manifest["parent_root"]) / "model-ledger")
            or summary.get("transport") != "direct_local_no_queue"
            or canonical(summary.get("policy")) != canonical(asdict(policy))
            or not summary.get("usage_complete") or summary.get("unknown_calls")
            or summary.get("claimed_calls") != totals["calls"]
            or summary.get("completed_receipts") != totals["calls"]
            or summary.get("known_observed_tokens") != totals["tokens"]):
        raise IntegrityError("Inherited dispatches omit or contradict the closed parent accounting")
    operations = store.json(manifest["parent_operations_ref"])
    if (not isinstance(operations, list) or any(not isinstance(row, dict) for row in operations)
            or len({row.get("key") for row in operations}) != len(operations)):
        raise IntegrityError("Invalid inherited controller-operation snapshot")
    # Verify the exact captured SQLite epoch, not merely a claimed list of rows.
    store.read(manifest["parent_snapshot_ref"])
    snapshot_path = store.root / manifest["parent_snapshot_ref"]
    database = sqlite3.connect(snapshot_path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    database.row_factory = sqlite3.Row
    try:
        snapshot_rows = [dict(row) for row in database.execute("SELECT * FROM live_operations ORDER BY key")]
        metadata = {row["key"]: json.loads(row["value"]) for row in database.execute(
            "SELECT key,value FROM metadata WHERE key IN ('live_spec_ref','live_deadline','live_pilot_model_accounting','inherited_model_accounting_ref')")}
    finally:
        database.close()
    if (snapshot_rows != sorted(operations, key=lambda row: row["key"])
            or metadata.get("live_spec_ref") != manifest["parent_spec_ref"]
            or metadata.get("live_deadline") != deadline
            or metadata.get("live_pilot_model_accounting") != manifest["parent_summary_ref"]
            or metadata.get("inherited_model_accounting_ref") is not None):
        raise IntegrityError("Inherited accounting does not match its single original SQLite epoch")
    for row in operations:
        actual = store.db.execute("SELECT * FROM live_operations WHERE key=?", (row["key"],)).fetchone()
        if actual is None or dict(actual) != row or row.get("status") == "running":
            raise IntegrityError("Inherited controller operation was lost, changed or remains ambiguous")
    models = [row for row in operations if row["kind"] == "model"]
    if len(models) < totals["calls"] or sum(row["tokens"] for row in models) > totals["tokens"]:
        raise IntegrityError("Inherited model dispatches do not reconcile with controller operations")
    return {**totals, "manifest_ref": ref, "parent_root": manifest["parent_root"],
            "parent_policy": asdict(policy), "parent_spec_ref": manifest["parent_spec_ref"],
            "operations_ref": manifest["parent_operations_ref"],
            "artifact_hashes": {name: digest(store.read(name)) for name in sorted(artifact_refs)}}


class LivePilotOptions(Record):
    model: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,99}$")
    max_calls: int = Field(ge=1, le=1000)
    token_stop_threshold: int = Field(gt=0, le=10_000_000)
    campaign_seconds: float = Field(gt=0, le=14_400)
    call_seconds: float = Field(gt=0, le=600)
    max_experiments: int = Field(default=30, ge=1, le=100)
    workflow: ResearchPolicy
    limitation_rounds: int = Field(default=1, ge=1, le=16)
    search_rounds: int = Field(default=2, ge=1, le=4)
    success_target: int = Field(default=4, ge=1, le=4)
    engineering_retries: int = Field(default=1, ge=0, le=2)
    model_retries: Literal[0] = 0
    agent_output_retries: int = Field(default=1, ge=0, le=1)
    ablation_refinements: int = Field(default=0, ge=0, le=1)
    review_rounds: int = Field(default=1, ge=0, le=2)
    meta_refinements: int = Field(default=1, ge=0, le=1)
    manuscript_revisions: int = Field(default=1, ge=0, le=2)
    supplementary_seed_count: int = Field(default=2, ge=1, le=32)
    max_prompt_chars: int = Field(default=600_000, ge=1000, le=1_000_000)
    history_code_chars: int = Field(default=4000, ge=0, le=100000)
    log_tail_chars: int = Field(default=4000, ge=0, le=100000)
    sources: list[Source] = Field(default_factory=list, max_length=20)
    frozen_literature: list[SourceRecord] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def explicit_live_scope(self):
        if self.model.startswith(("scripted", "fixture")):
            raise ValueError("Live pilot options require an explicit non-fixture model identifier")
        if self.workflow.profile == "offline-pilot" or self.workflow.network_literature:
            raise ValueError("A live pilot requires a research workflow with frozen local literature")
        literature = LiteratureConfig.model_validate(self.workflow.literature)
        if literature.providers != ["local"]:
            raise ValueError("This live pilot entrypoint permits only frozen local literature")
        ids = [source.id for source in [*self.sources, *self.frozen_literature]]
        if len(ids) != len(set(ids)) or "task" in ids:
            raise ValueError("Pilot literature/source IDs must be distinct; task is reserved")
        if any(source.content_ref is not None for source in self.frozen_literature):
            raise ValueError("Frozen pilot literature must contain inline passages, not foreign artifact references")
        return self

    def config_values(self):
        return {**self.model_dump(mode="json", exclude={"frozen_literature"}),
                "engine_version": "research-v2", "mode": "research"}


class DirectPilotExecutor:
    """Synchronous LocalService transport with durable call/usage receipts."""
    def __init__(self, root, config, deadline, *, executable="codex", executor=run_process,
                 allow_live=False, accounting_store=None):
        if not allow_live:
            raise ValueError("Direct live pilot execution requires explicit allow_live=True")
        command = shutil.which(str(executable))
        if command is None:
            raise ValueError("The explicitly selected Codex executable is unavailable")
        self.accounting_store, self.config, self.deadline = accounting_store, config, deadline
        if accounting_store is not None and Path(root).resolve() != accounting_store.root / "model-ledger":
            raise IntegrityError("Model ledger differs from its accounting store")
        if accounting_store is None and (Path(root).parent / "campaign.sqlite").is_file():
            db = sqlite3.connect((Path(root).parent / "campaign.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
            try:
                inherited = db.execute("SELECT value FROM metadata WHERE key='inherited_model_accounting_ref'").fetchone()
            finally:
                db.close()
            if inherited is not None:
                raise IntegrityError("Inherited model accounting requires its verified campaign store")
        self.inherited = (validate_inherited_model_accounting(accounting_store, config, deadline)
                          if accounting_store is not None else
                          {"manifest_ref": None, "calls": 0, "tokens": 0, "request_ids": [], "artifact_hashes": {}})
        calls = config.max_calls - self.inherited["calls"]
        tokens = config.token_stop_threshold - self.inherited["tokens"]
        if calls <= 0 or tokens <= 0:
            raise relay.RelayError("Inherited dispatches exhausted the unchanged global model budget")
        self.policy = relay.RelayPolicy(
            run_id="pilot-" + digest(str(Path(root).resolve()).encode())[:32],
            expires_at=deadline, model=config.model, allowed_models=(config.model,),
            source_sha256=digest(canonical(config.implementation)),
            relay_sha256=file_digest(Path(relay.__file__)), max_calls=calls,
            token_stop_threshold=tokens, call_seconds=config.call_seconds)
        self.service = relay.LocalService(Path(root), self.policy, executable=command,
                                          executor=executor, allow_live=True)

    def __call__(self, argv, timeout, *, stdin=None):
        if self.accounting_store is not None:
            if validate_inherited_model_accounting(self.accounting_store, self.config, self.deadline) != self.inherited:
                raise IntegrityError("Inherited accounting changed after executor construction")
            if self.accounting_store.get("contract_repair_manifest_ref"):
                from .contract_repair import require_repair_dispatch_gate
                require_repair_dispatch_gate(self.accounting_store)
        if not isinstance(argv, list) or "--cd" not in argv:
            raise relay.RelayError("Unexpected direct pilot invocation")
        directory = Path(argv[argv.index("--cd") + 1])
        images = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--image"]
        if argv != relay.cli_argv("codex", directory, self.policy.model, images=images):
            raise relay.RelayError("Direct pilot argv differs from the fixed response-only backend")
        relay.private_directory(directory)
        schema = relay.read_json(directory / "response-schema.json", relay.SCHEMA_LIMIT, private=False)
        role = relay.validate_prompt_schema(stdin, schema)
        media = relay._stage_request_media(images, directory)
        now = time.time()
        if not relay._finite(timeout) or not 0 < timeout <= self.policy.call_seconds:
            raise relay.RelayError("Direct model timeout exceeds the frozen call ceiling")
        deadline = min(now + timeout, self.policy.expires_at)
        if deadline <= now:
            return ProcessResult(**relay._result(124, timed_out=True, stderr="Direct pilot deadline expired"))
        request = {"schema_version": 2, "request_id": uuid.uuid4().hex,
                   "policy_sha256": self.policy.identity, "run_id": self.policy.run_id,
                   "model": self.policy.model, "role": role, "created_at": now,
                   "deadline": deadline, "timeout": timeout, "prompt": stdin, "schema": schema,
                   "source_sha256": self.policy.source_sha256, "media": media}
        response = self.service.serve(request)
        process, output = relay.validate_response(response, request)
        if output is not None:
            with (directory / "response.json").open("x") as stream:
                stream.write(output)
        return ProcessResult(**process)

    def accounting(self):
        receipts, unknown = [], []
        for entry in sorted(self.service.claims.iterdir()):
            if not (entry / "dispatch.json").is_file():
                unknown.append(entry.name)
                continue
            receipt = relay.read_json(entry / "dispatch.json", 16384)
            receipts.append(receipt)
            if receipt.get("usage_status") != "known":
                unknown.append(entry.name)
        summary = {"source": "durable-local-model-ledger", "ledger": str(self.service.state),
                "claimed_calls": len(list(self.service.claims.iterdir())),
                "completed_receipts": len(receipts), "unknown_calls": unknown,
                "known_observed_tokens": sum(r["observed_tokens"] for r in receipts if r.get("usage_status") == "known"),
                "usage_complete": not unknown, "policy": asdict(self.policy),
                "transport": "direct_local_no_queue"}
        if self.inherited["manifest_ref"] is not None:
            summary.update(source="composite-direct-local-model-ledger",
                inherited_model_accounting_ref=self.inherited["manifest_ref"],
                inherited_claimed_calls=self.inherited["calls"], inherited_known_observed_tokens=self.inherited["tokens"],
                local_claimed_calls=summary["claimed_calls"], local_completed_receipts=summary["completed_receipts"],
                local_known_observed_tokens=summary["known_observed_tokens"],
                global_max_calls=self.config.max_calls, global_token_stop_threshold=self.config.token_stop_threshold)
            summary["claimed_calls"] += self.inherited["calls"]
            summary["completed_receipts"] += self.inherited["calls"]
            summary["known_observed_tokens"] += self.inherited["tokens"]
        return summary


def run_prepared_live(store, config, adapter, runner, *, allow_live=False, executable="codex",
                      process_executor=run_process, allow_network_literature=False):
    """Run a prepared research-v2 campaign through durable local model accounting.

    Native benchmark and bundled-pilot entrypoints share this dispatch boundary.
    Preparation, offline validation and completed resume never create a backend.
    """
    if not allow_live:
        raise ValueError("Live research calls require explicit allow_live=True")
    if config.engine_version != "research-v2":
        raise ValueError("Durable prepared execution requires research-v2")
    if config.token_stop_threshold is None or config.model_retries != 0:
        raise ValueError("Live research requires an explicit positive token_stop_threshold and model_retries=0")
    if config.model.startswith(("scripted", "fixture")):
        raise ValueError("Live research requires an explicit non-fixture model identifier")
    if config.implementation != implementation_identity() or store.verify():
        raise IntegrityError("Live research source or registered evidence changed")
    if store.get("live_spec_ref") and store.json(store.get("live_spec_ref")) != config.model_dump(mode="json"):
        raise IntegrityError("Live research specification changed")
    if store.get("inherited_model_accounting_ref"):
        validate_inherited_model_accounting(store, config)
    state = store.get("live_state", {})
    if state.get("status") == "complete":
        # Even the backend/executor constructor is skipped on completed resume.
        return state
    if store.get("contract_repair_manifest_ref"):
        from .contract_repair import ContractRepairBudget
        runner.budget = ContractRepairBudget(runner.budget)
    runner.budget.remaining()  # Commit the one campaign deadline before dispatch.
    executor = DirectPilotExecutor(store.root / "model-ledger", config, store.get("live_deadline"),
                                   executable=executable, executor=process_executor, allow_live=True,
                                   accounting_store=store)
    backend = CodexBackend(store.root / "agent", config.model, allow_live=True,
                           timeout=config.call_seconds, executor=executor)
    from .controller import ResearchController
    controller = ResearchController(store, config, adapter, backend, runner, budget=runner.budget,
                                    allow_network_literature=allow_network_literature)
    try:
        return controller.run()
    finally:
        summary = executor.accounting()
        store.set("live_pilot_model_accounting", store.put_json(summary, "live-pilot-model-accounting"))


def _run_prepared(store, config, adapter, runner, *, allow_live, executable, process_executor):
    """Compatibility entrypoint for the original bundled live-pilot callers."""
    return run_prepared_live(store, config, adapter, runner, allow_live=allow_live,
                             executable=executable, process_executor=process_executor)


def run_live_pilot(store, task, profile, *, options, allow_live=False, runtime="docker",
                   image="scientisttwo-worker:0.1", context=None, executable="codex",
                   process_executor=run_process, container_runner=None, experiment_executor=run_process):
    """Run only after explicit opt-in; prepare/freeze options before any backend exists."""
    if not allow_live:
        raise ValueError("Live pilot calls require explicit allow_live=True")
    from .pilot_bridge import create_pilot_campaign
    config, adapter, runner = create_pilot_campaign(store, task, profile, runtime, image, context,
        live_options=LivePilotOptions.model_validate(options), container_runner=container_runner,
        executor=experiment_executor)
    return _run_prepared(store, config, adapter, runner, allow_live=True,
                         executable=executable, process_executor=process_executor)


def resume_live_pilot(store, *, allow_live=False, executable="codex", process_executor=run_process,
                      container_runner=None, experiment_executor=run_process):
    """Resume exact saved live options; never reset deadlines or model ledgers."""
    if not allow_live:
        raise ValueError("Live pilot resume requires explicit allow_live=True")
    if not store.get("live_pilot_options_ref"):
        raise ValueError("This is not a declared live pilot")
    task = TaskSpec.model_validate(store.json(store.get("task_ref")))
    profile = Profile.model_validate(store.json(store.get("profile_ref")))
    return run_live_pilot(store, task, profile,
        options=store.json(store.get("live_pilot_options_ref")), allow_live=True,
        **store.get("runtime"), executable=executable, process_executor=process_executor,
        container_runner=container_runner, experiment_executor=experiment_executor)
