"""Explicit operator recovery of one completed, rejected native writer binding.

This module is loaded outside the pinned campaign package. It does not pretend
the effective validator is unchanged: its own source and the pure normalizer are
hash-bound in an append-only receipt. The original response, source epoch, model
policy and all historical operation/call/event/artifact rows remain unchanged.
Auditing the recovered manuscript requires the same scoped operator wrapper.
"""

from contextlib import contextmanager
from copy import deepcopy
import inspect
import json
from pathlib import Path
import re
import sqlite3
import time

from ..live.adapters import implementation_identity
from ..live.models import LiveSpec
from ..live.runtime import validate_prepared
from ..models import AgentResult
from ..store import IntegrityError, canonical, digest, file_digest


VERSION = "native-writer-binding-operator-recovery-v1"
_TABLES = {"calls": "key", "jobs": "key", "live_operations": "key", "events": "key", "artifacts": "path"}
_PROTECTED = ("live_spec_ref", "live_sources_ref", "live_deadline", "live_adapter_identity",
              "adapter_sources_ref", "live_images", "research_writer_identity", "live_backend")


class BindingReplayReady(RuntimeError):
    """The exact saved response validated; no new work was admitted."""


class DryReplayBlocked(BaseException):
    """Uncaught by ordinary controller error handling; preserves parent state."""


def _json(path):
    return json.loads(Path(path).read_bytes())


def _module_identity(normalizer):
    path = Path(inspect.getsourcefile(normalizer)).resolve()
    return {"operator_sha256": file_digest(Path(__file__)), "normalizer_sha256": file_digest(path),
            "normalizer_name": normalizer.__qualname__, "normalizer_path": str(path)}


def _model_files(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError("Model ledger missing or unsafe")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise IntegrityError("Model ledger contains a symlink")
        if path.is_file() and path.name != "service.lock":
            result[path.relative_to(root).as_posix()] = file_digest(path)
    return result


def _sources(value):
    from .manuscript import _source_index
    return _source_index(value)


def _binding_identity(bindings, latex, ledger, sources, synthetic_agents, concerns):
    from ..integrations.paperorchestra.manuscript_binding import PaperOrchestraBindings
    parsed = PaperOrchestraBindings.model_validate(bindings).model_dump(mode="json")
    return {"bindings_sha256": digest(canonical(parsed)), "latex_sha256": digest(latex.encode()),
            "ledger_sha256": digest(canonical(ledger)), "sources_sha256": digest(canonical(_sources(sources or ledger.get("sources", {})))),
            "synthetic_agents": synthetic_agents, "concern_responses_sha256": digest(canonical(concerns))}


def _verify_snapshot_history(store, manifest, *, require_no_new_work=False):
    store.read(manifest["parent_snapshot_ref"])
    snapshot = (store.root / manifest["parent_snapshot_ref"]).resolve()
    with sqlite3.connect(snapshot.as_uri() + "?mode=ro&immutable=1", uri=True) as parent:
        parent.row_factory = sqlite3.Row
        for table, key in _TABLES.items():
            rows = list(parent.execute("SELECT * FROM " + table))
            for row in rows:
                current = store.db.execute("SELECT * FROM " + table + " WHERE " + key + "=?", (row[key],)).fetchone()
                if current is None or dict(current) != dict(row):
                    raise IntegrityError("Historical recovery evidence row changed: " + table + "/" + row[key])
            if require_no_new_work and table != "artifacts":
                count = store.db.execute("SELECT count(*) FROM " + table).fetchone()[0]
                if count != len(rows):
                    raise IntegrityError("Recovery supersession forbids added work: " + table)
    current_files = _model_files(store.root / "model-ledger")
    if any(current_files.get(key) != value for key, value in manifest["parent_model_files"].items()):
        raise IntegrityError("An original model receipt or policy changed")
    if require_no_new_work and current_files != manifest["parent_model_files"]:
        raise IntegrityError("Recovery supersession forbids added model claims")


def _post_render_failure(store, prior, supersedes_ref):
    """Recognize only the observed zero-model fallback-source wrapper failure."""
    expected = {"stage": "draft", "status": "blocked", "outcome": "blocked",
                "reason": "Overlapping source claim/value bindings are ambiguous"}
    if store.get("live_state") != expected or store.get("research_manuscript_bundle"):
        raise IntegrityError("Recovery supersession does not match the exact post-render binding failure")
    dry_ref = store.get("native_binding_recovery_dry_ref")
    proof = store.json(dry_ref) if dry_ref else {}
    if (proof.get("status") != "pass" or proof.get("manifest_ref") != supersedes_ref
            or proof.get("added_model_calls") != 0 or proof.get("added_operations") != 0):
        raise IntegrityError("Post-render supersession requires the prior successful zero-work dry proof")
    request = store.json(prior["approved_request_ref"])["inputs"]
    normalization = store.json(prior["normalization_proof_ref"])
    identity = prior["binding_identity"]
    config = LiveSpec.model_validate(store.json(store.get("live_spec_ref")))
    from ..integrations.paperorchestra.manuscript_binding import _authoritative_receipt
    matches = []
    for row in store.db.execute("SELECT path FROM artifacts WHERE kind='manuscript_bundle'"):
        bundle = store.json(row["path"])
        writer = bundle.get("writer", {})
        if (writer.get("type") != "paperorchestra" or writer.get("runtime_ref") != request.get("authoritative_runtime_ref")
                or writer.get("latex_sha256") != identity["latex_sha256"]):
            continue
        if (digest(canonical(store.json(writer["bindings_ref"]))) != identity["bindings_sha256"]
                or digest(canonical(store.json(bundle["ledger_ref"]))) != identity["ledger_sha256"]
                or digest(canonical(store.json(bundle["manuscript_ref"]))) != normalization["canonical_manuscript_sha256"]):
            raise IntegrityError("Post-render bundle differs from the exact approved binding and ledger")
        for member in bundle["files"].values():
            payload = store.read(member["ref"])
            if digest(payload) != member["sha256"] or len(payload) != member["bytes"]:
                raise IntegrityError("Post-render bundle files differ from their sealed identities")
        latex = store.read(bundle["files"]["main.tex"]["ref"])
        pdf = store.read(bundle["pdf_ref"])
        if digest(latex) != identity["latex_sha256"] or not pdf.startswith(b"%PDF-"):
            raise IntegrityError("Post-render publication changed its authoritative bytes")
        _, receipt_ref, source_name = _authoritative_receipt(store, store.json(writer["runtime_ref"]),
            digest(latex), digest(pdf), config.workflow.compiler_image)
        if (receipt_ref != writer.get("compilation_receipt_ref") or source_name != writer.get("source_entrypoint")
                or writer.get("source_pdf_ref") != bundle["pdf_ref"]):
            raise IntegrityError("Post-render publication has inconsistent compile provenance")
        matches.append(row["path"])
    if len(matches) != 1:
        raise IntegrityError("Post-render supersession requires exactly one unchanged sealed publication bundle")
    return {"kind": "exact_zero-work_post-render_sources-fallback_failure", "prior_dry_ref": dry_ref,
            "prior_state": prior["parent_state"], "observed_state": expected,
            "sealed_bundle_ref": matches[0], "sealed_bundle_sha256": digest(store.read(matches[0]))}


def prepare_recovery(store, *, approved_response_ref, authorization_ref, normalizer,
                     repair_id="redundant-numeric-claims", supersedes_ref=None):
    """Snapshot a closed blocked campaign. No model, process or runtime launch.

    Caller must hold the ordinary campaign lock throughout prepare/dry/continue.
    Existing source must already be the pinned original package, and normalizer
    must have captured that original binder before the wrapper is installed.
    """
    from ..integrations.paperorchestra import manuscript_binding
    from .minimal_live_acceptance import _model_accounting
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", repair_id) or not authorization_ref:
        raise ValueError("A bounded repair ID and explicit existing authorization reference are required")
    prior, supersession_cause = None, None
    existing = store.get("native_binding_recovery_manifest_ref")
    if existing and supersedes_ref is None:
        manifest = verify_recovery(store, existing, normalizer=normalizer)
        if manifest["approved_response_ref"] != approved_response_ref or manifest["authorization_ref"] != authorization_ref:
            raise IntegrityError("Existing native recovery has a different approved scope")
        return existing
    if supersedes_ref is not None:
        if existing != supersedes_ref:
            raise IntegrityError("Supersession must name the exact active recovery manifest")
        prior = store.json(supersedes_ref)
        identity = _module_identity(normalizer)
        if (prior.get("schema_version") != VERSION or prior.get("root") != str(store.root)
                or prior.get("approved_response_ref") != approved_response_ref
                or prior.get("authorization_ref") != authorization_ref
                or prior.get("normalizer_sha256") != identity["normalizer_sha256"]
                or prior.get("normalizer_name") != identity["normalizer_name"]
                or store.get("live_deadline") != prior.get("original_deadline")
                or any(store.get(key) != value for key, value in prior["protected_metadata"].items())):
            raise IntegrityError("Recovery supersession must preserve the closed state, approved scope, normalizer and limits")
        _verify_snapshot_history(store, prior, require_no_new_work=True)
        if store.get("live_state") != prior.get("parent_state"):
            supersession_cause = _post_render_failure(store, prior, supersedes_ref)
    config = LiveSpec.model_validate(store.json(store.get("live_spec_ref")))
    state = store.get("live_state", {})
    if (config.development.adapter != "testbed" or config.native.task_id != "vectorization"
            or config.workflow.writer_backend != "paperorchestra" or config.engine_version != "research-v2"
            or state.get("status") != "blocked" or state.get("stage") != "draft"
            or (supersession_cause is None and (state.get("outcome") != "invalid_agent_output"
                or not state.get("reason", "").startswith("paperorchestra_bindings:")))
            or any(store.get(key) for key in ("live_frozen", "live_final", "research_manuscript_bundle"))
            or config.implementation != implementation_identity() or store.verify()):
        raise IntegrityError("Recovery requires the exact closed pre-publication native binding checkpoint")
    if (store.db.execute("SELECT 1 FROM live_operations WHERE status='running' OR key LIKE 'final/%'").fetchone()
            or store.db.execute("SELECT 1 FROM jobs WHERE status='running'").fetchone()
            or time.time() >= store.get("live_deadline", 0)):
        raise IntegrityError("Running/unknown work, final evidence or expired deadline blocks recovery")
    if normalizer.__globals__.get("bind_paperorchestra_manuscript") is not manuscript_binding.bind_paperorchestra_manuscript:
        raise IntegrityError("Normalizer must capture the original pinned binder before wrapping")
    accounting = _model_accounting(store, config)
    rows = list(store.db.execute("SELECT key FROM calls WHERE output_ref=? AND key LIKE 'live-agent/%'", (approved_response_ref,)))
    if len(rows) != 1:
        raise IntegrityError("Approved response must name exactly one committed model completion")
    response_key = rows[0]["key"].removeprefix("live-agent/")
    request_ref = store.get("request/" + response_key)
    request = store.json(request_ref)
    response = AgentResult.model_validate(store.json(approved_response_ref))
    if request["role"] != "paperorchestra_bindings" or response.synthetic or response.model != config.model:
        raise IntegrityError("Approved response is not a live binding completion")
    if not store.db.execute("SELECT 1 FROM live_operations WHERE kind='model' AND status='success' AND result_ref=? AND request_hash=?",
                            (approved_response_ref, digest(canonical(request)))).fetchone():
        raise IntegrityError("Approved binding lacks its completed model operation")
    inputs = request["inputs"]
    if inputs.get("supplementary_coverage"):
        raise IntegrityError("This operator supports only the first pre-review manuscript binding")
    parsed = manuscript_binding.PaperOrchestraBindings.model_validate(response.payload)
    normalized = normalizer(parsed, inputs["latex"], inputs["ledger"], inputs.get("sources", []),
                            synthetic_agents=False, concern_responses=[])
    if not normalized["proof"].get("changed") or normalized["proof"].get("original_binder_validation") != "pass":
        raise IntegrityError("Approved binding must pass the original binder after narrow normalization")
    if prior and (canonical(normalized["proof"]) != canonical(store.json(prior["normalization_proof_ref"]))
                  or canonical(normalized["bindings"]) != canonical(store.json(prior["normalized_bindings_ref"]))
                  or accounting != prior["parent_accounting"]):
        raise IntegrityError("Recovery supersession changed normalization or original resource accounting")
    source_identity = _module_identity(normalizer)
    destination = store.root / "binding-recovery" / repair_id
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = destination / "parent.sqlite"
    with sqlite3.connect(snapshot) as database:
        store.db.backup(database)
    snapshot_ref = store.put_file(snapshot, "native-binding-recovery-parent-snapshot")
    proof_ref = store.put_json(normalized["proof"], "native-binding-normalization-proof")
    normalized_ref = store.put_json(normalized["bindings"], "native-binding-normalized-annotations")
    manifest = {"schema_version": VERSION, "repair_id": repair_id, "authorization_ref": authorization_ref,
        "operator_revision": 3, "supersedes_ref": supersedes_ref, "supersession_cause": supersession_cause,
        "created_at": time.time(), "root": str(store.root), **source_identity,
        "parent_snapshot_ref": snapshot_ref, "parent_state": state,
        "protected_metadata": {key: store.get(key) for key in _PROTECTED},
        "parent_model_files": _model_files(store.root / "model-ledger"),
        "parent_accounting": accounting, "original_deadline": store.get("live_deadline"),
        "approved_response_ref": approved_response_ref, "approved_response_key": response_key,
        "approved_request_ref": request_ref,
        "binding_identity": _binding_identity(parsed, inputs["latex"], inputs["ledger"], inputs.get("sources", []), False, []),
        "normalization_proof_ref": proof_ref, "normalized_bindings_ref": normalized_ref,
        "scope": "Explicit external annotation normalization; original model response, TeX, numbers, experiments, budgets and source epochs preserved; recovery-aware audit required"}
    ref = store.put_json(manifest, "native-binding-recovery-manifest")
    store.set("native_binding_recovery_manifest_ref", ref)
    return ref


def verify_recovery(store, manifest_ref, *, normalizer):
    manifest = store.json(manifest_ref)
    if (manifest.get("schema_version") != VERSION or manifest.get("root") != str(store.root)
            or any(manifest.get(key) != value for key, value in _module_identity(normalizer).items())
            or store.get("native_binding_recovery_manifest_ref") != manifest_ref
            or any(store.get(key) != value for key, value in manifest["protected_metadata"].items())
            or store.get("live_deadline") != manifest["original_deadline"]):
        raise IntegrityError("Recovery operator, original source, configuration or deadline changed")
    config = LiveSpec.model_validate(store.json(store.get("live_spec_ref")))
    if config.implementation != implementation_identity():
        raise IntegrityError("Pinned original runtime changed during external recovery")
    response = AgentResult.model_validate(store.json(manifest["approved_response_ref"]))
    inputs = store.json(manifest["approved_request_ref"])["inputs"]
    from ..integrations.paperorchestra.manuscript_binding import PaperOrchestraBindings
    parsed = PaperOrchestraBindings.model_validate(response.payload)
    normalized = normalizer(parsed, inputs["latex"], inputs["ledger"], inputs.get("sources", []),
                            synthetic_agents=False, concern_responses=[])
    if (canonical(normalized["proof"]) != canonical(store.json(manifest["normalization_proof_ref"]))
            or canonical(normalized["bindings"]) != canonical(store.json(manifest["normalized_bindings_ref"]))
            or _binding_identity(parsed, inputs["latex"], inputs["ledger"], inputs.get("sources", []), False, []) != manifest["binding_identity"]):
        raise IntegrityError("Recorded normalization differs from the exact approved response and original binder")
    _verify_snapshot_history(store, manifest)
    if store.verify():
        raise IntegrityError("Recovery artifact integrity failed")
    return manifest


class _StoreProxy:
    def __init__(self, store, manifest, *, dry=False):
        self.original, self.manifest, self.dry = store, manifest, dry
        self.overlay = {}

    def __getattr__(self, name):
        return getattr(self.original, name)

    def get(self, key, default=None):
        if key in self.overlay:
            return deepcopy(self.overlay[key])
        value = self.original.get(key, default)
        if key == "live_state" and value == self.manifest["parent_state"]:
            return {**value, "status": "running", "outcome": "binding_recovery_pending",
                    "reason": "Explicitly replaying the approved saved binding under an external normalization receipt"}
        shadow = self.original.get("binding-recovery-shadow/" + key)
        return shadow if shadow is not None else value

    def set(self, key, value):
        old = self.original.get(key)
        if self.dry:
            protected = key in _PROTECTED or key in {
                "engine_version", "authorized_source_snapshot", "runtime_probe_ref", "runtime_hardware",
                "native_binding_recovery_manifest_ref", "native_binding_recovery_dry_ref"} or key.startswith("prepared/")
            if protected and old != value:
                raise DryReplayBlocked("Dry replay attempted changed protected metadata: " + key)
            self.overlay[key] = deepcopy(value)
            return None
        if key == "agent-output/" + self.manifest["approved_response_key"] and old not in (None, value):
            # Keep the original rejected validation pointer and append the new
            # source-bound interpretation separately; never rewrite paid output.
            return self.original.set("binding-recovery-shadow/" + key, value)
        return self.original.set(key, value)

    def cache(self, key, request, value):
        if self.dry and self.original.cached(key, request) is None:
            raise DryReplayBlocked("Dry replay attempted a new completed call: " + key)
        return self.original.cache(key, request, value)

    def event(self, key, stage, reason):
        if self.dry and not self.original.db.execute("SELECT 1 FROM events WHERE key=?", (key,)).fetchone():
            raise DryReplayBlocked("Dry replay attempted a new event: " + key)
        return self.original.event(key, stage, reason)


class _ReplayBudget:
    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def reserve(self, key, kind, request):
        if not self.delegate.store.db.execute("SELECT 1 FROM live_operations WHERE key=?", (key,)).fetchone():
            raise DryReplayBlocked("Dry binding replay attempted new " + kind + " work: " + key)
        return self.delegate.reserve(key, kind, request)


class NoCallsBackend:
    name, synthetic = "codex", False

    def invoke(self, *args, **kwargs):
        raise DryReplayBlocked("Dry binding replay forbids model dispatch")


def recorded_preflight(runner, store):
    """Validate existing data/artifacts and restore recorded runtime identity."""
    from ..benchmarks.files import verify_tree
    if runner.adapter.identity() != store.get("live_adapter_identity"):
        raise IntegrityError("Prepared adapter changed before cached replay")
    images, hardware = store.get("live_images"), store.get("runtime_hardware")
    probe = store.json(store.get("runtime_probe_ref"))
    if (not images or not hardware or hardware.get("status") != "verified"
            or probe.get("report", {}).get("status") != "pass"
            or images["execution"] != runner.spec.image or images["grading"] != runner.spec.grader_image):
        raise IntegrityError("Exact verified recorded runtime identities are required")
    for field in ("public", "candidate", "evaluator"):
        verify_tree(getattr(runner.spec, field + "_dir"), getattr(runner.spec, field + "_manifest"))
    for phase in ("subset", "full"):
        validate_prepared(store.json(store.get("prepared/" + phase)), store.root / "development" / phase)
    runner.image, runner.hardware = images["execution"], deepcopy(hardware)


@contextmanager
def recovery_scope(store, manifest_ref, *, normalizer, dry=False):
    """Wrap only the approved binding hash; other inputs use the original guard."""
    from ..integrations.paperorchestra import manuscript_binding, controller_writer
    manifest = verify_recovery(store, manifest_ref, normalizer=normalizer)
    original = manuscript_binding.bind_paperorchestra_manuscript
    if normalizer.__globals__.get("bind_paperorchestra_manuscript") is not original:
        raise IntegrityError("Import the normalizer before installing the recovery wrapper")
    proxy = _StoreProxy(store, manifest, dry=dry)
    hit = {"validated": False}

    def wrapped(bindings, latex, ledger, sources=(), *, synthetic_agents, concern_responses=()):
        identity = _binding_identity(bindings, latex, ledger, sources, synthetic_agents, list(concern_responses))
        if identity != manifest["binding_identity"]:
            return original(bindings, latex, ledger, sources, synthetic_agents=synthetic_agents,
                            concern_responses=concern_responses)
        normalized = normalizer(bindings, latex, ledger, sources, synthetic_agents=synthetic_agents,
                                concern_responses=concern_responses)
        if canonical(normalized["bindings"]) != canonical(store.json(manifest["normalized_bindings_ref"])):
            raise IntegrityError("Normalized binding differs from its approved immutable result")
        result = original(normalized["bindings"], latex, ledger, sources, synthetic_agents=synthetic_agents,
                          concern_responses=concern_responses)
        hit["validated"] = True
        if dry:
            raise BindingReplayReady("Exact saved binding validates before render or new work")
        return result

    previous_writer = getattr(controller_writer, "bind_paperorchestra_manuscript", None)
    manuscript_binding.bind_paperorchestra_manuscript = wrapped
    if previous_writer is not None:
        controller_writer.bind_paperorchestra_manuscript = wrapped
    try:
        yield proxy, hit
    finally:
        manuscript_binding.bind_paperorchestra_manuscript = original
        if previous_writer is not None:
            controller_writer.bind_paperorchestra_manuscript = previous_writer


def dry_replay(store, manifest_ref, *, normalizer, adapter, runner):
    """Replay caches through the approved binding and stop before rendering."""
    from .controller import ResearchController
    config = LiveSpec.model_validate(store.json(store.get("live_spec_ref")))
    before = {table: [dict(row) for row in store.db.execute("SELECT * FROM " + table + " ORDER BY " + key)]
              for table, key in _TABLES.items() if table != "artifacts"}
    model_before = _model_files(store.root / "model-ledger")
    previous = runner.store, runner.budget, runner.preflight
    reached = False
    try:
        with recovery_scope(store, manifest_ref, normalizer=normalizer, dry=True) as (proxy, hit):
            runner.store, runner.budget = proxy, _ReplayBudget(runner.budget)
            runner.preflight = lambda: recorded_preflight(runner, store)
            controller = ResearchController(proxy, config, adapter, NoCallsBackend(), runner, budget=runner.budget)
            def forbidden_finish(*args, **kwargs):
                raise DryReplayBlocked("Dry replay attempted controller.finish before the binding gate: " + str(args))
            controller.finish = forbidden_finish
            original_transition = getattr(controller, "transition", None)
            if original_transition is not None:
                def cached_transition(key, stage, reason):
                    if not store.db.execute("SELECT 1 FROM events WHERE key=?", ("live/" + key,)).fetchone():
                        raise DryReplayBlocked("Dry replay attempted a new workflow transition: " + key)
                    return original_transition(key, stage, reason)
                controller.transition = cached_transition
            try:
                state = controller.run()
            except BindingReplayReady:
                reached = hit["validated"]
            else:
                raise IntegrityError("Dry replay failed before the binding gate: " + str(state))
    finally:
        runner.store, runner.budget, runner.preflight = previous
    after = {table: [dict(row) for row in store.db.execute("SELECT * FROM " + table + " ORDER BY " + key)] for table, key in _TABLES.items() if table != "artifacts"}
    if not reached or before != after or model_before != _model_files(store.root / "model-ledger"):
        raise IntegrityError("Dry replay changed historical work or did not reach approved normalization")
    verify_recovery(store, manifest_ref, normalizer=normalizer)
    proof = {"schema_version": VERSION, "manifest_ref": manifest_ref, "status": "pass", "verified_at": time.time(),
             "added_model_calls": 0, "added_operations": 0, "rendered_new_manuscript": False,
             "history_sha256": digest(canonical(before)), "model_files_sha256": digest(canonical(model_before))}
    ref = store.put_json(proof, "native-binding-recovery-dry-proof")
    store.set("native_binding_recovery_dry_ref", ref)
    return proof


def continue_recovery(store, manifest_ref, *, normalizer, adapter, runner, allow_live=False,
                      executable="codex", process_executor=None):
    """Continue only after the no-work dry proof; reuse the original live ledger."""
    from .live_pilot import run_prepared_live
    from ..process import run_process
    if not allow_live:
        raise ValueError("Actual recovery requires explicit allow_live=True")
    manifest = verify_recovery(store, manifest_ref, normalizer=normalizer)
    proof_ref = store.get("native_binding_recovery_dry_ref")
    if not proof_ref:
        raise IntegrityError("Actual recovery requires its successful cached dry replay")
    proof = store.json(proof_ref)
    if proof.get("status") != "pass" or proof.get("manifest_ref") != manifest_ref:
        raise IntegrityError("Actual recovery requires its successful cached dry replay")
    if time.time() >= manifest["original_deadline"]:
        raise IntegrityError("Original campaign deadline expired; recovery cannot extend it")
    config = LiveSpec.model_validate(store.json(store.get("live_spec_ref")))
    previous = runner.store, runner.preflight
    try:
        with recovery_scope(store, manifest_ref, normalizer=normalizer) as (proxy, _):
            runner.store = proxy
            runner.preflight = lambda: recorded_preflight(runner, store)
            return run_prepared_live(proxy, config, adapter, runner, allow_live=True,
                executable=executable, process_executor=process_executor or run_process)
    finally:
        runner.store, runner.preflight = previous
        verify_recovery(store, manifest_ref, normalizer=normalizer)
