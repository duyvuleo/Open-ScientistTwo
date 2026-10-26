"""Synthetic recovery contracts only: no actual models, containers or benchmark run."""

from copy import deepcopy
from types import SimpleNamespace
import time

import pytest

from scientisttwo.integrations.paperorchestra import manuscript_binding
from scientisttwo.integrations.paperorchestra.binding_normalization import normalize_redundant_numeric_claims as normalize
from scientisttwo.live.adapters import implementation_identity
from scientisttwo.live.models import DevelopmentSpec, LiveSpec
from scientisttwo.live.runtime import Budget
from scientisttwo.live.workspace_models import WorkspaceSpec
from scientisttwo.models import AgentResult
from scientisttwo.research import native_binding_recovery as recovery
from scientisttwo.research.policy import ResearchPolicy
from scientisttwo.store import IntegrityError, Store, canonical, digest
from test_native_timing_acceptance import _roles_fixture
from test_paperorchestra_binding_normalization import example


@pytest.fixture
def blocked(tmp_path):
    native = WorkspaceSpec(name="vectorization", task_id="vectorization", public_dir=tmp_path / "public",
        private_dir=tmp_path / "private", candidate_dir=tmp_path / "reference", evaluator_dir=tmp_path / "evaluator",
        command=["python", "train.py"], evaluator_command=["python", "timing_worker.py"],
        metric="latency_ratio", direction="minimize", image="sha256:" + "1" * 64, grader_image="sha256:" + "1" * 64)
    config = LiveSpec(native=native, development=DevelopmentSpec(adapter="testbed", metric="latency_ratio", direction="minimize"),
        engine_version="research-v2", mode="research", model="gpt-6-astra", max_calls=100,
        token_stop_threshold=2_000_000, campaign_seconds=7200, model_retries=0,
        workflow=ResearchPolicy(writer_backend="paperorchestra", paperorchestra_root=str(tmp_path / "upstream"),
            compiler_image="sha256:" + "2" * 64), implementation=implementation_identity())
    store = Store(tmp_path / "campaign")
    budget = Budget(store, config)
    store.set("live_spec_ref", store.put_json(config))
    store.set("live_deadline", time.time() + 7200)
    store.set("live_state", {"stage": "draft", "status": "blocked", "outcome": "invalid_agent_output",
        "reason": "paperorchestra_bindings: Overlapping source claim/value bindings are ambiguous; output-correction allowance exhausted"})
    _roles_fixture(store, config)
    bindings, latex, ledger = example()
    key = "synthetic/paperorchestra_bindings"
    inputs = {"latex": latex, "ledger": ledger, "sources": [], "synthetic_agents": False, "supplementary_coverage": []}
    request = {"role": "paperorchestra_bindings", "inputs": inputs}
    response = AgentResult(payload=bindings, backend="codex", model=config.model, synthetic=False,
                           usage={"input_tokens": 1, "output_tokens": 1})
    ref = store.put_json(response)
    store.set("request/" + key, store.put_json(request))
    with store.db:
        store.db.execute("UPDATE calls SET input_hash=?,output_ref=? WHERE key=?", (digest(canonical(request)), ref, "live-agent/" + key))
        store.db.execute("UPDATE live_operations SET request_hash=?,result_ref=? WHERE key=?", (digest(canonical(request)), ref, "model/" + key + "/0"))
    old_proof = store.put_json({"status": "rejected", "response_ref": ref})
    store.set("agent-output/" + key, old_proof)
    yield store, config, budget, ref, inputs, bindings
    store.close()


def prepare(blocked):
    store, _, _, ref, *_ = blocked
    return recovery.prepare_recovery(store, approved_response_ref=ref,
        authorization_ref="synthetic-test-only-no-live-authority", normalizer=normalize)


def test_exact_wrapper_preserves_original_response_guard_and_rows(blocked):
    store, _, _, ref, inputs, bindings = blocked
    original = manuscript_binding.bind_paperorchestra_manuscript
    response_bytes = store.read(ref)
    manifest_ref = prepare(blocked)
    state = store.get("live_state")
    with pytest.raises(ValueError, match="Overlapping"):
        original(bindings, inputs["latex"], inputs["ledger"], synthetic_agents=False)
    with recovery.recovery_scope(store, manifest_ref, normalizer=normalize) as (proxy, hit):
        assert proxy.get("live_state")["status"] == "running" and store.get("live_state") == state
        manuscript = manuscript_binding.bind_paperorchestra_manuscript(bindings, inputs["latex"], inputs["ledger"], synthetic_agents=False)
        assert len(manuscript.claims) == 1 and hit["validated"]
        changed = deepcopy(bindings)
        changed["title"] = "Absent title must still fail"
        with pytest.raises(ValueError, match="title"):
            manuscript_binding.bind_paperorchestra_manuscript(changed, inputs["latex"], inputs["ledger"], synthetic_agents=False)
        old = store.get("agent-output/synthetic/paperorchestra_bindings")
        accepted = store.put_json({"status": "accepted"})
        proxy.set("agent-output/synthetic/paperorchestra_bindings", accepted)
        assert store.get("agent-output/synthetic/paperorchestra_bindings") == old
        assert proxy.get("agent-output/synthetic/paperorchestra_bindings") == accepted
    assert manuscript_binding.bind_paperorchestra_manuscript is original
    assert store.read(ref) == response_bytes
    assert recovery.verify_recovery(store, manifest_ref, normalizer=normalize)["approved_response_ref"] == ref


@pytest.mark.parametrize("unexpected_work", [False, True])
def test_dry_replay_stops_before_new_work_or_render(blocked, monkeypatch, unexpected_work):
    store, config, budget, _, inputs, bindings = blocked
    manifest_ref = prepare(blocked)
    runner = SimpleNamespace(store=store, budget=budget, preflight=lambda: None)
    monkeypatch.setattr(recovery, "recorded_preflight", lambda *args: None)

    class Controller:
        def __init__(self, proxy, config, adapter, backend, runner, *, budget):
            self.store, self.budget, self.runner = proxy, budget, runner
            assert isinstance(backend, recovery.NoCallsBackend)

        def run(self):
            self.runner.preflight()
            self.store.set("live_best", {"id": "earlier-chronological-incumbent"})
            assert self.store.get("live_best") == {"id": "earlier-chronological-incumbent"}
            assert store.get("live_best") is None
            if unexpected_work:
                self.budget.reserve("model/unexpected/0", "model", {})
            manuscript_binding.bind_paperorchestra_manuscript(bindings, inputs["latex"], inputs["ledger"], synthetic_agents=False)
            pytest.fail("Dry replay must stop before manuscript rendering")

    monkeypatch.setattr("scientisttwo.research.controller.ResearchController", Controller)
    before = canonical(budget.summary())
    if unexpected_work:
        with pytest.raises(recovery.DryReplayBlocked, match="new model work"):
            recovery.dry_replay(store, manifest_ref, normalizer=normalize, adapter=None, runner=runner)
        assert store.get("native_binding_recovery_dry_ref") is None
    else:
        proof = recovery.dry_replay(store, manifest_ref, normalizer=normalize, adapter=None, runner=runner)
        assert proof["status"] == "pass" and proof["added_model_calls"] == proof["added_operations"] == 0
        assert proof["rendered_new_manuscript"] is False
    assert canonical(budget.summary()) == before
    assert store.get("live_best") is None


def test_continuation_requires_explicit_flag_and_successful_dry_proof(blocked):
    store, _, budget, *_ = blocked
    manifest_ref = prepare(blocked)
    runner = SimpleNamespace(store=store, budget=budget, preflight=lambda: None)
    with pytest.raises(ValueError, match="allow_live"):
        recovery.continue_recovery(store, manifest_ref, normalizer=normalize, adapter=None, runner=runner)
    with pytest.raises(IntegrityError, match="dry replay"):
        recovery.continue_recovery(store, manifest_ref, normalizer=normalize, adapter=None, runner=runner, allow_live=True)


@pytest.mark.parametrize("tamper", ["history", "deadline", "receipt"])
def test_original_evidence_and_deadline_cannot_change(blocked, tamper):
    store, *_ = blocked
    manifest_ref = prepare(blocked)
    if tamper == "history":
        with store.db:
            store.db.execute("UPDATE calls SET input_hash='changed' WHERE key='live-agent/synthetic/paperorchestra_bindings'")
    elif tamper == "deadline":
        store.set("live_deadline", store.get("live_deadline") + 1)
    else:
        receipt = next((store.root / "model-ledger/claims").iterdir()) / "dispatch.json"
        receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(IntegrityError):
        recovery.verify_recovery(store, manifest_ref, normalizer=normalize)


def test_dry_metadata_overlay_is_discarded_and_protected_values_stay_fixed(blocked):
    store, *_ = blocked
    manifest_ref = prepare(blocked)
    before = [(row["key"], row["value"]) for row in store.db.execute("SELECT * FROM metadata ORDER BY key")]
    with recovery.recovery_scope(store, manifest_ref, normalizer=normalize, dry=True) as (proxy, _):
        proxy.set("live_best", {"id": "intermediate"})
        proxy.set("new-ephemeral-checkpoint", [1, 2])
        assert proxy.get("live_best") == {"id": "intermediate"}
        assert proxy.get("new-ephemeral-checkpoint") == [1, 2]
        with pytest.raises(recovery.DryReplayBlocked, match="protected"):
            proxy.set("live_deadline", store.get("live_deadline") + 1)
    assert before == [(row["key"], row["value"]) for row in store.db.execute("SELECT * FROM metadata ORDER BY key")]


def test_operator_supersession_is_explicit_append_only_and_keeps_limits(blocked, monkeypatch):
    store, _, _, response_ref, *_ = blocked
    original_identity = recovery._module_identity(normalize)
    monkeypatch.setattr(recovery, "_module_identity", lambda _: {**original_identity, "operator_sha256": "1" * 64})
    first = prepare(blocked)
    original_bytes = store.read(first)
    first_record = store.json(first)
    monkeypatch.setattr(recovery, "_module_identity", lambda _: {**original_identity, "operator_sha256": "2" * 64})
    with pytest.raises(IntegrityError, match="operator"):
        prepare(blocked)
    second = recovery.prepare_recovery(store, approved_response_ref=response_ref,
        authorization_ref="synthetic-test-only-no-live-authority", normalizer=normalize,
        repair_id="redundant-numeric-claims-v2", supersedes_ref=first)
    current = recovery.verify_recovery(store, second, normalizer=normalize)
    assert current["supersedes_ref"] == first and current["operator_revision"] == 3
    assert current["protected_metadata"] == first_record["protected_metadata"]
    assert current["parent_accounting"] == first_record["parent_accounting"]
    assert store.read(first) == original_bytes


@pytest.mark.parametrize("change", ["operation", "event", "model_claim", "state", "authorization"])
def test_supersession_cannot_hide_intervening_work_or_change_scope(blocked, change):
    store, _, _, response_ref, *_ = blocked
    first = prepare(blocked)
    authorization = "synthetic-test-only-no-live-authority"
    if change == "operation":
        with store.db:
            store.db.execute("INSERT INTO live_operations VALUES(?,?,?,?,?,?,?)",
                ("experiment/added", "experiment", "request", "success", 1, 0, store.put_json({})))
    elif change == "event":
        store.event("new-work", "draft", "New event is not a zero-work dry replay")
    elif change == "model_claim":
        (store.root / "model-ledger/new-claim.json").write_text("{}")
    elif change == "state":
        store.set("live_state", {**store.get("live_state"), "status": "running"})
    else:
        authorization = "different-scope"
    with pytest.raises(IntegrityError, match="supersession"):
        recovery.prepare_recovery(store, approved_response_ref=response_ref, authorization_ref=authorization,
            normalizer=normalize, repair_id="v2", supersedes_ref=first)


def test_effective_source_fallback_survives_real_render_and_verify(tmp_path, monkeypatch):
    """Reproduce the omitted-sources call inside the actual publication verifier."""
    from test_paperorchestra_manuscript_binding import setup as publication_fixture, publication_inputs, LATEX, IMAGE
    from scientisttwo.integrations.paperorchestra.manuscript_binding import SourceClaimBinding, fixture_bindings
    from scientisttwo.research.manuscript import LedgerClaim, build_evidence_ledger

    fixture = publication_fixture.__wrapped__(tmp_path)
    store, budget, _ = next(fixture)
    try:
        source = {"id": "nonempty-source", "title": "Synthetic supplied source", "text": "Synthetic test context."}
        observation_ref = store.db.execute("SELECT output_ref FROM calls WHERE key='development/baseline'").fetchone()[0]
        ledger = build_evidence_ledger(store, [store.json(observation_ref)], sources=[source])
        sidecar = fixture_bindings(LATEX, ledger)
        evidence_id = next(key for key, entry in ledger["entries"].items() if entry.get("kind") == "measured")
        quote = r"Recorded mean & 0.6 & \\"
        start, number = LATEX.index(quote), LATEX.index("0.6")
        sidecar.claims = [
            SourceClaimBinding(start=start, end=start + len(quote), quote=quote,
                claim=LedgerClaim(id="interpretation", kind="interpretation", text=quote,
                    evidence_ids=[evidence_id], attribution_scope="package_comparison")),
            SourceClaimBinding(start=number, end=number + 3, quote="0.6",
                claim=LedgerClaim(id="numeric", kind="measured", text="0.6", value=.6,
                    evidence_ids=[evidence_id], attribution_scope="package_comparison")),
        ]
        normalized = normalize(sidecar, LATEX, ledger, [source], synthetic_agents=True)
        manifest = {"parent_state": {}, "approved_response_key": "fixture",
            "binding_identity": recovery._binding_identity(sidecar, LATEX, ledger, [source], True, []),
            "normalized_bindings_ref": store.put_json(normalized["bindings"])}
        monkeypatch.setattr(recovery, "verify_recovery", lambda *args, **kwargs: manifest)
        original = manuscript_binding.bind_paperorchestra_manuscript
        with pytest.raises(ValueError, match="Overlapping"):
            original(sidecar, LATEX, ledger, synthetic_agents=True)
        publication = publication_inputs((store, budget, ledger), tmp_path)
        before = canonical(budget.summary())
        with recovery.recovery_scope(store, "synthetic-component-manifest", normalizer=normalize):
            manuscript = manuscript_binding.bind_paperorchestra_manuscript(sidecar, LATEX, ledger, [source], synthetic_agents=True)
            rendered = manuscript_binding.render_bound_manuscript(store, manuscript, sidecar, ledger, [source],
                **publication)
            checks = manuscript_binding.verify_bound_manifest(store, store.json(rendered.manifest_ref), expected_image_digest=IMAGE)
            assert checks["authoritative_source_binding"] == "pass"
            assert store.read(rendered.latex_ref) == LATEX.encode()
        assert canonical(budget.summary()) == before
        with pytest.raises(ValueError, match="Overlapping"):
            manuscript_binding.verify_bound_manifest(store, store.json(rendered.manifest_ref))
    finally:
        fixture.close()


def test_only_bound_zero_work_post_render_failure_can_supersede(blocked, monkeypatch):
    store, _, _, response_ref, inputs, bindings = blocked
    # Bind the first request to a synthetic authoritative runtime, before the
    # immutable recovery snapshot is created.
    request_ref = store.get("request/synthetic/paperorchestra_bindings")
    request = store.json(request_ref)
    runtime_ref = store.put_json({"fixture": "synthetic compiled runtime"})
    request["inputs"]["authoritative_runtime_ref"] = runtime_ref
    store.set("request/synthetic/paperorchestra_bindings", store.put_json(request))
    with store.db:
        store.db.execute("UPDATE calls SET input_hash=? WHERE key='live-agent/synthetic/paperorchestra_bindings'", (digest(canonical(request)),))
        store.db.execute("UPDATE live_operations SET request_hash=? WHERE key='model/synthetic/paperorchestra_bindings/0'", (digest(canonical(request)),))
    first = prepare(blocked)
    prior = store.json(first)
    normalized = normalize(bindings, inputs["latex"], inputs["ledger"], synthetic_agents=False)
    manuscript = manuscript_binding.bind_paperorchestra_manuscript(normalized["bindings"], inputs["latex"], inputs["ledger"], synthetic_agents=False)
    latex_ref = store.put(inputs["latex"].encode(), "tex")
    pdf_ref = store.put(b"%PDF-synthetic", "pdf")
    receipt_ref = store.put_json({"fixture": "synthetic receipt"})
    writer = {"type": "paperorchestra", "runtime_ref": runtime_ref,
        "latex_sha256": prior["binding_identity"]["latex_sha256"], "bindings_ref": store.put_json(bindings),
        "source_pdf_ref": pdf_ref, "compilation_receipt_ref": receipt_ref, "source_entrypoint": "main.tex"}
    bundle = {"writer": writer, "ledger_ref": store.put_json(inputs["ledger"]), "manuscript_ref": store.put_json(manuscript),
        "pdf_ref": pdf_ref, "files": {"main.tex": {"ref": latex_ref, "sha256": digest(inputs["latex"].encode()),
                                                     "bytes": len(inputs["latex"].encode())}}}
    bundle_ref = store.put_json(bundle, "manuscript_bundle")
    store.set("native_binding_recovery_dry_ref", store.put_json({"status": "pass", "manifest_ref": first,
        "added_model_calls": 0, "added_operations": 0}))
    state = {"stage": "draft", "status": "blocked", "outcome": "blocked",
             "reason": "Overlapping source claim/value bindings are ambiguous"}
    store.set("live_state", state)
    monkeypatch.setattr(manuscript_binding, "_authoritative_receipt", lambda *args: ({}, receipt_ref, "main.tex"))
    second = recovery.prepare_recovery(store, approved_response_ref=response_ref,
        authorization_ref="synthetic-test-only-no-live-authority", normalizer=normalize,
        repair_id="source-fallback-v3", supersedes_ref=first)
    manifest = recovery.verify_recovery(store, second, normalizer=normalize)
    assert manifest["parent_state"] == state
    assert manifest["supersession_cause"]["sealed_bundle_ref"] == bundle_ref
    assert manifest["supersession_cause"]["prior_state"] == prior["parent_state"]
    assert manifest["parent_accounting"] == prior["parent_accounting"]
