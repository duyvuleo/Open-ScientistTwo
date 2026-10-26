"""Synthetic receipt graphs only; these tests are never live benchmark evidence."""

from dataclasses import asdict
from importlib.metadata import entry_points
import json
from pathlib import Path
import time

import pytest

from upstream_checkouts import require_upstream

from scientisttwo.live.audit import _metric_proof
from scientisttwo.live.adapters import load_development
from scientisttwo.live.models import Observation
from scientisttwo.live.paired_evaluation import execute_paired, execute_paired_final
from scientisttwo.live.runtime import Budget, DevelopmentRunner
from scientisttwo.live.workspace_models import WorkspaceSpec
from scientisttwo.models import AgentResult
from scientisttwo.research.native_timing_acceptance import (
    MANDATORY_ROLES, _observation, _prepared_contract, capture_native_resume_snapshot, verify_native_timing_live,
)
from scientisttwo.store import IntegrityError, Store, canonical, digest
from scientisttwo.studies import media_relay as relay
from scientisttwo.testbed.adapter import PreparedDevelopment
from scientisttwo.testbed.catalog import UPSTREAM_REVISION
from scientisttwo.testbed.preparation import prepare
from scientisttwo.testbed.validation import create_validation_spec
from test_paired_evaluation import synthetic_pair


def _roles_fixture(store, config):
    """Create explicitly synthetic known-usage receipts without launching a CLI."""
    ledger = relay.private_directory(store.root / "model-ledger")
    claims = relay.private_directory(ledger / "claims")
    policy = relay.RelayPolicy(run_id="synthetic-native-acceptance", expires_at=store.get("live_deadline"),
        model=config.model, allowed_models=(config.model,), source_sha256=digest(canonical(config.implementation)),
        relay_sha256=digest(Path(relay.__file__).read_bytes()), max_calls=config.max_calls,
        token_stop_threshold=config.token_stop_threshold, call_seconds=config.call_seconds)
    relay.publish_new(ledger / "policy.json", asdict(policy))
    for index, role in enumerate(sorted(MANDATORY_ROLES)):
        key, request = "synthetic/" + role, {"role": role, "inputs": {"fixture": "synthetic"}}
        response = AgentResult(payload={}, backend="codex", model=config.model, synthetic=False,
                               usage={"input_tokens": 1, "output_tokens": 1})
        ref = store.put_json(response)
        store.set("request/" + key, store.put_json(request))
        store.cache("live-agent/" + key, request, response)
        with store.db:
            store.db.execute("INSERT INTO live_operations VALUES(?,?,?,?,?,?,?)",
                ("model/" + key + "/0", "model", digest(canonical(request)), "success", .1, 2, ref))
        now = time.time()
        wire = {"schema_version": 2, "request_id": f"{index:032x}", "policy_sha256": policy.identity,
            "run_id": policy.run_id, "model": config.model, "role": role, "created_at": now,
            "deadline": now + 120, "timeout": 120, "source_sha256": policy.source_sha256, "media": [],
            "prompt": json.dumps({"role": role, "prompt_version": "scientisttwo-v1",
                                  "instructions": relay.INSTRUCTIONS, "inputs": {"fixture": "synthetic"}}),
            "schema": {"type": "object"}}
        entry = relay.private_directory(claims / wire["request_id"])
        request_hash = digest(canonical(wire))
        relay.publish_new(entry / "claim.json", {"request": wire, "request_sha256": request_hash})
        relay.publish_new(entry / "dispatch.json", {"request_sha256": request_hash,
            "status": "completed", "usage_status": "known", "synthetic": False, "model": config.model,
            "observed_tokens": 2, "returncode": 0, "started_at": now})
        relay.publish_new(entry / "response.json", relay.response_envelope(wire, relay._result(0), "{}"))
    store.set("live_pilot_model_accounting", store.put_json({"ledger": str(ledger),
        "transport": "direct_local_no_queue", "policy": asdict(policy), "usage_complete": True,
        "unknown_calls": [], "claimed_calls": len(MANDATORY_ROLES),
        "completed_receipts": len(MANDATORY_ROLES), "known_observed_tokens": 2 * len(MANDATORY_ROLES)}))


@pytest.fixture(scope="module")
def prepared_vectorization(tmp_path_factory):
    """Real pinned recipe preparation only; no candidate or container execution."""
    upstream = require_upstream("mlagentbench")
    if not upstream.is_dir():
        pytest.skip("Pinned local MLAgentBench checkout required for preparation-contract regression")
    root = tmp_path_factory.mktemp("native-acceptance-preparation") / "prepared"
    prepare("vectorization", upstream, root, provenance="actual_data")
    return root


@pytest.fixture
def native_evidence(tmp_path, prepared_vectorization):
    config = create_validation_spec(prepared_vectorization, image="sha256:" + "1" * 64,
                                    level="full", campaign_seconds=7200)
    config.model = "gpt-6-astra"
    config.max_calls, config.token_stop_threshold = 100, 1_000_000
    config.workflow.writer_backend = "paperorchestra"
    config.workflow.paperorchestra_root = str(tmp_path / "fixture-upstream")
    config.workflow.compiler_image = "sha256:" + "2" * 64
    store = Store(tmp_path / "campaign")
    budget = Budget(store, config)
    store.set("live_deadline", time.time() + config.campaign_seconds)
    adapter = load_development(config.native, config.development)
    runner = DevelopmentRunner(store, config, adapter, budget)
    runner.image, runner.hardware = config.native.image, {"fixture": "synthetic; not a runtime probe"}
    store.set("live_images", {"execution": config.native.image, "grading": config.native.grader_image})
    for phase in ("subset", "full"):
        store.set("prepared/" + phase, store.put_json(adapter.prepare(phase, store.root / "development" / phase)))
    identity = adapter.identity()
    store.set("live_spec_ref", store.put_json(config))
    store.set("live_adapter_identity", identity)
    store.set("live_backend", {"backend": "codex", "synthetic": False, "model": config.model})
    _roles_fixture(store, config)
    baseline = adapter.baseline()
    candidate = adapter.control()

    def measured(key, candidate=candidate, phase="full"):
        return execute_paired(runner, key, candidate, phase, run_pair=synthetic_pair)

    measured("baseline/subset", baseline, "subset")
    measured("baseline/full", baseline)
    for index in range(2):
        observed = measured(f"search/0/{index}/subset/0", phase="subset")
        store.cache("live-trial/" + observed.id, {}, {"id": observed.id, "parents": [],
            "result": observed.model_dump(), "decision": {"decision": "Good"}})
    selected = measured("search/1/0/full/0")
    store.cache("live-trial/" + selected.id, {}, {"id": selected.id, "parents": ["search/0/0/subset/0"],
        "result": selected.model_dump(), "decision": {"decision": "Good"}})
    matched, control = measured("method/0/ablation/0/matched"), measured("method/0/ablation/0/control", baseline)
    store.cache("control/method/0/ablation/0/0", {}, {"review": {"decision": "Accept"},
        "matched": matched.model_dump(), "control": control.model_dump(), "source_changes": {"changed": ["train.py"]}})
    bundle = {"manuscript_ref": store.put_json({"synthetic_agents": False}),
        "manifest_ref": store.put_json({"fixture_manifest": "synthetic-development"}),
        "latex_ref": store.put(b"Synthetic test TeX", "tex"),
        "pdf_ref": store.put(b"%PDF-Synthetic receipt graph", "pdf"), "compilation": {"status": "success"}}
    final_bundle = {**bundle, "manifest_ref": store.put_json({"fixture_manifest": "synthetic-final"})}
    store.set("research_manuscript_bundle", store.put_json(bundle))
    store.set("research_final_bundle", store.put_json(final_bundle))
    store.set("live_frozen", {"baseline": baseline.model_dump(), "improved": True,
        "best": {"id": selected.id, "candidate": candidate.model_dump(), "result": selected.model_dump()}})
    store.event("live/freeze", "freeze", "Synthetic fixture freeze")
    finals = {name: execute_paired_final(runner, name, source, run_pair=synthetic_pair)
              for name, source in (("baseline", baseline), ("selected", candidate))}
    store.set("live_final", finals)
    store.set("live_state", {"status": "complete", "stage": "complete", "outcome": "development_improvement"})
    proofs = []
    for row in store.db.execute("SELECT key,output_ref FROM calls WHERE key LIKE 'development/%' OR key LIKE 'final-paired/%'"):
        observed = Observation.model_validate(store.json(row["output_ref"]))
        proofs.append(_metric_proof(store, row["key"].replace("final-paired/", "final/", 1),
            "paired_output_recomputation", row["output_ref"], observed,
            {"predictions": "pass", "paired_timing": "pass", "metric": "pass"}))
    audit = {"status": "pass", "errors": [], "live_model_calls": 0,
        "execution_audit": {"status": "pass", "errors": [], "live_model_calls": 0,
            "verification": {"complete": True, "paired_correctness_recomputations": proofs}},
        "checks": {"manuscripts": {"status": "pass", "records": [
            {"bundle_ref": bundle["manifest_ref"], "status": "pass", "pdf": "pass",
                "authoritative_writer": {"authoritative_source_binding": "pass"}},
            {"bundle_ref": final_bundle["manifest_ref"], "status": "pass", "pdf": "pass",
                "authoritative_writer": {"paperorchestra_development_source_preserved": "pass"}}]}}}
    refs = {"audit_ref": store.put_json(audit)}
    refs["resume_before_ref"] = refs["resume_after_ref"] = store.put_json(capture_native_resume_snapshot(store))
    yield store, refs, audit
    store.close()


def test_complete_synthetic_native_graph_preserves_raw_statistics_and_boundaries(native_evidence):
    store, refs, _ = native_evidence
    before = capture_native_resume_snapshot(store)
    result = verify_native_timing_live(store, **refs)
    assert result["workflow"] == "pass", result
    assert result["task"]["execution_contract"] == "workspace"
    assert result["task"]["upstream_revision"] == UPSTREAM_REVISION
    assert result["task"]["workload_generation"]["final_seed"] != result["checks"]["protected_finals"]["final_timing_seeds"][0]
    assert len(result["checks"]) == 9
    assert result["model_accounting"]["calls"] == len(MANDATORY_ROLES)
    assert result["model_accounting"]["unknown_calls"] == 0
    assert result["selected_development"]["latency_ratio"] == .5
    assert len(result["selected_development"]["per_seed"]) == 3
    sample = result["selected_development"]["per_seed"][0]
    assert sample["baseline_median_seconds"] == .002
    assert sample["candidate_median_seconds"] == .001
    assert sample["candidate_iqr_seconds"] == 0
    assert sample["sample_count"] == 30
    assert len(sample["candidate_measurements_seconds"]) == 10
    assert result["checks"]["protected_finals"]["final_timing_seeds"] == [0]
    assert result["latency_reproducibility"].startswith("unverified")
    assert capture_native_resume_snapshot(store) == before


@pytest.mark.parametrize("change", ["missing_oracle_proof", "duplicate_oracle_proof", "wrong_observation_ref", "unchecked_output", "unknown_usage", "modified_final_score", "wrong_frozen_source", "model_after_final"])
def test_incomplete_native_evidence_never_passes(native_evidence, change):
    store, refs, audit = native_evidence
    if "oracle_proof" in change:
        proofs = audit["execution_audit"]["verification"]["paired_correctness_recomputations"]
        proofs.pop() if change == "missing_oracle_proof" else proofs.append(proofs[0])
    elif change == "wrong_observation_ref":
        proof = audit["execution_audit"]["verification"]["paired_correctness_recomputations"][0]
        content = store.json(proof["evidence_ref"])
        content["observation_ref"] = store.put_json({"fabricated": True})
        proof["evidence_ref"] = store.put_json(content)
    elif change == "unchecked_output":
        row = store.db.execute("SELECT output_ref FROM calls WHERE key='development/baseline/subset'").fetchone()
        observed = store.json(row[0])
        evidence = store.json(observed["timing_refs"][0])
        timing = store.json(evidence["report_ref"])
        timing["samples"][0]["correctness"]["valid"] = False
        # Deliberate immutable-artifact tampering must be detected before acceptance.
        (store.root / evidence["report_ref"]).write_bytes(canonical(timing))
    elif change == "unknown_usage":
        path = next((store.root / "model-ledger/claims").iterdir()) / "dispatch.json"
        receipt = relay.read_json(path, 16384)
        receipt.update(usage_status="unknown", observed_tokens=None)
        path.write_bytes(relay.canonical(receipt))
    elif change == "modified_final_score":
        finals = store.get("live_final")
        finals["selected"]["score"] = .01
        store.set("live_final", finals)
    elif change == "wrong_frozen_source":
        frozen = store.get("live_frozen")
        frozen["best"]["candidate"]["files"][0]["content"] += "\n# changed after freeze"
        store.set("live_frozen", frozen)
    elif change == "model_after_final":
        with store.db:
            store.db.execute("INSERT INTO live_operations VALUES(?,?,?,?,?,?,?)",
                ("model/forbidden/0", "model", "x", "success", .1, 0, store.put_json({})))
    refs["audit_ref"] = store.put_json(audit)
    if change != "unchecked_output":
        refs["resume_before_ref"] = refs["resume_after_ref"] = store.put_json(capture_native_resume_snapshot(store))
    result = verify_native_timing_live(store, **refs)
    assert result["workflow"] == "fail" and result.get("error"), result


def test_native_observation_requires_exact_timing_seeds(native_evidence):
    store, _, _ = native_evidence
    ref = store.db.execute("SELECT output_ref FROM calls WHERE key='development/baseline/full'").fetchone()[0]
    with pytest.raises(IntegrityError, match="phase/seed"):
        _observation(store, store.json(ref), "full", [0])


def test_completed_resume_must_preserve_recorded_work(native_evidence):
    store, refs, _ = native_evidence
    store.event("synthetic-late-event", "complete", "Must invalidate no-op resume proof")
    result = verify_native_timing_live(store, **refs)
    assert result["workflow"] == "fail" and "resume" in result["error"]


def test_actual_validation_spec_workspace_resolves_pinned_benchmark(tmp_path, prepared_vectorization):
    config = create_validation_spec(prepared_vectorization, image="sha256:" + "1" * 64, level="full")
    assert isinstance(config.native, WorkspaceSpec) and config.native.benchmark == "workspace"
    assert not hasattr(config.native, "revision")
    store = Store(tmp_path)
    try:
        store.set("live_adapter_identity", load_development(config.native, config.development).identity())
        record = _prepared_contract(store, config)
        assert record["benchmark"] == "mlagentbench" and record["task_id"] == "vectorization"
        assert record["upstream_revision"] == UPSTREAM_REVISION
        assert record["prepared_manifest_sha256"] == store.get("live_adapter_identity")["prepared_manifest"]
        wrong = store.get("live_adapter_identity")
        wrong["prepared_manifest"] = "0" * 64
        store.set("live_adapter_identity", wrong)
        with pytest.raises(IntegrityError, match="manifest or adapter binding"):
            _prepared_contract(store, config)
    finally:
        store.close()


def test_actual_installed_entrypoint_and_frozen_baseline_identity_are_required(tmp_path, prepared_vectorization):
    """Use installed metadata and loader wrappers, without mocked registration."""
    installed = list(entry_points(group="scientisttwo.native_development", name="testbed"))
    assert len(installed) == 1
    entry = installed[0]
    config = create_validation_spec(prepared_vectorization, image="sha256:" + "1" * 64, level="full")
    identity = load_development(config.native, config.development).identity()
    assert identity["plugin_entry_point"] == entry.name == "testbed"
    assert identity["plugin_entry_value"] == entry.value
    assert identity["plugin_distribution"] == entry.dist.metadata["Name"]
    assert identity["plugin_version"] == entry.dist.version
    assert identity["plugin_source"]
    assert identity["baseline_source/train.py"]
    store = Store(tmp_path)
    try:
        store.set("live_adapter_identity", identity)
        assert _prepared_contract(store, config)["adapter_identity"] == identity
        # A bare implementation omits installed package and frozen-baseline
        # provenance. Subset matching must not silently accept that record.
        store.set("live_adapter_identity", PreparedDevelopment(config.native, config.development).identity())
        with pytest.raises(IntegrityError, match="source closure"):
            _prepared_contract(store, config)
        for field in ("plugin_version", "plugin_source", "baseline_source/train.py"):
            store.set("live_adapter_identity", {**identity, field: "changed"})
            with pytest.raises(IntegrityError, match="source closure"):
                _prepared_contract(store, config)
    finally:
        store.close()
