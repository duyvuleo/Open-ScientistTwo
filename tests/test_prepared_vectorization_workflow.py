"""Complete prepared workflow with explicitly synthetic timing and role responses.

Only task preparation and controller/writer code are real in this test. No model,
container, convolution workload or readiness validation is executed. Its purpose
is to expose context/claim/final-appendix contract failures before a costly run.
"""

import json

import pytest

from upstream_checkouts import require_upstream

from scientisttwo.live.runtime import Budget, DevelopmentRunner, validate_prepared
from scientisttwo.models import AgentResult
from scientisttwo.research.audit import _ReplayStore, _manuscripts
from scientisttwo.research.controller import ResearchController
from scientisttwo.research.fixtures import ResearchFixtures
from scientisttwo.research.manuscript import verify_evidence_ledger
from scientisttwo.research.timing_evidence import verify_final_result_binding
from scientisttwo.store import Store, canonical
from scientisttwo.testbed.adapter import PreparedDevelopment
from scientisttwo.testbed.preparation import prepare
from scientisttwo.testbed.validation import ScriptedResearch, create_validation_spec
from test_paired_evaluation import synthetic_pair


def test_full_prepared_vectorization_fixture_workflow(tmp_path, monkeypatch, record_property):
    upstream = require_upstream("mlagentbench")
    if not upstream.is_dir():
        pytest.skip("Pinned local MLAgentBench checkout required for preparation-contract test")
    prepared_root = tmp_path / "prepared"
    prepare("vectorization", upstream, prepared_root, provenance="actual_data")
    config = create_validation_spec(prepared_root, image="sha256:" + "1" * 64,
        max_prompt_chars=600_000, level="full", campaign_seconds=600)
    assert config.native.benchmark == "workspace"
    assert config.max_prompt_chars == 600_000
    adapter = PreparedDevelopment(config.native, config.development)
    store = Store(tmp_path / "campaign")
    budget = Budget(store, config)

    def forbidden_process(*args, **kwargs):
        pytest.fail("Synthetic workflow must never launch a container, model or candidate process")

    runner = DevelopmentRunner(store, config, adapter, budget, executor=forbidden_process)

    def synthetic_preflight():
        runner.image = config.native.image
        runner.hardware = {"status": "unverified", "fixture": "synthetic; no runtime/device probe"}
        store.set("live_images", {"execution": config.native.image, "grading": config.native.grader_image})
        for phase in ("subset", "full"):
            destination = store.root / "development" / phase
            value = validate_prepared(adapter.prepare(phase, destination), destination)
            store.set("prepared/" + phase, store.put_json(value, "synthetic-workflow-prepared-partition"))

    monkeypatch.setattr(runner, "preflight", synthetic_preflight)
    monkeypatch.setattr(adapter, "_pair_runner", lambda: synthetic_pair)
    scripted = ScriptedResearch(adapter)
    backend = ResearchFixtures(scripted.respond, store=store, materialize=getattr(adapter, "materialize", None))
    controller = ResearchController(store, config, adapter, backend, runner, budget=budget)
    try:
        state = controller.run()
        contexts = {}
        for row in store.db.execute("SELECT key,value FROM metadata WHERE key LIKE 'context/%'"):
            proof = store.json(json.loads(row["value"]))
            request = store.json(proof["presented_request_ref"])
            role = request["role"]
            contexts[role] = max(contexts.get(role, 0), proof["presented_input_characters"])
        # Always print bounded aggregate diagnostics, including when the workflow
        # stopped before completion. These are exact input characters, not tokens.
        print("SYNTHETIC_WORKFLOW_CONTEXT_MAXIMA=" + json.dumps(contexts, sort_keys=True))
        record_property("synthetic_context_maxima", json.dumps(contexts, sort_keys=True))
        assert state["status"] == "complete" and state["outcome"] == "development_improvement", state
        assert max(contexts.values()) <= 600_000
        assert {
            "limitations", "verify_limitations", "initial_idea", "additional_idea", "novelty_checker",
            "baseline_plan", "implement", "critic", "full_set_coder", "selection_review", "evolve",
            "ablation_planner", "ablation_coder", "control_reviewer", "ablation_critic",
            "initial_drafter", "writing_critic", "draft_enhancer", "scientific_review",
            "rebuttal_planner", "rebuttal_coder", "meta", "method_disposition", "refine",
        } <= set(contexts)
        roles = [AgentResult.model_validate(store.json(row["output_ref"])) for row in store.db.execute(
            "SELECT output_ref FROM calls WHERE key LIKE 'live-agent/%'")]
        assert roles and all(result.synthetic for result in roles)
        ledgers = [store.json(json.loads(row["value"])) for row in store.db.execute(
            "SELECT value FROM metadata WHERE key LIKE 'manuscript-ledger/%'")]
        assert ledgers
        for ledger in ledgers:
            verify_evidence_ledger(store, ledger)
            assert any(entry.get("metric", "").startswith("paired.") for entry in ledger["entries"].values())
        for row in store.db.execute("SELECT output_ref FROM calls WHERE key LIKE 'development/%' OR key LIKE 'final-paired/%'"):
            observed = store.json(row["output_ref"])
            for reference in observed["timing_refs"]:
                timing = store.json(store.json(reference)["report_ref"])
                assert timing["warmup_rounds"] == 5 and timing["measurement_pairs"] == 10
                assert len(timing["samples"]) == 30
                for arm in ("baseline", "candidate"):
                    assert {"median_seconds", "iqr_seconds", "q1_seconds", "q3_seconds"} <= timing[arm].keys()
                    assert len(timing[arm]["seconds"]) == 10
        bundle = store.json(store.get("research_manuscript_bundle"))
        manuscript = store.json(bundle["manuscript_ref"])
        assert manuscript["synthetic_agents"] is True and len(manuscript["claims"]) <= 128
        final_bundle = store.json(store.get("research_final_bundle"))
        assert final_bundle["latex_ref"] != bundle["latex_ref"]
        finals = store.get("live_final")
        assert set(finals) == {"baseline", "selected"}
        for label, result in finals.items():
            assert verify_final_result_binding(store, label, result)["kind"] == "paired_observation_summary"
        replay = _ReplayStore(tmp_path / "manuscript-replay", store)
        try:
            records = _manuscripts(store, replay, config)
            assert records and all(item["status"] == "pass" for item in records)
            assert any(item["final_appendix"] for item in records)
        finally:
            replay.close()
        before = canonical(budget.summary())
        call_count = len(backend.seen)
        resumed = controller.run()
        assert resumed == state and canonical(budget.summary()) == before
        assert len(backend.seen) == call_count
        assert store.verify() == []
    finally:
        store.close()
