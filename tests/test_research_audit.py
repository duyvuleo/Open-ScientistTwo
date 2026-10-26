"""Current-engine mechanical audits with explicitly synthetic experiment traces."""

import json
import time
import pytest

from scientisttwo.models import Profile, TaskSpec
from scientisttwo.research.audit import audit_research
from scientisttwo.research.controller import ResearchController
from scientisttwo.research.fixtures import ResearchFixtures
from scientisttwo.research.pilot_bridge import create_pilot_campaign
from scientisttwo.store import Store, digest
from test_research_pilot_bridge import SyntheticContainer


@pytest.fixture
def campaign(tmp_path):
    store = Store(tmp_path)
    worker = SyntheticContainer(store)
    original_execute = worker.execute

    def scripted(key, idea, source, spec):
        worker.dataset_values = {
            d: (
                0.8
                if "estimator = SVC(" in source
                else 0.6
                if "estimator = KNeighbors" in source
                else 0.5
            )
            for d in spec.datasets
        }
        return original_execute(key, idea, source, spec)

    worker.execute = scripted
    config, adapter, runner = create_pilot_campaign(
        store, TaskSpec(), Profile(), container_runner=worker
    )
    backend = ResearchFixtures(adapter.fixture_response)
    state = ResearchController(store, config, adapter, backend, runner, budget=runner.budget).run()
    assert state["status"] == "complete", state
    yield store, config, adapter, runner, backend, worker
    store.close()


def test_complete_synthetic_cycle_has_mechanical_audit_not_scientific_certification(campaign):
    store, _, _, runner, backend, worker = campaign
    before_calls, before_experiments = len(backend.seen), len(worker.executions)
    before_operations = runner.budget.summary()
    report = audit_research(store.root, include_final=True)
    assert report["status"] == "pass", report["errors"]
    for category in (
        "literature_and_novelty",
        "full_stage_handoffs",
        "selection",
        "parent_edits",
        "manuscripts",
    ):
        assert report["checks"][category]["status"] == "pass"
    assert report["checks"]["diagnostics"]["status"] == "not_exercised"
    assert report["execution_audit"]["scoring"] == "unverified"
    assert report["scientific_validity"] == report["novelty_certification"] == "unverified"
    assert report["live_model_calls"] == 0
    assert len(backend.seen) == before_calls and len(worker.executions) == before_experiments
    assert runner.budget.summary() == before_operations


@pytest.mark.parametrize("exhausted", [False, True])
def test_requested_reproduction_cannot_report_pass_when_changed_or_not_run(campaign, monkeypatch, exhausted):
    store, _, _, _, _, worker = campaign
    from scientisttwo.research import pilot_bridge

    original_factory = pilot_bridge.create_pilot_campaign
    monkeypatch.setattr(pilot_bridge, "create_pilot_campaign", lambda *a, **k:
                        original_factory(*a, **k, container_runner=worker))
    worker.execute = SyntheticContainer.execute.__get__(worker)
    worker.dataset_values = {name: .2 for name in ("iris", "wine", "digits")}
    if exhausted:
        store.set("live_deadline", time.time() - 1)
    report = audit_research(store.root, reexecute=True)
    assert report["status"] == ("unverified" if exhausted else "fail")
    assert report["execution_audit"]["reexecution"] == ("unverified" if exhausted else "fail")
    assert report["live_model_calls"] == 0


def test_audit_detects_novelty_validation_rewritten_after_model_response(campaign):
    store, *_ = campaign
    row = store.db.execute(
        "SELECT key,value FROM metadata WHERE key LIKE 'novelty/%' LIMIT 1"
    ).fetchone()
    value = store.json(json.loads(row["value"]))
    value["assessment"]["score"] = 9.9
    store.set(row["key"], store.put_json(value))
    report = audit_research(store.root)
    assert report["status"] == "fail"
    assert report["checks"]["literature_and_novelty"]["status"] == "fail"


def test_audit_detects_selection_replaced_after_review(campaign):
    store, *_ = campaign
    value = store.json(store.get("selection_review"))
    value["selected_id"] = "fabricated-selection"
    store.set("selection_review", store.put_json(value))
    report = audit_research(store.root)
    assert report["status"] == "fail"
    assert report["checks"]["selection"]["status"] == "fail"


def test_audit_detects_modified_rendered_tex_without_running_it(campaign):
    store, *_ = campaign
    wrapper = store.json(store.get("research_manuscript_bundle"))
    from pathlib import Path

    path = Path(wrapper["project_path"]) / "main.tex"
    path.write_text(path.read_text() + "\nInjected alteration\n")
    report = audit_research(store.root)
    assert report["status"] == "fail"
    assert report["checks"]["manuscripts"]["status"] == "fail"


def test_audit_detects_changed_immutable_source_before_replay(campaign):
    store, *_ = campaign
    candidate = store.get("live_best")["result"]["code_ref"]
    (store.root / candidate).write_text("{}")
    report = audit_research(store.root)
    assert report["status"] == "fail"
    assert any("modified" in error for error in report["errors"])


def test_legacy_or_changed_implementation_is_read_only(campaign):
    store, config, *_ = campaign
    old = config.model_copy(update={"engine_version": "legacy-live-v1"})
    store.set("live_spec_ref", store.put_json(old))
    store.db.execute("PRAGMA wal_checkpoint(FULL)")
    before = digest((store.root / "campaign.sqlite").read_bytes())
    report = audit_research(store.root, reexecute=True)
    after = digest((store.root / "campaign.sqlite").read_bytes())
    assert report["status"] == "unverified" and report["requires_pinned_runtime"]
    assert not report["legacy_evidence_modified"] and before == after


def test_common_audit_replays_paired_diagnostic_from_exact_committed_inputs(campaign):
    from scientisttwo.research.diagnostics import (
        DiagnosticInputs,
        DiagnosticRequest,
        run_diagnostic,
    )

    store, config, *_ = campaign
    best = store.get("live_best")["result"]
    references = [
        store.db.execute(
            "SELECT output_ref FROM calls WHERE key=?", ("development/" + key,)
        ).fetchone()[0]
        for key in ("baseline/full", best["id"])
    ]
    request = DiagnosticRequest(
        id="audit-paired",
        kind="paired_differences",
        metric=config.development.metric,
        expected_seeds=config.development.seeds,
    )
    result = run_diagnostic(
        store,
        request,
        DiagnosticInputs(
            observation_refs=references,
            declared_kinds=["metric", "paired_differences", "paired_interval"],
        ),
    )
    store.set("research-diagnostic-record/" + request.id, result.artifact_ref)
    report = audit_research(store.root)
    assert report["status"] == "pass", report["errors"]
    assert report["checks"]["diagnostics"]["status"] == "pass"
    assert report["checks"]["diagnostics"]["records"][0]["execution_status"] == "success"


def test_common_audit_rejects_tampered_handoff_record(campaign):
    store, *_ = campaign
    row = store.db.execute(
        "SELECT key,value FROM metadata WHERE key LIKE 'full-handoff/%' LIMIT 1"
    ).fetchone()
    handoff = store.json(json.loads(row["value"]))
    handoff["reason"] = "Unrecorded replacement reason"
    store.set(row["key"], store.put_json(handoff))
    report = audit_research(store.root)
    assert report["status"] == "fail"
    assert report["checks"]["full_stage_handoffs"]["status"] == "fail"
