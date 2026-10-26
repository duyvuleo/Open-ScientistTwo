"""Explicitly synthetic controls test attribution routing, not scientific validity."""

from copy import deepcopy

import pytest

from scientisttwo.live.evidence_context import committed_control_context
from scientisttwo.research.manuscript import (
    LedgerClaim, build_evidence_ledger, derive_comparison, fixture_manuscript,
    render_manuscript, validate_manuscript, verify_evidence_ledger,
)
from scientisttwo.store import IntegrityError, Store
from test_research_manuscript import observation
from test_research_workflow import campaign


@pytest.fixture
def control_store(tmp_path):
    store = Store(tmp_path / "evidence")
    yield store
    store.close()


def controlled_ledger(store, *, decision="Accept", plan_confounds=(), review_confounds=()):
    matched = observation(store, "matched")
    control = observation(store, "control", values=(.4, .5, .6))
    record = {
        "matched": matched.model_dump(), "control": control.model_dump(),
        "plan": {"id": "shortcut", "component": "shortcut", "question": "Remove the shortcut",
                 "changed_files": ["train.py"], "invariants": ["Preserve initialization"],
                 "known_confounds": list(plan_confounds)},
        "review": {"decision": decision, "reason": "Synthetic design judgment",
                   "confounds": list(review_confounds)},
    }
    store.cache("control/synthetic", {"synthetic": True}, record)
    ledger = build_evidence_ledger(store, [matched, control])
    keys = {item["observation_id"]: key for key, item in ledger["entries"].items()}
    difference = derive_comparison(ledger, keys["control"], keys["matched"])
    manuscript = fixture_manuscript(ledger)
    manuscript.claims.append(LedgerClaim(
        id="component-result", kind="derived", text="The component comparison is recorded.",
        evidence_ids=[difference], attribution_scope="isolated_component",
    ))
    manuscript.ablations.claim_ids.append("component-result")
    return ledger, manuscript, record


@pytest.mark.parametrize("decision,plan_confounds,review_confounds", [
    ("DiagnosticOnly", [], []), ("Reject", [], []),
    ("Accept", ["Changed active capacity"], []), ("Accept", [], ["Changed initialization"]),
])
def test_declared_isolated_claim_cannot_use_confounded_controls(
    control_store, decision, plan_confounds, review_confounds
):
    ledger, manuscript, record = controlled_ledger(
        control_store, decision=decision, plan_confounds=plan_confounds,
        review_confounds=review_confounds,
    )
    context = ledger["control_context"][0]
    assert context["plan"] == record["plan"]
    assert context["review"] == record["review"]
    assert context["known_confounds"] == plan_confounds + review_confounds
    verify_evidence_ledger(control_store, ledger)
    result = validate_manuscript(manuscript, ledger)
    assert result.status == "fail" and result.checks["declared_attribution_scope"] == "fail"
    manuscript.claims[-1].attribution_scope = "package_comparison"
    assert validate_manuscript(manuscript, ledger).status == "pass"
    rendered = render_manuscript(control_store, manuscript, ledger)
    text = control_store.read(rendered.markdown_ref).decode()
    assert "Control interpretation limits" in text
    assert "Isolated-component attribution: prohibited" in text
    assert all(value in text for value in plan_confounds + review_confounds)
    assert context["control_record_ref"] in control_store.json(rendered.manifest_ref)["files"]


def test_clean_review_allows_declared_scope_without_certifying_causality(control_store):
    ledger, manuscript, _ = controlled_ledger(control_store)
    result = validate_manuscript(manuscript, ledger)
    assert result.status == "pass"
    assert result.checks["scientific_validity"] == "unverified"
    assert ledger["control_context"][0]["attribution_restriction"]["isolated_component"] == "eligible_for_review"
    manuscript.claims[-1].evidence_ids = [next(iter(ledger["entries"]))]
    result = validate_manuscript(manuscript, ledger)
    assert any("both arms" in error for error in result.errors)


def test_control_context_is_rebuilt_and_unknown_observations_are_rejected(control_store):
    ledger, _, _ = controlled_ledger(control_store, decision="DiagnosticOnly")
    altered = deepcopy(ledger)
    altered["control_context"][0]["review"]["decision"] = "Accept"
    with pytest.raises(IntegrityError, match="ledger differs"):
        verify_evidence_ledger(control_store, altered)
    assert committed_control_context(control_store, [{"id": "matched"}]) == []
    with control_store.db:
        control_store.db.execute("DELETE FROM calls WHERE key='development/control'")
    with pytest.raises(IntegrityError, match="uncommitted observation"):
        committed_control_context(control_store, [{"id": "matched"}, {"id": "control"}])


def test_diagnostic_only_confounds_reach_every_downstream_role(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path)
    original = agents.responder
    marker = "Synthetic known active capacity confound"

    def respond(role, inputs):
        value = original(role, inputs)
        if role == "control_reviewer":
            value["confounds"] = [marker]
        return value

    agents.responder = respond
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        for role in ("ablation_critic", "initial_drafter", "draft_enhancer", "scientific_review", "meta"):
            presented = [inputs for actual, inputs in agents.seen if actual == role]
            assert presented and all(marker in str(inputs) for inputs in presented), role
        rendered = store.json(store.get("research_manuscript_bundle"))
        ledger = store.json(rendered["ledger_ref"])
        verify_evidence_ledger(store, ledger)
        assert ledger["control_context"][0]["review"]["decision"] == "DiagnosticOnly"
    finally:
        store.close()
