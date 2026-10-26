"""Synthetic saved manuscripts; no candidate execution or model calls."""

import json
from types import SimpleNamespace

import pytest

from scientisttwo.live.reporting import write_report
from scientisttwo.live.runtime import Budget
from scientisttwo.store import Store


@pytest.fixture
def reporting_store(tmp_path):
    store = Store(tmp_path / "campaign")
    config = SimpleNamespace(
        mode="research", model="scripted-fixture-v1",
        native=SimpleNamespace(data_provenance="contract_fixture"),
    )
    Budget(store, config)
    store.set("live_state", {
        "stage": "complete", "status": "complete", "outcome": "manuscript_revision_limit",
        "reason": "Synthetic editorial revision allowance exhausted; concerns remain unresolved",
    })
    yield store, config
    store.close()


def draft(title, limitation, explanation):
    return {
        "title": title, "claims": [], "limitations": [limitation], "source_ids": ["task"],
        "concern_responses": [{
            "concern_id": "concern_1", "status": "unaddressed", "experiment_ids": [],
            "explanation": explanation,
        }],
    }


def render(store, config):
    write_report(store, config, synthetic=True)
    return (
        (store.root / "report.md").read_text(),
        (store.root / "manuscript.tex").read_text(),
        json.loads((store.root / "summary.json").read_text()),
    )


def test_revised_manuscript_prose_replaces_old_text_in_both_outputs(reporting_store):
    store, config = reporting_store
    old_ref = store.put_json(draft("Original title", "OBSOLETE LIMITATION", "OBSOLETE RESPONSE"))
    store.set("live_draft", old_ref)
    render(store, config)
    revised_ref = store.put_json(draft(
        "Revised title", "Paired differences do not establish shortcut superiority.",
        "Selection bias remains unresolved; repeated seeds are not independent replications.",
    ))
    store.set("live_draft", revised_ref)
    markdown, latex, _ = render(store, config)
    for output in (markdown, latex):
        assert "Revised title" in output
        assert "Paired differences do not establish shortcut superiority." in output
        assert "Selection bias remains unresolved; repeated seeds are not independent replications." in output
        assert "Original title" not in output
        assert "OBSOLETE LIMITATION" not in output and "OBSOLETE RESPONSE" not in output
        assert "Model-proposed limitations (unverified prose)" in output
        assert "Model-proposed concern responses (unverified prose)" in output
    assert store.json(old_ref)["title"] == "Original title"


def test_latex_escapes_limitation_and_concern_text(reporting_store):
    store, config = reporting_store
    store.set("live_draft", store.put_json(draft(
        "Comparison A&B", "A&B: 5% of x_y costs $2 #1 {group} ~ ^.",
        r"Do not execute \input{outside}; review evidence_2.",
    )))
    markdown, latex, _ = render(store, config)
    assert "A&B: 5% of x_y costs $2 #1 {group} ~ ^." in markdown
    assert r"A\&B: 5\% of x\_y costs \$2 \#1 \{group\} \textasciitilde{} \textasciicircum{}." in latex
    assert r"Do not execute \textbackslash{}input\{outside\}; review evidence\_2." in latex
    assert r"\input{outside}" not in latex
    assert r"concern\_1: unaddressed; experiments: none." in latex


@pytest.mark.parametrize("decision,status", [
    ("ReviseManuscript", "unresolved"), ("ReviseMethod", "unresolved"),
    ("Refine", "unresolved"), ("Accept", "accepted"),
])
def test_meta_status_is_explicit_and_revision_records_are_preserved(reporting_store, decision, status):
    store, config = reporting_store
    previous_ref = store.put_json(draft("Prior", "Prior limitation", "Prior response"))
    revised_ref = store.put_json(draft("Current", "Current limitation", "Current response"))
    store.set("live_draft", revised_ref)
    review = {"decision": decision, "reason": "Synthetic internal meta-review"}
    meta_ref = store.put_json(review)
    store.set("live_meta", meta_ref)
    record = {
        "status": "reviewed", "previous_draft_ref": previous_ref, "revised_draft_ref": revised_ref,
        "selected_source_refs": ["synthetic-source-ref"], "feedback_ref": "synthetic-feedback-ref",
        "review_ref": "synthetic-review-ref", "meta_ref": meta_ref,
        "no_new_experiments": True, "editorial_adequacy": "unverified",
    }
    record_ref = store.put_json(record)
    store.set("manuscript-revision/method/0/editorial/0", record_ref)
    markdown, latex, summary = render(store, config)
    assert summary["manuscript_revisions"] == [{
        "key": "method/0/editorial/0", "evidence_ref": record_ref, "record": record,
    }]
    assert summary["meta_review"] == {
        "status": status, "decision": decision, "evidence_ref": meta_ref, "review": review,
        "scope": "Recorded internal meta-review only; not conference acceptance or verified scientific validity",
    }
    for output in (markdown, latex):
        assert f"Internal meta-review: {status}; recorded decision: {decision}." in output
        assert "revision-limit exhaustion do not imply review acceptance" in output
        assert "editorial adequacy remains unverified" in output
    if decision != "Accept":
        assert "Internal meta-review: accepted" not in markdown
        assert "Internal meta-review: accepted" not in latex
    assert summary["experiment_attempts"] == 0


def test_missing_meta_review_is_unverified_despite_complete_state(reporting_store):
    store, config = reporting_store
    markdown, latex, summary = render(store, config)
    assert summary["meta_review"]["status"] == "unverified"
    assert summary["meta_review"]["decision"] is None
    assert summary["manuscript_revisions"] == []
    for output in (markdown, latex):
        assert "Internal meta-review: unverified; recorded decision: none." in output
