"""Synthetic arithmetic/provenance contracts; no models or experiments execute."""

from copy import deepcopy

import pytest

from scientisttwo.research.manuscript import build_evidence_ledger, verify_evidence_ledger
from scientisttwo.research import writer_comparisons as comparisons
from scientisttwo.research.parent_comparison import resolve_parent_comparison
from scientisttwo.live.models import Candidate
from scientisttwo.live.source_tree import candidate_snapshot
from scientisttwo.store import IntegrityError, Store, canonical, digest
from test_research_manuscript import observation
from test_research_workflow import campaign


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "evidence")
    yield value
    value.close()


def inputs(store):
    subset = observation(store, "baseline/subset", (.5,), phase="subset", seeds=(0,))
    full = observation(store, "baseline/full", (.4, .5, .6))
    seed = observation(store, "seed/subset", (.7,), phase="subset", seeds=(0,))
    grown = observation(store, "candidate/full", (.6, .7, .8))
    matched = observation(store, "matched", (.7, .8, .9))
    control = observation(store, "control", (.4, .5, .6))
    store.cache("control/synthetic", {"fixture": True}, {
        "matched": matched.model_dump(), "control": control.model_dump(),
        "plan": {"component": "synthetic component", "known_confounds": ["Synthetic capacity confound"]},
        "review": {"decision": "DiagnosticOnly", "reason": "Synthetic diagnostic", "confounds": []}})
    references = {"subset": subset, "full": full}
    offered = comparisons.include_phase_baselines([seed, grown, matched, control, full], references)
    return references, build_evidence_ledger(store, offered)


def indexed(ledger):
    return {identifier: (entry["observation_id"], entry["metric"])
            for identifier, entry in ledger["entries"].items() if entry["kind"] == "measured"}


def arithmetic(ledger):
    names = indexed(ledger)
    return {(names[entry["input_ids"][0]], names[entry["input_ids"][1]]): entry
            for entry in ledger["entries"].values() if entry["kind"] == "derived"}


def test_registers_both_signs_from_exact_baselines_and_matched_controls(store):
    references, ledger = inputs(store)
    original = deepcopy(ledger)
    before = [dict(row) for row in store.db.execute("SELECT * FROM calls")]
    updated, proof = comparisons.register_writer_comparisons(store, ledger, references)
    values = arithmetic(updated)
    assert len(values) == 10
    assert values[(('seed/subset', 'accuracy'), ('baseline/subset', 'accuracy'))]["value"] == pytest.approx(.2)
    assert values[(('baseline/subset', 'accuracy'), ('seed/subset', 'accuracy'))]["value"] == pytest.approx(-.2)
    assert values[(('matched', 'accuracy'), ('control', 'accuracy'))]["value"] == pytest.approx(.3)
    assert values[(('control', 'accuracy'), ('matched', 'accuracy'))]["value"] == pytest.approx(-.3)
    assert (('candidate/full', 'accuracy'), ('control', 'accuracy')) not in values  # Not all-pairs.
    assert all(entry["operation"] == "difference" for entry in values.values())
    assert all("no statistical significance or causal claim" in entry["interpretation"] for entry in values.values())
    assert updated["control_context"] == ledger["control_context"]
    assert updated["control_context"][0]["attribution_restriction"]["isolated_component"] == "prohibited"
    assert proof["baseline_observation_ids"] == {"subset": "baseline/subset", "full": "baseline/full"}
    assert len(proof["entries"]) == 10
    verify_evidence_ledger(store, updated)
    assert ledger == original and [dict(row) for row in store.db.execute("SELECT * FROM calls")] == before
    repeated, _ = comparisons.register_writer_comparisons(store, updated, references)
    assert repeated == updated


def test_mismatched_protocol_seeds_final_and_metadata_never_create_comparisons(store):
    reference = observation(store, "base", (.5,), phase="subset", seeds=(0,))
    different_phase = observation(store, "different-phase", (.6,), phase="full", seeds=(0,))
    different_seed = observation(store, "different-seed", (.7,), phase="subset", seeds=(1,))
    protected = observation(store, "protected", (.8,), phase="final", seeds=(0,))
    ledger = build_evidence_ledger(store, [reference, different_phase, different_seed, protected],
        method_facts={"metadata": {"text": "Synthetic metadata says 100", "value": 100,
                                   "evidence_ref": store.put_json({"metadata_only": True})}})
    updated, proof = comparisons.register_writer_comparisons(store, ledger, {"subset": reference})
    assert not proof["entries"] and not arithmetic(updated)
    assert "protected" not in updated["observations"]
    final = build_evidence_ledger(store, [protected], include_final=True, method_frozen=True)
    with pytest.raises(IntegrityError, match="exclude protected final"):
        comparisons.register_writer_comparisons(store, final, {})


def test_multi_metric_differences_keep_metrics_separate(store):
    baseline = observation(store, "baseline")
    candidate = observation(store, "candidate", (.7, .8, .9))
    for value, losses in ((baseline, [3., 2., 1.]), (candidate, [2., 1., .5])):
        value.metrics = {"accuracy": value.value, "loss": sum(losses) / 3}
        value.metric_values = {"accuracy": value.values, "loss": losses}
        with store.db:
            store.db.execute("UPDATE calls SET output_ref=? WHERE key=?",
                             (store.put_json(value), "development/" + value.id))
    ledger = build_evidence_ledger(store, [baseline, candidate])
    updated, _ = comparisons.register_writer_comparisons(store, ledger, {"full": baseline})
    values = arithmetic(updated)
    assert len(values) == 4
    assert values[(('candidate', 'accuracy'), ('baseline', 'accuracy'))]["value"] == pytest.approx(.2)
    assert values[(('candidate', 'loss'), ('baseline', 'loss'))]["value"] == pytest.approx(-5 / 6)
    assert all(left[1] == right[1] for left, right in values)


def test_uncommitted_or_conflicting_baselines_fail_without_changing_ledger(store):
    references, ledger = inputs(store)
    wrong = references["subset"].model_copy(update={"value": .1})
    before = canonical(ledger)
    with pytest.raises(IntegrityError, match="committed ledger observation"):
        comparisons.register_writer_comparisons(store, ledger, {"subset": wrong})
    with pytest.raises(IntegrityError, match="conflicts with its declared baseline"):
        comparisons.include_phase_baselines([references["subset"]], {"subset": wrong})
    with pytest.raises(IntegrityError, match="declared subset/full"):
        comparisons.include_phase_baselines([], {"final": references["full"]})
    assert canonical(ledger) == before


def test_limit_exhaustion_is_explicit_and_does_not_partially_mutate_ledger(store, monkeypatch):
    references, ledger = inputs(store)
    before = canonical(ledger)
    monkeypatch.setattr(comparisons, "MAX_DERIVED_COMPARISONS", 1)
    with pytest.raises(IntegrityError, match="derivation limit"):
        comparisons.register_writer_comparisons(store, ledger, references)
    assert canonical(ledger) == before


def test_modified_derived_number_fails_trusted_rederivation(store):
    references, ledger = inputs(store)
    updated, proof = comparisons.register_writer_comparisons(store, ledger, references)
    updated["entries"][proof["entries"][0]["evidence_id"]]["value"] += .1
    with pytest.raises(IntegrityError, match="differs from committed measurements or derivations"):
        verify_evidence_ledger(store, updated)


def parent_comparison_ledger(store, change=None):
    def bound(name, values, phase, seeds):
        value = observation(store, name, values, phase=phase, seeds=seeds)
        candidate = Candidate.model_validate(store.json(value.code_ref))
        value.source_tree_ref = store.put_json(candidate_snapshot(store, candidate))
        with store.db:
            store.db.execute("UPDATE calls SET output_ref=? WHERE key=?",
                             (store.put_json(value), "development/" + name))
        return value

    parent = bound("parent/full", (.82, .82, .82), "full", (0, 1, 2))
    subset_parent = bound("parent/subset", (.8166574048346764,), "subset", (0,))
    child = bound("child/subset", (.8,), "subset", (0,))
    context, reference = resolve_parent_comparison(store, {"id": parent.id, "result": parent.model_dump()},
        parent.source_tree_ref, child, lambda *args: {"valid": False, "delta": 999, "reason": "Synthetic failed parent guardrail"})
    assert reference.id == subset_parent.id
    if change == "wrong_reference":
        context["reference"]["id"] = parent.id
        context["reference"]["observation_ref"] = store.put_json(parent)
    request = {"role": "critic", "inputs": {"parent_comparison": context}}
    key = "child/subset/critic"
    store.set("request/" + key, store.put_json(request))
    store.cache("live-agent/" + key, request, {"fixture": "Synthetic critic completion"})
    if change == "request_hash":
        with store.db:
            store.db.execute("UPDATE calls SET input_hash=? WHERE key=?",
                             (digest(b"different request"), "live-agent/" + key))
    return build_evidence_ledger(store, [parent, subset_parent, child])


def test_parent_relative_drop_uses_the_verified_subset_reference_not_full_parent_or_claimed_delta(store):
    ledger = parent_comparison_ledger(store)
    updated, proof = comparisons.register_writer_comparisons(store, ledger, {})
    values = arithmetic(updated)
    assert len(values) == 2
    forward = values[(('parent/subset', 'accuracy'), ('child/subset', 'accuracy'))]
    assert forward["value"] == pytest.approx(.01665740483467637)
    assert values[(('child/subset', 'accuracy'), ('parent/subset', 'accuracy'))]["value"] == -forward["value"]
    assert not any('parent/full' in (left[0], right[0]) for left, right in values)
    assert proof["entries"][0]["pair_sources"] == ["verified_parent_comparison"]
    assert len(proof["parent_comparison_request_refs"]) == 1
    verify_evidence_ledger(store, updated)


@pytest.mark.parametrize("change", ["wrong_reference", "request_hash"])
def test_modified_parent_reference_or_request_cannot_register_arithmetic(store, change):
    ledger = parent_comparison_ledger(store, change)
    before = canonical(ledger)
    with pytest.raises(IntegrityError, match="parent comparison|Parent comparison"):
        comparisons.register_writer_comparisons(store, ledger, {})
    assert canonical(ledger) == before


def test_complete_scripted_workflow_freezes_registered_arithmetic_before_writing(tmp_path):
    controller, evidence, _, agents, _ = campaign(tmp_path)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        requests = [inputs for role, inputs in agents.seen if role == "initial_drafter"]
        assert requests
        for request in requests:
            ledger = request["ledger"]
            assert controller.baselines["subset"].id in ledger["observations"]
            assert any(entry["kind"] == "derived" and entry["operation"] == "difference"
                       for entry in ledger["entries"].values())
            assert request["writer_comparison_contract"]["schema_version"] == comparisons.VERSION
            verify_evidence_ledger(evidence, ledger)
        frozen = evidence.json(evidence.get("research_manuscript_bundle"))
        verify_evidence_ledger(evidence, evidence.json(frozen["ledger_ref"]))
    finally:
        evidence.close()
