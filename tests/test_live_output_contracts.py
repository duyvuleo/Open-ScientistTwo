"""Purely synthetic model-output and command-contract regressions.

All trial IDs, hypotheses, observations, responses and usage are invented test
inputs. No historical model output or experiment record is included or executed.
"""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scientisttwo.backends import BackendError, FixtureBackend
from scientisttwo.live.controller import AgentOutputError, LiveController
from scientisttwo.live.models import Hypothesis, LiveSpec, Observation, Revision, Seeds
from scientisttwo.live.runtime import AmbiguousCall, Budget, LiveBudgetExhausted
from scientisttwo.models import CriticDecision
from scientisttwo.store import IntegrityError, Store
from test_live import Adapter, Agents, Runner, candidate, idea, setup


def synthetic_contract():
    """Minimal invented cases for namespace validation and command enforcement."""
    attempts = [
        {'id': 'synthetic/0/subset', 'phase': 'subset', 'status': 'success'},
        {'id': 'synthetic/0/full', 'phase': 'full', 'status': 'success'},
        {'id': 'synthetic/1/invalid', 'phase': 'subset', 'status': 'invalid'},
        {'id': 'synthetic/1/repaired', 'phase': 'subset', 'status': 'success'},
        {'id': 'synthetic/1/full', 'phase': 'full', 'status': 'success'},
    ]
    return {
        'fixture_kind': 'synthetic_unit_test',
        'allowed_source_ids': ['task'],
        'attempt_projection': attempts,
        'evolution_output': {
            'title': 'Synthetic mixed-namespace proposal',
            'hypothesis': 'Trial IDs belong to evidence references, not literature references.',
            'novelty_score': 0,
            'novelty_scope': 'Invented contract-test inputs; no scientific claim.',
            'source_ids': ['task', attempts[1]['id'], attempts[2]['id']],
        },
        'expected_candidate_command': ['python', 'train.py'],
        'invalid_candidate_command': ['python', 'train.py', '--synthetic-extra', '1'],
    }


class Responses(FixtureBackend):
    """A deterministic fake model with nonzero synthetic usage and event evidence."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.seen = []
        self.events = []
        super().__init__(self.respond)

    def respond(self, role, inputs):
        self.seen.append((role, deepcopy(inputs)))
        self.events = [{"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20}}]
        payload = self.payloads.pop(0)
        if isinstance(payload, BaseException):
            raise payload
        return deepcopy(payload)

    def invoke(self, role, inputs, schema):
        response = super().invoke(role, inputs, schema)
        return response.model_copy(update={"usage": {"input_tokens": 100, "output_tokens": 20}})


def commit_trial(controller, key, *, phase="full", status="success"):
    """Commit synthetic trial evidence; never execute candidate code."""
    controller.store.set("live_state", {"stage": "search", "status": "running", "reason": "Synthetic contract test"})
    proposal = candidate(8)
    observation = Observation(
        id=key, phase=phase, status=status, protocol_hash="synthetic-contract-protocol",
        code_ref=controller.store.put_json(proposal), metric="process_seconds",
        value=8.0 if status == "success" else None,
        values=[8.0] if status == "success" else [],
        prediction_refs=[controller.store.put_json({"synthetic": True, "purpose": "Unexecuted contract fixture"})]
        if status == "success" else [],
        reason="Synthetic test observation; not measured",
    )
    return controller.record(key, idea("Synthetic evidence"), proposal, observation,
                             CriticDecision(decision="Good" if status == "success" else "Engineer",
                                            reason="Synthetic test decision"))


def inputs_for(controller, attempts):
    return {"attempts": deepcopy(attempts), "sources": deepcopy(controller.sources),
            "contract": controller.adapter.context()}


def corrected_fixture_payload():
    fixture = synthetic_contract()
    payload = deepcopy(fixture["evolution_output"])
    payload["source_ids"] = fixture["allowed_source_ids"]
    payload["evidence_ids"] = [row["id"] for row in fixture["attempt_projection"]]
    return payload


def model_rows(controller):
    return [row for row in controller.budget.summary() if row["kind"] == "model"]


def reopen(controller, store, backend):
    config, root = controller.config, store.root
    store.close()
    restored = Store(root)
    budget = Budget(restored, config)
    runner = Runner(restored, budget)
    return LiveController(restored, config, Adapter(), backend, runner, budget=budget), restored


def test_synthetic_attempt_ids_are_allowed_only_in_evidence_namespace(tmp_path):
    payload = corrected_fixture_payload()
    backend = Responses([payload])
    controller, store, _, _ = setup(tmp_path, backend, agent_output_retries=0)
    trials = [commit_trial(controller, row["id"], phase=row["phase"], status=row["status"])
              for row in synthetic_contract()["attempt_projection"]]
    result = controller.ask("synthetic-evolve", "evolve", inputs_for(controller, trials), Hypothesis)
    assert result.evidence_ids == payload["evidence_ids"]
    assert result.source_ids == ["task"]
    reference = backend.seen[0][1]["reference_contract"]
    assert set(reference["source_ids"]) == {"task"}
    assert set(reference["evidence_ids"]) == set(payload["evidence_ids"])
    assert any(row["status"] == "invalid" for row in synthetic_contract()["attempt_projection"])
    assert sum(row["tokens"] for row in model_rows(controller)) == 120
    store.close()


def test_synthetic_command_failure_exposes_the_same_authoritative_argv_it_enforces(tmp_path):
    from scientisttwo.live.source_tree import freeze_source_tree
    from scientisttwo.testbed.adapter import PreparedDevelopment
    from scientisttwo.testbed.preparation import prepare
    from scientisttwo.testbed.validation import create_validation_spec
    from test_testbed_families import family_bundle

    prepared = tmp_path / "prepared"
    prepare("cifar10", family_bundle(tmp_path / "synthetic-data"), prepared,
            provenance="synthetic_fixture", source_format="bundle")
    config = create_validation_spec(prepared, image="synthetic:torch", gpus="device=0")
    adapter = PreparedDevelopment(config.native, config.development)
    store = Store(tmp_path / "source-store")
    descriptor = adapter.source_tree()
    adapter.bind_source(freeze_source_tree(store, Path(descriptor["path"]), descriptor["editable_paths"]))
    context = adapter.context()
    fixture = synthetic_contract()
    assert context["command"] == adapter.family.command("train") == fixture["expected_candidate_command"]
    bad = adapter.baseline().model_copy(update={"command": fixture["invalid_candidate_command"]})
    with pytest.raises(ValueError) as error:
        adapter.validate(bad)
    assert "train.py" in str(error.value)
    repaired = bad.model_copy(update={"command": context["command"]})
    adapter.validate(repaired)
    assert str(config.native.private_dir) not in json.dumps(context)
    store.close()


@pytest.mark.parametrize("violation", [
    "wrong_namespace", "unknown", "future", "hidden", "text_only", "final",
    "mislabeled_final", "document_as_evidence",
])
def test_reference_contract_rejects_unavailable_or_wrong_namespace_ids(tmp_path, violation):
    payload = idea("Synthetic reference test").model_dump()
    payload["evidence_ids"] = []
    backend = Responses([payload])
    controller, store, _, _ = setup(tmp_path, backend, agent_output_retries=0)
    trial = commit_trial(controller, "known-invalid", status="invalid")
    attempts = [trial]
    if violation == "wrong_namespace":
        payload["source_ids"] = ["task", trial["id"]]
    elif violation == "unknown":
        payload["evidence_ids"] = ["not-an-observation"]
    elif violation == "future":
        future = deepcopy(trial)
        future["id"] = future["result"]["id"] = "not-yet-committed"
        attempts.append(future)
        payload["evidence_ids"] = [future["id"]]
    elif violation == "hidden":
        commit_trial(controller, "committed-but-not-presented")
        payload["evidence_ids"] = ["committed-but-not-presented"]
    elif violation == "text_only":
        commit_trial(controller, "mentioned-in-prose-only")
        payload["evidence_ids"] = ["mentioned-in-prose-only"]
    elif violation in {"final", "mislabeled_final"}:
        final = commit_trial(controller, "protected-final", phase="final")
        if violation == "mislabeled_final":
            final = deepcopy(final)
            final["result"]["phase"] = "full"
        attempts.append(final)
        payload["evidence_ids"] = [final["id"]]
    else:
        payload["evidence_ids"] = ["task"]
    backend.payloads[0] = payload
    inputs = inputs_for(controller, attempts)
    if violation == "text_only":
        inputs["note"] = "A prose mention of mentioned-in-prose-only supplies no observation."
    with pytest.raises(AgentOutputError):
        controller.ask("bad-reference", "evolve", inputs, Hypothesis)
    assert len(backend.seen) == 1
    assert len(model_rows(controller)) == 1 and model_rows(controller)[0]["tokens"] == 120
    store.close()


def test_seed_ideas_cannot_cite_experiment_history(tmp_path):
    payload = idea("Synthetic seed").model_dump()
    payload["evidence_ids"] = ["prior-trial"]
    backend = Responses([{"ideas": [payload]}])
    controller, store, _, _ = setup(tmp_path, backend, agent_output_retries=0)
    trial = commit_trial(controller, "prior-trial")
    with pytest.raises(AgentOutputError):
        controller.ask("new-seeds", "seeds", inputs_for(controller, [trial]), Seeds)
    store.close()


def test_revision_idea_uses_same_evidence_contract(tmp_path):
    payload = idea("Synthetic revision").model_dump()
    payload["evidence_ids"] = ["known-trial"]
    backend = Responses([{"idea": payload, "candidate": candidate(7).model_dump()}])
    controller, store, _, _ = setup(tmp_path, backend, agent_output_retries=0)
    trial = commit_trial(controller, "known-trial")
    result = controller.ask("revision", "refine", {"incumbent": trial, "sources": controller.sources}, Revision)
    assert result.idea.evidence_ids == ["known-trial"]
    store.close()


def test_one_charged_correction_is_cached_across_restart_and_preserves_bad_output(tmp_path):
    fixture = synthetic_contract()
    invalid = deepcopy(fixture["evolution_output"])
    backend = Responses([invalid, corrected_fixture_payload()])
    controller, store, _, _ = setup(tmp_path, backend, agent_output_retries=1)
    trials = [commit_trial(controller, row["id"], phase=row["phase"], status=row["status"])
              for row in fixture["attempt_projection"]]
    inputs = inputs_for(controller, trials)
    expected = controller.ask("synthetic-evolve", "evolve", inputs, Hypothesis)
    before = controller.budget.summary()
    deadline = store.get("live_deadline")
    assert len(backend.seen) == 2
    rows = model_rows(controller)
    assert len(rows) == 2 and sum(row["tokens"] for row in rows) == 240
    results = [store.json(row["result_ref"]) for row in rows]
    assert results[0]["payload"]["source_ids"] == invalid["source_ids"]
    assert results[1]["payload"]["source_ids"] == ["task"]
    event_rows = list(store.db.execute("SELECT value FROM metadata WHERE key LIKE 'events/model/%'"))
    assert len(event_rows) == 2
    assert all(store.json(json.loads(row[0])) for row in event_rows)
    controller, store = reopen(controller, store, backend)
    assert controller.ask("synthetic-evolve", "evolve", inputs, Hypothesis) == expected
    assert controller.budget.summary() == before and store.get("live_deadline") == deadline
    assert len(backend.seen) == 2
    store.close()


def test_exhausted_correction_does_not_restart_after_reopen(tmp_path):
    invalid = idea("Synthetic invalid").model_dump()
    invalid["source_ids"] = ["unknown-source"]
    backend = Responses([invalid, invalid])
    controller, store, _, _ = setup(tmp_path, backend, agent_output_retries=1)
    inputs = inputs_for(controller, [])
    with pytest.raises(AgentOutputError):
        controller.ask("exhausted", "evolve", inputs, Hypothesis)
    before = controller.budget.summary()
    assert len(backend.seen) == 2 and sum(row["tokens"] for row in model_rows(controller)) == 240
    controller, store = reopen(controller, store, backend)
    with pytest.raises(AgentOutputError):
        controller.ask("exhausted", "evolve", inputs, Hypothesis)
    assert len(backend.seen) == 2 and controller.budget.summary() == before
    store.close()


def test_budget_exhaustion_cannot_create_a_free_correction(tmp_path):
    payload = idea("Invalid within one-call budget").model_dump()
    payload["source_ids"] = ["unknown"]
    backend = Responses([payload])
    controller, store, _, _ = setup(tmp_path, backend, max_calls=1, agent_output_retries=1)
    with pytest.raises(LiveBudgetExhausted):
        controller.ask("bounded", "evolve", inputs_for(controller, []), Hypothesis)
    assert len(backend.seen) == 1
    assert len(model_rows(controller)) == 1 and model_rows(controller)[0]["tokens"] == 120
    store.close()


def test_unknown_model_dispatch_never_becomes_a_correction_retry(tmp_path):
    backend = Responses([KeyboardInterrupt()])
    controller, store, _, _ = setup(tmp_path, backend, agent_output_retries=1)
    inputs = inputs_for(controller, [])
    with pytest.raises(KeyboardInterrupt):
        controller.ask("uncertain", "evolve", inputs, Hypothesis)
    controller, store = reopen(controller, store, backend)
    with pytest.raises(AmbiguousCall):
        controller.ask("uncertain", "evolve", inputs, Hypothesis)
    assert len(backend.seen) == 1 and model_rows(controller)[0]["status"] == "running"
    store.close()


def test_unknown_correction_dispatch_is_not_repeated_after_restart(tmp_path):
    invalid = idea("Synthetic invalid first response").model_dump()
    invalid["source_ids"] = ["unknown"]
    backend = Responses([invalid, KeyboardInterrupt()])
    controller, store, _, _ = setup(tmp_path, backend, model_retries=0, agent_output_retries=1)
    inputs = inputs_for(controller, [])
    with pytest.raises(KeyboardInterrupt):
        controller.ask("uncertain-correction", "evolve", inputs, Hypothesis)
    before, deadline = controller.budget.summary(), store.get("live_deadline")
    rows = model_rows(controller)
    assert len(backend.seen) == len(rows) == 2
    assert rows[0]["tokens"] == 120 and rows[1]["status"] == "running"
    assert rows[1]["result_ref"] is None  # Unknown usage/completion is not a free retry.
    controller, store = reopen(controller, store, backend)
    with pytest.raises(AmbiguousCall):
        controller.ask("uncertain-correction", "evolve", inputs, Hypothesis)
    assert len(backend.seen) == 2
    assert controller.budget.summary() == before and store.get("live_deadline") == deadline
    store.close()


def test_transport_failure_does_not_consume_semantic_correction_allowance(tmp_path):
    backend = Responses([BackendError("Synthetic transport failure")])
    controller, store, _, _ = setup(tmp_path, backend, model_retries=0, agent_output_retries=1)
    with pytest.raises(BackendError):
        controller.ask("transport", "evolve", inputs_for(controller, []), Hypothesis)
    assert len(backend.seen) == 1 and len(model_rows(controller)) == 1
    store.close()


def test_provenance_integrity_error_never_triggers_correction(tmp_path):
    backend = Responses([idea("Synthetic valid structure").model_dump()])
    original = backend.invoke
    backend.invoke = lambda *args: original(*args).model_copy(update={"model": "wrong-model"})
    controller, store, _, _ = setup(tmp_path, backend, agent_output_retries=1)
    with pytest.raises(IntegrityError, match="provenance"):
        controller.ask("provenance", "evolve", inputs_for(controller, []), Hypothesis)
    assert len(backend.seen) == 1
    store.close()


def test_corrupt_store_is_not_treated_as_repairable_agent_output(tmp_path):
    controller, store, backend, _ = setup(tmp_path, agent_output_retries=1)
    ref = store.put_json({"synthetic": True})
    (store.root / ref).write_text('{"synthetic":false}')
    with pytest.raises(IntegrityError, match="modified"):
        controller.run()
    assert not backend.seen
    store.close()


def test_invalid_evolution_sets_terminal_blocked_and_preserves_incumbent(tmp_path):
    backend = Agents()
    respond = backend.responder

    def invalid_evolution(role, inputs):
        output = respond(role, inputs)
        if role == "evolve":
            output["idea"]["source_ids"] = ["task", inputs["attempts"][0]["id"]]
        return output

    backend.responder = invalid_evolution
    controller, store, _, runner = setup(tmp_path, backend, agent_output_retries=0)
    result = controller.run()
    assert result["status"] == "blocked" and result["outcome"] == "invalid_agent_output"
    assert store.get("live_state")["status"] == "blocked"
    assert store.get("live_best")["id"].startswith("search/0/")
    assert not store.get("live_frozen") and not runner.finals
    assert all(not key.startswith("search/1/") for key in runner.executions)
    before, incumbent, count = controller.budget.summary(), store.get("live_best"), len(backend.seen)
    assert controller.run() == result
    assert controller.budget.summary() == before and store.get("live_best") == incumbent
    assert len(backend.seen) == count
    store.close()


def test_correction_policy_is_explicit_bounded_and_defaults_off(tmp_path):
    controller, store, _, _ = setup(tmp_path)
    serialized = controller.config.model_dump(mode="json")
    serialized.pop("agent_output_retries", None)
    assert LiveSpec.model_validate(serialized).agent_output_retries == 0
    for invalid in (-1, 2):
        with pytest.raises(ValidationError):
            LiveSpec.model_validate({**serialized, "agent_output_retries": invalid})
    store.close()
