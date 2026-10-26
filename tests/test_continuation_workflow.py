"""Writer recovery through the real controller using explicitly synthetic doubles.

The scientific predictions, agent replies, container transports and PDF bytes
are fixture contracts. These tests establish checkpoint/recovery behavior, not
scientific utility, GPU execution, or live PaperOrchestra quality.
"""

from copy import deepcopy

import pytest

from scientisttwo.backends import BackendError
from scientisttwo.integrations.paperorchestra.controller_writer import PaperOrchestraResponse
from scientisttwo.live.runtime import Budget, DevelopmentRunner
from scientisttwo.research.controller import ResearchController
from scientisttwo.store import IntegrityError, Store, canonical, digest

from test_paperorchestra_research_workflow import (
    assert_authoritative_reviews,
    paper_campaign,
)


def operation_rows(store, kind=None):
    query = "SELECT * FROM live_operations"
    parameters = ()
    if kind is not None:
        query += " WHERE kind=?"
        parameters = (kind,)
    return [dict(row) for row in store.db.execute(query + " ORDER BY rowid", parameters)]


def stop_at_first_writer(tmp_path):
    controller, store, runner, agents, execution = paper_campaign(
        tmp_path, writing_refinements=0)
    controller.config.token_stop_threshold = 1_500_000
    previous = agents.responder
    failures = []

    def fail_first_writer(role, inputs):
        if role == "paperorchestra_model" and not failures:
            failures.append(deepcopy(inputs))
            raise BackendError("Synthetic completed dispatch with unknown token usage")
        return previous(role, inputs)

    agents.responder = fail_first_writer
    state = controller.run()
    assert state["status"] == "blocked", state
    assert state["stage"] == "draft", state
    assert len(failures) == 1
    failed = [row for row in operation_rows(store, "model") if row["status"] == "failed"]
    assert len(failed) == 1
    assert "manuscript/upstream/0001" in failed[0]["key"]
    assert store.get("live_frozen") is None
    assert store.get("live_final") is None
    assert all(not row["key"].startswith("final/") for row in operation_rows(store))
    roles = {role for role, _ in agents.seen}
    assert {"baseline_plan", "critic", "selection_review", "ablation_planner",
            "ablation_coder", "ablation_critic"} <= roles
    assert "scientific_review" not in roles and "meta" not in roles
    agents.responder = previous
    return controller, store, runner, agents, execution, failed[0], failures[0]


def authorize(store, failed, *, continuation_id="synthetic-writer-repair"):
    from scientisttwo.continuation import admit_continuation, prepare_plan

    plan_ref = prepare_plan(store, failed["key"], continuation_id)
    allowance = {
        "schema_version": "scientisttwo-continuation-allowance-v1",
        "plan_ref": plan_ref,
        "max_new_calls": 30,
        "token_stop_threshold": 10_000,
        "usage_disposition": "acknowledge_unknown_with_segment_limit",
        "authorization_ref": "synthetic-test-authorization:no-live-dispatch",
    }
    admission_ref = admit_continuation(store, plan_ref, allowance)
    return plan_ref, admission_ref


def reopen_campaign(controller, store, agents, execution):
    """Discard controller/budget/SQLite state while preserving fixture counters."""
    root, config, adapter = store.root, controller.config, controller.adapter
    store.close()
    store = Store(root)
    budget = Budget(store, config)
    runner = DevelopmentRunner(store, config, adapter, budget, executor=execution)
    return ResearchController(store, config, adapter, agents, runner, budget=budget), store


def test_authorized_writer_continuation_reuses_experiments_and_finishes_workflow(tmp_path):
    from scientisttwo.continuation import audit_continuation, install_continuation

    controller, store, runner, agents, execution, failed, first_inputs = stop_at_first_writer(tmp_path)
    try:
        experiments = operation_rows(store, "experiment")
        original_operations = operation_rows(store)
        stages = deepcopy(execution.scientific.stages)
        prior_calls = len(agents.seen)
        deadline = store.get("live_deadline")
        failed_bytes = store.read(failed["result_ref"])
        request_key = failed["key"].removeprefix("model/").rsplit("/", 1)[0]
        original_request_ref = store.get("request/" + request_key)
        original_request = store.json(original_request_ref)
        assert failed["request_hash"] == digest(canonical(original_request))
        assert store.db.execute("SELECT 1 FROM calls WHERE key=?",
                                ("live-agent/" + request_key,)).fetchone() is None

        # Ordinary resume must remain closed to exhausted/possibly paid calls.
        assert controller.run()["status"] == "blocked"
        assert operation_rows(store) == original_operations
        assert execution.scientific.stages == stages
        assert len(agents.seen) == prior_calls

        plan_ref, admission_ref = authorize(store, failed)
        controller, store = reopen_campaign(controller, store, agents, execution)
        install_continuation(controller, plan_ref, admission_ref)
        previous = agents.responder
        retried_inputs = []

        def observe_writer_retry(role, inputs):
            if role == "paperorchestra_model" and not retried_inputs:
                # Reconstructing the controller's chronological context is free:
                # no completed development train/infer/score stage is repeated.
                assert operation_rows(store, "experiment") == experiments
                assert execution.scientific.stages == stages
                assert store.get("live_final") is None
                assert store.get("live_frozen") is None
                retried_inputs.append(deepcopy(inputs))
            return previous(role, inputs)

        agents.responder = observe_writer_retry
        state = controller.run()
        assert state["status"] == "complete", state
        assert state["outcome"] == "development_improvement"
        assert retried_inputs == [first_inputs]
        assert store.get("live_deadline") == deadline
        assert [row for row in operation_rows(store, "experiment")
                if not row["key"].startswith("final/")] == experiments
        assert all(row in operation_rows(store) for row in original_operations)
        assert dict(store.db.execute("SELECT * FROM live_operations WHERE key=?",
                                     (failed["key"],)).fetchone()) == failed
        assert store.read(failed["result_ref"]) == failed_bytes
        assert store.get("request/" + request_key) == original_request_ref

        # A new durable operation owns the replacement. The original ask cache
        # is bound to that same request; the failed row is never changed to pass.
        replacement = [row for row in operation_rows(store, "model")
                       if row["key"] != failed["key"] and row["status"] == "success"
                       and row["request_hash"] == failed["request_hash"]]
        assert len(replacement) == 1
        cached = store.db.execute("SELECT * FROM calls WHERE key=?",
                                  ("live-agent/" + request_key,)).fetchone()
        assert cached and cached["input_hash"] == failed["request_hash"]
        assert store.json(cached["output_ref"]) == store.json(replacement[0]["result_ref"])
        assert_authoritative_reviews(store, agents)
        assert store.get("live_frozen") and store.get("research_final_bundle")
        assert store.get("live_final")["selected"]["status"] == "completed"
        assert "FINAL_STAGED_LABELS_MUST_NOT_ENTER_REFINEMENT" not in str(agents.seen)
        assert not any("'phase': 'final'" in str(inputs) for _, inputs in agents.seen)
        assert not store.verify()

        audit = audit_continuation(store, plan_ref, admission_ref)
        assert audit["integrity"] == "pass", audit
        assert audit["original_total_token_ceiling_verified"] is False
        before_resume = (operation_rows(store), len(agents.seen), len(execution.writer.calls),
                         len(execution.appendix.calls), deepcopy(execution.scientific.stages))
        controller, store = reopen_campaign(controller, store, agents, execution)
        install_continuation(controller, plan_ref, admission_ref)
        assert controller.run() == state
        assert before_resume == (operation_rows(store), len(agents.seen), len(execution.writer.calls),
                                 len(execution.appendix.calls), execution.scientific.stages)
        assert audit_continuation(store, plan_ref, admission_ref)["integrity"] == "pass"
    finally:
        store.close()


def test_continuation_refuses_changed_writer_request_without_backend_dispatch(tmp_path):
    from scientisttwo.continuation import install_continuation

    controller, store, _, agents, execution, failed, _ = stop_at_first_writer(tmp_path)
    try:
        plan_ref, admission_ref = authorize(store, failed)
        install_continuation(controller, plan_ref, admission_ref)
        request_key = failed["key"].removeprefix("model/").rsplit("/", 1)[0]
        inputs = store.json(store.get("request/" + request_key))["inputs"]
        inputs["prompt"] += "\nChanged synthetic request must not inherit authorization."
        operations, calls, stages = operation_rows(store), len(agents.seen), deepcopy(execution.scientific.stages)
        with pytest.raises(IntegrityError):
            controller.ask(request_key, "paperorchestra_model", inputs, PaperOrchestraResponse)
        assert operation_rows(store) == operations
        assert len(agents.seen) == calls
        assert execution.scientific.stages == stages
        assert store.get("live_final") is None
    finally:
        store.close()


def test_continuation_refuses_modified_saved_experiment_before_dispatch(tmp_path):
    from scientisttwo.continuation import install_continuation

    controller, store, _, agents, execution, failed, _ = stop_at_first_writer(tmp_path)
    try:
        plan_ref, admission_ref = authorize(store, failed)
        experiment = operation_rows(store, "experiment")[0]
        path = store.root / experiment["result_ref"]
        path.write_bytes(path.read_bytes() + b" ")
        operations, calls, stages = operation_rows(store), len(agents.seen), deepcopy(execution.scientific.stages)
        with pytest.raises(IntegrityError):
            install_continuation(controller, plan_ref, admission_ref)
        assert operation_rows(store) == operations
        assert len(agents.seen) == calls
        assert execution.scientific.stages == stages
        assert store.get("live_final") is None
    finally:
        store.close()


def test_continuation_refuses_changed_controller_source_before_dispatch(tmp_path, monkeypatch):
    from scientisttwo.continuation import install_continuation
    from scientisttwo.research import controller as controller_module

    controller, store, _, agents, execution, failed, _ = stop_at_first_writer(tmp_path)
    try:
        plan_ref, admission_ref = authorize(store, failed)
        actual_identity = controller_module.implementation_identity()
        monkeypatch.setattr(controller_module, "implementation_identity",
                            lambda: {**actual_identity, "synthetic_source_change": "f" * 64})
        operations, calls, stages = operation_rows(store), len(agents.seen), deepcopy(execution.scientific.stages)
        # The repair is an explicit sidecar over the original implementation;
        # it cannot authorize changing the pinned research controller.
        with pytest.raises(IntegrityError):
            install_continuation(controller, plan_ref, admission_ref)
            controller.run()
        assert operation_rows(store) == operations
        assert len(agents.seen) == calls
        assert execution.scientific.stages == stages
        assert store.get("live_final") is None
    finally:
        store.close()
