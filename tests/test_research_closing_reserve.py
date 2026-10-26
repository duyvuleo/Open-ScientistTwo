"""Synthetic admission traces: never fabricate completion of deferred research."""

from copy import deepcopy

import pytest

from scientisttwo.live.runtime import AmbiguousCall
from scientisttwo.research.controller import ResearchController
from test_research_workflow import campaign


@pytest.mark.parametrize("limit,expected_block", [
    (22, "method/0/ablation/0/planning"),
    (26, "method/0/ablation/0/revision"),
    (30, "method/0/ablation/0/revision"),
])
def test_optional_scientific_work_defers_to_real_closure(tmp_path, limit, expected_block):
    controller, store, _, agents, _ = campaign(tmp_path, refine=True)
    controller.config.max_calls = limit
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        assert state["outcome"] == "development_improvement"
        assert state["workflow_coverage"] == "partial_budget_deferred"
        assert "not completed scientific evidence" in state["reason"]
        deferred = store.json(store.get("research-deferred/" + expected_block))
        assert deferred["status"] == "deferred" and deferred["reserved_closing_calls"] >= 6
        assert store.get("live_best")["id"] == "search/0/0/full/0"
        assert store.get("live_frozen")["best"]["id"] == "search/0/0/full/0"
        assert store.get("research_manuscript")
        assert all(value["status"] == "completed" for value in store.get("live_final").values())
        assert not any(role in {"method_disposition", "refine"} for role, _ in agents.seen)
        writer = next(inputs for role, inputs in agents.seen if role == "initial_drafter")
        assert writer["deferred_research_work"][0]["key"] == expected_block
        operations = controller.budget.summary()
        assert sum(row["kind"] == "model" for row in operations) <= limit
        assert not any("/revision" in row["key"] for row in operations)
        calls = len(agents.seen)
        assert controller.run() == state
        assert controller.budget.summary() == operations and len(agents.seen) == calls
    finally:
        store.close()


def test_deferred_search_closes_reference_with_explicit_partial_coverage(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, reject=True)
    controller.config.max_calls = 15
    controller.config.max_experiments = 3  # Two baseline phases and one protected final.
    try:
        state = controller.run()
        assert state["status"] == "complete" and state["outcome"] == "no_improvement", state
        assert state["workflow_coverage"] == "partial_budget_deferred"
        assert store.get("live_best") is None
        assert store.get("live_frozen")["best"]["id"] == "baseline"
        assert set(store.get("live_final")) == {"baseline"}
        assert store.get("live_final")["baseline"]["status"] == "completed"
        assert not any(role in {"implement", "critic", "selection_review"} for role, _ in agents.seen)
        writer = next(inputs for role, inputs in agents.seen if role == "initial_drafter")
        assert writer["search_outcome"] == "no_approved_improvement"
        assert writer["scientific_history"] == []
        assert writer["deferred_research_work"][0]["key"] == "search/0/0"
    finally:
        store.close()


def test_resume_replays_deferral_without_changing_earlier_writer_inputs(tmp_path):
    controller, store, runner, agents, _ = campaign(tmp_path, refine=True)
    controller.config.max_calls = 26
    original = controller.ask

    def interrupt_after_draft(key, role, *args, **kwargs):
        value = original(key, role, *args, **kwargs)
        if role == "initial_drafter":
            raise KeyboardInterrupt
        return value

    controller.ask = interrupt_after_draft
    try:
        with pytest.raises(KeyboardInterrupt):
            controller.run()
        before = deepcopy(agents.seen)
        resumed = ResearchController(store, controller.config, controller.adapter, agents,
                                     runner, budget=controller.budget)
        state = resumed.run()
        assert state["status"] == "complete", state
        assert agents.seen[:len(before)] == before
        assert sum(role == "initial_drafter" for role, _ in agents.seen) == 1
        assert state["workflow_coverage"] == "partial_budget_deferred"
        assert len(state["deferred_research_refs"]) == 1
    finally:
        store.close()


def test_actual_token_exhaustion_is_not_an_optional_deferral(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path)
    controller.config.token_stop_threshold = 100
    controller.config.workflow.estimated_tokens_per_call = 1
    original = agents.invoke

    def costly(*args, **kwargs):
        return original(*args, **kwargs).model_copy(update={"usage": {"input_tokens": 101}})

    agents.invoke = costly
    try:
        state = controller.run()
        assert state["outcome"] == "budget_exhausted"
        assert store.get("live_frozen") is None
        assert not controller.research_deferrals
        assert [row["tokens"] for row in controller.budget.summary() if row["kind"] == "model"] == [101]
    finally:
        store.close()


def test_ambiguous_started_operation_is_not_reclassified_as_deferred(tmp_path):
    controller, store, runner, _, _ = campaign(tmp_path)
    original = runner.execute

    def unknown(key, candidate, phase, **kwargs):
        if key.startswith("search/"):
            controller.budget.reserve("experiment/" + key, "experiment", {"synthetic_unknown": True})
            raise AmbiguousCall("Synthetic uncommitted execution")
        return original(key, candidate, phase, **kwargs)

    runner.execute = unknown
    try:
        state = controller.run()
        assert state["status"] == "blocked" and "uncommitted execution" in state["reason"]
        assert store.get("live_frozen") is None and not controller.research_deferrals
        assert controller.budget.summary()[-1]["status"] == "running"
    finally:
        store.close()


def test_required_rebuttal_cannot_consume_reserved_final_experiments(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, rebuttal=True)
    controller.config.max_experiments = 7
    try:
        state = controller.run()
        assert state["outcome"] == "budget_exhausted"
        assert "rebuttal" in state["reason"]
        assert any(role == "rebuttal_planner" for role, _ in agents.seen)
        assert store.get("live_frozen") is None
        assert not any("rebuttal" in row["key"] and row["kind"] == "experiment"
                       for row in controller.budget.summary())
    finally:
        store.close()
