"""Synthetic review routing traces; no models, containers or empirical claims."""
from copy import deepcopy

import pytest

from scientisttwo.live.supplementary import SupplementaryPlan, SupplementaryPlans
from scientisttwo.research.controller import ResearchController
from test_research_workflow import campaign


CONCERN = {
    "id": "editorial-caption", "category": "clarity", "severity": "minor",
    "text": "Clarify the synthetic table caption using its existing measurements.",
    "requested_action": "revise_manuscript", "confidence": 1,
    "evidence_ids": [], "claim_ids": [], "resolution": "open", "resolution_reason": "",
}


def editorial_campaign(tmp_path, *, allowance=1, persistent=False, review_rounds=1):
    controller, store, runner, agents, transport = campaign(tmp_path, writing_refinements=0)
    controller.config.manuscript_revisions = allowance
    controller.config.review_rounds = review_rounds
    original, observations = agents.respond, []

    def respond(role, inputs):
        result = original(role, inputs)
        if role == "scientific_review":
            observations.append({
                "inputs": deepcopy(inputs),
                "experiments": [dict(r) for r in store.db.execute(
                    "SELECT * FROM live_operations WHERE kind='experiment' ORDER BY key")],
                "best": deepcopy(store.get("live_best")),
            })
            if persistent or len(observations) == 1:
                result.update(score=7.5, score_scale=10, decision="ReviseManuscript",
                    summary="Synthetic manuscript-only feedback; no scientific experiment requested.",
                    concerns=[deepcopy(CONCERN)])
        return result

    agents.responder = respond
    return controller, store, runner, agents, transport, observations


@pytest.mark.parametrize("review_rounds", [0, 1])
def test_editorial_scientific_review_uses_existing_revision_allowance_without_science(
    tmp_path, review_rounds
):
    controller, store, _, agents, _, observations = editorial_campaign(tmp_path, review_rounds=review_rounds)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        roles = [role for role, _ in agents.seen]
        assert "rebuttal_planner" not in roles and "rebuttal_coder" not in roles
        assert "refine" not in roles and "method_disposition" not in roles
        assert roles.count("draft_enhancer") == roles.count("meta") == 1
        assert roles.count("scientific_review") == 2
        assert observations[0]["experiments"] == observations[1]["experiments"]
        assert observations[0]["best"] == observations[1]["best"] == store.get("live_frozen")["best"]
        routing = store.json(store.get("review-routing/method/0/0"))
        assert routing["route"] == "manuscript_revision"
        assert routing["decision_origin"] == "scientific_review"
        assert routing["concern_ids"] == [CONCERN["id"]]
        assert not routing["new_experiments"] and not routing["method_change"]
        assert not controller.supplementary_records
        revision = store.json(store.get("manuscript-revision/method/0/manuscript/0"))
        assert revision["no_new_experiments"] and revision["status"] == "reviewed"
        feedback = store.json(revision["feedback_ref"])
        assert feedback["review"]["review_ref"] == routing["review_ref"]
        assert "Controller editorial route" in feedback["meta"]["reason"]
        assert store.get("live_manuscript_revisions_used") == 1
        before, seen = controller.budget.summary(), len(agents.seen)
        assert controller.run() == state
        assert controller.budget.summary() == before and len(agents.seen) == seen
    finally:
        store.close()


@pytest.mark.parametrize("allowance", [0, 1, 2])
def test_unresolved_editorial_request_cannot_bypass_limit_or_turn_into_method_work(tmp_path, allowance):
    controller, store, _, agents, _, observations = editorial_campaign(
        tmp_path, allowance=allowance, persistent=True)
    try:
        state = controller.run()
        assert state["status"] == "blocked" and state["outcome"] == "manuscript_revision_limit", state
        assert not store.get("live_frozen") and not store.get("live_final")
        assert len(observations) == allowance + 1
        assert all(item["experiments"] == observations[0]["experiments"] for item in observations)
        roles = [role for role, _ in agents.seen]
        assert roles.count("draft_enhancer") == allowance
        assert not {"rebuttal_planner", "rebuttal_coder", "refine", "method_disposition"} & set(roles)
        assert store.get("live_best") == observations[0]["best"]
        before, seen = controller.budget.summary(), len(agents.seen)
        assert controller.run() == state
        assert controller.budget.summary() == before and len(agents.seen) == seen
    finally:
        store.close()


@pytest.mark.parametrize("role_to_interrupt", ["draft_enhancer", "scientific_review"])
def test_editorial_resume_reuses_completed_roles_and_preserves_experiment_ledger(tmp_path, role_to_interrupt):
    controller, store, runner, agents, _, observations = editorial_campaign(tmp_path)
    original, interrupted = controller.ask, False

    def after_commit(key, role, *args, **kwargs):
        nonlocal interrupted
        value = original(key, role, *args, **kwargs)
        if "/manuscript/" in key and role == role_to_interrupt and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return value

    controller.ask = after_commit
    try:
        with pytest.raises(KeyboardInterrupt):
            controller.run()
        calls = deepcopy(agents.seen)
        resumed = ResearchController(store, controller.config, controller.adapter, agents,
                                     runner, budget=controller.budget)
        state = resumed.run()
        assert state["status"] == "complete", state
        assert agents.seen[:len(calls)] == calls
        assert sum(role == "draft_enhancer" for role, _ in agents.seen) == 1
        assert len(observations) == 2 and observations[0]["experiments"] == observations[1]["experiments"]
        assert store.get("live_manuscript_revisions_used") == 1
    finally:
        store.close()


@pytest.mark.parametrize("kind", ["empty", "unaddressed", "unsupported"])
def test_zero_accepted_supplementary_actions_preserve_coverage_without_execution_admission(
    tmp_path, kind
):
    controller, store, runner, _, _, _ = editorial_campaign(tmp_path)
    try:
        assert controller.run()["status"] == "complete"
        best = store.get("live_best")
        before = controller.budget.summary()
        controller.config.max_experiments = sum(row["kind"] == "experiment" for row in before)

        def plans(key, incumbent, inputs, seeds, metrics):
            if kind != "unsupported":
                return SupplementaryPlans(plans=[], unaddressed_concerns=[] if not inputs["concerns"] else [
                    {"concern_ids": [c["id"] for c in inputs["concerns"]], "reason": "Synthetic unsupported request"}])
            return SupplementaryPlans(plans=[SupplementaryPlan(id="too-many-seeds",
                concern_ids=[c["id"] for c in inputs["concerns"]], purpose="Synthetic unsupported exact repeat",
                seed_count=len(seeds) + 1, execution_kind="repeat", evaluation_split="development",
                diagnostics=[{"kind": "metric", "metric": metrics[0]}], changed_file=None,
                candidate=best["candidate"])], unaddressed_concerns=[])

        def forbidden(*args, **kwargs):
            raise AssertionError("No accepted action may not admit execution or invoke experiments")

        controller.research_supplementary_plans = plans
        controller.admit_block = forbidden
        runner.execute = forbidden
        result = controller.supplementary_controls("empty-contract", best,
            {"concerns": [] if kind == "empty" else ["Synthetic unsupported experimental request"]})
        assert result == [] and controller.budget.summary() == before
        assert store.get("live_best") == best
        record = store.json(store.get("supplementary/empty-contract"))
        assert record["executions"] == []
        assert all(c["status"] == "unfulfilled" for c in record["coverage"])
        assert bool(record["coverage"]) == (kind != "empty")
        route = store.json(store.get("supplementary-routing/empty-contract"))
        assert route["route"] == "no_accepted_actions"
        assert not route["new_experiments"] and not route["execution_admission_required"]
        assert not store.get("research-admission/empty-contract/execution")
    finally:
        store.close()
