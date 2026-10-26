"""Phase-gate semantics through the real controller with synthetic transports.

No LLM or container executes. A small explicit critic interprets structured gate
fields, allowing later-stage evidence to be collected while rejecting current
violations. These are orchestration tests, not evidence of live critic quality.
"""

from copy import deepcopy

import pytest

from scientisttwo.live.models import MetricSpec
from scientisttwo.store import IntegrityError, canonical, digest

from test_research_workflow import campaign, with_offset


def install_gate_critic(agents):
    original = agents.legacy_response
    seen = []

    def respond(role, inputs):
        if role != "critic":
            return original(role, inputs)
        seen.append(deepcopy(inputs))
        stage = inputs["stage_contract"]
        current = stage["candidate_protocol"]
        comparison = inputs["comparison"]
        # Future experiments remain absent here. Promotion is appropriate only
        # when the supplied gate does not require their completion now.
        eligible = (comparison["valid"] and comparison["improved"]
            and current["seeds"] == stage["declared_stage_seeds"]
            and current["phase"] == stage["current_phase"]
            and not stage["missing_deferred_evidence_alone_requires_rejection"]
            and not stage["promotion"]["certifies_mechanism"])
        return {"decision": "Good" if eligible else "Bad",
                "reason": "Synthetic current-stage gate interpretation; future mechanism claim remains unverified"}

    agents.legacy_response = respond
    return seen


def assert_protocol_bound(store, inputs):
    stage = inputs["stage_contract"]
    result = inputs["evidence"]["observation"]
    assert stage["candidate_observation_id"] == result["id"]
    assert stage["current_phase"] == result["phase"]
    for field in ("candidate_protocol", "reference_protocol"):
        protocol = stage[field]
        frozen = store.json(protocol["ref"])
        assert protocol["status"] == "recorded"
        assert protocol["hash"] == digest(canonical(frozen))
        assert protocol["seeds"] == frozen["seeds"]
        assert protocol["phase"] == frozen["phase"]
    assert not stage["promotion"]["certifies_final_test_performance"]


def test_screening_promotes_to_full_before_any_mechanism_control(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path)
    controller.config.development.seeds = [11, 23, 37]
    seen = install_gate_critic(agents)
    try:
        result = controller.run()
        assert result["outcome"] == "development_improvement", result
        subset = next(value for value in seen if value["stage_contract"]["current_phase"] == "subset")
        full = next(value for value in seen if value["stage_contract"]["current_phase"] == "full")
        for value in (subset, full):
            assert_protocol_bound(store, value)
            assert value["evidence_package"]["matched_controls"] == []
            assert "no_unsupported_assertions_presented_as_established_findings" in value["stage_contract"]["required_current_evidence"]
        assert subset["stage_contract"]["declared_stage_seeds"] == [11]
        assert subset["stage_contract"]["promotion"]["authorizes"] == "full_development"
        future = {item["kind"]: item for item in subset["stage_contract"]["deferred_evidence"]}
        assert future["full_development"]["seeds"] == [11, 23, 37]
        assert full["stage_contract"]["declared_stage_seeds"] == [11, 23, 37]
        assert full["stage_contract"]["promotion"]["authorizes"] == "incumbent_comparison"
        assert "full_development" not in {item["kind"] for item in full["stage_contract"]["deferred_evidence"]}
        operations = controller.budget.summary()
        full_index = next(i for i, row in enumerate(operations) if row["key"] == "experiment/search/0/0/full/0")
        ablation_index = next(i for i, row in enumerate(operations) if row["kind"] == "experiment" and "/ablation/" in row["key"])
        assert full_index < ablation_index
        assert store.get("live_best")["decision"]["decision"] == "Good"
        assert "FINAL_STAGED_LABELS_MUST_NOT_ENTER_REFINEMENT" not in str(seen)
    finally:
        store.close()


def test_scientific_handoff_rescreen_uses_subset_gate_then_runs_full(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, handoff="scientific")
    controller.config.development.seeds = [5, 7, 9]
    seen = install_gate_critic(agents)
    try:
        result = controller.run()
        assert result["outcome"] == "development_improvement", result
        rescreen = next(value for value in seen if value["stage_contract"]["evaluation_purpose"] == "full_handoff_rescreen")
        assert_protocol_bound(store, rescreen)
        assert rescreen["stage_contract"]["current_phase"] == "subset"
        assert rescreen["stage_contract"]["declared_stage_seeds"] == [5]
        assert rescreen["stage_contract"]["promotion"]["authorizes"] == "full_development"
        assert store.get("live_best")["parents"] == [rescreen["stage_contract"]["candidate_observation_id"]]
        assert store.get("live_best")["result"]["phase"] == "full"
    finally:
        store.close()


def test_method_revision_compares_full_result_to_incumbent_not_baseline(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, refine=True)
    controller.config.development.seeds = [2, 4, 8]
    seen = install_gate_critic(agents)
    try:
        result = controller.run()
        assert result["outcome"] == "development_improvement", result
        revision = next(value for value in seen if value["stage_contract"]["evaluation_purpose"] == "method_revision")
        assert_protocol_bound(store, revision)
        stage = revision["stage_contract"]
        assert stage["reference_role"] == "incumbent" and stage["current_phase"] == "full"
        assert stage["reference_observation_id"].startswith("search/")
        assert stage["declared_stage_seeds"] == [2, 4, 8]
        assert stage["promotion"]["authorizes"] == "incumbent_comparison"
        assert not stage["promotion"]["certifies_mechanism"]
        assert store.get("live_best")["id"] == stage["candidate_observation_id"]
    finally:
        store.close()


def test_critic_retains_discretion_to_reject_unsupported_established_assertion(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path)
    original = agents.legacy_response
    reason = "Unsupported established assertion of universal causal proof; numerical gain alone cannot support it"

    def respond(role, inputs):
        if role == "seeds":
            value = original(role, inputs)
            for idea in value["ideas"]:
                idea["hypothesis"] = "This candidate has already proven a universal causal mechanism."
            return value
        if role == "critic":
            assert inputs["comparison"]["improved"]
            assert not inputs["stage_contract"]["missing_deferred_evidence_alone_requires_rejection"]
            return {"decision": "Bad", "reason": reason}
        return original(role, inputs)

    agents.legacy_response = respond
    try:
        result = controller.run()
        assert result["outcome"] == "no_improvement"
        assert all(trial["decision"] == {"decision": "Bad", "reason": reason} for trial in controller.trials)
        assert not any(row["key"].startswith("experiment/search/") and "/full/" in row["key"]
                       for row in controller.budget.summary())
        assert store.get("live_best") is None
    finally:
        store.close()


@pytest.mark.parametrize("violation", ["non_improvement", "hard_guardrail"])
def test_good_cannot_override_current_numerical_or_constraint_failure(tmp_path, violation):
    controller, store, _, agents, _ = campaign(tmp_path)
    original = agents.legacy_response
    if violation == "hard_guardrail":
        controller.config.development.metric_specs = {"mse": MetricSpec(direction="minimize", hard_max=.1)}

    def respond(role, inputs):
        if role == "implement" and violation == "non_improvement":
            return with_offset(controller.adapter.baseline(), 1.).model_dump()
        if role == "critic":
            assert "hard_constraints_and_guardrails" in inputs["stage_contract"]["required_current_evidence"]
            assert not inputs["comparison"]["improved"]
            return {"decision": "Good", "reason": "Deliberately invalid fixture approval"}
        return original(role, inputs)

    agents.legacy_response = respond
    try:
        result = controller.run()
        assert result["outcome"] == "no_improvement"
        assert all(trial["decision"]["decision"] == "Bad" for trial in controller.trials)
        assert store.get("live_best") is None
        assert not any(row["key"].startswith("experiment/search/") and "/full/" in row["key"]
                       for row in controller.budget.summary())
    finally:
        store.close()


def test_stage_contract_never_manufactures_seed_completeness(tmp_path):
    controller, store, runner, _, _ = campaign(tmp_path)
    controller.config.development.seeds = [3, 6, 9]
    try:
        runner.preflight()
        observation = runner.execute("candidate", controller.adapter.baseline(), "full")
        before = controller.budget.summary()
        current = controller.critic_stage_contract(observation, observation)
        assert current["candidate_protocol"]["seeds"] == current["declared_stage_seeds"] == [3, 6, 9]
        incomplete = observation.model_copy(update={"protocol_ref": None})
        unverified = controller.critic_stage_contract(incomplete, observation)
        assert unverified["candidate_protocol"]["status"] == "unverified"
        assert unverified["candidate_protocol"]["seeds"] == []
        assert unverified["declared_stage_seeds"] == [3, 6, 9]
        protocol = store.json(observation.protocol_ref)
        protocol["seeds"] = [3]
        altered = observation.model_copy(update={"protocol_ref": store.put_json(protocol), "protocol_hash": digest(canonical(protocol))})
        with pytest.raises(IntegrityError, match="frozen observation protocol"):
            controller.critic_stage_contract(altered, observation)
        assert controller.budget.summary() == before
    finally:
        store.close()


def test_failed_refinement_preserves_incumbent_despite_full_gate(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, refine=True)
    original = agents.legacy_response

    def respond(role, inputs):
        if role == "implement" and "parent" in inputs:
            before = next(item["content"] for item in inputs["parent"]["files"] if item["path"] == "custom.py")
            after = before.replace("SYNTHETIC_TEST_OFFSET = 0.5", "SYNTHETIC_TEST_OFFSET = 1.5")
            return {"parent_source_ref": inputs["parent"]["parent_source_ref"],
                "edits": [{"path": "custom.py", "before": before, "after": after, "reason": "Synthetic adverse revision"}],
                "description": "Synthetic worse method"}
        return original(role, inputs)

    agents.legacy_response = respond
    seen = install_gate_critic(agents)
    try:
        result = controller.run()
        assert result["outcome"] == "development_improvement"
        revision = next(value for value in seen if value["stage_contract"]["evaluation_purpose"] == "method_revision")
        assert not revision["comparison"]["improved"]
        assert store.get("live_best")["id"] == revision["stage_contract"]["reference_observation_id"]
        failed = next(trial for trial in controller.trials if trial["id"] == revision["stage_contract"]["candidate_observation_id"])
        assert failed["decision"]["decision"] == "Bad"
    finally:
        store.close()
