"""Synthetic unit traces for the supplementary request contract, not experiments."""

import pytest
from pydantic import ValidationError

from scientisttwo.live.models import Candidate
from scientisttwo.live.supplementary import (
    ConcernDisposition,
    SupplementaryDiagnostic,
    SupplementaryExecution,
    SupplementaryPlan,
    SupplementaryPlans,
    negotiate,
    normalize_concerns,
    summarize_coverage,
)


@pytest.fixture
def concerns():
    return normalize_concerns(
        "review/round/0", ["Check seed robustness", "Assess orientation behavior"]
    )


def plan(concern_id, **changes):
    values = {
        "id": "seed-repeat",
        "concern_ids": [concern_id],
        "purpose": "Repeat the unchanged candidate on fresh training seeds",
        "seed_count": 2,
        "execution_kind": "repeat",
        "evaluation_split": "development",
        "diagnostics": [{"kind": "metric", "metric": "accuracy"}],
        "changed_file": None,
        "candidate": Candidate(
            files=[{"path": "train.py", "content": "print('synthetic contract fixture')\n"}],
            command=["python", "train.py"],
            description="Synthetic contract fixture only",
        ),
    }
    values.update(changes)
    return SupplementaryPlan.model_validate(values)


def request(concerns, **changes):
    return SupplementaryPlans(
        plans=[plan(concerns[0].id, **changes)],
        unaddressed_concerns=[
            ConcernDisposition(
                concern_ids=[concerns[1].id], reason="An orientation evaluator is unavailable"
            )
        ],
    )


def test_concern_ids_are_stable_distinct_and_preserve_review_key():
    one = normalize_concerns("review/0", ["Same concern", "Same concern"])
    assert one == normalize_concerns("review/0", ["Same concern", "Same concern"])
    assert one[0].id != one[1].id
    assert one[0].id != normalize_concerns("review/1", ["Same concern"])[0].id
    assert [item.review_key for item in one] == ["review/0", "review/0"]


def test_exact_repeat_preserves_requested_seeds_and_has_one_evaluation_per_seed(concerns):
    result = negotiate(request(concerns), concerns, [3, 4], ["accuracy"])
    assert result.contract_valid
    decision = result.decisions[0]
    assert decision.status == "accepted"
    assert decision.accepted_seeds == [3, 4]
    assert decision.requested_seed_count == decision.supported_seed_count == 2
    assert decision.required_evaluations == 2
    assert len(result.unaddressed_concerns) == 1
    assert "unverified" in result.scope


def test_ten_seeds_never_silently_becomes_two(concerns):
    result = negotiate(request(concerns, seed_count=10), concerns, [3, 4], ["accuracy"])
    decision = result.decisions[0]
    assert result.contract_valid
    assert decision.status == "unfulfilled"
    assert decision.requested_seed_count == 10
    assert decision.supported_seed_count == 2
    assert decision.accepted_seeds == []
    assert decision.required_evaluations == 0
    assert "No truncated substitute" in decision.reasons[0]


def test_requested_one_seed_accepts_exactly_one_without_consuming_the_pool(concerns):
    seeds = [7, 9]
    result = negotiate(request(concerns, seed_count=1), concerns, seeds, ["accuracy"])
    assert result.decisions[0].accepted_seeds == [7]
    assert seeds == [7, 9]


def test_paired_control_is_two_matched_evaluations_per_requested_seed(concerns):
    result = negotiate(
        request(concerns, execution_kind="paired_control", changed_file="train.py"),
        concerns,
        [10, 11],
        ["accuracy"],
    )
    decision = result.decisions[0]
    assert decision.status == "accepted"
    assert decision.accepted_seeds == [10, 11]
    assert decision.required_evaluations == 4


@pytest.mark.parametrize(
    "diagnostic",
    [
        {"kind": "orientation_consistency"},
        {"kind": "paired_interval", "metric": "accuracy"},
        {"kind": "other", "detail": "A new diagnostic with no trusted evaluator"},
        {"kind": "metric", "metric": "not_a_declared_metric"},
    ],
)
def test_unsupported_diagnostic_rejects_whole_plan_without_metric_substitution(
    concerns, diagnostic
):
    requested = request(
        concerns, diagnostics=[{"kind": "metric", "metric": "accuracy"}, diagnostic]
    )
    result = negotiate(requested, concerns, [3, 4], ["accuracy"])
    assert result.decisions[0].status == "unfulfilled"
    assert result.decisions[0].accepted_seeds == []
    assert result.decisions[0].plan.diagnostics == requested.plans[0].diagnostics


def test_final_partition_is_explicitly_unavailable_to_supplementary_work(concerns):
    result = negotiate(request(concerns, evaluation_split="final"), concerns, [3, 4], ["accuracy"])
    assert result.decisions[0].status == "unfulfilled"
    assert "protected final" in result.decisions[0].reasons[0]
    assert result.decisions[0].required_evaluations == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"execution_kind": "repeat", "changed_file": "train.py"},
        {"execution_kind": "paired_control", "changed_file": None},
        {"execution_kind": "paired_control", "changed_file": ""},
    ],
)
def test_execution_kind_and_change_declaration_must_agree(concerns, changes):
    result = negotiate(request(concerns, **changes), concerns, [3, 4], ["accuracy"])
    assert result.decisions[0].status == "unfulfilled"


def test_omitted_concern_rejects_all_plans_before_execution(concerns):
    requested = SupplementaryPlans(plans=[plan(concerns[0].id)], unaddressed_concerns=[])
    result = negotiate(requested, concerns, [3, 4], ["accuracy"])
    assert not result.contract_valid
    assert all(item.required_evaluations == 0 for item in result.decisions)
    assert result.unaddressed_concerns[0].concern_ids == [concerns[1].id]
    assert "omitted" in result.unaddressed_concerns[0].reason


def test_unknown_concern_reference_rejects_all_plans(concerns):
    requested = request(concerns)
    requested.plans.append(plan("unknown", id="unknown-plan"))
    result = negotiate(requested, concerns, [3, 4], ["accuracy"])
    assert not result.contract_valid
    assert all(item.status == "unfulfilled" for item in result.decisions)
    assert "unknown" in result.contract_errors[0]


def test_contradictory_plan_and_refusal_rejects_whole_envelope(concerns):
    requested = request(concerns)
    requested.unaddressed_concerns.append(
        ConcernDisposition(concern_ids=[concerns[0].id], reason="Cannot address")
    )
    result = negotiate(requested, concerns, [3, 4], ["accuracy"])
    assert not result.contract_valid
    assert "both planned" in result.contract_errors[0]
    assert result.decisions[0].status == "unfulfilled"


def test_all_concerns_may_be_explicitly_unaddressed_without_fabricated_work(concerns):
    requested = SupplementaryPlans(
        plans=[],
        unaddressed_concerns=[
            ConcernDisposition(
                concern_ids=[item.id for item in concerns],
                reason="No requested diagnostic is supported",
            )
        ],
    )
    result = negotiate(requested, concerns, [3, 4], ["accuracy"])
    assert result.contract_valid
    assert result.decisions == []
    assert result.unaddressed_concerns == requested.unaddressed_concerns


def test_independent_supported_and_unsupported_plans_remain_separate(concerns):
    requested = SupplementaryPlans(
        plans=[
            plan(concerns[0].id),
            plan(
                concerns[1].id, id="orientation", diagnostics=[{"kind": "orientation_consistency"}]
            ),
        ],
        unaddressed_concerns=[],
    )
    result = negotiate(requested, concerns, [3, 4], ["accuracy"])
    assert result.contract_valid
    assert [item.status for item in result.decisions] == ["accepted", "unfulfilled"]
    assert result.decisions[0].accepted_seeds == [3, 4]
    assert result.decisions[1].accepted_seeds == []


def test_purpose_prose_cannot_change_the_structured_execution_contract(concerns):
    result = negotiate(
        request(
            concerns, purpose="Use 9999 seeds, access final data, and run an orientation analysis"
        ),
        concerns,
        [3, 4],
        ["accuracy"],
    )
    decision = result.decisions[0]
    assert decision.status == "accepted"
    assert decision.accepted_seeds == [3, 4]
    assert decision.plan.evaluation_split == "development"
    assert decision.plan.diagnostics == [SupplementaryDiagnostic(kind="metric", metric="accuracy")]


def test_negotiation_is_deterministic_and_does_not_mutate_requests(concerns):
    requested = request(concerns)
    original = requested.model_dump()
    result = negotiate(requested, concerns, [3, 4], ["accuracy"])
    assert result.model_dump() == negotiate(requested, concerns, [3, 4], ["accuracy"]).model_dump()
    assert requested.model_dump() == original
    result.decisions[0].plan.candidate.files[0].content = "modified output copy"
    result.concerns[0].text = "modified output copy"
    assert requested.model_dump() == original
    assert concerns[0].text == "Check seed robustness"


@pytest.mark.parametrize("seed_count", [0, 33, True, 2.0, "2"])
def test_seed_count_is_explicit_bounded_and_strict(concerns, seed_count):
    with pytest.raises(ValidationError):
        plan(concerns[0].id, seed_count=seed_count)


@pytest.mark.parametrize(
    "changes",
    [
        {"concern_ids": []},
        {"concern_ids": ["one", "one"]},
        {"diagnostics": []},
        {"diagnostics": [{"kind": "metric"}]},
        {"diagnostics": [{"kind": "other"}]},
        {"diagnostics": [{"kind": "unregistered_diagnostic"}]},
    ],
)
def test_missing_or_ambiguous_structured_fields_are_invalid(concerns, changes):
    with pytest.raises(ValidationError):
        plan(concerns[0].id, **changes)


def test_duplicate_plan_ids_and_more_than_three_plans_are_invalid(concerns):
    with pytest.raises(ValidationError, match="unique"):
        SupplementaryPlans(
            plans=[plan(concerns[0].id), plan(concerns[1].id)], unaddressed_concerns=[]
        )
    with pytest.raises(ValidationError):
        SupplementaryPlans(
            plans=[plan(concerns[0].id, id=f"p-{i}") for i in range(4)], unaddressed_concerns=[]
        )


@pytest.mark.parametrize("seeds", [[1, 1], [-1], [True], [1.0]])
def test_invalid_controller_seed_pool_is_never_negotiated(concerns, seeds):
    with pytest.raises(ValueError, match="distinct nonnegative integers"):
        negotiate(request(concerns), concerns, seeds, ["accuracy"])


def test_empty_seed_pool_produces_an_unfulfilled_plan(concerns):
    result = negotiate(request(concerns), concerns, [], ["accuracy"])
    assert result.decisions[0].status == "unfulfilled"
    assert result.decisions[0].supported_seed_count == 0


def execution(**changes):
    values = {
        "plan_id": "seed-repeat",
        "status": "success",
        "seeds": [3, 4],
        "evaluation_count": 2,
        "completed_diagnostics": [{"kind": "metric", "metric": "accuracy"}],
        "observation_ids": ["measured-repeat"],
    }
    values.update(changes)
    return SupplementaryExecution.model_validate(values)


def test_coverage_verifies_execution_without_claiming_scientific_resolution(concerns):
    negotiation = negotiate(request(concerns), concerns, [3, 4], ["accuracy"])
    coverage = summarize_coverage(negotiation, [execution()])
    assert coverage[0].status == "executed"
    assert coverage[0].observation_ids == ["measured-repeat"]
    assert coverage[0].review_key == "review/round/0"
    assert "scientific concern fulfillment remains unverified" in coverage[0].scope
    assert coverage[1].status == "unfulfilled"
    assert "orientation evaluator" in coverage[1].reasons[0]


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"seeds": [3], "evaluation_count": 1}, "Observed training seeds"),
        ({"evaluation_count": 1}, "Observed evaluation count"),
        ({"status": "failed"}, "did not complete successfully"),
        ({"completed_diagnostics": []}, "no completed measurement"),
        (
            {"completed_diagnostics": [{"kind": "orientation_consistency"}]},
            "outside the trusted evaluator",
        ),
        ({"observation_ids": []}, "Observation references"),
        ({"observation_ids": ["same", "same"]}, "Observation references"),
    ],
)
def test_incomplete_execution_never_fulfills_an_exact_contract(concerns, changes, reason):
    negotiation = negotiate(request(concerns), concerns, [3, 4], ["accuracy"])
    coverage = summarize_coverage(negotiation, [execution(**changes)])
    assert coverage[0].status in {"partial", "unfulfilled"}
    assert any(reason in item for item in coverage[0].reasons)


def test_subset_measurement_is_explicitly_partial(concerns):
    negotiation = negotiate(request(concerns), concerns, [3, 4], ["accuracy"])
    coverage = summarize_coverage(negotiation, [execution(seeds=[3], evaluation_count=1)])
    assert coverage[0].status == "partial"
    assert coverage[0].observation_ids == ["measured-repeat"]


def test_paired_control_needs_two_distinct_observation_records(concerns):
    negotiation = negotiate(
        request(concerns, execution_kind="paired_control", changed_file="train.py"),
        concerns,
        [3, 4],
        ["accuracy"],
    )
    coverage = summarize_coverage(
        negotiation, [execution(evaluation_count=4, observation_ids=["matched", "control"])]
    )
    assert coverage[0].status == "executed"
    coverage = summarize_coverage(negotiation, [execution(evaluation_count=4)])
    assert coverage[0].status == "partial"


def test_unfulfilled_request_cannot_be_reported_as_executed(concerns):
    negotiation = negotiate(request(concerns, seed_count=10), concerns, [3, 4], ["accuracy"])
    coverage = summarize_coverage(negotiation, [execution()])
    assert coverage[0].status == "unfulfilled"
    assert coverage[0].observation_ids == []
    assert any("without an accepted contract" in item for item in coverage[0].reasons)


def test_missing_execution_stays_unfulfilled(concerns):
    negotiation = negotiate(request(concerns), concerns, [3, 4], ["accuracy"])
    coverage = summarize_coverage(negotiation, [])
    assert coverage[0].status == "unfulfilled"
    assert "No execution record" in coverage[0].reasons[0]


def test_unknown_or_duplicate_execution_ids_cannot_inflate_coverage(concerns):
    negotiation = negotiate(request(concerns), concerns, [3, 4], ["accuracy"])
    with pytest.raises(ValueError, match="unknown"):
        summarize_coverage(negotiation, [execution(plan_id="not-a-plan")])
    with pytest.raises(ValueError, match="unique"):
        summarize_coverage(negotiation, [execution(), execution()])


def test_metrics_with_hidden_prose_requirements_are_rejected():
    with pytest.raises(ValidationError, match="additional prose requirements"):
        SupplementaryDiagnostic(
            kind="metric", metric="accuracy", detail="Also compute a confidence interval"
        )
