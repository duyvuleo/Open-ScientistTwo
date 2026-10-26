"""Structured review contract tests with explicitly synthetic model judgments."""

import pytest

from scientisttwo.research.reviews import (
    ConcernProgress, ReviewConcern, ScientificReview, WritingCriticReview,
    fixture_review, review_gate, validate_review_references,
)


def concern(**changes):
    return ReviewConcern.model_validate({
        "id": "concern-one", "category": "claim_support", "severity": "blocking",
        "text": "Synthetic concern about unsupported causal attribution.",
        "requested_action": "revise_manuscript", "evidence_ids": ["result-one"],
        "claim_ids": ["claim-one"], "confidence": .8, **changes,
    })


def test_high_score_does_not_override_integrity():
    result = review_gate(fixture_review(score=10), integrity_passed=False)
    assert not result.accepted
    assert result.mechanical_status == "fail"
    assert result.scientific_status == "unverified"


def test_score_and_decision_both_gate_acceptance():
    assert review_gate(fixture_review(), integrity_passed=True).accepted
    assert not review_gate(fixture_review(score=7), integrity_passed=True).accepted
    assert not review_gate(fixture_review(decision="ReviseMethod"), integrity_passed=True).accepted


def test_blocker_cannot_be_silently_dropped_between_reviews():
    previous = [concern()]
    review = fixture_review(score=10)
    assert not review_gate(review, integrity_passed=True, previous_concerns=previous).accepted
    review.concerns = [concern(resolution="resolved", resolution_reason="Synthetic model judgment")]
    gate = review_gate(review, integrity_passed=True, previous_concerns=previous)
    assert gate.accepted and gate.scientific_status == "unverified"
    assert gate.acceptance_scope == "internal_workflow_only"


def test_review_ids_and_resolution_are_structural_requirements():
    with pytest.raises(ValueError, match="explanation"):
        concern(resolution="resolved")
    review = fixture_review().model_dump()
    review["concerns"] = [concern().model_dump(), concern().model_dump()]
    with pytest.raises(ValueError, match="unique"):
        ScientificReview.model_validate(review)


def test_writing_critic_cannot_request_method_execution():
    with pytest.raises(ValueError, match="editorial"):
        WritingCriticReview(decision="ReviseManuscript", summary="Synthetic writing feedback", confidence=.5,
                            concerns=[concern(requested_action="revise_method")])


def test_concern_ids_preserve_identity():
    review = fixture_review()
    review.concerns = [concern(text="A different scientific concern")]
    assert "changed identity" in validate_review_references(
        review, evidence_ids={"result-one"}, claim_ids={"claim-one"}, prior_concerns=[concern()]
    )[0]
    review.concerns = [concern()]
    assert validate_review_references(review, evidence_ids={"result-one"}, claim_ids={"claim-one"}) == []
    assert len(validate_review_references(review, evidence_ids=set(), claim_ids=set())) == 2


def test_execution_response_and_resolution_are_independent():
    state = ConcernProgress(concern_id="one", execution_status="executed", response_status="provided",
                            reviewer_resolution="unresolved", evidence_ids=["result-one"],
                            explanation="Executed work does not resolve the objection.")
    assert state.execution_status == "executed"
    assert state.reviewer_resolution == "unresolved"
