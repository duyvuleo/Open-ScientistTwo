"""Explicit scientific review judgments, separate from mechanical evidence checks.

A model score is a recorded opinion. These records cannot certify novelty,
causal isolation, correctness, or conference acceptance.
"""

from typing import Literal

from pydantic import Field, model_validator

from ..models import Record


class RubricRating(Record):
    score: float = Field(ge=1, le=10, description="Rating on the declared 1–10 scale, not a five-point scale.")
    rationale: str = Field(min_length=1, max_length=5000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=128)


class ReviewRubric(Record):
    empirical_validity: RubricRating
    claim_support: RubricRating
    novelty_positioning: RubricRating
    reproducibility: RubricRating
    clarity: RubricRating


class ReviewConcern(Record):
    id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")
    category: Literal[
        "empirical_validity", "claim_support", "novelty_positioning", "reproducibility", "clarity"
    ]
    severity: Literal["blocking", "major", "minor"]
    text: str = Field(min_length=1, max_length=10000)
    requested_action: Literal[
        "revise_manuscript", "supplementary_experiment", "revise_method", "clarify", "none"
    ]
    evidence_ids: list[str] = Field(default_factory=list, max_length=128)
    claim_ids: list[str] = Field(default_factory=list, max_length=128)
    confidence: float = Field(ge=0, le=1)
    resolution: Literal["open", "resolved", "unresolved"] = "open"
    resolution_reason: str = Field(default="", max_length=5000)

    @model_validator(mode="after")
    def resolved_reason(self):
        if self.resolution != "open" and not self.resolution_reason.strip():
            raise ValueError("A resolution judgment requires an explanation")
        return self


class ScientificReview(Record):
    schema_version: Literal["scientific-review-v1"] = "scientific-review-v1"
    score: float = Field(ge=1, le=10, description="Overall score on a 1–10 scale; 8/10 is the workflow early-stop threshold.")
    score_scale: Literal[10] | None = Field(default=None,
        description="New reviews must explicitly declare 10. Null is reserved for historical reviews with an unspecified scale; it never authorizes automatic rescaling.")
    confidence: float = Field(ge=0, le=1)
    rubric: ReviewRubric
    concerns: list[ReviewConcern] = Field(default_factory=list, max_length=64)
    summary: str = Field(min_length=1, max_length=10000)
    decision: Literal["Accept", "ReviseManuscript", "SupplementaryExperiments", "ReviseMethod"]
    judgment_scope: Literal["model_scientific_judgment_unverified"] = (
        "model_scientific_judgment_unverified"
    )

    @model_validator(mode="after")
    def unique_concerns(self):
        identifiers = [item.id for item in self.concerns]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Review concern IDs must be unique")
        return self


class WritingCriticReview(Record):
    schema_version: Literal["writing-critic-v1"] = "writing-critic-v1"
    decision: Literal["Accept", "ReviseManuscript"]
    summary: str = Field(min_length=1, max_length=10000)
    concerns: list[ReviewConcern] = Field(default_factory=list, max_length=64)
    confidence: float = Field(ge=0, le=1)
    judgment_scope: Literal["model_scientific_judgment_unverified"] = (
        "model_scientific_judgment_unverified"
    )

    @model_validator(mode="after")
    def writing_scope(self):
        if len({item.id for item in self.concerns}) != len(self.concerns):
            raise ValueError("Review concern IDs must be unique")
        if any(item.requested_action not in {"revise_manuscript", "clarify", "none"}
               for item in self.concerns):
            raise ValueError("The writing critic requests editorial work, not method execution")
        return self


class ConcernProgress(Record):
    concern_id: str
    execution_status: Literal["not_requested", "pending", "executed", "unfulfilled"]
    response_status: Literal["missing", "provided"]
    reviewer_resolution: Literal["not_reviewed", "resolved", "unresolved"]
    evidence_ids: list[str] = Field(default_factory=list)
    explanation: str


class ReviewGate(Record):
    accepted: bool
    mechanical_status: Literal["pass", "fail"]
    reasons: list[str]
    scientific_status: Literal["unverified"] = "unverified"
    acceptance_scope: Literal["internal_workflow_only"] = "internal_workflow_only"


def review_gate(
    review: ScientificReview | dict,
    *,
    integrity_passed: bool,
    threshold: float = 8,
    previous_concerns: list[ReviewConcern] | None = None,
) -> ReviewGate:
    """A high score cannot waive evidence failures or silently drop blockers."""
    review = ScientificReview.model_validate(review)
    reasons = []
    if not integrity_passed:
        reasons.append("Mechanical evidence integrity did not pass")
    if review.score < threshold:
        reasons.append("Review score is below the configured early-stop threshold")
    if review.decision != "Accept":
        reasons.append("Reviewer requested further work")
    current = {item.id: item for item in review.concerns}
    for concern in [*(previous_concerns or []), *review.concerns]:
        if concern.severity == "blocking":
            latest = current.get(concern.id)
            if latest is None or latest.resolution != "resolved":
                reason = f"Blocking concern {concern.id} remains unresolved"
                if reason not in reasons:
                    reasons.append(reason)
    return ReviewGate(
        accepted=not reasons,
        mechanical_status="pass" if integrity_passed else "fail",
        reasons=reasons,
    )


def validate_review_references(
    review: ScientificReview | WritingCriticReview,
    *,
    evidence_ids: set[str],
    claim_ids: set[str],
    prior_concerns: list[ReviewConcern] | None = None,
) -> list[str]:
    """Validate identifiers and stable concern identity, never the model's judgment."""
    errors = []
    for concern in review.concerns:
        if set(concern.evidence_ids) - evidence_ids:
            errors.append(f"Concern {concern.id} references unknown evidence")
        if set(concern.claim_ids) - claim_ids:
            errors.append(f"Concern {concern.id} references unknown claims")
    if isinstance(review, ScientificReview):
        for name in ReviewRubric.model_fields:
            if set(getattr(review.rubric, name).evidence_ids) - evidence_ids:
                errors.append(f"Rubric {name} references unknown evidence")
    prior = {concern.id: concern for concern in prior_concerns or []}
    for concern in review.concerns:
        previous = prior.get(concern.id)
        if previous and (previous.category != concern.category or previous.text != concern.text):
            errors.append(f"Concern {concern.id} changed identity instead of recording a resolution")
    return errors


def validate_live_review_scale(review: ScientificReview) -> list[str]:
    """Require the scale explicitly without modifying the model's judgment."""
    if review.score_scale != 10:
        return ["Declare score_scale=10 and assess overall and rubric scores on a 1–10 scale. Historical or five-point scores are not automatically converted. Preserve honest concerns and do not invent supplementary work to justify a score."]
    return []


def fixture_review(*, score: float = 8, decision: str = "Accept") -> ScientificReview:
    """Labeled synthetic response for contract tests and scripted validation only."""
    rating = RubricRating(score=score, rationale="Synthetic fixture judgment; not scientific review.")
    return ScientificReview(
        score=score,
        score_scale=10,
        confidence=0,
        rubric=ReviewRubric(**{name: rating for name in ReviewRubric.model_fields}),
        concerns=[],
        summary="Synthetic fixture reviewer response; scientific quality remains unverified.",
        decision=decision,
    )
