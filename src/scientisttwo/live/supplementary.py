"""Negotiate executable supplementary work before any experiment is dispatched.

Purpose and concern text are explanatory prose, never an execution language. The
controller executes only accepted structured fields. Acceptance is permission to
attempt the exact request, not evidence that the concern has been resolved.
"""

from hashlib import sha256
import json
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from ..models import Record
from .models import Candidate


Identifier = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
]
Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class Concern(Record):
    id: Identifier
    review_key: Nonempty
    text: Nonempty


def normalize_concerns(review_key: str, texts: list[str]) -> list[Concern]:
    """Give each review concern a stable ID without interpreting its prose."""
    result = []
    for index, text in enumerate(texts):
        normalized = text.strip()
        payload = json.dumps([review_key, index, normalized], ensure_ascii=False).encode()
        identifier = "concern-" + sha256(payload).hexdigest()[:20]
        result.append(Concern(id=identifier, review_key=review_key, text=normalized))
    return result


class SupplementaryDiagnostic(Record):
    kind: Literal["metric", "orientation_consistency", "paired_interval", "paired_differences",
                  "classification", "regression_residuals", "timing_distribution", "input_consistency", "other"]
    metric: str | None = None
    detail: Nonempty | None = None
    parameters: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def metric_required(self):
        if self.kind == "metric" and (self.metric is None or not self.metric.strip()):
            raise ValueError("Metric diagnostics require an explicit metric")
        if self.kind == "metric" and self.detail is not None:
            raise ValueError("Metric diagnostics cannot carry additional prose requirements")
        if self.kind == "metric" and self.parameters:
            raise ValueError("Metric diagnostics cannot carry additional analysis parameters")
        if self.kind == "other" and self.detail is None:
            raise ValueError("Other diagnostics require an explicit description")
        return self


class SupplementaryPlan(Record):
    id: Identifier
    concern_ids: list[Identifier] = Field(min_length=1, max_length=32)
    purpose: Nonempty = Field(
        description="Explanation only; all executable requirements must be in structured fields."
    )
    seed_count: int = Field(
        strict=True,
        ge=1,
        le=32,
        description="Exact number of fresh training seeds requested; never silently truncated.",
    )
    execution_kind: Literal["paired_control", "repeat"]
    evaluation_split: Literal["development", "final"]
    diagnostics: list[SupplementaryDiagnostic] = Field(min_length=1, max_length=8)
    changed_file: str | None
    candidate: Candidate

    @model_validator(mode="after")
    def unique_references(self):
        if len(set(self.concern_ids)) != len(self.concern_ids):
            raise ValueError("Supplementary concern references must be unique")
        diagnostics = [json.dumps(item.model_dump(), sort_keys=True) for item in self.diagnostics]
        if len(set(diagnostics)) != len(diagnostics):
            raise ValueError("Supplementary diagnostics must be unique")
        return self


class ConcernDisposition(Record):
    concern_ids: list[Identifier] = Field(min_length=1, max_length=32)
    reason: Nonempty

    @model_validator(mode="after")
    def unique_references(self):
        if len(set(self.concern_ids)) != len(self.concern_ids):
            raise ValueError("Unaddressed concern references must be unique")
        return self


class SupplementaryPlans(Record):
    plans: list[SupplementaryPlan] = Field(max_length=3)
    unaddressed_concerns: list[ConcernDisposition] = Field(max_length=32)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({plan.id for plan in self.plans}) != len(self.plans):
            raise ValueError("Supplementary plan IDs must be unique")
        refused = [item for entry in self.unaddressed_concerns for item in entry.concern_ids]
        if len(set(refused)) != len(refused):
            raise ValueError("Each unaddressed concern requires one disposition")
        return self


class SupplementaryDecision(Record):
    plan: SupplementaryPlan
    status: Literal["accepted", "unfulfilled"]
    reasons: list[str]
    accepted_seeds: list[int]
    requested_seed_count: int
    supported_seed_count: int
    required_evaluations: int


class SupplementaryNegotiation(Record):
    contract_version: Literal[1] = 1
    contract_valid: bool
    contract_errors: list[str]
    concerns: list[Concern]
    available_fresh_seeds: list[int]
    declared_metrics: list[str]
    supported_diagnostics: list[str] = ["metric"]
    decisions: list[SupplementaryDecision]
    unaddressed_concerns: list[ConcernDisposition]
    scope: Literal[
        "Exact execution requests only; scientific concern fulfillment remains unverified"
    ] = "Exact execution requests only; scientific concern fulfillment remains unverified"


class SupplementaryExecution(Record):
    """Controller-derived observations, never an agent's claimed completion."""

    plan_id: Identifier
    status: Literal["success", "failed", "not_run"]
    seeds: list[Annotated[int, Field(strict=True, ge=0)]]
    evaluation_count: int = Field(strict=True, ge=0)
    completed_diagnostics: list[SupplementaryDiagnostic]
    observation_ids: list[Nonempty]
    diagnostic_refs: list[str] = Field(default_factory=list)


class ConcernCoverage(Record):
    concern_id: Identifier
    review_key: Nonempty
    status: Literal["executed", "partial", "unfulfilled"]
    plan_ids: list[Identifier]
    observation_ids: list[str]
    reasons: list[str]
    scope: Literal[
        "Execution accounting only; scientific concern fulfillment remains unverified"
    ] = "Execution accounting only; scientific concern fulfillment remains unverified"


def summarize_coverage(
    negotiation: SupplementaryNegotiation,
    executions: list[SupplementaryExecution],
    *, store=None,
) -> list[ConcernCoverage]:
    """Compare observed work with exact accepted contracts, without scoring it."""
    by_id = {item.plan_id: item for item in executions}
    if len(by_id) != len(executions):
        raise ValueError("Execution records must be unique per supplementary plan")
    known_plan_ids = {item.plan.id for item in negotiation.decisions}
    if set(by_id) - known_plan_ids:
        raise ValueError("Execution records refer to unknown supplementary plans")
    plan_results = {}
    for decision in negotiation.decisions:
        execution = by_id.get(decision.plan.id)
        issues = list(decision.reasons)
        observed = []
        if decision.status != "accepted":
            if execution is not None and execution.status != "not_run":
                issues.append("An unfulfilled request was executed without an accepted contract")
        elif execution is None:
            issues.append("No execution record is available for the accepted plan")
        else:
            if execution.status != "success":
                issues.append("The accepted plan did not complete successfully")
            if execution.seeds != decision.accepted_seeds:
                issues.append("Observed training seeds do not match the exact accepted seeds")
            if execution.evaluation_count != decision.required_evaluations:
                issues.append("Observed evaluation count does not match the exact accepted count")
            def identity(item):
                return json.dumps(item.model_dump(mode="json"), sort_keys=True)
            requested = {identity(item) for item in decision.plan.diagnostics}
            completed = {identity(item) for item in execution.completed_diagnostics}
            supported = all(
                item.kind in negotiation.supported_diagnostics
                and (item.kind != "metric" or item.metric in negotiation.declared_metrics)
                for item in execution.completed_diagnostics
            )
            if supported:
                supported = _verified_diagnostics(store, decision, execution, negotiation)
            if not supported:
                issues.append(
                    "Execution reports a diagnostic outside the trusted evaluator contract"
                )
            if not requested <= completed:
                issues.append("At least one requested diagnostic has no completed measurement")
            lanes = 2 if decision.plan.execution_kind == "paired_control" else 1
            ids_valid = (
                len(set(execution.observation_ids)) == len(execution.observation_ids)
                and len(execution.observation_ids) == lanes
            )
            if not ids_valid:
                issues.append("Observation references do not match the declared execution kind")
            # A partial record may support a limitation, never completion of the
            # exact request. The controller supplies IDs of measured results.
            if (
                execution.status != "not_run"
                and execution.evaluation_count > 0
                and execution.seeds
                and len(set(execution.seeds)) == len(execution.seeds)
                and set(execution.seeds) <= set(decision.accepted_seeds)
                and supported
                and requested & completed
                and 0 < len(execution.observation_ids) <= lanes
                and len(set(execution.observation_ids)) == len(execution.observation_ids)
            ):
                observed = list(execution.observation_ids)
        plan_results[decision.plan.id] = {
            "complete": decision.status == "accepted" and not issues,
            "observation_ids": observed,
            "reasons": issues,
        }

    coverage = []
    for concern in negotiation.concerns:
        plans = [
            item.plan.id for item in negotiation.decisions if concern.id in item.plan.concern_ids
        ]
        refs = list(
            dict.fromkeys(
                ref for identifier in plans for ref in plan_results[identifier]["observation_ids"]
            )
        )
        reasons = [reason for identifier in plans for reason in plan_results[identifier]["reasons"]]
        reasons.extend(
            item.reason
            for item in negotiation.unaddressed_concerns
            if concern.id in item.concern_ids
        )
        all_complete = bool(plans) and all(
            plan_results[identifier]["complete"] for identifier in plans
        )
        status = "executed" if all_complete else "partial" if refs else "unfulfilled"
        if not plans and not reasons:
            reasons.append("No plan or recorded disposition addresses this concern")
        coverage.append(
            ConcernCoverage(
                concern_id=concern.id,
                review_key=concern.review_key,
                status=status,
                plan_ids=plans,
                observation_ids=refs,
                reasons=list(dict.fromkeys(reasons)),
            )
        )
    return coverage


def negotiate(
    request: SupplementaryPlans,
    concerns: list[Concern],
    available_fresh_seeds: list[int],
    declared_metrics: list[str],
    supported_diagnostics: list[str] | None = None,
) -> SupplementaryNegotiation:
    """Validate the whole request without I/O, model calls, or candidate execution.

    Unsupported individual plans are explicitly unfulfilled; valid independent
    plans may still run. An incomplete or contradictory concern mapping rejects
    the entire envelope. Source-tree isolation and experiment-budget admission
    remain mandatory controller checks before executing any accepted plan.
    """
    if len({concern.id for concern in concerns}) != len(concerns):
        raise ValueError("Supplied review concern IDs must be unique")
    if len(set(available_fresh_seeds)) != len(available_fresh_seeds) or any(
        type(seed) is not int or seed < 0 for seed in available_fresh_seeds
    ):
        raise ValueError("Available fresh seeds must be distinct nonnegative integers")
    if not declared_metrics or any(not metric.strip() for metric in declared_metrics):
        raise ValueError("Declared metrics must be nonempty names")
    if len(set(declared_metrics)) != len(declared_metrics):
        raise ValueError("Declared metrics must be unique")
    supported_diagnostics = ["metric"] if supported_diagnostics is None else list(supported_diagnostics)
    if len(set(supported_diagnostics)) != len(supported_diagnostics):
        raise ValueError("Supported diagnostic kinds must be distinct")

    known = {concern.id for concern in concerns}
    planned = {item for plan in request.plans for item in plan.concern_ids}
    refused = {
        item for disposition in request.unaddressed_concerns for item in disposition.concern_ids
    }
    errors = []
    if (planned | refused) - known:
        errors.append("The response refers to unknown review concern IDs")
    if known - (planned | refused):
        errors.append("Every review concern must have a plan or explicit unaddressed disposition")
    if planned & refused:
        errors.append("A concern cannot be both planned and explicitly unaddressed")

    decisions = []
    for plan in request.plans:
        reasons = list(errors)
        if plan.evaluation_split != "development":
            reasons.append("Supplementary refinement cannot access protected final evaluation data")
        if plan.seed_count > len(available_fresh_seeds):
            reasons.append(
                f"Requested {plan.seed_count} fresh training seeds; the configured allowance supports "
                f"{len(available_fresh_seeds)}. No truncated substitute will execute"
            )
        if plan.execution_kind == "repeat" and plan.changed_file is not None:
            reasons.append("An unchanged repeat cannot declare a changed file")
        if plan.execution_kind == "paired_control" and not plan.changed_file:
            reasons.append("A paired control must declare one existing changed file")
        for diagnostic in plan.diagnostics:
            if diagnostic.metric is not None and diagnostic.metric not in declared_metrics:
                reasons.append("Diagnostic metric is absent from the declared evaluator contract")
            if diagnostic.kind not in supported_diagnostics or diagnostic.kind in {"other", "orientation_consistency"}:
                reasons.append(
                    f"Diagnostic {diagnostic.kind!r} has no supported evaluator; it will not be substituted with a metric"
                )
            elif diagnostic.kind == "metric" and diagnostic.metric not in declared_metrics:
                reasons.append(
                    f"Requested metric {diagnostic.metric!r} is not declared by the trusted evaluator"
                )
            elif diagnostic.kind in {"paired_interval", "paired_differences"}:
                if plan.execution_kind != "paired_control":
                    reasons.append("Paired diagnostics require matched baseline and control executions")
                if diagnostic.metric is not None and diagnostic.metric not in declared_metrics:
                    reasons.append("Paired diagnostic metric is absent from the evaluator contract")
                if diagnostic.kind == "paired_interval":
                    if plan.seed_count < 3:
                        reasons.append("Paired uncertainty requires at least three distinct matched seeds")
                    allowed = {"confidence", "resamples", "diagnostic_seed"}
                    parameters = diagnostic.parameters
                    confidence = parameters.get("confidence", .95)
                    if (set(parameters) - allowed or parameters.get("resamples", 10000) != 10000
                            or parameters.get("diagnostic_seed", 1729) != 1729
                            or type(confidence) not in {int, float} or not 0 < confidence < 1):
                        reasons.append("Paired uncertainty requires supported exact parameters; no precision substitution")
                elif diagnostic.parameters:
                    reasons.append("Paired differences have no optional parameters")
            elif diagnostic.parameters:
                reasons.append("This declared diagnostic accepts no additional parameters")
        accepted = not reasons
        decisions.append(
            SupplementaryDecision(
                plan=plan.model_copy(deep=True),
                status="accepted" if accepted else "unfulfilled",
                reasons=reasons,
                accepted_seeds=list(available_fresh_seeds[: plan.seed_count]) if accepted else [],
                requested_seed_count=plan.seed_count,
                supported_seed_count=len(available_fresh_seeds),
                required_evaluations=plan.seed_count
                * (2 if plan.execution_kind == "paired_control" else 1)
                if accepted
                else 0,
            )
        )

    unaddressed = [item.model_copy(deep=True) for item in request.unaddressed_concerns]
    missing = sorted(known - (planned | refused))
    if missing:
        unaddressed.append(
            ConcernDisposition(
                concern_ids=missing,
                reason="Agent omitted a plan or explicit disposition for these concerns",
            )
        )
    return SupplementaryNegotiation(
        contract_valid=not errors,
        contract_errors=errors,
        concerns=[concern.model_copy(deep=True) for concern in concerns],
        available_fresh_seeds=list(available_fresh_seeds),
        declared_metrics=list(declared_metrics),
        supported_diagnostics=supported_diagnostics,
        decisions=decisions,
        unaddressed_concerns=unaddressed,
    )


def _verified_diagnostics(store, decision, execution, negotiation):
    """Nonmetric completion requires actual committed artifacts, never a kind name."""
    nonmetric = [item for item in execution.completed_diagnostics if item.kind != "metric"]
    if not execution.completed_diagnostics:
        return True
    if not nonmetric and store is None:
        return True  # Historical callers preserve their original metric-only contract.
    if store is None:
        return False
    from ..research.diagnostics import DiagnosticResult, committed_observation
    from ..store import IntegrityError
    observations = {}
    for identifier in execution.observation_ids:
        row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("development/" + identifier,)).fetchone()
        if row is None:
            return False
        value, protocol = committed_observation(store, row[0])
        if protocol["seeds"] != execution.seeds:
            return False
        observations[row[0]] = value
    for diagnostic in execution.completed_diagnostics:
        if diagnostic.kind == "metric" and any(diagnostic.metric not in (value.metric_values or {value.metric: value.values}) for value in observations.values()):
            return False
    measured = []
    for ref in execution.diagnostic_refs:
        result = DiagnosticResult.model_validate(store.json(ref))
        row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("research-diagnostic/" + result.request.id,)).fetchone()
        if row is None or row[0] != ref:
            raise IntegrityError("Coverage diagnostic is not its committed result")
        if (result.status != "success" or result.request.expected_seeds != execution.seeds
                or result.request.concern_ids != decision.plan.concern_ids
                or result.request.evaluation_split != "development"):
            continue
        linked = set(result.input_refs) & set(observations)
        if not linked or result.request.kind in {"paired_interval", "paired_differences"} and linked != set(observations):
            continue
        measured.append((result.request, linked))
    for item in nonmetric:
        linked = set().union(*(refs for result, refs in measured
            if result.kind == item.kind and result.parameters == item.parameters
            and (item.metric is None or result.metric == item.metric)))
        if linked != set(observations):
            return False
    return True
