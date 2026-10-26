"""Task-independent trusted metric aggregation and comparison policy.

Adapters calculate scores from execution evidence. These helpers validate that
evidence against the frozen declaration; they never accept an agent's verdict as
an override. Legacy observations may omit the metric mappings only when the
protocol declares a single metric.
"""

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from statistics import fmean

from .models import DevelopmentSpec, LEGACY_METRIC_DIRECTIONS, LiveSpec, MetricSpec, Observation


def development(config: DevelopmentSpec | LiveSpec) -> DevelopmentSpec:
    return config.development if isinstance(config, LiveSpec) else config


def metric_specs(config: DevelopmentSpec | LiveSpec) -> dict[str, MetricSpec]:
    """Resolve the declaration without modifying legacy manifests in memory."""
    config = development(config)
    declared = dict(config.metric_specs)
    if config.metric not in declared:
        direction = config.direction or LEGACY_METRIC_DIRECTIONS.get(config.metric)
        if direction is None:
            raise ValueError("Custom primary metrics require an explicit direction")
        declared[config.metric] = MetricSpec(direction=direction)
    return declared


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Evaluator metrics must be numeric, not strings or booleans")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Evaluator returned a nonfinite metric")
    return value


def normalize_scores(config: DevelopmentSpec | LiveSpec, scores) -> dict[str, float]:
    """Require exactly the declared metrics from one successful execution."""
    config = development(config)
    declared = metric_specs(config)
    if not isinstance(scores, Mapping):
        if len(declared) != 1:
            raise ValueError("A multi-metric protocol requires all declared evaluator metrics")
        scores = {config.metric: scores}
    if set(scores) != set(declared):
        missing, unexpected = set(declared) - set(scores), set(scores) - set(declared)
        raise ValueError(
            f"Evaluator metric names differ from declaration: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected, key=str)}"
        )
    return {name: _number(scores[name]) for name in declared}


def aggregate_scores(
    config: DevelopmentSpec | LiveSpec, scores: Sequence
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Return mean aggregates and complete per-execution values, in that order."""
    if not scores:
        raise ValueError("Successful evaluation requires at least one execution")
    rows = [normalize_scores(config, row) for row in scores]
    values = {name: [row[name] for row in rows] for name in metric_specs(config)}
    try:
        metrics = {name: _number(fmean(series)) for name, series in values.items()}
    except OverflowError as error:
        raise ValueError("Metric aggregation overflowed") from error
    return metrics, values


def observation_metrics(
    config: DevelopmentSpec | LiveSpec, observation: Observation
) -> dict[str, float]:
    """Validate successful stored evidence and return its aggregate metric map."""
    config = development(config)
    if observation.status != "success":
        raise ValueError("Execution did not produce a successful observation")
    if observation.metric != config.metric:
        raise ValueError("Observation primary metric differs from the declared objective")
    if not observation.values or len(observation.prediction_refs) != len(observation.values):
        raise ValueError("Metric/prediction evidence cardinality mismatch")
    declared = metric_specs(config)
    if observation.metrics or observation.metric_values:
        scores = normalize_scores(config, observation.metrics)
        values = observation.metric_values
        if set(values) != set(declared):
            raise ValueError("Per-execution metric names differ from the declaration")
        if (
            scores[config.metric] != observation.value
            or values[config.metric] != observation.values
        ):
            raise ValueError("Primary metric aliases differ from complete metric evidence")
    else:
        if len(declared) != 1:
            raise ValueError("Successful observation is missing declared secondary metrics")
        scores = normalize_scores(config, observation.value)
        values = {config.metric: observation.values}
    for name, series in values.items():
        if len(series) != len(observation.values):
            raise ValueError("Every metric requires one value per execution")
        try:
            actual = _number(fmean(_number(value) for value in series))
        except OverflowError as error:
            raise ValueError("Metric aggregation overflowed") from error
        if not math.isclose(scores[name], actual, rel_tol=0, abs_tol=declared[name].tolerance):
            raise ValueError(f"Stored aggregate differs from the execution metrics: {name}")
    return scores


def compare_observations(
    config: DevelopmentSpec | LiveSpec, result: Observation, reference: Observation
) -> dict:
    """Compare mean objectives with candidate bounds and relative guardrails.

    Hard bounds constrain the candidate, including when the reference violates a
    bound: a candidate may repair an infeasible baseline. Guardrails compare each
    declared metric against this explicit reference. Tolerances apply to evidence
    consistency and constraint boundaries; numerical improvement remains strictly
    greater than the separately declared primary epsilon.
    """
    config = development(config)
    invalid = {"valid": False, "improved": False}
    if result.status != "success" or reference.status != "success":
        return {**invalid, "reason": "Invalid execution; scientific comparison unavailable"}
    if result.protocol_hash != reference.protocol_hash or result.phase != reference.phase:
        return {**invalid, "reason": "Incomparable evaluation protocols or phases"}
    try:
        candidate, baseline = (
            observation_metrics(config, result),
            observation_metrics(config, reference),
        )
    except ValueError as error:
        return {**invalid, "reason": str(error)}
    deltas, failures = {}, []
    for name, spec in metric_specs(config).items():
        value = candidate[name]
        try:
            delta = _number(value - baseline[name])
        except ValueError:
            return {**invalid, "reason": f"Metric comparison overflowed: {name}"}
        deltas[name] = delta if spec.direction == "maximize" else -delta
        if spec.hard_min is not None and value < spec.hard_min - spec.tolerance:
            failures.append(f"{name} is below hard_min")
        if spec.hard_max is not None and value > spec.hard_max + spec.tolerance:
            failures.append(f"{name} is above hard_max")
        if spec.max_degradation is not None:
            if deltas[name] < -spec.max_degradation - spec.tolerance:
                failures.append(f"{name} exceeds max_degradation")
    delta = deltas[config.metric]
    return {
        "valid": not failures,
        "improved": not failures and delta > config.epsilon,
        "delta": delta,
        "metric_deltas": deltas,
        "constraint_failures": failures,
        "reason": "; ".join(failures)
        if failures
        else "Declared development comparison; not statistical significance",
    }
