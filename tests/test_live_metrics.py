"""Synthetic metric-contract traces; these are not experimental outcomes."""

import pytest
from pydantic import ValidationError

from scientisttwo.live.evaluation import (
    aggregate_scores,
    compare_observations,
    metric_specs,
    normalize_scores,
    observation_metrics,
)
from scientisttwo.live.models import Candidate, DevelopmentSpec, MetricSpec, Observation, SourceFile


def config(**changes):
    return DevelopmentSpec(adapter="custom", metric="utility", direction="maximize", **changes)


def observation(spec, rows, **changes):
    aggregates, values = aggregate_scores(spec, rows)
    fields = {
        "id": "synthetic",
        "phase": "full",
        "status": "success",
        "protocol_hash": "same-protocol",
        "code_ref": "synthetic-source",
        "metric": spec.metric,
        "value": aggregates[spec.metric],
        "values": values[spec.metric],
        "metrics": aggregates,
        "metric_values": values,
        "prediction_refs": [f"synthetic-prediction-{i}" for i in range(len(rows))],
    }
    fields.update(changes)
    return Observation(**fields)


def test_custom_direction_required_and_conflict_rejected():
    with pytest.raises(ValidationError, match="explicit direction"):
        DevelopmentSpec(metric="custom_objective")
    spec = DevelopmentSpec(
        metric="custom_objective", metric_specs={"custom_objective": {"direction": "minimize"}}
    )
    assert metric_specs(spec)["custom_objective"].direction == "minimize"
    with pytest.raises(ValidationError, match="conflicts"):
        DevelopmentSpec(
            metric="utility",
            direction="maximize",
            metric_specs={"utility": {"direction": "minimize"}},
        )


@pytest.mark.parametrize(
    "name,direction",
    [
        ("process_seconds", "minimize"),
        ("rmse", "minimize"),
        ("log_loss", "minimize"),
        ("roc_auc", "maximize"),
        ("accuracy", "maximize"),
    ],
)
def test_legacy_direction_without_mutating_manifest(name, direction):
    spec = DevelopmentSpec(metric=name)
    before = spec.model_dump()
    assert metric_specs(spec)[name].direction == direction
    assert spec.model_dump() == before


@pytest.mark.parametrize("name", ["", " ", " score", "score "])
def test_metric_names_nonempty(name):
    with pytest.raises(ValidationError):
        DevelopmentSpec(metric=name, direction="maximize")


def test_metric_contract_bounds_and_aggregation():
    with pytest.raises(ValidationError, match="hard_min"):
        MetricSpec(direction="maximize", hard_min=3, hard_max=2)
    with pytest.raises(ValidationError):
        MetricSpec(direction="maximize", tolerance=float("nan"))
    with pytest.raises(ValidationError):
        config(aggregation="median")
    assert aggregate_scores(config(), [0.2, 0.6]) == ({"utility": 0.4}, {"utility": [0.2, 0.6]})
    with pytest.raises(ValueError, match="at least one"):
        aggregate_scores(config(), [])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "0.5", True, None])
def test_no_nonfinite_or_coerced_evaluator_scores(value):
    with pytest.raises(ValueError):
        normalize_scores(config(), value)


def test_declared_metrics_are_complete_and_no_unexpected_scores():
    spec = config(metric_specs={"latency": {"direction": "minimize"}})
    for bad in [1.0, {"utility": 1.0}, {"utility": 1.0, "latency": 1.0, "invented": 2.0}]:
        with pytest.raises(ValueError):
            normalize_scores(spec, bad)
    assert normalize_scores(spec, {"utility": 0.8, "latency": 3}) == {
        "latency": 3.0,
        "utility": 0.8,
    }


def test_comparison_uses_custom_objective_and_minimize_direction():
    spec = DevelopmentSpec(metric="energy", direction="minimize", epsilon=0.01)
    baseline, candidate = observation(spec, [8.0, 10.0]), observation(spec, [7.0, 9.0])
    comparison = compare_observations(spec, candidate, baseline)
    assert comparison["valid"] and comparison["improved"]
    assert comparison["delta"] == 1.0


def test_guardrail_vetoes_better_primary_metric():
    spec = config(metric_specs={"accuracy": {"direction": "maximize", "max_degradation": 0.01}})
    baseline = observation(spec, [{"utility": 0.6, "accuracy": 0.9}])
    candidate = observation(spec, [{"utility": 0.8, "accuracy": 0.88}])
    comparison = compare_observations(spec, candidate, baseline)
    assert comparison["delta"] > 0
    assert not comparison["valid"] and not comparison["improved"]
    assert comparison["constraint_failures"] == ["accuracy exceeds max_degradation"]


def test_minimize_guardrail_and_hard_bounds_apply_to_secondary_metrics():
    spec = config(
        metric_specs={
            "latency": {
                "direction": "minimize",
                "hard_min": 0,
                "hard_max": 2,
                "max_degradation": 0.2,
            }
        }
    )
    baseline = observation(spec, [{"utility": 0.6, "latency": 1.0}])
    for latency, text in [
        (-0.1, "below hard_min"),
        (2.1, "above hard_max"),
        (1.3, "max_degradation"),
    ]:
        candidate = observation(spec, [{"utility": 0.8, "latency": latency}])
        result = compare_observations(spec, candidate, baseline)
        assert not result["improved"]
        assert text in result["reason"]


def test_candidate_may_repair_infeasible_baseline():
    spec = DevelopmentSpec(
        metric="energy",
        direction="minimize",
        metric_specs={
            "energy": {"direction": "minimize", "hard_max": 2.0},
        },
    )
    assert compare_observations(spec, observation(spec, [1.5]), observation(spec, [3.0]))[
        "improved"
    ]


def test_tolerance_for_guardrail_and_strict_primary_epsilon():
    spec = config(
        epsilon=0.1,
        metric_specs={
            "latency": {"direction": "minimize", "max_degradation": 0.1, "tolerance": 0.001},
        },
    )
    baseline = observation(spec, [{"utility": 0.0, "latency": 1.0}])
    at_epsilon = observation(spec, [{"utility": 0.1, "latency": 1.1005}])
    above_epsilon = observation(spec, [{"utility": 0.11, "latency": 1.1005}])
    assert compare_observations(spec, at_epsilon, baseline)["valid"]
    assert not compare_observations(spec, at_epsilon, baseline)["improved"]
    assert compare_observations(spec, above_epsilon, baseline)["improved"]


@pytest.mark.parametrize("changes", [{"protocol_hash": "other"}, {"phase": "subset"}])
def test_incomparable_protocols_or_phases(changes):
    spec = config()
    assert not compare_observations(
        spec, observation(spec, [2.0], **changes), observation(spec, [1.0])
    )["valid"]


def test_failed_execution_has_no_scientific_scores():
    spec = config()
    failed = Observation(
        id="failed",
        phase="full",
        status="timeout",
        protocol_hash="same-protocol",
        code_ref="source",
        metric="utility",
    )
    assert not compare_observations(spec, failed, observation(spec, [1.0]))["valid"]
    with pytest.raises(ValidationError, match="failures"):
        Observation(**{**failed.model_dump(), "metrics": {"utility": 1.0}})


def test_legacy_observation_single_metric_only():
    spec = config()
    legacy = observation(spec, [1.0, 2.0]).model_dump()
    legacy.pop("metrics")
    legacy.pop("metric_values")
    result = Observation.model_validate(legacy)
    assert observation_metrics(spec, result) == {"utility": 1.5}
    multi = config(metric_specs={"latency": {"direction": "minimize"}})
    with pytest.raises(ValueError, match="secondary metrics"):
        observation_metrics(multi, result)


@pytest.mark.parametrize(
    "changes",
    [
        {"value": 99.0},
        {"values": [99.0]},
        {"metrics": {"other": 1.0}},
        {"metric_values": {"utility": [1.0, 2.0]}},
        {"metrics": {"utility": float("nan")}},
        {"metrics": {"utility": True}},
    ],
)
def test_observation_aliases_finite_and_shape_checked(changes):
    result = observation(config(), [1.0]).model_dump()
    with pytest.raises(ValidationError):
        Observation.model_validate({**result, **changes})


def test_fabricated_aggregate_and_evidence_cardinality_fail_comparison():
    spec = config()
    good = observation(spec, [1.0, 2.0])
    fabricated = good.model_copy(update={"value": 5.0, "metrics": {"utility": 5.0}})
    assert not compare_observations(spec, fabricated, good)["valid"]
    missing = good.model_copy(update={"prediction_refs": ["one-only"]})
    assert not compare_observations(spec, missing, good)["valid"]


def test_finite_inputs_cannot_produce_infinite_delta():
    spec = config()
    assert not compare_observations(spec, observation(spec, [1e308]), observation(spec, [-1e308]))[
        "valid"
    ]


@pytest.mark.parametrize(
    "path,command",
    [
        ("model.R", ["Rscript", "model.R"]),
        ("model.js", ["node", "model.js"]),
        ("Makefile", ["make"]),
        ("train.jl", ["julia", "train.jl"]),
    ],
)
def test_candidate_command_is_adapter_declared_not_python_only(path, command):
    assert (
        Candidate(
            files=[SourceFile(path=path, content="text")],
            command=command,
            description="Synthetic non-Python contract",
        ).command
        == command
    )


@pytest.mark.parametrize(
    "path", ["../escape.py", "/absolute.R", ".secret", "nested/.secret", "auth.json", "data.bin"]
)
def test_source_paths_and_credential_names_remain_restricted(path):
    with pytest.raises(ValidationError):
        SourceFile(path=path, content="text")


def test_source_is_bounded_utf8_text_and_argv_has_no_nul():
    for content in ["text\x00binary", "é" * 125_001]:
        with pytest.raises(ValidationError):
            SourceFile(path="model.R", content=content)
    for command in [[""], ["node", "model.js\x00other"]]:
        with pytest.raises(ValidationError):
            Candidate(
                files=[SourceFile(path="model.js", content="text")],
                command=command,
                description="Synthetic malformed command",
            )
