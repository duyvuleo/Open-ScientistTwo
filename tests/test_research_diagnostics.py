"""Numerical tests use real NumPy computations; transport records are synthetic."""

import json

import numpy as np
import pytest

from scientisttwo.live.models import Observation
from scientisttwo.research.diagnostics import (
    ConsistencyDeclaration, DiagnosticInputs, DiagnosticRequest, classification_diagnostics,
    diagnostic_capabilities, input_consistency, paired_differences, paired_interval,
    regression_residuals, run_diagnostic, timing_distribution,
)
from scientisttwo.store import IntegrityError, Store, canonical, digest


def observation(store, identifier, values=(.5, .6, .7), *, seeds=(0, 1, 2), phase="full", extra=None):
    protocol = {"phase": phase, "seeds": list(seeds), "data": "synthetic", **(extra or {})}
    value = Observation(id=identifier, phase=phase, status="success", protocol_hash=digest(canonical(protocol)),
                        protocol_ref=store.put_json(protocol), code_ref=store.put_json({"code": identifier}),
                        metric="accuracy", value=float(np.mean(values)), values=list(values),
                        prediction_refs=[store.put_json({"synthetic_predictions": True})] * len(values))
    store.cache("development/" + identifier, {}, value)
    ref = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("development/" + identifier,)).fetchone()[0]
    return value, ref


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "campaign")
    yield value
    value.close()


def test_paired_bootstrap_has_reproducible_exact_seed_unit_and_figure_inputs():
    result = paired_interval([.1, .2, .3], [.2, .4, .7], [11, 12, 13])
    assert result == paired_interval([.1, .2, .3], [.2, .4, .7], [11, 12, 13])
    assert result["summary"]["mean_difference"] == pytest.approx(7 / 30)
    assert result["summary"]["resamples"] == 10_000
    assert result["summary"]["lower"] == pytest.approx(.1)
    assert result["summary"]["upper"] == pytest.approx(.4)
    assert sum(result["arrays"]["bootstrap_histogram_counts"]) == 10_000
    assert result["summary"]["resampling_unit"] == "distinct matched training seed"


@pytest.mark.parametrize("baseline,candidate,seeds", [([1, 2], [2, 3], [0, 1]),
    ([1, 2, 3], [2, 3, 4], [0, 0, 1]), ([1, np.nan, 3], [2, 3, 4], [0, 1, 2]),
    ([1, 2, 3], [2, 3], [0, 1, 2])])
def test_uncertainty_rejects_insufficient_nonfinite_unmatched_or_duplicate_seeds(baseline, candidate, seeds):
    with pytest.raises(ValueError):
        paired_interval(baseline, candidate, seeds)


@pytest.mark.parametrize("parameters", [{"resamples": 9999}, {"diagnostic_seed": 0}, {"confidence": float("nan")}])
def test_bootstrap_never_silently_changes_requested_precision(parameters):
    with pytest.raises(ValueError):
        paired_interval([0, 1, 2], [1, 2, 3], [0, 1, 2], **parameters)


def test_differences_do_not_assume_metric_direction():
    result = paired_differences([1, 2], [.5, 1.5], [0, 1])
    assert result["summary"]["mean_difference"] == -.5
    assert "improved" not in canonical(result).decode()


def test_classification_confusion_per_class_and_absent_class():
    result = classification_diagnostics([0, 0, 1, 1], [0, 1, 1, 1], classes=[0, 1, 2])
    assert result["arrays"]["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]
    assert result["summary"]["accuracy"] == .75
    assert result["tables"]["per_class"][1]["precision"] == pytest.approx(2 / 3)
    assert result["tables"]["per_class"][2]["f1"] == 0
    assert "targets" not in result and "predictions" not in result


@pytest.mark.parametrize("truth,predicted", [([0, np.inf], [0, 1]), ([0, .5], [0, 1]), ([0, 1], [0])])
def test_classification_requires_valid_matched_class_ids(truth, predicted):
    with pytest.raises(ValueError):
        classification_diagnostics(truth, predicted)


def test_regression_six_target_rmse_is_mean_of_target_rmse():
    actual = np.zeros((4, 6))
    predicted = np.broadcast_to(np.arange(1, 7), (4, 6))
    result = regression_residuals(actual, predicted, target_names=list("abcdef"))
    assert result["summary"]["mean_target_rmse"] == 3.5
    assert [row["rmse"] for row in result["tables"]["per_target_residuals"]] == [1, 2, 3, 4, 5, 6]
    assert all(sum(value) == 4 for key, value in result["arrays"].items() if key.endswith("counts"))


def test_timing_dispersion_retains_matching_samples_and_separates_startup():
    result = timing_distribution([1, 2, 3, 4], [.5, 1, 1.5, 2])
    assert result["summary"]["median_latency_ratio"] == .5
    assert result["tables"]["timing"][0]["iqr_seconds"] == 1.5
    assert "startup" in result["limitations"][0]


def test_consistency_requires_a_task_declaration_and_does_not_claim_general_invariance():
    declaration = ConsistencyDeclaration(id="reflect-v1", task_id="test", description="Declared test fixture",
                                        expectation="allclose", atol=.01, source_ref="frozen-contract")
    result = input_consistency([1, 2], [1.001, 2], declaration)
    assert result["summary"]["all_matched"]
    assert "does not establish generic invariance" in result["limitations"][0]
    assert not input_consistency([1, 2], [1.1, 2], declaration)["summary"]["all_matched"]


def test_capabilities_require_both_task_declaration_and_available_input_kind():
    assert diagnostic_capabilities() == []
    assert diagnostic_capabilities(["paired_interval"]) == []
    assert [d.kind for d in diagnostic_capabilities(["paired_interval", "unknown"], input_kinds=["committed_observations"])] == ["paired_interval"]


def test_committed_pair_diagnostic_resume_is_idempotent_and_tampering_detected(store):
    _, baseline = observation(store, "baseline")
    _, candidate = observation(store, "candidate", (.6, .7, .8))
    request = DiagnosticRequest(id="review/interval", kind="paired_interval", metric="accuracy", expected_seeds=[0, 1, 2])
    inputs = DiagnosticInputs(observation_refs=[baseline, candidate], declared_kinds=["paired_interval"])
    result = run_diagnostic(store, request, inputs)
    assert result.status == "success" and result.artifact_ref
    calls = store.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    assert run_diagnostic(store, request, inputs) == result
    assert store.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == calls
    (store.root / baseline).write_text("{}")
    with pytest.raises(IntegrityError):
        run_diagnostic(store, request, inputs)


@pytest.mark.parametrize("issue", ["final", "seeds", "protocol", "precision", "metric", "unsupported"])
def test_requests_are_unfulfilled_instead_of_silently_reduced(store, issue):
    _, baseline = observation(store, "base")
    _, candidate = observation(store, "candidate", (.6, .7, .8), extra={"runtime": "different"} if issue == "protocol" else None)
    request = DiagnosticRequest(id="test", kind="paired_interval", metric="missing" if issue == "metric" else "accuracy",
        expected_seeds=list(range(10)) if issue == "seeds" else [0, 1, 2],
        evaluation_split="final" if issue == "final" else "development",
        parameters={"resamples": 100} if issue == "precision" else {})
    result = run_diagnostic(store, request, DiagnosticInputs(observation_refs=[baseline, candidate],
        declared_kinds=[] if issue == "unsupported" else ["paired_interval"]))
    assert result.status == "unfulfilled" and result.reasons
    assert result.request == request
    assert not result.tables and not result.arrays


def test_fabricated_registered_result_is_not_a_committed_observation(store):
    baseline, _ = observation(store, "base")
    forged = store.put_json({**baseline.model_dump(), "id": "forged"})
    request = DiagnosticRequest(id="test", kind="paired_interval", metric="accuracy", expected_seeds=[0, 1, 2])
    with pytest.raises(IntegrityError, match="committed"):
        run_diagnostic(store, request, DiagnosticInputs(observation_refs=[forged, forged], declared_kinds=["paired_interval"]))


def test_candidate_written_diagnostic_cannot_be_reported_as_trusted(store):
    request = DiagnosticRequest(id="class", kind="classification", expected_seeds=[0])
    ref = store.put_json({"request": request.model_dump(), "output": {"summary": {"accuracy": 1}}})
    with pytest.raises(IntegrityError, match="trusted scoring"):
        run_diagnostic(store, request, DiagnosticInputs(trusted_result_refs=[ref], declared_kinds=["classification"]))


def test_request_json_remains_free_of_unrequested_resampling_defaults():
    request = DiagnosticRequest(id="x", kind="paired_interval", metric="accuracy", expected_seeds=[0, 1, 2])
    assert json.loads(canonical(request))["parameters"] == {}


def test_timing_diagnostic_reads_actual_envelope_and_replays_measured_samples(tmp_path, monkeypatch):
    from test_paired_evaluation import setup, synthetic_pair
    from scientisttwo.live.paired_evaluation import execute_paired
    store, adapter, runner = setup(tmp_path, monkeypatch)
    try:
        candidate = adapter.baseline()
        candidate.files[0].content += "\n# Synthetic timing candidate\n"
        measured = execute_paired(runner, "timing", candidate, seeds=[0], run_pair=synthetic_pair)
        ref = store.put_json(measured)
        request = DiagnosticRequest(id="timing-distribution", kind="timing_distribution", expected_seeds=[0])
        result = run_diagnostic(store, request, DiagnosticInputs(observation_refs=[ref], declared_kinds=["timing_distribution"]))
        assert result.status == "success"
        assert result.tables["per_seed"][0]["median_latency_ratio"] == .5
        assert len(result.arrays["seed_0/baseline_seconds"]) == 10
        assert "warmup" not in result.arrays
        timing = store.json(measured.timing_refs[0])
        (store.root / timing["report_ref"]).write_text("{}")
        with pytest.raises(IntegrityError):
            run_diagnostic(store, request, DiagnosticInputs(observation_refs=[ref], declared_kinds=["timing_distribution"]))
    finally:
        store.close()
