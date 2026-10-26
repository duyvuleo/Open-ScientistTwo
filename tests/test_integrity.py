import pytest
from pydantic import ValidationError

from scientisttwo.data import prepare, split_indices
from scientisttwo.evaluation import compare, score_predictions
from scientisttwo.models import (
    EvalSpec,
    ExperimentResult,
    Idea,
    ManuscriptDraft,
    MetricRow,
    TaskSpec,
    ExperimentPlan,
)
from scientisttwo.reporting import validate_claims
from scientisttwo.store import IntegrityError, Store
from scientisttwo.tasks import ClassificationTask


def result(value=0.8, accuracy=0.8, protocol="same"):
    return ExperimentResult(
        id="experiment",
        idea_id="candidate",
        status="success",
        code_ref="code",
        spec_ref="spec",
        protocol_hash=protocol,
        observations_ref="observations",
        rows=[
            MetricRow(
                dataset="iris",
                seed=0,
                metrics={"macro_f1": value, "accuracy": accuracy},
                training_seconds=1,
                inference_seconds=0.1,
                count=30,
            )
        ],
    )


def test_comparison_guardrails_and_protocols():
    task = TaskSpec()
    assert compare(result(0.9), result(), task).improved
    assert not compare(result(0.9, 0.7), result(), task).improved
    assert not compare(result(protocol="changed"), result(), task).valid
    assert not compare(result(0.80001), result(), task).improved


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_metric_rejected(value):
    with pytest.raises(ValidationError):
        result(value)


def test_missing_primary_metric_rejected():
    bad = result().model_copy(deep=True)
    bad.rows[0].metrics.pop("macro_f1")
    assert not compare(bad, result(), TaskSpec()).valid


def test_artifacts_and_checkpoints_are_input_bound(tmp_path):
    store = Store(tmp_path)
    ref = store.put_json({"number": 1})
    store.cache("call", {"input": 1}, {"output": 2})
    with pytest.raises(IntegrityError, match="inputs changed"):
        store.cached("call", {"input": 2})
    (tmp_path / ref).write_text('{"number": 2}')
    assert store.verify()
    with pytest.raises(IntegrityError):
        store.read(ref)


def test_split_is_disjoint_and_candidate_has_no_eval_labels():
    from sklearn.datasets import load_iris

    _, labels = load_iris(return_X_y=True)
    train, dev, test = split_indices(labels, 0)
    assert (len(train), len(dev), len(test)) == (90, 30, 30)
    assert not set(train) & set(test) and not set(dev) & set(test) and not set(train) & set(dev)
    spec = EvalSpec.for_phase(TaskSpec(), "subset")
    _, _, arrays, y = next(prepare(spec))
    assert set(arrays) == {"x_train", "y_train", "x_eval", "row_ids"}
    assert len(arrays["x_train"]) == 45 and len(y) == 30


def test_trusted_scoring_checks_row_ids_and_schema():
    task = TaskSpec(datasets=["iris"], subset_datasets=["iris"], seeds=[0])
    spec = EvalSpec.for_phase(task, "full")
    _, _, arrays, labels = next(prepare(spec))
    observations = {
        "iris:0": {
            "row_ids": arrays["row_ids"].tolist(),
            "predictions": labels.tolist(),
            "training_seconds": 0.1,
            "inference_seconds": 0.1,
        }
    }
    assert score_predictions(spec, observations)[0].metrics["macro_f1"] == 1
    observations["iris:0"]["macro_f1"] = 0.99
    with pytest.raises(ValueError, match="scores are forbidden"):
        score_predictions(spec, observations)
    del observations["iris:0"]["macro_f1"]
    observations["iris:0"]["row_ids"] = list(reversed(arrays["row_ids"].tolist()))
    with pytest.raises(ValueError, match="row IDs changed"):
        score_predictions(spec, observations)


def test_claim_requires_actual_matching_evidence():
    draft = ManuscriptDraft(
        title="Test",
        claims=[
            dict(experiment_id="experiment", dataset="iris", seed=0, metric="macro_f1", value=0.999)
        ],
    )
    with pytest.raises(IntegrityError, match="Unsupported"):
        validate_claims(draft, [result()])


def test_ablation_cannot_change_multiple_components():
    task = ClassificationTask()
    idea = Idea(id="a", method="svc", hypothesis="Test", parameters={"strength": 3})
    bad = Idea(
        id="b",
        method="svc",
        hypothesis="Bad ablation",
        parameters={"strength": 10, "scaled": False},
    )
    with pytest.raises(ValueError, match="isolate"):
        task.validate_plan(
            idea, ExperimentPlan(idea=bad, purpose="Test", changed_component="scaling")
        )


def test_preprocessor_fits_training_only():
    from scientisttwo.candidates import source_for

    # Inspect the trusted fixture source/worker contract, never execute candidate code on the host.
    source = source_for(Idea(id="a", method="svc", hypothesis="Test"))
    assert "make_pipeline(StandardScaler(), estimator)" in source
    from pathlib import Path
    from scientisttwo import worker

    text = Path(worker.__file__).read_text()
    assert 'estimator.fit(arrays["x_train"], arrays["y_train"])' in text
    assert 'estimator.predict(arrays["x_eval"])' in text
