import pytest

from scientisttwo.models import TaskSpec, Idea, ExperimentPlan, ExperimentResult, MetricRow
from scientisttwo.numeric_tasks import RegressionTask, OptimizationTask
from scientisttwo.splits import prepare_arrays


def spec_for(adapter, name):
    task = TaskSpec(
        adapter=adapter.name,
        datasets=[name],
        subset_datasets=[name],
        seeds=[0],
        primary_metric=adapter.primary,
        metric_directions={adapter.primary: "minimize"},
        guardrails={},
    )
    adapter.configure(task)
    return adapter.evaluation(task, "full")


@pytest.mark.parametrize(
    "adapter,name", [(RegressionTask(), "quadratic"), (OptimizationTask(), "box-mild")]
)
def test_scores_come_from_observations_and_reject_fabricated_fields(adapter, name):
    spec = spec_for(adapter, name)
    _, seed, arrays, target = next(prepare_arrays(spec, adapter.loader))
    payload = {
        f"{name}:{seed}": {
            "predictions": target.tolist(),
            "row_ids": arrays["row_ids"].tolist(),
            "training_seconds": 0,
            "inference_seconds": 0.01,
        }
    }
    rows = adapter.score(spec, payload)
    assert abs(rows[0].metrics[adapter.primary]) < 1e-10
    payload[f"{name}:{seed}"]["score"] = 123
    with pytest.raises(ValueError, match="observations"):
        adapter.score(spec, payload)


def test_infeasible_optimization_cannot_win_with_better_objective():
    adapter = OptimizationTask()
    spec = spec_for(adapter, "simplex-mild")
    _, seed, arrays, target = next(prepare_arrays(spec, adapter.loader))
    d = target.shape[1]
    infeasible = arrays["x_eval"][:, 1 + d :]
    metrics = adapter.metrics(infeasible, target, arrays["x_eval"])
    assert metrics["objective_gap"] < 0 and metrics["max_violation"] > 1e-8

    def result(values):
        return ExperimentResult(
            id="x",
            idea_id="x",
            status="success",
            protocol_hash="same",
            code_ref="code",
            spec_ref="spec",
            observations_ref="obs",
            rows=[
                MetricRow(
                    dataset="simplex-mild",
                    seed=0,
                    metrics=values,
                    count=48,
                    training_seconds=0,
                    inference_seconds=0,
                )
            ],
        )

    comparison = adapter.compare(
        result(metrics), result({"objective_gap": 1, "max_violation": 0}), spec.task
    )
    assert comparison.valid and not comparison.improved and not comparison.guardrails_pass


def test_numeric_ablations_preserve_matched_components():
    adapter = RegressionTask()
    incumbent = Idea(id="a", method="polynomial", hypothesis="fixture", parameters={"degree": 2})
    other = incumbent.model_copy(update={"id": "b", "parameters": {"degree": 1, "scaled": False}})
    with pytest.raises(ValueError, match="isolate"):
        adapter.validate_plan(
            incumbent,
            ExperimentPlan(idea=other, purpose="fixture", changed_component="polynomial_features"),
        )
