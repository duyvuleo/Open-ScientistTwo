"""Only trusted evaluator code can produce scientific metrics."""

from collections import defaultdict
from statistics import mean

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .data import prepare
from .models import Comparison, EvalSpec, ExperimentResult, MetricRow, TaskSpec


def score_predictions(spec: EvalSpec, predictions: dict) -> list[MetricRow]:
    expected_keys = {f"{name}:{seed}" for name in spec.datasets for seed in spec.seeds}
    if set(predictions) != expected_keys:
        raise ValueError("Prediction grid differs from the immutable protocol")
    rows = []
    for name, seed, arrays, labels in prepare(spec):
        payload = predictions[f"{name}:{seed}"]
        if set(payload) != {"row_ids", "predictions", "training_seconds", "inference_seconds"}:
            raise ValueError(
                "Unexpected prediction fields; candidate-supplied scores are forbidden"
            )
        values = np.asarray(payload["predictions"])
        ids = np.asarray(payload["row_ids"])
        if values.shape != labels.shape or not np.isfinite(values).all():
            raise ValueError("Missing, nonfinite, or incorrectly shaped predictions")
        if not np.array_equal(ids, arrays["row_ids"]):
            raise ValueError("Evaluation row IDs changed")
        if not set(values.tolist()) <= set(arrays["y_train"].tolist()):
            raise ValueError("Predictions contain unknown classes")
        rows.append(
            MetricRow(
                dataset=name,
                seed=seed,
                metrics={
                    "macro_f1": f1_score(labels, values, average="macro", zero_division=0),
                    "accuracy": accuracy_score(labels, values),
                },
                count=len(labels),
                training_seconds=payload["training_seconds"],
                inference_seconds=payload["inference_seconds"],
            )
        )
    return rows


def compare(candidate: ExperimentResult, baseline: ExperimentResult, task: TaskSpec) -> Comparison:
    if candidate.status != "success" or baseline.status != "success":
        return Comparison(valid=False, reason="Execution did not produce valid comparable evidence")
    if candidate.protocol_hash != baseline.protocol_hash:
        return Comparison(valid=False, reason="Evaluation protocols differ")

    def grid(r):
        return {(x.dataset, x.seed, x.count) for x in r.rows}

    if grid(candidate) != grid(baseline):
        return Comparison(valid=False, reason="Metric grids differ")
    c_acc, b_acc = defaultdict(list), defaultdict(list)
    for r in candidate.rows:
        for metric, value in r.metrics.items():
            c_acc[(r.dataset, metric)].append(value)
    for r in baseline.rows:
        for metric, value in r.metrics.items():
            b_acc[(r.dataset, metric)].append(value)
    required = {task.primary_metric, *task.guardrails}
    if any(not required <= r.metrics.keys() for r in candidate.rows + baseline.rows):
        return Comparison(valid=False, reason="Required metrics missing")

    def sign(metric):
        return 1 if task.metric_directions[metric] == "maximize" else -1

    guard = all(
        sign(metric) * (mean(c_acc[(dataset, metric)]) - mean(b_acc[(dataset, metric)]))
        >= -tolerance
        for dataset in {r.dataset for r in baseline.rows}
        for metric, tolerance in task.guardrails.items()
    )
    delta = sign(task.primary_metric) * (
        mean(r.metrics[task.primary_metric] for r in candidate.rows)
        - mean(r.metrics[task.primary_metric] for r in baseline.rows)
    )
    improved = guard and delta > task.improvement_epsilon
    return Comparison(
        valid=True,
        improved=improved,
        guardrails_pass=guard,
        delta=delta,
        reason="Valid improvement" if improved else "Improvement rule or guardrail not met",
    )


def quality_matches(a: ExperimentResult, b: ExperimentResult, tolerance: float) -> bool:
    if a.status != "success" or b.status != "success" or a.protocol_hash != b.protocol_hash:
        return False
    left = {(r.dataset, r.seed): r for r in a.rows}
    right = {(r.dataset, r.seed): r for r in b.rows}
    return left.keys() == right.keys() and all(
        left[k].metrics.keys() == right[k].metrics.keys()
        and all(abs(left[k].metrics[m] - right[k].metrics[m]) <= tolerance for m in left[k].metrics)
        and left[k].count == right[k].count
        for k in left
    )
