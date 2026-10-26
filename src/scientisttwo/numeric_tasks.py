"""Regression and constrained optimization adapters using the general controller/runner."""

from pathlib import Path
import numpy as np
from pydantic import Field

from . import catalog, evaluation
from .models import Record, Idea, EvalSpec, MetricRow, Comparison
from .splits import make_manifest, bind_spec, prepare_arrays
from .store import digest


class RegressionParameters(Record):
    scaled: bool = True
    degree: int = Field(default=2, ge=1, le=3)
    strength: float = Field(default=1, gt=0, le=100)
    depth: int = Field(default=8, ge=1, le=16)


class OptimizationParameters(Record):
    steps: int = Field(default=10, ge=1, le=200)
    rate: float = Field(default=1, gt=0, le=2)
    momentum: float = Field(default=0, ge=0, le=0.9)
    project: bool = True


def regression_source(idea):
    p = RegressionParameters.model_validate(idea.parameters)
    models = {
        "ridge": f"Ridge(alpha={p.strength!r})",
        "polynomial": f"make_pipeline(PolynomialFeatures(degree={p.degree}, include_bias=False), Ridge(alpha={p.strength!r}))",
        "forest": f"RandomForestRegressor(n_estimators=60, max_depth={p.depth}, random_state=seed, n_jobs=1)",
        "svr": f"SVR(C={p.strength!r}, epsilon=0.1)",
    }
    if idea.method not in models:
        raise ValueError("Unknown regression fixture method")
    return f"""# Scripted, established regression method; no novelty claim.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def build(seed):
    estimator = {models[idea.method]}
    return make_pipeline(StandardScaler(), estimator) if {p.scaled!r} else estimator
"""


def optimization_source(idea):
    p = OptimizationParameters.model_validate(idea.parameters)
    if idea.method != "projected_gradient":
        raise ValueError("Unknown optimization fixture method")
    return f"""# Scripted projected-gradient solver; scientific metrics are computed outside this module.
import numpy as np

def simplex_project(x):
    ordered = np.sort(x, axis=1)[:, ::-1]
    shifted = np.cumsum(ordered, axis=1) - 1
    ranks = np.arange(1, x.shape[1]+1)
    active = ordered - shifted/ranks > 0
    rho = active.sum(axis=1) - 1
    theta = shifted[np.arange(len(x)), rho] / (rho+1)
    return np.maximum(x-theta[:, None], 0)

class Solver:
    def fit(self, x, y):
        return self

    def predict(self, features):
        d = (features.shape[1]-1)//2
        mode, q, center = features[:, 0], features[:, 1:1+d], features[:, 1+d:]
        state = np.zeros_like(center)
        simplex = mode == 1
        state[simplex] = 1/d
        previous = state.copy()
        for _ in range({p.steps}):
            proposal = state - {p.rate!r} * q * (state-center) / q.max(axis=1)[:, None]
            proposal = proposal + {p.momentum!r} * (state-previous)
            if {p.project!r}:
                proposal[~simplex] = np.clip(proposal[~simplex], 0, 1)
                if simplex.any():
                    proposal[simplex] = simplex_project(proposal[simplex])
            previous, state = state, proposal
        return state

def build(seed):
    return Solver()
"""


class NumericTask:
    worker_path = Path(__file__).with_name("worker.py")
    task = None

    def configure(self, task):
        self.task = task
        if task.primary_metric != self.primary:
            raise ValueError(f"{self.name} requires primary_metric={self.primary}")
        for name in task.datasets:
            self.loader(name, task.data_seed)

    def split_manifest(self, task):
        return make_manifest(task, self.loader)

    def evaluation(self, task, phase):
        return bind_spec(EvalSpec.for_phase(task, phase), self.split_manifest(task))

    def prepare_inputs(self, spec, directory):
        cells = []
        for name, seed, arrays, _ in prepare_arrays(spec, self.loader):
            filename = f"{name}-{seed}.npz"
            np.savez(directory / filename, **arrays)
            cells.append({"key": f"{name}:{seed}", "seed": seed, "file": filename})
        return {"cells": cells}

    def score(self, spec, observations):
        expected = {f"{d}:{s}" for d in spec.datasets for s in spec.seeds}
        if set(observations) != expected:
            raise ValueError("Observation grid differs from the declared experiment")
        rows = []
        for name, seed, arrays, truth in prepare_arrays(spec, self.loader):
            item = observations[f"{name}:{seed}"]
            if set(item) != {"predictions", "row_ids", "training_seconds", "inference_seconds"}:
                raise ValueError("Worker must supply observations, not computed scores")
            values = np.asarray(item["predictions"])
            if values.shape != truth.shape or not np.isfinite(values).all():
                raise ValueError("Invalid observation shape or nonfinite values")
            if not np.array_equal(item["row_ids"], arrays["row_ids"]):
                raise ValueError("Evaluation row IDs changed")
            rows.append(
                MetricRow(
                    dataset=name,
                    seed=seed,
                    count=len(truth),
                    metrics=self.metrics(values, truth, arrays["x_eval"]),
                    training_seconds=item["training_seconds"],
                    inference_seconds=item["inference_seconds"],
                )
            )
        return rows

    def compare(self, candidate, baseline, task):
        return evaluation.compare(candidate, baseline, task)

    def identity(self):
        return {
            name: digest(Path(__file__).with_name(name).read_bytes())
            for name in (
                "numeric_tasks.py",
                "numeric_fixtures.py",
                "catalog.py",
                "splits.py",
                "evaluation.py",
                "worker.py",
            )
        }

    def sources(self):
        return [
            {
                "id": self.name + "-fixture",
                "title": self.name + " offline fixture specification",
                "verification": "local_specification",
                "note": "Generated control tasks; not an external novelty reference",
            }
        ]

    def fixture_response(self, role, inputs):
        from .numeric_fixtures import response

        return response(self, role, inputs)

    def audit_method(self, idea, source):
        return "pass" if source == self.source(idea) else "unverified"


class RegressionTask(NumericTask):
    name = "regression"
    primary = "mse"
    loader = staticmethod(catalog.regression_data)
    source = staticmethod(regression_source)
    parameters = RegressionParameters

    def baseline(self):
        return Idea(
            id="baseline",
            method="ridge",
            hypothesis="Fixed standardized linear regression",
            origin="baseline",
        )

    def metrics(self, prediction, truth, features):
        return {
            "mse": float(np.mean((prediction - truth) ** 2)),
            "mae": float(np.mean(np.abs(prediction - truth))),
        }

    def validate_plan(self, incumbent, plan):
        before = self.parameters.model_validate(incumbent.parameters).model_dump()
        after = self.parameters.model_validate(plan.idea.parameters).model_dump()
        differences = {key for key in before if before[key] != after[key]}
        expected = {"polynomial_features": {"degree"}, "scaling": {"scaled"}, "repeat": set()}
        if incumbent.method != plan.idea.method or differences != expected.get(
            plan.changed_component
        ):
            raise ValueError("Regression ablation does not isolate its declared component")


class OptimizationTask(NumericTask):
    name = "optimization"
    primary = "objective_gap"
    loader = staticmethod(catalog.optimization_data)
    source = staticmethod(optimization_source)
    parameters = OptimizationParameters

    def baseline(self):
        return Idea(
            id="baseline",
            method="projected_gradient",
            parameters={"steps": 1},
            hypothesis="One projected-gradient step with curvature-scaled step size",
            origin="baseline",
        )

    def metrics(self, prediction, truth, features):
        d = truth.shape[1]
        q, center = features[:, 1 : 1 + d], features[:, 1 + d :]

        def objective(x):
            return 0.5 * np.sum(q * (x - center) ** 2, axis=1)

        negative = np.maximum(-prediction, 0).max(axis=1)
        box = np.maximum(prediction - 1, 0).max(axis=1)
        simplex = np.abs(prediction.sum(axis=1) - 1)
        violation = np.maximum(negative, np.where(features[:, 0] == 0, box, simplex))
        return {
            "objective_gap": float(np.mean(objective(prediction) - objective(truth))),
            "max_violation": float(violation.max()),
        }

    def compare(self, candidate, baseline, task):
        result = super().compare(candidate, baseline, task)
        if result.valid and any(
            r.metrics.get("max_violation", float("inf")) > 1e-8 for r in candidate.rows
        ):
            return Comparison(
                valid=True,
                improved=False,
                guardrails_pass=False,
                reason="Hard feasibility violation: a constraint exceeds 1e-8",
            )
        return result

    def validate_plan(self, incumbent, plan):
        before = self.parameters.model_validate(incumbent.parameters).model_dump()
        after = self.parameters.model_validate(plan.idea.parameters).model_dump()
        differences = {key for key in before if before[key] != after[key]}
        expected = {"momentum": {"momentum"}, "projection": {"project"}, "repeat": set()}
        if incumbent.method != plan.idea.method or differences != expected.get(
            plan.changed_component
        ):
            raise ValueError("Optimization ablation does not isolate its declared component")
