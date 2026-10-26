"""Task extension boundary: the controller never assumes a dataset, model, or metric."""

from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

import numpy as np

from . import candidates, data, evaluation
from .models import (
    Comparison,
    EvalSpec,
    ExperimentPlan,
    ExperimentResult,
    Idea,
    MetricRow,
    TaskSpec,
)
from .store import digest
from .catalog import classification_data
from .splits import make_manifest, bind_spec


class TaskAdapter(Protocol):
    name: str
    worker_path: Path

    def baseline(self) -> Idea: ...
    def split_manifest(self, task: TaskSpec) -> dict: ...
    def evaluation(self, task: TaskSpec, phase: str) -> EvalSpec: ...
    def prepare_inputs(self, spec: EvalSpec, directory: Path) -> dict: ...
    def score(self, spec: EvalSpec, observations: dict) -> list[MetricRow]: ...
    def compare(
        self, candidate: ExperimentResult, baseline: ExperimentResult, task: TaskSpec
    ) -> Comparison: ...
    def validate_plan(self, incumbent: Idea, plan: ExperimentPlan) -> None: ...
    def identity(self) -> dict: ...
    def sources(self) -> list[dict]: ...
    def fixture_response(self, role: str, inputs: dict) -> dict: ...
    def audit_method(self, idea: Idea, source: str) -> str: ...


class ClassificationTask:
    name = "classification"
    worker_path = Path(__file__).with_name("worker.py")

    def baseline(self):
        return candidates.baseline_idea()

    def evaluation(self, task, phase):
        return bind_spec(EvalSpec.for_phase(task, phase), self.split_manifest(task))

    def split_manifest(self, task):
        return make_manifest(task, classification_data, stratified=True)

    def prepare_inputs(self, spec, directory):
        cells = []
        for dataset, seed, arrays, _ in data.prepare(spec):
            filename = f"{dataset}-{seed}.npz"
            np.savez(directory / filename, **arrays)
            cells.append({"key": f"{dataset}:{seed}", "seed": seed, "file": filename})
        return {"cells": cells}

    def score(self, spec, observations):
        return evaluation.score_predictions(spec, observations)

    def compare(self, candidate, baseline, task):
        return evaluation.compare(candidate, baseline, task)

    def validate_plan(self, incumbent, plan):
        original = candidates.FixtureParameters.model_validate(incumbent.parameters).model_dump()
        changed = candidates.FixtureParameters.model_validate(plan.idea.parameters).model_dump()
        if plan.idea.method != incumbent.method:
            raise ValueError("Matched control cannot change estimator family")
        differences = {k for k in original if original[k] != changed[k]}
        expected = {"scaled"} if plan.changed_component == "scaling" else set()
        if differences != expected or plan.changed_component not in {"scaling", "repeat"}:
            raise ValueError("Ablation does not isolate the declared component")

    def identity(self):
        directory = Path(__file__).parent
        return {
            name: digest((directory / name).read_bytes())
            for name in (
                "tasks.py",
                "data.py",
                "evaluation.py",
                "worker.py",
                "candidates.py",
                "fixtures.py",
                "splits.py",
                "catalog.py",
            )
        }

    def fixture_response(self, role, inputs):
        from .fixtures import classification_response

        return classification_response(role, inputs)

    def sources(self):
        from .fixtures import REFERENCES

        return REFERENCES

    def audit_method(self, idea, source):
        return "pass" if source == candidates.source_for(idea) else "unverified"


def load_task_adapter(name: str, task: TaskSpec | None = None) -> TaskAdapter:
    if name == "classification":
        return ClassificationTask()
    if name in {"regression", "optimization"}:
        from .numeric_tasks import RegressionTask, OptimizationTask

        adapter = RegressionTask() if name == "regression" else OptimizationTask()
        if task is not None:
            adapter.configure(task)
        return adapter
    entries = list(entry_points(group="scientisttwo.tasks", name=name))
    if len(entries) != 1:
        raise ValueError(f"Unknown or ambiguous task adapter: {name}")
    return entries[0].load()()
