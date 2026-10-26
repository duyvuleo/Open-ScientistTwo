"""Synthetic minimization task: tests the general controller without ML or code execution."""

import time
from pathlib import Path

from scientisttwo.evaluation import compare
from scientisttwo.models import EvalSpec, ExperimentResult, Idea, MetricRow
from scientisttwo.store import canonical, digest


class ScalarTask:
    name = "scalar-test"
    worker_path = Path(__file__)

    def __init__(
        self,
        *,
        reject=False,
        refine_ablation=False,
        refine_meta=False,
        improved_revision=True,
        engineering=False,
    ):
        self.reject = reject
        self.refine_ablation = refine_ablation
        self.refine_meta = refine_meta
        self.improved_revision = improved_revision
        self.engineering = engineering
        self.received = []

    def split_manifest(self, task):
        return {
            "version": 2,
            "datasets": {
                d: {
                    "count": 10,
                    "data_hash": "synthetic-test-only",
                    "train": list(range(6)),
                    "development": [6, 7],
                    "test": [8, 9],
                }
                for d in task.datasets
            },
        }

    def baseline(self):
        return Idea(
            id="baseline", method="quadratic", hypothesis="Reference", parameters={"loss": 10}
        )

    def evaluation(self, task, phase):
        return EvalSpec.for_phase(task, phase)

    def identity(self):
        return {"adapter": "synthetic-scalar-v1"}

    def sources(self):
        return [{"id": "fixture", "title": "Synthetic test source", "verification": "unverified"}]

    def score(self, spec, observations):
        return [MetricRow.model_validate(row) for row in observations["rows"]]

    def compare(self, a, b, task):
        return compare(a, b, task)

    def audit_method(self, idea, source):
        return "pass" if source == "SYNTHETIC TEST ONLY" else "unverified"

    def validate_plan(self, incumbent, plan):
        if plan.changed_component not in {"coefficient", "repeat"}:
            raise ValueError("Unexpected test component")

    def fixture_response(self, role, inputs):
        self.received.append((role, inputs))
        if role in {"extract_limitations", "verify_limitations"}:
            return {"items": ["Synthetic limitation"], "sufficient": True}
        if role == "seed_ideas":
            return {
                "ideas": [
                    Idea(
                        id=f"seed-{i}",
                        hypothesis="Synthetic hypothesis",
                        method="quadratic",
                        parameters={"loss": 8 + i},
                        novelty=5 - i,
                    ).model_dump()
                    for i in range(inputs["count"])
                ]
            }
        if role == "novelty":
            return {
                "score": inputs["idea"]["novelty"],
                "references": ["fixture"],
                "scope": "Synthetic",
            }
        if role == "implement":
            return {"source": "SYNTHETIC TEST ONLY", "description": "Never executed"}
        if role == "critic":
            decision = (
                "Bad"
                if self.reject
                else ("Engineer" if self.engineering and inputs["can_engineer"] else "Good")
            )
            return {"decision": decision, "reason": "Synthetic branch test"}
        if role in {"evolve", "refine", "engineer"}:
            parent = inputs.get("idea") or inputs["traces"][0]["idea"]
            loss = parent["parameters"]["loss"] - 1 if self.improved_revision else 11
            return Idea(
                id=inputs["new_id"],
                hypothesis="Synthetic revision",
                method="quadratic",
                parameters={"loss": loss},
                parent_ids=[parent["id"]],
                origin="evolved",
            ).model_dump()
        if role == "select":
            best = min(inputs["candidates"], key=lambda x: x["idea"]["parameters"]["loss"])
            return {"idea_id": best["idea"]["id"], "reason": "Minimize loss"}
        if role in {"plan_ablations", "plan_rebuttal"}:
            idea = Idea.model_validate(inputs["idea"])
            idea = idea.model_copy(
                update={
                    "id": inputs["new_id"],
                    "parent_ids": [idea.id],
                    "parameters": {"loss": idea.parameters["loss"] + 1},
                }
            )
            return {
                "plans": [
                    {
                        "idea": idea.model_dump(),
                        "purpose": "Synthetic check",
                        "changed_component": "coefficient"
                        if role == "plan_ablations"
                        else "repeat",
                    }
                ]
            }
        if role == "ablation_critic":
            return {"decision": "Refine" if self.refine_ablation else "Good", "reason": "Synthetic"}
        if role == "draft":
            return {"title": "Synthetic draft", "claims": [], "synthetic": True}
        if role == "review":
            return {
                "score": 8 if inputs["has_rebuttal"] else 7,
                "concerns": ["Repeat"],
                "synthetic": True,
            }
        if role == "meta_review":
            return {"decision": "Refine" if self.refine_meta else "Accept", "reason": "Synthetic"}
        raise ValueError(role)


class SyntheticRunner:
    def __init__(self, store, *, fail_baseline=False, interrupt_at=None):
        self.store = store
        self.fail_baseline = fail_baseline
        self.interrupt_at = interrupt_at
        self.executed = []

    def preflight(self):
        pass

    def execute(self, key, idea, source, spec, **kwargs):
        request = {"idea": idea.model_dump(), "source": source, "spec": spec.model_dump()}
        cached = self.store.cached("test-run/" + key, request)
        if cached:
            return ExperimentResult.model_validate(cached)
        if key == self.interrupt_at:
            self.interrupt_at = None
            raise KeyboardInterrupt
        self.executed.append(key)
        loss = idea.parameters["loss"]
        rows = [
            MetricRow(
                dataset=d,
                seed=s,
                metrics={"loss": loss},
                training_seconds=0.1,
                inference_seconds=0.01,
                count=10,
            )
            for d in spec.datasets
            for s in spec.seeds
        ]
        failed = self.fail_baseline and idea.id == "baseline"
        result = ExperimentResult(
            id=key,
            idea_id=idea.id,
            code_ref=self.store.put(source.encode(), "py"),
            spec_ref=self.store.put_json(spec),
            split_manifest_ref=self.store.get("split_manifest_ref"),
            protocol_hash=digest(canonical(spec)),
            status="execution_failed" if failed else "success",
            rows=[] if failed else rows,
            observations_ref=None
            if failed
            else self.store.put_json({"rows": [r.model_dump() for r in rows]}),
            duration_seconds=0.11,
            measurement_origin="synthetic_test",
        )
        ref = self.store.put_json(result)
        with self.store.db:
            self.store.db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?,?)",
                (key, digest(canonical(request)), result.status, time.time(), 1, 0.11, ref),
            )
        self.store.cache("test-run/" + key, request, result)
        return result
