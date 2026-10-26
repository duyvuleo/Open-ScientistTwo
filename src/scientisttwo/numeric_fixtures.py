"""Scripted hypotheses for measured regression and optimization experiments."""

from statistics import mean
from .models import Idea


def response(adapter, role, inputs):
    metric = adapter.primary
    if role in {"extract_limitations", "verify_limitations"}:
        return {
            "items": [
                "The fixed baseline has limited capacity or an insufficient optimization budget.",
                "A development improvement may not transfer to the frozen test set.",
            ],
            "sufficient": role == "verify_limitations",
        }
    if role == "seed_ideas":
        if adapter.name == "regression":
            candidates = [
                ("polynomial", {"degree": 2, "strength": 0.1}),
                ("forest", {}),
                ("svr", {"strength": 10}),
                ("ridge", {"strength": 0.1}),
                ("polynomial", {"degree": 3, "strength": 1}),
            ]
        else:
            candidates = [
                ("projected_gradient", {"steps": 10}),
                ("projected_gradient", {"steps": 30, "momentum": 0.5}),
                ("projected_gradient", {"steps": 10, "project": False}),
                ("projected_gradient", {"steps": 50}),
                ("projected_gradient", {"steps": 100}),
            ]
        return {
            "ideas": [
                Idea(
                    id=f"seed-{i}",
                    method=method,
                    parameters=params,
                    hypothesis="Test an established alternative under the fixed protocol.",
                    novelty=5 - i,
                ).model_dump()
                for i, (method, params) in enumerate(candidates[: inputs["count"]])
            ]
        }
    if role == "novelty":
        return {
            "score": inputs["idea"]["novelty"],
            "references": [x["id"] for x in inputs["sources"]],
            "scope": "Scripted exploration ranking; established methods with no novelty claim",
        }
    if role == "implement":
        return {
            "source": adapter.source(Idea.model_validate(inputs["idea"])),
            "description": "Explicit offline fixture implementation",
        }
    if role == "critic":
        c = inputs["comparison"]
        decision = (
            "Good"
            if c["valid"] and c["improved"] and c["guardrails_pass"]
            else ("Engineer" if inputs["can_engineer"] else "Bad")
        )
        return {"decision": decision, "reason": c["reason"]}
    if role in {"engineer", "evolve", "refine"}:
        if "idea" in inputs:
            parent = inputs["idea"]
        else:
            usable = [
                t
                for t in inputs["traces"]
                if t["result"]["status"] == "success" and t["decision"]["decision"] == "Good"
            ]
            usable.sort(key=lambda t: mean(r["metrics"][metric] for r in t["result"]["rows"]))
            parent = (usable or inputs["traces"])[0]["idea"]
        idea = Idea.model_validate(parent)
        p = adapter.parameters.model_validate(idea.parameters).model_dump()
        if adapter.name == "optimization":
            p.update(steps=min(p["steps"] * 2, 200), project=True)
        elif idea.method == "polynomial":
            p.update(degree=min(3, p["degree"] + 1))
        elif idea.method == "forest":
            p.update(depth=min(16, p["depth"] + 2))
        else:
            p.update(strength=min(100, p["strength"] * 2))
        return idea.model_copy(
            update={
                "id": inputs["new_id"],
                "parameters": p,
                "parent_ids": [idea.id],
                "origin": "evolved" if role == "evolve" else "engineered",
            }
        ).model_dump()
    if role == "select":
        best = min(
            inputs["candidates"],
            key=lambda t: mean(r["metrics"][metric] for r in t["result"]["rows"]),
        )
        return {
            "idea_id": best["idea"]["id"],
            "reason": "Best primary metric among valid full-development trials",
        }
    if role in {"plan_ablations", "plan_rebuttal"}:
        idea = Idea.model_validate(inputs["idea"])
        params = adapter.parameters.model_validate(idea.parameters).model_dump()
        component = "repeat"
        if role == "plan_ablations":
            if adapter.name == "optimization":
                if params["momentum"]:
                    params["momentum"], component = 0, "momentum"
                else:
                    params["project"], component = False, "projection"
            elif idea.method == "polynomial" and params["degree"] > 1:
                params["degree"], component = 1, "polynomial_features"
            else:
                params["scaled"], component = not params["scaled"], "scaling"
        revised = idea.model_copy(
            update={
                "id": inputs["new_id"],
                "parameters": params,
                "parent_ids": [idea.id],
                "origin": "ablation",
            }
        )
        return {
            "plans": [
                {
                    "idea": revised.model_dump(),
                    "changed_component": component,
                    "purpose": "Controlled component check"
                    if role == "plan_ablations"
                    else "Additional training-seed repetitions on fixed development rows",
                }
            ]
        }
    if role == "ablation_critic":
        return {
            "decision": "Refine"
            if any(c["improved"] and c["guardrails_pass"] for c in inputs["comparisons"])
            else "Good",
            "reason": "Use measured ablation evidence; retain unsuccessful controls",
        }
    if role == "draft":
        claims = [
            {
                "experiment_id": r["id"],
                "dataset": row["dataset"],
                "seed": row["seed"],
                "metric": metric,
                "value": row["metrics"][metric],
            }
            for r in inputs["evidence"]
            if r["status"] == "success"
            for row in r["rows"]
        ]
        return {
            "title": adapter.name + " offline engineering validation",
            "claims": claims,
            "synthetic": True,
        }
    if role == "review":
        return {
            "score": 8 if inputs["has_rebuttal"] else 7,
            "synthetic": True,
            "concerns": []
            if inputs["has_rebuttal"]
            else ["Check reproducibility with additional training seeds."],
        }
    if role == "meta_review":
        refine = bool(adapter.task and adapter.task.settings.get("exercise_meta_refinement", False))
        return {
            "decision": "Refine" if refine else "Accept",
            "reason": "Declared scripted meta-refinement control"
            if refine
            else "Synthetic workflow acceptance, not publication readiness",
        }
    raise ValueError(role)
