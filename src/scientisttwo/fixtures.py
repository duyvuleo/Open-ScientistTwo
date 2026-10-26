"""Scripted classification agents, deliberately separate from the general controller."""

from statistics import mean

from .candidates import FixtureParameters, source_for
from .models import Idea


REFERENCES = [
    {
        "id": "scientisttwo",
        "title": "ScientistTwo",
        "url": "https://arxiv.org/abs/2609.19644",
        "verification": "unverified",
        "note": "Bundled metadata; no live citation lookup in offline mode",
    },
    {
        "id": "sklearn-toy",
        "title": "scikit-learn toy datasets",
        "url": "https://scikit-learn.org/stable/datasets/toy_dataset.html",
        "verification": "unverified",
        "note": "Bundled metadata; no live citation lookup in offline mode",
    },
]


def classification_response(role, inputs):
    if role in {"extract_limitations", "verify_limitations"}:
        return {
            "items": [
                "A linear decision rule cannot represent every nonlinear class boundary.",
                "Screening on two small datasets does not establish generalization.",
            ],
            "sufficient": role == "verify_limitations",
        }
    if role == "seed_ideas":
        ideas = [
            Idea(
                id="seed-svc",
                method="svc",
                parameters={"strength": 3},
                hypothesis="Test a nonlinear kernel under the same data protocol.",
                novelty=3,
            ),
            Idea(
                id="seed-knn",
                method="knn",
                parameters={"neighbors": 5},
                hypothesis="Test whether local neighborhoods improve classification.",
                novelty=2,
            ),
            Idea(
                id="seed-forest",
                method="forest",
                hypothesis="Test bounded-depth feature interactions.",
                novelty=1,
            ),
            Idea(
                id="seed-logistic",
                method="logistic",
                parameters={"strength": 0.1},
                hypothesis="Test stronger regularization as a control.",
                novelty=0,
            ),
            Idea(
                id="seed-svc-low",
                method="svc",
                parameters={"strength": 0.5},
                hypothesis="Test stronger kernel regularization as a control.",
                novelty=0,
            ),
        ]
        return {"ideas": [i.model_dump() for i in ideas[: inputs["count"]]]}
    if role == "novelty":
        return {
            "score": inputs["idea"]["novelty"],
            "references": [r["id"] for r in REFERENCES],
            "scope": "Scripted ranking only; these are established methods, not novel discoveries",
        }
    if role == "implement":
        return {
            "source": source_for(Idea.model_validate(inputs["idea"])),
            "description": "Known classification fixture; estimator fit uses training data only",
        }
    if role == "critic":
        comparison = inputs["comparison"]
        decision = (
            "Good"
            if comparison["valid"] and comparison["improved"]
            else ("Engineer" if inputs["can_engineer"] else "Bad")
        )
        return {"decision": decision, "reason": comparison["reason"]}
    if role in {"engineer", "evolve", "refine"}:
        if "idea" in inputs:
            idea = Idea.model_validate(inputs["idea"])
        else:
            # Use the best valid trace; when none exist, retain failure evidence and try a new kernel.
            usable = [t for t in inputs["traces"] if t["result"]["status"] == "success"]
            usable.sort(
                key=lambda t: mean(r["metrics"]["macro_f1"] for r in t["result"]["rows"]),
                reverse=True,
            )
            idea = Idea.model_validate((usable or inputs["traces"])[0]["idea"])
        parameters = FixtureParameters.model_validate(idea.parameters).model_dump()
        if idea.method == "svc":
            parameters["strength"] = min(parameters["strength"] * 2, 100)
        elif idea.method == "knn":
            parameters["neighbors"] = max(1, parameters["neighbors"] - 2)
        elif idea.method == "logistic":
            parameters["strength"] = min(parameters["strength"] * 2, 100)
        else:
            parameters["scaled"] = not parameters["scaled"]
        return idea.model_copy(
            update={
                "id": inputs["new_id"],
                "parameters": parameters,
                "origin": "evolved" if role == "evolve" else "engineered",
                "parent_ids": [idea.id],
                "hypothesis": idea.hypothesis + " Refined using recorded trial feedback.",
            }
        ).model_dump()
    if role == "select":
        best = max(
            inputs["candidates"],
            key=lambda t: mean(r["metrics"]["macro_f1"] for r in t["result"]["rows"]),
        )
        return {
            "idea_id": best["idea"]["id"],
            "reason": "Highest mean development macro-F1 among valid candidates",
        }
    if role in {"plan_ablations", "plan_rebuttal"}:
        idea = Idea.model_validate(inputs["idea"])
        parameters = FixtureParameters.model_validate(idea.parameters).model_dump()
        if role == "plan_ablations":
            parameters["scaled"] = not parameters["scaled"]
        revised = idea.model_copy(
            update={
                "id": inputs["new_id"],
                "parameters": parameters,
                "parent_ids": [idea.id],
                "origin": "ablation",
            }
        )
        return {
            "plans": [
                {
                    "idea": revised.model_dump(),
                    "purpose": "Isolate feature scaling"
                    if role == "plan_ablations"
                    else "Check repeatability on additional development seeds",
                    "changed_component": "scaling" if role == "plan_ablations" else "repeat",
                }
            ]
        }
    if role == "ablation_critic":
        return {
            "decision": "Refine" if any(c["improved"] for c in inputs["comparisons"]) else "Good",
            "reason": "Inspect component evidence; retain unsuccessful ablations in the report",
        }
    if role == "review":
        return {
            "score": 8 if inputs["has_rebuttal"] else 7,
            "concerns": []
            if inputs["has_rebuttal"]
            else ["Repeat the method on additional development seeds."],
            "synthetic": True,
        }
    if role == "meta_review":
        return {
            "decision": "Accept" if inputs["review"]["score"] >= 8 else "Refine",
            "reason": "Scripted workflow decision; no judgment of publication readiness",
        }
    if role == "draft":
        claims = []
        for result in inputs["evidence"]:
            if result["status"] == "success":
                for row in result["rows"]:
                    claims.append(
                        {
                            "experiment_id": result["id"],
                            "dataset": row["dataset"],
                            "seed": row["seed"],
                            "metric": "macro_f1",
                            "value": row["metrics"]["macro_f1"],
                        }
                    )
        return {
            "title": "Offline classification workflow demonstration",
            "claims": claims,
            "synthetic": True,
        }
    raise ValueError(f"Unknown fixture role: {role}")
