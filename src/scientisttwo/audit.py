"""Explicit reproducibility audit; does not feed observations back to research agents."""

from .evaluation import quality_matches
from .models import EvalSpec, ExperimentResult, Idea
from .reporting import evidence_audit
from .store import digest


def audit_campaign(
    store, adapter, runner, reexecute=True, include_final=False, screening_sample_size=0
):
    audit = evidence_audit(store, adapter)
    best = store.get("best")
    audit["reexecution_target"] = "selected_candidate" if best else "unavailable"
    if best is None:
        baseline = store.db.execute(
            "SELECT result_ref FROM jobs WHERE key LIKE 'baseline/full/%' AND status='success' ORDER BY key LIMIT 1"
        ).fetchone()
        if baseline:
            best = {"idea": adapter.baseline().model_dump(), "result": store.json(baseline[0])}
            audit["reexecution_target"] = "baseline_only_no_selected_candidate"
    if audit["errors"] or audit["manuscript_claims"] == "fail":
        audit["reexecution"] = "unverified"
    elif reexecute and best:
        original = ExperimentResult.model_validate(best["result"])
        spec = EvalSpec.model_validate(store.json(original.spec_ref))
        key = "audit/" + digest(original.id.encode())[:16]
        reproduced = runner.execute(
            key,
            Idea.model_validate(best["idea"]),
            store.read(original.code_ref).decode(),
            spec,
            audit=True,
        )
        audit["reexecution"] = (
            "pass" if quality_matches(original, reproduced, spec.task.quality_tolerance) else "fail"
        )
        audit["reexecution_result"] = reproduced.model_dump(mode="json")
        audit["timing_comparison"] = "informational; no timing-equivalence claim"
    audit["final_reexecution"] = "unverified"
    if include_final and reexecute and not audit["errors"] and store.get("freeze_ref"):
        frozen = store.json(store.get("freeze_ref"))
        final_checks = {}
        for label in ("baseline", "candidate"):
            saved = store.get("final_" + label)
            if not saved or saved["status"] != "success":
                final_checks[label] = "unverified"
                continue
            original = ExperimentResult.model_validate(saved)
            spec = EvalSpec.model_validate(store.json(original.spec_ref))
            idea = (
                adapter.baseline() if label == "baseline" else Idea.model_validate(frozen["idea"])
            )
            repeated = runner.execute(
                "audit/final/" + label,
                idea,
                store.read(original.code_ref).decode(),
                spec,
                audit=True,
            )
            final_checks[label] = (
                "pass"
                if quality_matches(original, repeated, spec.task.quality_tolerance)
                else "fail"
            )
        audit["final_checks"] = final_checks
        audit["final_reexecution"] = (
            "pass" if all(v == "pass" for v in final_checks.values()) else "fail"
        )
    audit["screening_checks"] = []
    if screening_sample_size and reexecute and not audit["errors"]:
        baseline_row = store.db.execute(
            "SELECT result_ref FROM jobs WHERE key LIKE 'baseline/full/%' AND status='success' ORDER BY key LIMIT 1"
        ).fetchone()
        rejected = {}
        for row in store.db.execute("SELECT output_ref FROM calls WHERE key LIKE 'trial/%'"):
            trial = store.json(row[0])
            result = ExperimentResult.model_validate(trial["result"])
            if trial["decision"]["decision"] == "Bad" and result.status == "success":
                spec = EvalSpec.model_validate(store.json(result.spec_ref))
                if spec.phase == "subset":
                    rejected[trial["idea"]["id"]] = trial
        selected = sorted(
            rejected.values(),
            key=lambda t: digest((store.root.name + "/" + t["idea"]["id"]).encode()),
        )[:screening_sample_size]
        audit["screening_eligible"] = len(rejected)
        audit["screening_selection"] = (
            "Predeclared SHA256(campaign ID / idea ID) ordering; no full-set result used for selection"
        )
        if baseline_row:
            baseline = ExperimentResult.model_validate(store.json(baseline_row[0]))
            full_spec = EvalSpec.model_validate(store.json(baseline.spec_ref))
            for trial in selected:
                idea = Idea.model_validate(trial["idea"])
                source = store.read(trial["result"]["code_ref"]).decode()
                result = runner.execute(
                    "audit/screening/" + idea.id, idea, source, full_spec, audit=True
                )
                comparison = adapter.compare(result, baseline, full_spec.task)
                audit["screening_checks"].append(
                    {
                        "idea_id": idea.id,
                        "result": result.model_dump(mode="json"),
                        "comparison": comparison.model_dump(mode="json"),
                        "scope": "Post-campaign full DEVELOPMENT check; not final-test confirmation",
                    }
                )
    audit["audit_ref"] = store.put_json(audit, "reproduction-audit")
    store.set("latest_audit", audit)
    return audit
