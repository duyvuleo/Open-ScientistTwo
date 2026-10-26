"""Mechanically grounded reports; narratives cannot introduce unsupported metric claims."""

from .models import ExperimentResult, Idea, ManuscriptDraft
from .store import IntegrityError, canonical
from typing import Protocol


class WriterAdapter(Protocol):
    def write(self, store, adapter) -> dict: ...


class EvidenceWriter:
    def write(self, store, adapter):
        return write_report(store, adapter)


def validate_claims(draft: ManuscriptDraft, evidence: list[ExperimentResult]):
    lookup = {
        (result.id, row.dataset, row.seed, metric): value
        for result in evidence
        if result.status == "success"
        for row in result.rows
        for metric, value in row.metrics.items()
    }
    for claim in draft.claims:
        key = (claim.experiment_id, claim.dataset, claim.seed, claim.metric)
        if key not in lookup or abs(lookup[key] - claim.value) > 1e-10:
            raise IntegrityError(f"Unsupported manuscript claim: {key}")


def experiments(store):
    return [
        ExperimentResult.model_validate(store.json(row[0]))
        for row in store.db.execute(
            "SELECT result_ref FROM jobs WHERE result_ref IS NOT NULL ORDER BY key"
        )
    ]


def evidence_audit(store, adapter):
    errors = store.verify()
    protocol_errors = []
    score_errors = []
    from .models import EvalSpec
    from .store import digest
    from .runner import evaluator_identity
    from .splits import validate_partitions

    split_errors = []
    split_ref = store.get("split_manifest_ref")
    if split_ref and not errors:
        try:
            manifest = store.json(split_ref)
            for record in manifest["datasets"].values():
                validate_partitions(record)
        except (ValueError, KeyError, TypeError) as error:
            split_errors.append(str(error))

    if (
        store.get("evaluator") != evaluator_identity()
        or store.get("adapter_identity") != adapter.identity()
    ):
        protocol_errors.append("Evaluator/adapter version changed")
    if not errors:
        for result in experiments(store):
            if result.status != "success":
                continue
            try:
                spec = EvalSpec.model_validate(store.json(result.spec_ref))
                if result.protocol_hash != digest(canonical(spec)):
                    protocol_errors.append(result.id)
                if split_ref and (
                    result.split_manifest_ref != split_ref
                    or spec.split_manifest_hash != digest(store.read(split_ref))
                ):
                    split_errors.append(f"{result.id}: partition identity mismatch")
                rows = adapter.score(spec, store.json(result.observations_ref))
                if canonical(rows_as_dict(rows)) != canonical(rows_as_dict(result.rows)):
                    score_errors.append(result.id)
            except (ValueError, KeyError, TypeError, OSError) as error:
                score_errors.append(f"{result.id}: {error}")
    draft_ref = store.get("draft_ref")
    claims = "unverified"
    if draft_ref and not errors:
        try:
            validate_claims(
                ManuscriptDraft.model_validate(store.json(draft_ref)), experiments(store)
            )
            claims = "pass"
        except (ValueError, IntegrityError):
            claims = "fail"
    best = store.get("best")
    alignment = "unverified"
    if best and not errors:
        alignment = adapter.audit_method(
            Idea.model_validate(best["idea"]), store.read(best["result"]["code_ref"]).decode()
        )
    return {
        "artifact_integrity": "fail" if errors else "pass",
        "partition_isolation": "fail" if split_errors else "pass" if split_ref else "unverified",
        "specification_compliance": "fail" if protocol_errors else "pass",
        "score_recalculation": "fail"
        if score_errors or errors
        else ("pass" if any(r.status == "success" for r in experiments(store)) else "unverified"),
        "manuscript_claims": claims,
        "method_code_alignment": alignment,
        "reference_existence": "unverified",
        "reexecution": "unverified",
        "errors": errors + protocol_errors + score_errors + split_errors,
        "limitations": [
            "Method-code pass covers known fixture implementations only.",
            "Offline source metadata is not a live citation-existence check.",
            "Protocol checks are technical checks, not proof of scientific validity.",
        ],
    }


def rows_as_dict(rows):
    return [row.model_dump(mode="json") for row in rows]


def escape_tex(text):
    substitutions = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(substitutions.get(c, c) for c in str(text))


def write_report(store, adapter):
    state = store.state()
    results = experiments(store)
    audit = evidence_audit(store, adapter)
    audit_ref = store.put_json(audit, "audit")
    trials = [
        store.json(row[0])
        for row in store.db.execute(
            "SELECT output_ref FROM calls WHERE key LIKE 'trial/%' ORDER BY key"
        )
    ]
    agents = [
        store.json(row[0])
        for row in store.db.execute("SELECT output_ref FROM calls WHERE key LIKE 'agent/%'")
    ]
    live_calls = sum(not agent["synthetic"] for agent in agents)
    summary = {
        "state": state.model_dump(mode="json"),
        "synthetic_agents": live_calls == 0,
        "live_calls": live_calls,
        "scripted_agent_invocations": len(agents) - live_calls,
        "api_cost_usd": 0 if live_calls == 0 else None,
        "split_manifest_ref": store.get("split_manifest_ref"),
        "experiments": len(results),
        "successful_executions": sum(r.status == "success" for r in results),
        "charged_execution_seconds": sum(r.duration_seconds for r in results),
        "audit_ref": audit_ref,
        "trials": trials,
        "final_candidate": store.get("final_candidate"),
        "final_baseline": store.get("final_baseline"),
        "boundary": "Scripted agent workflow with real container experiments; no autonomous discovery claim",
    }
    origins = sorted({r.measurement_origin for r in results})
    summary["measurement_origins"] = origins
    summary_ref = store.put_json(summary, "run-summary")
    lines = [
        "# ScientistTwo campaign report",
        "",
        f"Outcome: **{state.outcome}**",
        "",
        state.reason,
        "",
        "Agent responses are scripted fixtures. Measurement provenance: "
        + ", ".join(origins or ["no experiments"]),
        "Automated review acceptance is an internal workflow status, not conference acceptance.",
        "",
        f"Run summary: [{summary_ref}]({summary_ref})",
        f"Audit: [{audit_ref}]({audit_ref})",
        "",
        "## Experimental evidence",
        "",
        "Development results were available during refinement. Final-test results were collected only after freezing the candidate.",
        "",
        "| Experiment | Dataset | Seed | Metric | Value | Train (s) | Infer (s) | Evidence |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    tex_rows = []
    for result in results:
        for row in result.rows:
            for metric, value in row.metrics.items():
                lines.append(
                    f"| {result.id} | {row.dataset} | {row.seed} | {metric} | {value:.8f} | "
                    f"{row.training_seconds:.6f} | {row.inference_seconds:.6f} | "
                    f"[observations]({result.observations_ref}) / [code]({result.code_ref}) |"
                )
                tex_rows.append(
                    " & ".join(
                        map(escape_tex, [result.id, row.dataset, row.seed, metric, f"{value:.8f}"])
                    )
                    + r" \\"
                )
    lines += ["", "## Failed executions and rejected ideas", ""]
    for result in results:
        if result.status != "success":
            lines.append(f"- {result.id}: {result.status}. {result.reason}")
    for trial in trials:
        lines.append(
            f"- {trial['idea']['id']}: {trial['decision']['decision']}. {trial['decision']['reason']}"
        )
    lines += [
        "",
        "## Audit status",
        "",
        *[f"- {k}: {v}" for k, v in audit.items() if isinstance(v, str)],
        "",
        "## Limits",
        "",
        "- Small fixture tasks do not establish research generality or novelty.",
        "- Quality thresholds are workflow rules, not statistical-significance tests.",
        "- Timing measurements are environment-dependent and are not scientific promotion criteria.",
        "- Citation existence and live autonomous-agent quality remain unverified.",
    ]
    report = "\n".join(lines) + "\n"
    tex = "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage[margin=1cm]{geometry}",
            r"\usepackage{longtable}",
            r"\begin{document}",
            r"\section*{ScientistTwo offline workflow demonstration}",
            "Scripted agent responses. Measurement provenance: "
            + escape_tex(", ".join(origins))
            + ". No autonomous discovery claim.",
            "Outcome: " + escape_tex(state.outcome) + ".",
            r"\tiny\begin{longtable}{p{7cm}lllr}",
            r"Experiment & Dataset & Seed & Metric & Value \\",
            r"\hline",
            *tex_rows,
            r"\end{longtable}",
            r"\end{document}",
        ]
    )
    report_ref = store.put(report.encode(), "md", "report")
    tex_ref = store.put(tex.encode(), "tex", "manuscript")
    store.set("report_ref", report_ref)
    store.set("summary_ref", summary_ref)
    store.set("manuscript_ref", tex_ref)
    # Friendly copies are derived outputs, never inputs to scientific decisions.
    for name, content in (
        ("report.md", report),
        ("manuscript.tex", tex),
        ("summary.json", canonical(summary).decode()),
    ):
        (store.root / name).write_text(content)
    return summary
