"""Deterministic empirical tables; model prose and mechanism claims remain unverified."""

import json


def tex(value):
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(c, c) for c in str(value))


def write_report(store, config, synthetic):
    state = store.get("live_state")
    operations = [dict(r) for r in store.db.execute("SELECT * FROM live_operations ORDER BY rowid")]
    calls = [r for r in operations if r["kind"] == "model"]
    results = []
    for row in store.db.execute(
        "SELECT output_ref FROM calls WHERE key LIKE 'development/%' ORDER BY key"
    ):
        results.append((store.json(row[0]), row[0]))
    draft_ref = store.get("live_draft")
    draft = store.json(draft_ref) if draft_ref else None
    meta_ref = store.get("live_meta")
    meta = store.json(meta_ref) if meta_ref else None
    meta_decision = meta.get("decision") if meta else None
    meta_status = ("accepted" if meta_decision == "Accept" else
                   "unresolved" if meta_decision else "unverified")
    meta_review = {
        "status": meta_status,
        "decision": meta_decision,
        "evidence_ref": meta_ref,
        "review": meta,
        "scope": "Recorded internal meta-review only; not conference acceptance or verified scientific validity",
    }
    manuscript_revisions = [
        {"key": row[0].removeprefix("manuscript-revision/"),
         "evidence_ref": json.loads(row[1]), "record": store.json(json.loads(row[1]))}
        for row in store.db.execute(
            "SELECT key,value FROM metadata WHERE key LIKE 'manuscript-revision/%' ORDER BY key"
        )
    ]
    supplementary = [
        {"key": row[0].removeprefix("supplementary/"), "evidence_ref": json.loads(row[1]),
         **store.json(json.loads(row[1]))}
        for row in store.db.execute("SELECT key,value FROM metadata WHERE key LIKE 'supplementary/%' ORDER BY key")
    ]
    parent_edits = [{"key": row[0], "evidence_ref": row[1]}
                    for row in store.db.execute("SELECT key,output_ref FROM calls WHERE key LIKE 'parent-edit/%' ORDER BY key")]
    summary = {
        "state": state,
        "mode": config.mode,
        "model": config.model,
        "synthetic_agents": synthetic,
        "model_attempts": len(calls),
        "live_model_calls": 0 if synthetic else len(calls),
        "observed_tokens": sum(r["tokens"] for r in calls),
        "unknown_usage_attempts": sum(r["status"] != "success" for r in calls),
        "experiment_attempts": sum(r["kind"] == "experiment" for r in operations),
        "selected": store.get("live_best"),
        "final": store.get("live_final"),
        "data_provenance": config.native.data_provenance,
        "leaderboard_eligible": False,
        "checks": {
            "artifacts": "fail" if store.verify() else "pass",
            "structured_claims": "pass" if draft else "unverified",
            "citation_id_resolution": "pass" if draft else "unverified",
            "citation_existence": "unverified",
            "method_code_consistency": "unverified",
            "semantic_ablation_isolation": "unverified",
        },
        "operations": operations,
        "supplementary_requests": supplementary,
        "parent_edit_proofs": parent_edits,
        "manuscript_revisions": manuscript_revisions,
        "meta_review": meta_review,
    }
    bundle_ref = store.get("research_final_bundle") or store.get("research_manuscript_bundle")
    bundle = store.json(bundle_ref) if bundle_ref else None
    if bundle:
        validation = store.json(bundle["validation_ref"])
        summary["checks"].update(validation["checks"])
        summary["research_manuscript"] = {"bundle_ref": bundle_ref, **bundle}
    summary["engine_version"] = getattr(config, "engine_version", "legacy-live-v1")
    summary["workflow_profile"] = config.workflow.profile if summary["engine_version"] == "research-v2" else "legacy"
    ref = store.put_json(summary, "live-summary")
    store.set("live_summary_ref", ref)
    (store.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    title = draft["title"] if draft else "Native benchmark research run"
    review_status_text = (
        f"Internal meta-review: {meta_status}; recorded decision: {meta_decision or 'none'}. "
        "Campaign termination and revision-limit exhaustion do not imply review acceptance."
    )
    lines = [
        f"# {title}",
        "",
        f"Status: **{state['status']}**. Outcome: **{state['outcome']}**.",
        "",
        state["reason"],
        "",
        review_status_text,
        "",
        f"Mode: {config.mode}. Model: {config.model}. Synthetic agents: {synthetic}.",
        f"Model attempts: {len(calls)}. Observed tokens: {summary['observed_tokens']}. Failed/interrupted call usage can be unknown.",
        "",
        f"Data provenance: {config.native.data_provenance}. Leaderboard eligible: false.",
        "",
        "Development measurements follow the declared local protocol; they are not official benchmark scores.",
        "",
        "| Experiment | Status | Metric | Value | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    if bundle:
        from pathlib import Path
        relative = Path(bundle["project_path"]).relative_to(store.root).as_posix()
        lines[2:2] = ["Complete manuscript: [Markdown](" + relative + "/manuscript.md), "
                      "[LaTeX project](" + relative + "/main.tex). PDF: "
                      + bundle["compilation"]["status"] + ".", ""]
    latex_rows = []
    for result, evidence_ref in results:
        metrics = result.get("metrics") or {result["metric"]: result["value"]}
        for metric, measured in metrics.items():
            value = measured if measured is not None else "unavailable"
            lines.append(
                f"| {result['id']} | {result['status']} | {metric} | {value} | [result]({evidence_ref}) |"
            )
            label = f"{result['id']} / {metric}"
            latex_rows.append(f"{tex(label)} & {tex(value)} & \\path{{{evidence_ref}}} \\\\")
    lines += ["", "## Frozen final evaluation", ""]
    for label, result in (store.get("live_final") or {}).items():
        lines.append(
            f"- {label}: {result['status']}; score {result.get('score')}. [Native report](final/{label}/report.md)."
        )
    latex_prose = []
    if draft:
        lines += [
            "",
            "## Model-proposed limitations (unverified prose)",
            "",
            *[f"- {x}" for x in draft["limitations"]],
            "",
            f"[Validated structured claims]({draft_ref})",
        ]
        latex_prose += [
            r"\section*{Model-proposed limitations (unverified prose)}",
            *[tex(limitation) + r"\par" for limitation in draft["limitations"]],
        ]
        if draft.get("concern_responses"):
            lines += ["", "## Model-proposed concern responses (unverified prose)", ""]
            latex_prose.append(r"\section*{Model-proposed concern responses (unverified prose)}")
            for response in draft["concern_responses"]:
                text = (f"{response['concern_id']}: {response['status']}; "
                        f"experiments: {', '.join(response['experiment_ids']) or 'none'}. "
                        f"{response['explanation']}")
                lines.append(f"- {text}")
                latex_prose.append(tex(text) + r"\par")
    if manuscript_revisions:
        lines += ["", "## Manuscript revision records", "",
                  "Recorded routing and execution facts; editorial adequacy remains unverified.", ""]
        latex_prose += [r"\section*{Manuscript revision records}",
                        tex("Recorded routing and execution facts; editorial adequacy remains unverified.")]
        for revision in manuscript_revisions:
            record = revision["record"]
            text = (f"{revision['key']}: {record.get('status', 'unverified')}; "
                    f"recorded no_new_experiments: {record.get('no_new_experiments', 'unverified')}.")
            lines.append(f"- {text} [Revision record]({revision['evidence_ref']}).")
            latex_prose.append(tex(text) + r"\par")
    if supplementary:
        lines += ["", "## Supplementary request coverage", "",
                  "Counts and diagnostics below describe actual execution, not scientific resolution. Purpose text is not executable.", ""]
        for record in supplementary:
            for decision in record["negotiation"]["decisions"]:
                plan = decision["plan"]
                lines.append(f"- {plan['id']}: {decision['status']}; requested {plan['seed_count']} seeds; "
                             f"accepted seeds {decision['accepted_seeds']}; diagnostics {plan['diagnostics']}. "
                             f"Reasons: {'; '.join(decision['reasons']) or 'request accepted, see execution coverage'}. "
                             f"[Request and execution record]({record['evidence_ref']}).")
            for concern in record["coverage"]:
                lines.append(f"- {concern['concern_id']}: **{concern['status']}**; "
                             f"observations {concern['observation_ids']}; {'; '.join(concern['reasons'])}.")
    if parent_edits:
        lines += ["", "## Parent source fidelity", "",
                  "Changes inherit exact parent bytes and command outside their declared anchored edits. Scientific intent remains unverified.",
                  *[f"- [{row['key']}]({row['evidence_ref']})" for row in parent_edits]]
    lines += [
        "",
        "## Interpretation",
        "",
        "Native grading receipts cover execution only; the outer campaign records agent provenance. Final grades never enter model prompts. Review acceptance is internal. File-scoped controls do not certify causal mechanism isolation. Timing includes process/container startup. This run does not establish general autonomous research quality.",
        "",
        f"[Complete summary]({ref})",
        "",
    ]
    if config.native.data_provenance == "contract_fixture":
        lines += ["CONTRACT FIXTURE DATA: no competition performance claim.", ""]
    (store.root / "report.md").write_text("\n".join(lines))
    (store.root / "manuscript.tex").write_text(
        "\n".join(
            [
                r"\documentclass{article}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{longtable,url}",
                r"\begin{document}",
                r"\title{" + tex(title) + "}",
                r"\maketitle",
                tex("Evidence draft. " + state["reason"]),
                r"\par",
                tex(review_status_text),
                r"\begin{longtable}{p{0.29\linewidth}p{0.15\linewidth}p{0.46\linewidth}}",
                "Experiment & Development value & Evidence \\\\ ",
                *latex_rows,
                r"\end{longtable}",
                *latex_prose,
                tex(
                    "Method-code consistency and semantic ablation isolation remain unverified. Review scores do not indicate conference acceptance."
                ),
                r"\end{document}",
                "",
            ]
        )
    )
