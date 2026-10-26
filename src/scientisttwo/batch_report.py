"""Batch aggregation retains every assigned task, failure, and negative outcome."""

from collections import Counter, defaultdict
import json
from pathlib import Path

from .store import Store, IntegrityError, canonical, digest


def read_receipt(root, entry):
    directory = Path(root) / "campaigns" / entry.id
    path = directory / "batch-result.json"
    if not path.exists():
        return {"status": "pending"}
    result = json.loads(path.read_text())
    if result.get("status") in {"not_started_budget_exhausted", "interrupted"}:
        return result
    if not (directory / "campaign.sqlite").exists():
        return {"status": "error", "error": "Campaign receipt has no supporting evidence store"}
    store = Store(directory)
    try:
        ref = store.get("batch_result_ref")
        if not ref or store.json(ref) != result:
            raise IntegrityError("Batch receipt differs from its immutable artifact")
        if store.get("batch_entry") != entry.model_dump(mode="json"):
            raise IntegrityError("Batch receipt belongs to a different campaign assignment")
        return result
    except (IntegrityError, ValueError) as error:
        return {"status": "error", "error": str(error)}
    finally:
        store.close()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def summarize(root, manifest):
    root = Path(root)
    results = []
    for entry in manifest.campaigns:
        result = read_receipt(root, entry)
        results.append(
            {
                "campaign_id": entry.id,
                "task_id": entry.task_id,
                "family": entry.task.adapter,
                "repetition": entry.repetition,
                "metric": entry.task.primary_metric,
                "data_origin": entry.data_origin,
                "expected_control": entry.expected_control,
                **result,
            }
        )
    family_stats = {}
    for family in sorted({r["family"] for r in results}):
        rows = [r for r in results if r["family"] == family]
        family_stats[family] = {
            "assigned_campaigns": len(rows),
            "unique_tasks": len({r["task_id"] for r in rows}),
            "completed": sum(r["status"] == "complete" for r in rows),
            "outcomes": dict(Counter(r.get("outcome", r["status"]) for r in rows)),
            "audit_passes": sum(r.get("audit_pass", False) for r in rows),
            "experiment_jobs": sum(r.get("experiment_jobs", 0) for r in rows),
            "audit_jobs": sum(r.get("audit_jobs", 0) for r in rows),
            "screening_audited": sum(r.get("screening_audited", 0) for r in rows),
            "screening_missed_development_improvements": sum(
                r.get("screening_missed_development_improvements", 0) for r in rows
            ),
        }
    stages = Counter()
    for row in results:
        stages.update(set(row.get("stages", [])))
    by_task = defaultdict(list)
    for row in results:
        by_task[row["task_id"]].append(row)
    task_outcomes = {
        name: {
            "assigned_repetitions": len(rows),
            "outcomes": dict(Counter(r.get("outcome", r["status"]) for r in rows)),
            "partition_hashes": sorted(
                {r["observed_split_hash"] for r in rows if r.get("observed_split_hash")}
            ),
        }
        for name, rows in sorted(by_task.items())
    }
    complete = sum(r["status"] == "complete" for r in results)
    summary = {
        "manifest_hash": digest(canonical(manifest)),
        "assigned_campaigns": len(results),
        "unique_tasks": len(by_task),
        "completed_campaigns": complete,
        "status_counts": dict(Counter(r["status"] for r in results)),
        "families": family_stats,
        "stage_coverage_campaigns": dict(stages),
        "task_outcomes": task_outcomes,
        "experiment_jobs": sum(r.get("experiment_jobs", 0) for r in results),
        "audit_jobs": sum(r.get("audit_jobs", 0) for r in results),
        "execution_seconds": sum(r.get("execution_seconds", 0) for r in results),
        "live_model_calls": sum(r.get("live_model_calls", 0) for r in results),
        "api_cost_usd": 0,
        "results": results,
        "boundary": "Scripted-agent engineering validation on bundled and generated tasks. Repetitions are not independent scientific discoveries.",
    }
    atomic_json(root / "summary.json", summary)
    lines = [
        "# Offline cross-domain validation",
        "",
        summary["boundary"],
        "",
        f"Completed {complete}/{len(results)} assigned campaigns across {len(by_task)} distinct task specifications.",
        "",
        "| Family | Tasks | Assigned | Completed | Supported final result | No improvement | Audits passed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, values in family_stats.items():
        lines.append(
            f"| {family} | {values['unique_tasks']} | {values['assigned_campaigns']} | {values['completed']} | "
            f"{values['outcomes'].get('candidate_supported', 0)} | {values['outcomes'].get('no_improvement', 0)} | {values['audit_passes']} |"
        )
    lines += [
        "",
        "## Per-campaign evidence",
        "",
        "Final deltas are baseline-relative and sign-adjusted so positive means improvement. Units remain task-specific; values are not pooled across metrics.",
        "",
        "| Campaign | Status / outcome | Primary metric | Final delta | Audit | Evidence |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in results:
        delta = f"{row['final_delta']:.8g}" if row.get("final_delta") is not None else "unavailable"
        evidence = (
            f"[report](campaigns/{row['campaign_id']}/report.md)"
            if row["status"] == "complete"
            else row.get("error", "pending")
        )
        lines.append(
            f"| {row['campaign_id']} | {row.get('outcome', row['status'])} | {row['metric']} | {delta} | "
            f"{'pass' if row.get('audit_pass') else 'unverified/fail'} | {evidence} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- All assigned campaigns remain in the denominator, including failures and negative outcomes.",
        "- Three repetitions vary estimator seeds and repeat execution; they do not triple the number of independent tasks.",
        "- Generated controls test known mathematical and statistical mechanisms. Positive scores do not establish novelty or general research ability.",
        "- Final-test observations are collected after method freeze; audit reruns never return to the research loop.",
        "- A predeclared hash-based sample of rejected screening candidates is checked on full development data. Any recovered improvement is an audit observation, not a new final-test success.",
        "- Citation existence and live scientific reasoning remain unverified. No paid model calls were enabled.",
        "- [Frozen manifest](manifest.json) and [machine-readable summary](summary.json) contain the task, split, implementation, and per-run evidence identifiers.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return summary
