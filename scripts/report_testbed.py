"""Render readiness and per-task measurements; never combine heterogeneous scores."""

import argparse
import json
from pathlib import Path

from scientisttwo.store import Store
from scientisttwo.testbed.readiness import readiness


def link(path):
    return f"[evidence](<{Path(path).resolve()}>)"


def report(root):
    status = readiness(root)
    lines = ["# MLAgentBench audited testbed", "",
             f"Canonical task readiness: **{status['ready_count']}/13**. "
             f"Full audited-testbed ready: **{status['full_audited_testbed_ready']}**.",
             "", "Validation uses scripted agents; live autonomous research is unevaluated.", "",
             "Workflow families passed: " + (", ".join(status["research_families_exercised"]) or "none") +
             ". Missing: " + (", ".join(status["research_families_missing"]) or "none") + ".", "",
             "| Task | Variant | Data | Runtime | Execution | Ready |",
             "|---|---|---|---|---|---|"]
    for row in status["tasks"]:
        gates = row["gates"]
        lines.append(f"| {row['task_id']} | {row['variant'] or 'not prepared'} | {gates['data']} | "
                     f"{gates['runtime']} | {gates['execution']} | {row['ready']} |")
    for row in status["tasks"]:
        task = root / row["task_id"]
        receipt_path = task / "validation/receipt.json"
        lines.extend(["", "## " + row["task_id"], "",
                      f"Original starter: `{row['original_starter_status']}`. "
                      f"Reference: `{row['reference_baseline']}`.", "",
                      "Upstream compatibility availability: `" + row["tracks"]["upstream_compatibility"] +
                      "`; audited-v1: `" + row["tracks"]["audited_v1"] + "`."])
        for reason in row["blockers"]:
            lines.extend(["", "Blocker: " + reason])
        for deviation in row["deviations"]:
            lines.extend(["", "Deviation: " + deviation])
        if not receipt_path.is_file() or row["gates"]["audit"] != "pass":
            continue
        receipt = json.loads(receipt_path.read_text())
        lines.extend(["", f"Measured receipt: {link(receipt_path)}. Outcome: `{receipt['outcome']['outcome']}`. "
                      f"Data provenance: **{receipt['provenance']}**.",
                      "", "| Method | Metric | Matched seed values | Final |",
                      "|---|---|---|---|"])
        for name, measured in receipt["observations"].items():
            values = ", ".join(f"{v:.9g}" for v in measured["values"])
            final = receipt["finals"][name].get("score")
            lines.append(f"| {name} | {measured['metric']} | {values} | {final} |")
        store = Store(task / "validation/campaign")
        try:
            rows = store.db.execute("SELECT kind,status,count(*),sum(seconds),sum(tokens) "
                                    "FROM live_operations GROUP BY kind,status").fetchall()
            lines.extend(["", "Resource ledger (container scorer audits are separately recorded):", "",
                          "| Operation kind | Status | Count | Charged seconds | Observed tokens |",
                          "|---|---|---|---|---|"])
            for kind, state, count, seconds, tokens in rows:
                lines.append(f"| {kind} | {state} | {count} | {seconds:.3f} | {tokens} |")
            for name, measured in receipt["observations"].items():
                for ref in measured.get("timing_refs", []):
                    timing = store.json(ref)
                    raw = store.json(timing["report_ref"])
                    base, candidate = raw["baseline"], raw["candidate"]
                    lines.extend(["", f"{name}, seed {timing['seed']}: baseline median "
                                  f"{base['median_seconds']:.9g} s (IQR {base['iqr_seconds']:.9g}); "
                                  f"method median {candidate['median_seconds']:.9g} s "
                                  f"(IQR {candidate['iqr_seconds']:.9g}). " + link(store.root / timing["report_ref"])])
        finally:
            store.close()
    lines.extend(["", "No aggregate score is calculated across tasks or differing protocols.", ""])
    return status, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".testbed"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status, content = report(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    args.output.with_suffix(".json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
