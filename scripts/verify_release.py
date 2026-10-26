"""Portable release checks. Never downloads images/data or invokes live agents.

Use the package command with a wheel installed in a fresh environment outside
the checkout. The container command requires an explicitly built local image;
its candidate processes use the application's offline container boundary.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata, resources
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_trailing_json_object(output):
    """Read the final CLI object after progress logs; reject a non-JSON suffix."""
    decoder = json.JSONDecoder()
    position = output.rfind("{")
    while position >= 0:
        try:
            value, end = decoder.raw_decode(output, position)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(value, dict) and not output[end:].strip():
                return value
        position = output.rfind("{", 0, position)
    raise ValueError("CLI output does not end with a complete JSON object")


def cli(arguments, *, cwd, output, timeout=2100):
    """Use this interpreter's installed package, ignoring PYTHONPATH and cwd."""
    result = subprocess.run(
        [sys.executable, "-I", "-m", "scientisttwo.cli", *map(str, arguments)],
        cwd=cwd, capture_output=True, text=True, timeout=timeout,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".stdout.txt").write_text(result.stdout)
    output.with_suffix(".stderr.txt").write_text(result.stderr)
    require(result.returncode == 0,
            f"CLI {arguments[0]} failed with exit {result.returncode}; inspect {output.parent}")
    payload = parse_trailing_json_object(result.stdout)
    write_json(output.with_suffix(".json"), payload)
    return payload


def verify_package(args):
    import scientisttwo
    from scientisttwo.integrations.paperorchestra.runtime import (
        SOURCE_HASHES,
        UPSTREAM_REVISION,
        bundled_upstream_root,
        verify_upstream,
    )

    checkout = args.source_root.resolve()
    package = Path(scientisttwo.__file__).resolve().parent
    require(not package.is_relative_to(checkout),
            "Install the wheel in a fresh environment outside the source checkout")
    dist = metadata.distribution("scientisttwo")
    direct_url = dist.read_text("direct_url.json")
    require(not direct_url or not json.loads(direct_url).get("dir_info", {}).get("editable"),
            "The release check requires an installed wheel, not an editable install")
    source = checkout / "src" / "scientisttwo"
    expected = {"runtime_locks.json", "worker.py", "live/worker.py",
                "live/workspace_worker.py", "live/timing_worker.py",
                "research/compiler_worker.py", "research/diagnostic_worker.py",
                "integrations/paperorchestra/pdf_worker.py"}
    # Verify the complete installed implementation, not only resource presence.
    expected.update(str(path.relative_to(source)) for path in source.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                    and path.suffix != ".pyc")
    expected.update("_vendor/paperorchestra/" + name for name in SOURCE_HASHES)
    expected.add("_vendor/paperorchestra/PROVENANCE.json")
    for directory in ("task_families/templates", "research/templates", "_vendor/paperorchestra"):
        files = [path for path in (source / directory).rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"]
        require(files, f"Source resource directory is missing or empty: {directory}")
        expected.update(str(path.relative_to(source)) for path in files)
    resource_root = resources.files("scientisttwo")
    for relative in sorted(expected):
        installed = resource_root.joinpath(relative)
        require(installed.is_file(), f"Wheel is missing runtime resource: {relative}")
        require(installed.read_bytes() == (source / relative).read_bytes(),
                f"Wheel resource differs from the source: {relative}")
    json.loads(resource_root.joinpath("runtime_locks.json").read_text())
    bundle = package / "_vendor" / "paperorchestra"
    require(bundled_upstream_root().resolve() == bundle.resolve(),
            "Default PaperOrchestra source does not resolve inside the installed package")
    verified = verify_upstream(bundle)
    require(verified == SOURCE_HASHES, "Bundled PaperOrchestra hash closure differs from its lock")
    provenance = json.loads((bundle / "PROVENANCE.json").read_text())
    require(provenance.get("revision") == UPSTREAM_REVISION,
            "Bundled PaperOrchestra provenance does not match the pinned revision")
    require(provenance.get("files") == SOURCE_HASHES,
            "Bundled PaperOrchestra provenance differs from the verified file closure")
    bundled_python = {str(path.relative_to(bundle)) for path in bundle.rglob("*.py")}
    locked_python = {name for name in SOURCE_HASHES if name.endswith(".py")}
    require(bundled_python == locked_python,
            "Bundled PaperOrchestra contains missing or unreviewed Python modules")
    entries = {entry.name: entry.value for entry in dist.entry_points
               if entry.group == "scientisttwo.native_development"}
    require(entries.get("testbed") == "scientisttwo.testbed.adapter:PreparedDevelopment",
            "Installed native adapter entry point is missing")
    with tempfile.TemporaryDirectory(prefix="scientisttwo-wheel-check-") as temporary:
        check = subprocess.run(
            [sys.executable, "-I", "-m", "scientisttwo.cli", "--help"],
            cwd=temporary, capture_output=True, text=True, timeout=60,
        )
        require(check.returncode == 0 and "paperorchestra" in check.stdout,
                "Installed CLI help failed outside the checkout")
    return {"status": "pass", "version": dist.version, "resource_count": len(expected),
            "resources": sorted(expected), "live_model_calls": 0,
            "bundled_paperorchestra": {"status": "pass", "revision": UPSTREAM_REVISION,
                                       "files": len(verified), "sha256": verified},
            "scope": "Installed wheel imports, entry points, and runtime resource bytes"}


def campaign_snapshot(root):
    """Compare canonical ledger content and evidence bytes.

    Metadata uses INSERT OR REPLACE, so unchanged values may acquire new rowids.
    Ordinary table row order is immaterial. Operation rowids remain significant:
    the research audit uses their order to enforce the protected-final boundary.
    """
    with sqlite3.connect(f"file:{root / 'campaign.sqlite'}?mode=ro", uri=True) as db:
        schema = db.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name"
        ).fetchall()
        tables = {}
        for kind, name, _, _ in schema:
            if kind != "table":
                continue
            identifier = '"' + name.replace('"', '""') + '"'
            rows = db.execute("SELECT * FROM " + identifier).fetchall()
            # Preserve SQLite value types, binary payloads, and duplicate rows.
            # Hex floats also serialize infinities without nonstandard JSON.
            tables[name] = sorted(json.dumps(
                [[type(value).__name__, value.hex() if isinstance(value, (bytes, float))
                  else value] for value in row], separators=(",", ":"), ensure_ascii=True,
            ) for row in rows)
        operation_order = (db.execute(
            "SELECT rowid,key FROM live_operations ORDER BY rowid"
        ).fetchall() if "live_operations" in tables else [])
        canonical = json.dumps({"schema": schema, "tables": tables,
                                "operation_order": operation_order},
                               sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        ledger = hashlib.sha256(canonical.encode()).hexdigest()
    files = {str(path.relative_to(root)): sha256(path) for path in root.rglob("*")
             if path.is_file() and path.name not in {
                 "campaign.sqlite", "campaign.sqlite-shm", "campaign.sqlite-wal", ".campaign.lock"}}
    return {"ledger_sha256": ledger, "files": files}


def verify_offline(inspected):
    require(inspected.get("engine_version") == "research-v2", "Shared research engine not used")
    require(inspected.get("backend", {}).get("synthetic") is True,
            "Release E2E must use the explicitly synthetic fixture backend")
    require(inspected.get("live_calls") is False and inspected.get("live_model_calls") == 0,
            "Release E2E unexpectedly reports live model calls")


def prompt_budget(args):
    if args.max_prompt_chars is not None:
        return args.max_prompt_chars
    return 1_000_000 if args.writer_backend == "paperorchestra" else 200_000


def bounded_prompt_chars(value):
    value = int(value)
    if not 1_000 <= value <= 1_000_000:
        raise argparse.ArgumentTypeError("Prompt budget must be between 1000 and 1000000 characters")
    return value


def verify_paperorchestra(root, args):
    """Require the requested writer and real PDF services in committed evidence."""
    from scientisttwo.integrations.paperorchestra.runtime import (
        SOURCE_HASHES,
        UPSTREAM_REVISION,
        bundled_upstream_root,
    )

    with sqlite3.connect(f"file:{root / 'campaign.sqlite'}?mode=ro", uri=True) as db:
        saved = dict(db.execute("SELECT key,value FROM metadata"))
    config = json.loads((root / json.loads(saved["pilot_live_spec_ref"])).read_text())
    workflow = config["workflow"]
    require(config.get("max_prompt_chars") == prompt_budget(args),
            "Campaign did not freeze the requested bounded prompt budget")
    expected_root = args.paperorchestra_root or bundled_upstream_root()
    require(workflow.get("writer_backend") == "paperorchestra",
            "Campaign did not freeze the requested PaperOrchestra writer")
    require(Path(workflow["paperorchestra_root"]).resolve() == expected_root.resolve(),
            "Campaign did not use the requested bundled or override PaperOrchestra source")
    reports = [json.loads((root / json.loads(value)).read_text())
               for key, value in saved.items() if key.startswith("paperorchestra-attempt/")]
    successful = [report for report in reports if report.get("status") == "success"]
    require(successful, "Campaign has no successful committed PaperOrchestra runtime report")
    operations = set()
    for report in reports:
        require(report.get("upstream", {}).get("revision") == UPSTREAM_REVISION
                and report["upstream"].get("files") == SOURCE_HASHES,
                "Executed PaperOrchestra source differs from the pinned bundle")
        require(report.get("synthetic_agents") is True and report.get("synthetic_pdf") is False,
                "PaperOrchestra E2E requires scripted agents and real PDF services")
        require(report.get("config", {}).get("allow_live") is False,
                "PaperOrchestra report unexpectedly permits live calls")
        for operation in report.get("controller_pdf_receipts", []):
            receipt = operation["receipt"]
            require(receipt.get("status") == "success"
                    and receipt.get("image_digest") == args.compiler_image
                    and receipt.get("network_disabled") is True,
                    "PDF receipt does not confirm successful execution in the offline image")
            operations.add(receipt["operation"])
    require({"compile", "text", "render"} <= operations,
            "PaperOrchestra did not exercise real PDF compilation, extraction, and rendering")
    return {"source": "override" if args.paperorchestra_root else "bundled",
            "revision": UPSTREAM_REVISION, "runtime_attempts": len(reports),
            "successful_runtime_attempts": len(successful), "pdf_operations": sorted(operations)}


def verify_container(args):
    task = args.task.resolve()
    require(task.is_file(), f"Task file does not exist: {task}")
    output = args.work_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Each invocation owns a new evidence directory; failed attempts are retained.
    work = Path(tempfile.mkdtemp(prefix="container-", dir=output))
    run_args = ["run", task, "--profile", "offline", "--runtime", args.engine,
                "--image", args.image, "--runs-dir", work / "runs",
                "--max-prompt-chars", prompt_budget(args)]
    if args.context:
        run_args.extend(["--context", args.context])
    if args.writer_backend == "paperorchestra":
        require(args.compiler_image is not None,
                "PaperOrchestra E2E requires an existing --compiler-image")
        run_args.extend(["--writer-backend", "paperorchestra"])
        if args.paperorchestra_root is not None:
            run_args.extend(["--paperorchestra-root", args.paperorchestra_root.resolve()])
    if args.compiler_image:
        run_args.extend(["--compiler-image", args.compiler_image, "--require-pdf"])
    print(f"Running offline CPU campaign; evidence: {work}", flush=True)
    run = cli(run_args, cwd=work, output=work / "run")
    return verify_completed_campaign(args, work, run)


def verify_completed_campaign(args, work, run):
    """Audit/resume a completed run without repeating the initial CPU campaign.

    Pass the original arguments, owned work directory, and final object parsed
    from its preserved run.stdout.txt using this same installed interpreter.
    The frozen implementation and runtime identities are checked by the audit.
    """
    task = args.task.resolve()
    work = Path(work).resolve()
    require(task.is_file(), f"Task file does not exist: {task}")
    require(work.is_dir(), f"Verification work directory does not exist: {work}")
    require(run.get("status") == "complete", "CPU campaign did not complete")
    require(run.get("outcome") in {"development_improvement", "no_improvement"},
            "CPU campaign has no completed research outcome")
    root = Path(run["directory"]).resolve()
    require(root.is_relative_to(work / "runs"), "CLI returned an unexpected campaign directory")
    write_json(work / "run.json", run)
    inspected = cli(["inspect", root], cwd=work, output=work / "inspect-before")
    verify_offline(inspected)
    # Check resume against the just-completed campaign. Audit deliberately adds
    # a reexecution operation; a later resume would refresh reports to include
    # that new evidence and would no longer be a byte-for-byte no-op baseline.
    before = campaign_snapshot(root)
    print("Checking completed-campaign resume adds no operations or evidence", flush=True)
    resumed = cli(["resume", root], cwd=work, output=work / "resume")
    require(resumed == run, "Resume changed the completed campaign result")
    require(campaign_snapshot(root) == before, "Completed resume changed ledger or evidence bytes")
    inspected_after = cli(["inspect", root], cwd=work, output=work / "inspect-after")
    verify_offline(inspected_after)
    print("Reexecuting the selected development experiment for the audit", flush=True)
    audit = cli(["audit", root, "--include-final"], cwd=work, output=work / "audit")
    require(audit.get("status") == "pass", "Mechanical evidence audit did not pass")
    require(audit.get("live_model_calls") == 0, "Audit unexpectedly reports live model calls")
    execution = audit.get("execution_audit", {})
    require(execution.get("reexecution") == "pass", "Selected experiment did not reproduce")
    require(execution.get("reexecution_measurement_origin") == "container",
            "Audit did not reexecute a real container")
    require(execution.get("scoring") == "pass", "Trusted score recalculation did not pass")
    observations = execution.get("observations", [])
    require(observations and all(item.get("measurement_origin") == "container"
                                for item in observations),
            "E2E observations contain missing or synthetic measurements")
    summary = {"status": "pass", "task": json.loads(task.read_text())["name"],
               "engine": args.engine, "writer_backend": args.writer_backend,
               "outcome": run["outcome"], "observations": len(observations),
               "audit": "pass", "real_container_reexecution": "pass",
               "no_op_resume": "pass", "live_model_calls": 0,
               "verification_order": ["campaign", "inspect", "completed_resume",
                                      "audit_reexecution"],
               "pdf_required": bool(args.compiler_image),
               "max_prompt_chars": prompt_budget(args),
               "scope": "Generated CPU task with scripted agent responses; mechanical workflow "
                        "verification, not live-agent research or scientific validity"}
    if args.writer_backend == "paperorchestra":
        summary["paperorchestra"] = verify_paperorchestra(root, args)
    write_json(work / "summary.json", summary)
    print(f"Evidence preserved at {work}", flush=True)
    return summary


def verify_junit(args):
    root = ET.parse(args.input).getroot()
    cases = root.findall(".//testcase")
    require(cases, "Test report contains no executed test cases")
    skipped = [case.attrib.get("name", "unknown") for case in cases
               if case.find("skipped") is not None]
    require(not skipped, "Required integration coverage skipped tests: " + ", ".join(skipped))
    require(not any(case.find("failure") is not None or case.find("error") is not None
                    for case in cases), "Test report contains failures or errors")
    return {"status": "pass", "tests": len(cases), "skipped": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    package = subparsers.add_parser("package", help="Check a wheel installed outside the checkout")
    package.add_argument("--source-root", type=Path, required=True)
    package.set_defaults(verify=verify_package)
    container = subparsers.add_parser("container", help="Run, audit, and resume a real offline task")
    container.add_argument("--task", type=Path, default=Path("examples/quickstart.json"))
    container.add_argument("--image", required=True, help="Existing local worker image")
    container.add_argument("--engine", choices=("docker", "podman"), default="docker")
    container.add_argument("--context", help="Optional Docker context; never starts a daemon")
    container.add_argument("--work-dir", type=Path, default=Path(".release-validation"))
    container.add_argument("--writer-backend", choices=("canonical", "paperorchestra"),
                           default="canonical")
    container.add_argument("--paperorchestra-root", type=Path,
                           help="Optional verified source override; defaults to bundled engine")
    container.add_argument("--compiler-image", help="Existing local image; requires a real PDF")
    container.add_argument("--max-prompt-chars", type=bounded_prompt_chars,
                           help="Frozen input bound: defaults to 1000000 for PaperOrchestra, "
                                "200000 otherwise; maximum 1000000")
    container.set_defaults(verify=verify_container)
    junit = subparsers.add_parser("junit", help="Reject missing, skipped, or failed integration tests")
    junit.add_argument("--input", type=Path, required=True)
    junit.set_defaults(verify=verify_junit)
    for command in (package, container, junit):
        command.add_argument("--output", type=Path, help="Optional JSON summary path")
    args = parser.parse_args()
    try:
        result = args.verify(args)
    except (OSError, ValueError, sqlite3.Error, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Release verification failed: {error}\n")
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
