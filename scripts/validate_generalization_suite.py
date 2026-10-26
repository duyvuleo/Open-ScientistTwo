"""Validate the default suite executor with fixture agents and real containers.

Only the CodexBackend factory and trusted fixture-plugin registration are
substituted. The controller, suite, budgets, runners, graders and process
execution are unchanged. This script cannot invoke a live model endpoint.
"""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time


def operations(path, table):
    db = sqlite3.connect((path / "campaign.sqlite").as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY rowid")]
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--image", default="scientisttwo-native:0.1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "tests"))
    import pytest
    from test_workspace_task import (
        FINAL_CANARY,
        ResearchFixtures,
        development,
        install_plugin,
        make_workspace,
    )
    from scientisttwo.live.adapters import implementation_identity
    from scientisttwo.live.audit import audit_live
    from scientisttwo.live.models import LiveSpec
    from scientisttwo.live.suite import (
        SuiteAssignment,
        SuiteSpec,
        inspect_suite,
        run_suite,
        suite_identity,
    )
    from scientisttwo.store import Store, file_digest

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    output = root / "suite"
    kinds = {"regression": "regression", "text": "text"}
    factories = []
    backends = {}
    started = time.monotonic()

    def fixture_factory(workspace, model, *, allow_live=False, timeout=120):
        assignment = Path(workspace).parent.name
        assert assignment in kinds and model == "scripted-fixture-v1"
        assert allow_live is True
        factories.append(assignment)
        backend = ResearchFixtures(kind=kinds[assignment], meta_improves=assignment == "regression")
        assert backend.synthetic is True
        backends[assignment] = backend
        return backend

    with pytest.MonkeyPatch.context() as patch:
        install_plugin(patch)
        patch.setattr("scientisttwo.backends.CodexBackend", fixture_factory)
        assignments = []
        for assignment, kind in kinds.items():
            native = make_workspace(
                root / "inputs" / assignment, kind=kind, context=args.context, image=args.image
            )
            campaign = LiveSpec(
                native=native,
                development=development(kind),
                model="scripted-fixture-v1",
                mode="research",
                max_calls=100,
                max_experiments=100,
                campaign_seconds=600,
                model_retries=0,
                search_rounds=2,
                engineering_retries=0,
                ablation_refinements=1,
                review_rounds=1,
                meta_refinements=1,
                implementation=implementation_identity(),
            )
            assignments.append(SuiteAssignment(id=assignment, repetition=0, campaign=campaign))
        config = SuiteSpec(
            name="synthetic-generalization-suite",
            wall_seconds=1200,
            max_calls=200,
            max_experiments=200,
            assignments=assignments,
            implementation=suite_identity(),
        )
        (root / "suite-manifest.json").write_text(config.model_dump_json(indent=2) + "\n")
        # Use the real default executor; no executor/runner/process double is supplied.
        result = run_suite(config, output, allow_live=True)
        assert result["status"] == "complete", result
        assert result["development_improvements"] == result["declared_assignments"] == 2
        shared = operations(output, "suite_operations")
        assert all(row["status"] != "running" for row in shared)
        cases = []
        child_operations = {}
        final_hashes = {}
        receipts_verified = 0
        required_roles = {
            "evolve",
            "ablations",
            "ablation_critic",
            "refine",
            "draft",
            "review",
            "rebuttal",
            "meta",
        }
        for assignment in assignments:
            child = output / "campaigns" / assignment.id
            backend = backends[assignment.id]
            roles = {role for role, _ in backend.seen}
            assert required_roles <= roles
            assert FINAL_CANARY not in json.dumps(backend.seen)
            assert str(assignment.campaign.native.private_dir) not in json.dumps(backend.seen)
            assert str(assignment.campaign.native.evaluator_dir) not in json.dumps(backend.seen)
            records = operations(child, "live_operations")
            child_operations[assignment.id] = records
            paired = [row for row in shared if row["assignment"] == assignment.id]
            assert len(paired) == len(records)
            for row in records:
                mirror = next(r for r in paired if r["key"] == row["key"])
                assert all(mirror[key] == value for key, value in row.items())
            store = Store(child)
            try:
                assert not store.verify()
                assert store.get("suite_owner")["assignment"] == assignment.id
                provenance = store.get("live_backend")
                assert provenance == {
                    "backend": "fixture",
                    "synthetic": True,
                    "model": "scripted-fixture-v1",
                }
                best = store.get("live_best")
                images = store.get("live_images")
            finally:
                store.close()
            audit = audit_live(child)
            assert audit["status"] == "pass", audit
            final = json.loads((child / "final" / "selected" / "result.json").read_text())
            assert final["status"] == "completed" and final["live_model_calls"] == 0
            for path in sorted(child.glob("final/*/result.json")):
                final_hashes[str(path.relative_to(output))] = file_digest(path)
            for directory in [child, *sorted((child / "final").iterdir())]:
                receipt_store = Store(directory)
                try:
                    assert not receipt_store.verify()
                    for row in receipt_store.db.execute(
                        "SELECT path FROM artifacts WHERE kind='native-execution'"
                    ):
                        receipt = receipt_store.json(row[0])
                        argv = receipt["argv"]
                        assert "--network=none" in argv and "--read-only" in argv
                        mounts = {
                            argv[i + 1].rsplit(":", 2)[1]
                            for i, arg in enumerate(argv[:-1])
                            if arg == "-v"
                        }
                        if "/workspace" in mounts:
                            assert "/data/private" not in mounts and "/evaluator" not in mounts
                        else:
                            assert {
                                "/submission",
                                "/data/private",
                                "/evaluator",
                                "/grade-output",
                            } <= mounts
                        assert receipt["returncode"] == 0 and not receipt["timed_out"]
                        receipts_verified += 1
                finally:
                    receipt_store.close()
            cases.append(
                {
                    "id": assignment.id,
                    "kind": kinds[assignment.id],
                    "roles": sorted(roles),
                    "synthetic_agents": True,
                    "synthetic_task_data": True,
                    "fixture_role_invocations": len(backend.seen),
                    "live_model_calls": 0,
                    "logical_experiment_attempts": sum(r["kind"] == "experiment" for r in records),
                    "best": best["result"],
                    "final": final,
                    "audit": audit,
                    "images": images,
                    "shared_ledger_matches_child": True,
                }
            )
        assert len(shared) == sum(len(rows) for rows in child_operations.values())
        assert result["model_attempts"] == sum(len(backend.seen) for backend in backends.values())
        count = len(factories)
        resumed = run_suite(config, output, allow_live=True)
        assert resumed == result and len(factories) == count
        assert operations(output, "suite_operations") == shared
        for assignment in assignments:
            child = output / "campaigns" / assignment.id
            assert operations(child, "live_operations") == child_operations[assignment.id]
        for relative, expected in final_hashes.items():
            assert file_digest(output / relative) == expected
        inspected_config, inspected = inspect_suite(output)
        assert inspected_config == config and inspected == result
        assert len(factories) == count and operations(output, "suite_operations") == shared
        verification = {
            "status": "pass",
            "scope": "Default suite executor and aggregate budgets; synthetic fixture agents/data, real isolated CPU experiments",
            "synthetic_agents": True,
            "synthetic_task_data": True,
            "live_model_calls": 0,
            "fixture_role_invocations": result["model_attempts"],
            "logical_experiment_attempts": result["experiment_attempts"],
            "suite_model_attempts_are_fixture_reservations_not_live_LLM_calls": True,
            "factory_instantiations": factories,
            "container_receipts_verified": receipts_verified,
            "shared_ledger_matches_all_children": True,
            "resume_added_factory_instantiations": 0,
            "resume_added_operations": 0,
            "resume_changed_final_results": 0,
            "inspect_added_model_calls": 0,
            "protected_final_canary_absent_from_role_inputs": True,
            "protected_host_paths_absent_from_role_inputs": True,
            "container_network_and_private_mount_checks": "pass",
            "context": args.context,
            "image": args.image,
            "elapsed_seconds": time.monotonic() - started,
            "implementation": implementation_identity(),
            "validation_script_sha256": file_digest(Path(__file__)),
            "fixture_source_sha256": file_digest(project / "tests" / "test_workspace_task.py"),
            "suite_summary": result,
            "cases": cases,
        }
        (root / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": "pass",
                    "output": str(root),
                    "live_model_calls": 0,
                    "fixture_role_invocations": result["model_attempts"],
                    "logical_experiment_attempts": result["experiment_attempts"],
                    "container_receipts_verified": receipts_verified,
                    "elapsed_seconds": verification["elapsed_seconds"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
