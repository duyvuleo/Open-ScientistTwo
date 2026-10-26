"""Readiness is a verified evidence gate, never a catalog-entry count."""

import json
import math
from pathlib import Path
import re

from ..artifacts import load_bundle
from ..benchmarks.files import tree_manifest
from ..live.audit import audit_live
from ..live.models import Candidate
from ..live.source_tree import candidate_snapshot
from ..store import Store
from .catalog import get_recipe, recipes
from .preparation import CPU_TASKS, manifest_hash, verify_prepared
from .validation import sealed_evidence


def _require_immutable_images(images):
    """Accept Docker and Podman config-ID spelling without rewriting evidence."""
    if (not isinstance(images, dict) or not images
            or any(not isinstance(value, str)
                   or re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", value) is None
                   for value in images.values())):
        raise ValueError("Validation runtime images are not complete immutable digests")


def _metrics_equal(left, right):
    return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
            and all(isinstance(left[key], (float, int)) and isinstance(right[key], (float, int))
                    and not isinstance(left[key], bool) and not isinstance(right[key], bool)
                    and math.isfinite(left[key]) and math.isfinite(right[key])
                    and math.isclose(left[key], right[key], rel_tol=1e-10, abs_tol=1e-10) for key in left))


def _verify_audit_proof(store, saved, current):
    if saved.get("complete") is not True:
        raise ValueError("The saved audit did not complete applicable metric/evaluator recomputations")
    required = set(current.get("required_container_replays", []))
    rows = saved.get("container_replays", [])
    if required != {row["id"] for row in rows} or len(rows) != len(required):
        raise ValueError("Saved audit does not cover every required trusted-container replay")
    sources = current.get("replay_sources", {})
    for row in rows:
        proof = store.json(row["evidence_ref"])
        source = sources[row["id"]]
        if (proof.get("verification_kind") != "container_evaluator_replay" or proof.get("status") != "pass"
                or proof.get("id") != row["id"] or proof.get("source_grade_ref") != source["grade_ref"]
                or not _metrics_equal(proof.get("metrics"), row.get("metrics"))
                or not _metrics_equal(row.get("metrics"), source["metrics"])):
            raise ValueError("Container replay proof differs from current frozen score evidence")
        executions = proof.get("execution_refs", [])
        if not executions:
            raise ValueError("Container replay has no retained actual execution receipts")
        for ref in executions:
            execution = store.json(ref)
            if execution.get("returncode") != 0 or execution.get("timed_out") or execution.get("limit_exceeded"):
                raise ValueError("Container replay did not complete successfully")
            store.read(execution["log_ref"])
            store.json(execution["job_ref"])
        grade = store.json(proof["replayed_grade_ref"])
        if grade.get("valid_submission") is not True or not _metrics_equal(grade.get("metrics"), source["metrics"]):
            raise ValueError("Replayed trusted grade differs from its proof")
    total = len(rows)
    for field, kind in (("host_metric_recomputations", "trusted_host_metric_recomputation"),
                        ("paired_correctness_recomputations", "paired_output_recomputation")):
        expected = {item["id"]: item for item in current.get(field, [])}
        records = saved.get(field, [])
        if set(expected) != {item["id"] for item in records} or len(expected) != len(records):
            raise ValueError("Saved host metric/correctness proofs do not cover the current observations")
        for row in records:
            proof, actual = store.json(row["evidence_ref"]), store.json(expected[row["id"]]["evidence_ref"])
            if (proof.get("verification_kind") != kind or proof.get("status") != "pass"
                    or proof.get("id") != row["id"] or proof.get("observation_ref") != actual.get("observation_ref")
                    or not _metrics_equal(proof.get("metrics"), row.get("metrics"))
                    or not _metrics_equal(row.get("metrics"), expected[row["id"]].get("metrics"))):
                raise ValueError("Host recomputation proof differs from current observation evidence")
        total += len(records)
    if total == 0:
        raise ValueError("Audit requires at least one actual metric or correctness recomputation")


def verify_validation(prepared, output):
    prepared, output = Path(prepared), Path(output)
    manifest = verify_prepared(prepared)
    receipt = json.loads((output / "receipt.json").read_text())
    if receipt.get("task_id") != manifest["task_id"] or receipt.get("recipe_sha256") != manifest["recipe_sha256"]:
        raise ValueError("Validation receipt names a different recipe")
    if receipt.get("prepared_manifest_sha256") != manifest_hash(manifest):
        raise ValueError("Validation receipt belongs to different prepared inputs")
    current_evidence = sealed_evidence(output / "campaign")
    if not receipt.get("evidence"):
        raise ValueError("Validation receipt does not bind any retained evidence")
    for name, expected in receipt.get("evidence", {}).items():
        if name.endswith("campaign.sqlite"):
            if name not in current_evidence:
                raise ValueError("Validation evidence database is missing")
        elif current_evidence.get(name) != expected:
            raise ValueError("Validation evidence changed or is missing")
    if receipt.get("provenance") != manifest["provenance"]:
        raise ValueError("Validation data provenance differs from prepared manifest")
    if receipt.get("dataset_identity") != manifest.get("dataset_identity"):
        raise ValueError("Dataset provider identity differs from the prepared source evidence")
    if receipt.get("live_model_calls") != 0 or receipt.get("synthetic_agents") is not True:
        raise ValueError("Readiness validation requires scripted agents and zero live calls")
    audit = audit_live(output / "campaign")
    if audit["status"] != "pass":
        raise ValueError("Saved validation fails current audit")
    store = Store(output / "campaign")
    try:
        _verify_audit_proof(store, receipt.get("audit", {}).get("verification", {}), audit.get("verification", {}))
        for label, observation in receipt.get("observations", {}).items():
            row = store.db.execute("SELECT output_ref FROM calls WHERE key=?",
                                   (f"development/validation/{label}/full",)).fetchone()
            if row is None or store.json(row[0]) != observation:
                raise ValueError("Reported development measurement lacks matching execution evidence")
        for label, final in receipt.get("finals", {}).items():
            key = receipt.get("final_operation_keys", {}).get(label, f"final/validation-{label}")
            if key not in {"final/baseline", "final/selected", f"final/validation-{label}"}:
                raise ValueError("Validation final alias refers to an unrelated operation")
            row = store.db.execute("SELECT result_ref FROM live_operations WHERE key=?",
                                   (key,)).fetchone()
            if final.get("observation_ref"):
                bound = store.json(final["observation_ref"])
                matches = (row is not None and store.json(row[0]) == bound
                           and final.get("score") == bound["value"]
                           and final.get("protocol_hash") == bound["protocol_hash"])
            else:
                matches = row is not None and store.json(row[0]) == final
            if not matches:
                raise ValueError("Reported final score lacks matching protected grading evidence")
            if key in {"final/baseline", "final/selected"}:
                frozen = store.get("live_frozen")
                proposed = frozen["baseline"] if key == "final/baseline" else frozen["best"]["candidate"]
                candidate = Candidate.model_validate(proposed)
                snapshot = candidate_snapshot(store, candidate)
                files = {name: {"sha256": item.sha256, "size": item.size}
                         for name, item in load_bundle(store, snapshot.bundle_ref).files.items()}
                if files != tree_manifest(prepared / ("reference" if label == "reference" else "control")) or candidate.command != ["python", "train.py"]:
                    raise ValueError("Final alias source/command differs from the declared prepared method")
        backend = store.get("live_backend")
        if backend != {"backend": "fixture", "synthetic": True, "model": "scripted-fixture-v1"}:
            raise ValueError("Validation backend evidence differs from scripted execution")
        images = store.get("live_images")
        if not images or images != receipt.get("images"):
            raise ValueError("Frozen runtime image evidence is missing")
        _require_immutable_images(images)
        probe_ref = store.get("runtime_probe_ref")
        probe = store.json(probe_ref) if probe_ref else None
        if probe_ref != receipt.get("runtime_probe_ref") or probe != receipt.get("runtime_probe"):
            raise ValueError("Runtime validation proof differs from sealed execution evidence")
        if probe:
            execution = store.json(probe["execution_ref"])
            if execution.get("image_id") != images["execution"] or execution.get("returncode") != 0 or execution.get("timed_out"):
                raise ValueError("Runtime probe did not successfully execute in the frozen runtime image")
            if probe.get("profile", {}).get("name") != get_recipe(manifest["task_id"]).runtime_profile:
                raise ValueError("Runtime probe used a different recipe environment profile")
        if store.get("runtime_hardware") != receipt.get("hardware"):
            raise ValueError("Hardware identity differs from sealed runtime evidence")
        recovery_ref = store.get("validation_recovery")
        if receipt.get("gates", {}).get("recovery"):
            if not recovery_ref:
                raise ValueError("Recovery claim has no sealed operation-ledger proof")
            proof = store.json(recovery_ref)
            if proof["before"] != proof["after"] or proof["state_before"] != proof["state_after"]:
                raise ValueError("Recovery repeated work or changed the completed outcome")
            if [dict(row) for row in store.db.execute("SELECT * FROM live_operations ORDER BY rowid")] != proof["after"]:
                raise ValueError("Validation operation ledger changed after the no-duplicate-work recovery proof")
        if store.get("live_state") != receipt.get("outcome"):
            raise ValueError("Campaign state differs from its recorded validation outcome")
    finally:
        store.close()
    expected = {"reference", "control"}
    valid = (set(receipt.get("observations", {})) == expected and set(receipt.get("finals", {})) == expected
             and all(row["status"] == "success" for row in receipt["observations"].values())
             and all(row["status"] == "completed" for row in receipt["finals"].values())
             and receipt.get("gates", {}).get("recovery") is True
             and (receipt.get("runtime_probe") or {}).get("report", {}).get("status") == "pass"
             and (receipt.get("hardware") or {}).get("status") == "verified"
             and receipt.get("audit", {}).get("verification", {}).get("complete") is True)
    if receipt.get("level") == "full":
        valid = valid and all(len(row["values"]) == 3 for row in receipt["observations"].values())
    actual = (valid and manifest["provenance"] == "actual_data" and receipt.get("level") == "full"
              and manifest.get("dataset_identity", {}).get("status") == "verified")
    if actual != receipt.get("actual_task_validated"):
        raise ValueError("Actual task readiness claim differs from execution evidence")
    return receipt


def readiness(root):
    """Always report every canonical slot, including absent data and failed tasks."""
    root = Path(root)
    available_sources = {}
    for path in sorted((root / "downloads").glob("*/source_receipt.json")):
        try:
            if path.is_symlink() or path.stat().st_size > 128_000_000:
                continue
            receipt = json.loads(path.read_text())
            if receipt.get("status", "completed") == "completed" and receipt.get("task_id"):
                available_sources.setdefault(receipt["task_id"], []).append(path.parent)
        except (ValueError, OSError):
            continue
    rows = []
    for recipe in recipes():
        task = root / recipe.task_id
        row = {
            "task_id": recipe.task_id, "family": recipe.family,
            "adapter_family": recipe.adapter_family,
            "variant": None, "recipe_version": recipe.recipe_version,
            "original_starter_status": recipe.original_starter_status,
            "reference_baseline": recipe.reference_baseline,
            "tracks": {"upstream_compatibility": recipe.upstream_grading,
                       "audited_v1": "not_validated"},
            "gates": {"recipe": "pass", "implementation": "pass" if recipe.task_id in CPU_TASKS | {"vectorization"} else "unverified",
                      "data": "missing", "assets": "missing" if recipe.assets else "not_required",
                      "runtime": "unverified", "hardware": "unverified",
                      "evaluator": "unverified", "execution": "unverified", "audit": "unverified",
                      "recovery": "unverified", "research_cycle": "unverified", "live_agents": "unevaluated"},
            "ready": False, "blockers": [], "deviations": recipe.deviations,
            "provider_sources": [{"path": str(path.resolve()), "status": "receipt_available_not_yet_prepared"}
                                 for path in available_sources.get(recipe.task_id, [])],
        }
        prepared = task / "prepared"
        try:
            if not (prepared / "manifest.json").is_file():
                if row["provider_sources"]:
                    row["gates"]["data"] = "acquired_not_prepared"
                    row["blockers"].append("Provider data have been acquired but canonical protected preparation has not completed")
                    missing_assets = [name for name in recipe.assets
                                      if not any((source / "assets" / name).is_dir()
                                                 for source in available_sources[recipe.task_id])]
                    if missing_assets:
                        row["blockers"].append("Required local asset caches are missing: " + ", ".join(missing_assets))
                else:
                    row["blockers"].append("Prepared task data and assets are missing")
            else:
                manifest = verify_prepared(prepared)
                if manifest["task_id"] != recipe.task_id:
                    raise ValueError("Prepared task stored under the wrong canonical slot")
                row["variant"] = manifest["variant"]
                row["gates"]["assets"] = "frozen_not_runtime_verified" if recipe.assets else "not_required"
                identity = manifest.get("dataset_identity", {"status": "unverified"})
                row["dataset_identity"] = identity
                row["gates"]["data"] = ("pass" if manifest["provenance"] == "actual_data" and identity["status"] == "verified"
                                         else "synthetic_fixture_only" if manifest["provenance"] == "synthetic_fixture"
                                         else "provider_identity_unverified")
                if manifest["provenance"] != "actual_data":
                    row["blockers"].append("Synthetic fixtures do not establish actual-task readiness")
                elif identity["status"] != "verified":
                    row["blockers"].append("Dataset provider identity is unverified; a caller's actual_data assertion does not establish canonical task readiness")
                output = task / "validation"
                if not (output / "receipt.json").is_file():
                    row["blockers"].append("Actual execution, audit and recovery evidence is missing")
                    if recipe.runtime_profile != "cpu-tabular":
                        row["blockers"].append("A compatible runtime image and allocated GPU must pass the actual framework/hardware probe")
                else:
                    receipt = verify_validation(prepared, output)
                    for name in ("runtime", "hardware"):
                        row["gates"][name] = "pass" if receipt["gates"].get(name) else "unverified"
                    row["gates"]["evaluator"] = "pass" if receipt["gates"]["final"] else "fail"
                    if receipt["gates"].get("runtime") and recipe.assets:
                        row["gates"]["assets"] = "pass"
                    row["gates"]["execution"] = "pass" if receipt["gates"]["development"] else "fail"
                    for name in ("audit", "recovery"):
                        row["gates"][name] = "pass" if receipt["gates"][name] else "fail"
                    role_keys = receipt.get("roles", [])
                    complete_cycle = (receipt.get("level") == "full"
                                      and receipt.get("outcome", {}).get("status") == "complete"
                                      and receipt.get("outcome", {}).get("outcome") in {"no_improvement", "development_improvement"}
                                      and all(receipt["gates"].values()))
                    row["gates"]["research_cycle"] = "pass" if complete_cycle else "not_exercised"
                    row["research_roles_exercised"] = role_keys
                    row["downstream_roles_not_exercised"] = [name for name in ("ablations", "draft", "review", "rebuttal", "meta") if name not in role_keys]
                    row["gates"]["downstream_branches"] = "pass" if complete_cycle and not row["downstream_roles_not_exercised"] else "not_exercised"
                    row["ready"] = receipt["actual_task_validated"]
                    if row["ready"]:
                        row["gates"]["implementation"] = "pass"
                    row["tracks"]["audited_v1"] = "validated" if row["ready"] else "partial_validation"
                    if not row["ready"]:
                        row["blockers"].append("Full actual-data validation gates have not all passed")
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
            row["gates"]["audit"] = "fail"
            row["blockers"].append(str(error))
        rows.append(row)
    from .workflow_contracts import verified_workflow_contracts
    contracts = verified_workflow_contracts(root)
    families = {row["adapter_family"] for row in rows}
    exercised = {name for name, proof in contracts.items() if proof["status"] == "pass"}
    return {
        "schema_version": 1, "canonical_tasks": 13, "assigned": 13,
        "ready_count": sum(row["ready"] for row in rows),
        "full_audited_testbed_ready": all(row["ready"] for row in rows) and families <= exercised,
        "research_families_exercised": sorted(exercised),
        "research_families_missing": sorted(families - exercised),
        "workflow_contracts": contracts,
        "live_model_calls": 0, "live_autonomous_research": "unevaluated", "tasks": rows,
    }


def recipe_readiness(task_id, root):
    get_recipe(task_id)
    return next(row for row in readiness(root)["tasks"] if row["task_id"] == task_id)
