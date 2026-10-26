"""Reusable execution-family workflow evidence, independent of task improvement."""

import json
from pathlib import Path

from ..benchmarks.files import file_hash
from ..live.source_tree import compare_source_trees
from ..store import Store
from .catalog import get_recipe
from .preparation import canonical, manifest_hash

ADAPTER_FAMILIES = {"cpu_frames", "checkpoint_pipeline", "paired_timing"}
REQUIRED_ROLES = {"evolve", "ablations", "draft", "review", "rebuttal", "meta", "refine"}
REQUIRED_RESEARCH_ROLES = {
    "initial_idea", "additional_idea", "novelty_checker", "baseline_plan", "full_set_coder",
    "selection_review", "evolve", "ablation_planner", "ablation_coder", "control_reviewer",
    "initial_drafter", "writing_critic", "draft_enhancer", "scientific_review",
    "rebuttal_planner", "rebuttal_coder", "meta", "method_disposition", "refine",
}


def workflow_proof(validation_output):
    """Require measured matched controls, not merely scripted role invocations."""
    from .readiness import verify_validation

    output = Path(validation_output).resolve()
    configuration = json.loads((output / "validation-spec.json").read_text())
    prepared = Path(configuration["development"]["settings"]["prepared_root"])
    receipt = verify_validation(prepared, output)
    family = get_recipe(receipt["task_id"]).adapter_family
    research = configuration.get("engine_version") == "research-v2"
    missing = (REQUIRED_RESEARCH_ROLES if research else REQUIRED_ROLES) - set(receipt.get("roles", []))
    if receipt.get("level") != "full" or missing or not all(receipt["gates"].values()):
        raise ValueError("Full measured workflow branches are not covered: " + ", ".join(sorted(missing)))
    store = Store(output / "campaign")
    try:
        ablations, rebuttals = [], []
        for key, ref in store.db.execute("SELECT key,output_ref FROM calls WHERE key LIKE 'control/%'"):
            evidence = store.json(ref)
            matched, control = evidence["matched"], evidence["control"]
            if matched["status"] != "success" or control["status"] != "success":
                continue
            if matched["protocol_hash"] != control["protocol_hash"]:
                raise ValueError("Workflow controls did not use a matched evaluation protocol")
            if not matched.get("log_refs") or not control.get("log_refs"):
                raise ValueError("Workflow branch lacks real execution receipts")
            if matched["phase"] == "full" and control["phase"] == "full":
                changed = compare_source_trees(store, matched["source_tree_ref"], control["source_tree_ref"])
                if research:
                    declared = evidence.get("plan", {}).get("changed_files", [])
                    if (changed and set(changed) == set(declared)
                            and evidence.get("review", {}).get("decision") in {"Accept", "DiagnosticOnly"}):
                        ablations.append({"key": key, "evidence_ref": ref, "changed_files": changed,
                                          "interpretation": evidence["review"]})
                elif len(changed) == 1:
                    ablations.append({"key": key, "evidence_ref": ref, "changed_file": changed[0]})
            if matched["phase"] == "rebuttal" and control["phase"] == "rebuttal" and len(matched["values"]) >= 2:
                rebuttals.append({"key": key, "evidence_ref": ref})
        if not ablations or not rebuttals:
            raise ValueError("Workflow coverage requires executed component-removal and supplementary matched experiments")
        trials = [store.json(row[0]) for row in store.db.execute("SELECT output_ref FROM calls WHERE key LIKE 'live-trial/%'")]
        rejected = [item for item in trials if item["decision"]["decision"] == "Bad" and item["result"]["status"] == "success"]
        failed_revisions = [item for item in rejected if item["id"].startswith("meta/")]
        incumbent = store.get("live_best")
        if not rejected or not failed_revisions or not incumbent or any(item["id"] == incumbent["id"] for item in failed_revisions):
            raise ValueError("Workflow coverage requires measured rejection and unsuccessful revision preserving the incumbent")
        # Ensure the representative actually used the execution contract it claims.
        observations = [store.json(row[0]) for row in store.db.execute("SELECT output_ref FROM calls WHERE key LIKE 'development/%'")]
        successful = [item for item in observations if item["status"] == "success"]
        if family == "paired_timing" and not all(item.get("timing_refs") for item in successful):
            raise ValueError("Paired timing workflow proof lacks actual paired measurements")
        if family == "checkpoint_pipeline" and not all(item.get("evaluation_refs") for item in successful):
            raise ValueError("Checkpoint workflow proof lacks isolated inference/scoring receipts")
    finally:
        store.close()
    return {"schema_version": 2 if research else 1,
            **({"engine_version": "research-v2", "roles": sorted(REQUIRED_RESEARCH_ROLES)} if research else {}),
            "adapter_family": family, "validation_output": str(output),
            "validation_receipt_sha256": file_hash(output / "receipt.json"),
            "prepared_manifest_sha256": receipt["prepared_manifest_sha256"],
            "data_provenance": receipt["provenance"], "synthetic_agents": True,
            "live_model_calls": 0, "ablations": ablations, "rebuttals": rebuttals,
            "rejected_trials": [item["id"] for item in rejected],
            "unsuccessful_revisions": [item["id"] for item in failed_revisions],
            "preserved_incumbent": incumbent["id"],
            "scope": "Real measured workflow branch coverage for this reusable execution interface; not proof of canonical task data or live autonomous research quality"}


def register_workflow_contract(root, validation_output):
    proof = workflow_proof(validation_output)
    path = Path(root) / "workflow-contracts" / proof["adapter_family"] / (manifest_hash(proof) + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != canonical(proof):
        raise ValueError("Workflow contract proof changed at its content-addressed path")
    if not path.exists():
        path.write_bytes(canonical(proof))
    return {"status": "pass", "path": str(path.resolve()), **proof}


def verified_workflow_contracts(root):
    results = {}
    for family in sorted(ADAPTER_FAMILIES):
        result = {"status": "missing", "proofs": [], "errors": []}
        for path in sorted((Path(root) / "workflow-contracts" / family).glob("*.json")):
            try:
                saved = json.loads(path.read_text())
                actual = workflow_proof(saved["validation_output"])
                if saved != actual or actual["adapter_family"] != family or path.stem != manifest_hash(actual):
                    raise ValueError("Workflow proof differs from current execution evidence")
                result["proofs"].append(str(path.resolve()))
                result["status"] = "pass"
            except (ValueError, OSError, KeyError) as error:
                result["errors"].append(str(error))
        results[family] = result
    return results
