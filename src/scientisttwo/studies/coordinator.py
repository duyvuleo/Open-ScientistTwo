"""Predeclared two-arm study, immutable selections, and a protected-final barrier.

The coordinator never constructs a backend or proposes an improvement. A failed
research arm stays failed; completed fixtures are never autonomous evidence.
"""

from copy import deepcopy
from pathlib import Path
import time
from types import SimpleNamespace

from ..artifacts import load_bundle
from ..live.models import Candidate, Observation
from ..live.source_tree import candidate_snapshot, freeze_source_tree
from ..live.workspace_models import WorkspaceSpec
from ..research.controller import ResearchController
from ..store import IntegrityError, canonical, digest, file_digest
from .runner import candidate_source_hash, validate_observation_receipts, verify_final_aggregation

ARM_NAMES = ("scientist", "direct")
MATCHED_FIELDS = (
    "model", "max_calls", "max_experiments", "campaign_seconds", "call_seconds",
    "token_stop_threshold", "max_prompt_chars", "history_code_chars", "log_tail_chars",
    "model_retries", "agent_output_retries", "sources", "native", "development", "implementation",
)


def _operations(store):
    return [dict(row) for row in store.db.execute(
        "SELECT * FROM live_operations WHERE kind IN ('model','experiment') AND key NOT LIKE 'final/%' ORDER BY key")]


def _no_unknown_operations(store):
    if any(row["status"] == "running" or not row["result_ref"] for row in _operations(store)):
        raise IntegrityError("Cannot freeze an arm with unknown model or experiment completion")


def _bind_source(owner):
    declaration = owner.adapter.source_tree()
    ref = freeze_source_tree(owner.store, Path(declaration["path"]), declaration["editable_paths"])
    if owner.store.get("authorized_source_snapshot") not in (None, ref):
        raise IntegrityError("Arm source snapshot differs from the common baseline")
    owner.store.set("authorized_source_snapshot", ref)
    owner.adapter.bind_source(ref)


def _common(owner):
    from ..live.adapters import implementation_identity
    config = owner.config.model_dump(mode="json")
    if not owner.backend.synthetic and owner.config.implementation != implementation_identity():
        raise IntegrityError("Live arm implementation differs from its pinned source identity")
    if owner.config.workflow.network_literature:
        raise ValueError("The paired study requires the same frozen literature; disable network retrieval")
    if owner.config.workflow.baseline_mode != "prepared_reference":
        raise ValueError("The paired study requires an immutable prepared reference")
    if owner.config.development.seeds != [0, 1, 2]:
        raise ValueError("The paired study requires matched seeds [0,1,2]")
    if not owner.backend.synthetic and owner.config.token_stop_threshold is None:
        raise ValueError("Live paired arms require an explicit observed-token stop threshold")
    compute = owner.store.get("study_compute_budget")
    if not compute or not callable(getattr(owner.budget, "experiment_scope", None)):
        raise ValueError("Each arm needs a configured StudyBudget with explicit compute/final ceilings")
    return {"config": {name: config[name] for name in MATCHED_FIELDS},
            "backend": {"name": owner.backend.name, "synthetic": owner.backend.synthetic},
            "adapter": owner.adapter.identity(), "task_information": owner.adapter.context(),
            "literature": deepcopy(owner.config.workflow.literature),
            "compute_budget": compute,
            "baseline_source_sha256": candidate_source_hash(owner.store, owner.adapter.baseline())}


def create_study_plan(study_store, research, direct, *, direct_max_iterations):
    """Freeze comparability before arm model/training operations; source binding is local only."""
    if type(direct_max_iterations) is not int or direct_max_iterations < 1:
        raise ValueError("Declare a positive direct-control iteration ceiling")
    if not research.backend.synthetic and research.config.workflow.writer_backend != "paperorchestra":
        raise ValueError("This live end-to-end study requires the integrated PaperOrchestra writer")
    owners = {"scientist": research, "direct": direct}
    if research.store.root == direct.store.root or study_store.root in {research.store.root, direct.store.root}:
        raise ValueError("Study and arm evidence require separate stores")
    saved = study_store.get("study_plan_ref")
    if not saved:
        if any(owner.store.db.execute("SELECT 1 FROM live_operations WHERE kind IN ('model','experiment')").fetchone()
               for owner in owners.values()):
            raise IntegrityError("Study plan must be frozen before any arm model or experiment operation")
        for owner in owners.values():
            _bind_source(owner)
    common = {name: _common(owner) for name, owner in owners.items()}
    if canonical(common["scientist"]) != canonical(common["direct"]):
        raise IntegrityError("Study arms differ in model, information, baseline, protocol or resource budgets")
    plan = {"schema_version": "cifar-paired-study-v1", "common": common["scientist"],
            "arm_order": ["scientist", "direct"], "direct_max_iterations": direct_max_iterations,
            "arms": {name: str(owner.store.root) for name, owner in owners.items()},
            "arm_configurations": {name: owner.config.model_dump(mode="json") for name, owner in owners.items()},
            "comparison": {"metric": "accuracy", "minimum_reference_gain": 0.02,
                           "positive_difference_each_seed": True, "seeds": [0, 1, 2],
                           "statistical_significance_claim": False},
            "coordinator_sha256": file_digest(Path(__file__)),
            "final_policy": "Both complete development selections freeze before any protected final access; one common reference and one selected method per arm, three matched seeds each"}
    ref = study_store.put_json(plan, "paired-study-plan")
    if saved not in (None, ref):
        raise IntegrityError("Frozen study plan changed")
    study_store.set("study_plan_ref", ref)
    for owner in owners.values():
        if owner.store.get("study_plan_ref") not in (None, ref):
            raise IntegrityError("Arm belongs to a different frozen study")
        owner.store.put_json(plan, "paired-study-plan")
        owner.store.set("study_plan_ref", ref)
    return {"plan_ref": ref, "plan_sha256": digest(canonical(plan)), "status": "planned",
            "live_model_calls": 0, "final_access": False}


def _normalized_protocol(value):
    result = deepcopy(value)
    # Controller-owned development directories differ between arms; their exact
    # data manifests remain compared. Evaluator source/image/resource fields stay.
    if result.get("pipeline"):
        result["pipeline"].pop("private_dir", None)
    return result


def _verified_observation(store, candidate, observation):
    observed = Observation.model_validate(observation)
    if observed.status != "success" or observed.phase != "full" or observed.metric != "accuracy":
        raise IntegrityError("Final selection requires successful full-development accuracy")
    proof = validate_observation_receipts(store, observed, expected_seeds=[0, 1, 2])
    row = store.db.execute("SELECT * FROM live_operations WHERE key=?", ("experiment/" + observed.id,)).fetchone()
    if (row is None or row["kind"] != "experiment" or row["status"] != "success"
            or not row["result_ref"] or canonical(store.json(row["result_ref"])) != canonical(observed)):
        raise IntegrityError("Selection observation differs from its committed experiment")
    actual_source = store.json(observed.source_tree_ref)
    expected_source = candidate_snapshot(store, candidate)
    if (actual_source["bundle_ref"] != expected_source.bundle_ref
            or actual_source["command"] != candidate.command):
        raise IntegrityError("Selected source differs from the actual full-development execution")
    if len(observed.evaluation_refs) != 3 or len(observed.prediction_refs) != 3 or len(observed.log_refs) != 3:
        raise IntegrityError("Selection lacks isolated scoring evidence for all three seeds")
    protocol = store.json(observed.protocol_ref)
    images = protocol.get("runtime", {}).get("images", {})
    for seed, ref in enumerate(observed.evaluation_refs):
        evaluation = store.json(ref)
        grade = store.json(evaluation["grade_ref"])
        if (evaluation.get("seed") != seed or evaluation.get("input_bundle") != observed.bundle_refs[seed]
                or evaluation.get("prediction_bundle") != observed.prediction_refs[seed]
                or grade.get("valid_submission") is not True
                or evaluation.get("metrics") != grade.get("metrics")
                or grade.get("metrics", {}).get("accuracy") != observed.values[seed]):
            raise IntegrityError("Selected accuracy differs from trusted per-seed evaluation evidence")
        load_bundle(store, observed.prediction_refs[seed])
        receipts = evaluation.get("receipts", [])
        if len(receipts) != 2:
            raise IntegrityError("Selection requires separate inference and trusted scoring receipts")
        for receipt_ref, expected_image in zip([observed.log_refs[seed], *receipts],
                                               [images.get("execution"), images.get("execution"), images.get("grading")], strict=True):
            receipt = store.json(receipt_ref)
            if (not expected_image or receipt.get("image_id") != expected_image
                    or receipt.get("returncode") != 0 or receipt.get("timed_out") is not False or receipt.get("limit_exceeded")):
                raise IntegrityError("Selected score references an unsuccessful evaluation stage")
    return {"candidate_ref": store.put_json(candidate, "study-sealed-candidate"),
            "source_sha256": candidate_source_hash(store, candidate),
            "observation_ref": row["result_ref"], "training_proof_ref": store.put_json(proof),
            "protocol_sha256": digest(canonical(_normalized_protocol(protocol))),
            "values": observed.values}


def _baseline_observation(store, key):
    row = store.db.execute("SELECT result_ref FROM live_operations WHERE key=?", (key,)).fetchone()
    if row is None or not row[0]:
        raise IntegrityError("The shared baseline lacks complete full-development evidence")
    return store.json(row[0])


def seal_research_arm(controller, frozen):
    store = controller.store
    if not store.get("study_plan_ref") or canonical(store.get("live_frozen")) != canonical(frozen):
        raise IntegrityError("Research selection must be committed under the frozen study plan")
    if not store.get("study_research_freeze_ref"):
        if store.get("live_state", {}).get("stage") != "freeze" or not store.get("research_manuscript_bundle"):
            raise IntegrityError("Research must complete its manuscript/review path before sealing a selection")
    _no_unknown_operations(store)
    selected = Candidate.model_validate(frozen["best"]["candidate"])
    baseline = Candidate.model_validate(frozen["baseline"])
    result = {"arm": "scientist", "status": "frozen", "plan_ref": store.get("study_plan_ref"),
              "frozen_ref": store.put_json(frozen, "study-research-selection"),
              "selected": _verified_observation(store, selected, frozen["best"]["result"]),
              "baseline": _verified_observation(store, baseline, _baseline_observation(store, "experiment/baseline/full")),
              "nonfinal_operations": _operations(store), "final_access": False}
    ref = store.put_json(result, "study-research-freeze")
    if store.get("study_research_freeze_ref") not in (None, ref):
        raise IntegrityError("Research selection or development ledger changed after freeze")
    store.set("study_research_freeze_ref", ref)
    return ref, result


class StudyResearchController(ResearchController):
    """The ordinary workflow ends at a selection seal until both arms are ready."""

    def run(self):
        if not self.store.get("study_plan_ref"):
            raise IntegrityError("Create the paired study plan before starting live research")
        return super().run()

    def finalize(self, frozen):
        ref, _ = seal_research_arm(self, frozen)
        state = {**self.store.get("live_state", {}), "stage": "freeze", "status": "frozen",
                 "outcome": "development_frozen", "reason": "Waiting for both study arms before protected final evaluation",
                 "study_freeze_ref": ref, "final_evaluation": "not_started"}
        self.store.set("live_state", state)
        return state


def _direct_seal(control):
    store = control.store
    ref = store.get("direct_frozen_ref")
    if not ref:
        raise IntegrityError("Direct control has not frozen a completed selection")
    frozen = store.json(ref)
    identity = store.json(frozen["identity_ref"])
    if frozen.get("selection_basis") != "full_development_only" or frozen.get("seeds") != [0, 1, 2]:
        raise IntegrityError("Direct control selection is not a matched full-development result")
    _no_unknown_operations(store)
    selected = Candidate.model_validate(store.json(frozen["selected_candidate_ref"]))
    result = {"arm": "direct", "status": "frozen", "plan_ref": store.get("study_plan_ref"),
              "frozen_ref": ref, "direct_max_iterations": identity["max_iterations"],
              "selected": _verified_observation(store, selected, store.json(frozen["selected_observation_ref"])),
              "baseline": _verified_observation(store, control.adapter.baseline(), store.json(frozen["baseline_observation_ref"])),
              "nonfinal_operations": _operations(store), "final_access": False}
    return result


def _verify_arms(study_store, research, direct):
    plan_ref = study_store.get("study_plan_ref")
    if not plan_ref:
        raise IntegrityError("No frozen paired-study plan")
    plan = study_store.json(plan_ref)
    if plan["coordinator_sha256"] != file_digest(Path(__file__)):
        raise IntegrityError("Study coordinator changed after plan freeze")
    for name, owner in (("scientist", research), ("direct", direct)):
        if (str(owner.store.root) != plan["arms"][name] or owner.store.get("study_plan_ref") != plan_ref
                or canonical(_common(owner)) != canonical(plan["common"])
                or owner.config.model_dump(mode="json") != plan["arm_configurations"][name]):
            raise IntegrityError("Arm configuration, baseline or executable protocol changed after plan freeze")
    frozen = research.store.get("live_frozen")
    ref, scientist = seal_research_arm(research, frozen)
    direct_seal = _direct_seal(direct)
    if direct_seal["direct_max_iterations"] != plan["direct_max_iterations"]:
        raise IntegrityError("Direct iteration ceiling differs from the predeclared study")
    all_methods = [scientist["baseline"], scientist["selected"], direct_seal["baseline"], direct_seal["selected"]]
    if len({item["protocol_sha256"] for item in all_methods}) != 1:
        raise IntegrityError("Arms' development scores use incomparable protocols")
    if scientist["baseline"]["source_sha256"] != direct_seal["baseline"]["source_sha256"]:
        raise IntegrityError("Study arms did not execute the same reference baseline")
    return plan, {"scientist": {"seal_ref": ref, **scientist}, "direct": direct_seal}


def grant_final_access(study_store, research, direct):
    """Create a durable grant only after both selections and ledgers are sealed."""
    plan, arms = _verify_arms(study_store, research, direct)
    prior = study_store.get("study_final_grant_ref")
    if prior:
        grant = study_store.json(prior)
        if canonical(grant["arms"]) != canonical(arms):
            raise IntegrityError("Study selections or non-final evidence changed after final release")
        return prior
    for owner in (research, direct):
        if owner.store.db.execute("SELECT 1 FROM live_operations WHERE key LIKE 'final/%'").fetchone():
            raise IntegrityError("Protected final evaluation began before both arm selections were frozen")
    allowance = plan["common"]["compute_budget"].get("max_final_wall_seconds")
    if type(allowance) not in (int, float) or allowance <= 0:
        raise ValueError("Declare the independent final wall-time allowance before starting the study")
    grant = {"schema_version": "cifar-paired-final-grant-v1", "status": "BOTH_ARMS_FROZEN",
             "plan_ref": study_store.get("study_plan_ref"), "arms": arms, "seeds": [0, 1, 2],
             "final_deadline": time.time() + allowance,
             "authorized": {"scientist": {name: arms["scientist"][name]["source_sha256"] for name in ("baseline", "selected")},
                            "direct": {"selected": arms["direct"]["selected"]["source_sha256"]}}}
    ref = study_store.put_json(grant, "paired-final-grant")
    study_store.set("study_final_grant_ref", ref)
    return ref


def final_grant_verifier(study_store, research, direct):
    """Return the callback required by StudyDevelopmentRunner; no boolean bypass."""
    def verify(runner, grant_ref, label, source_hash):
        if grant_ref != study_store.get("study_final_grant_ref"):
            raise IntegrityError("Final grant is not the current immutable study release")
        grant = study_store.json(grant_ref)
        _, arms = _verify_arms(study_store, research, direct)
        if canonical(arms) != canonical(grant["arms"]):
            raise IntegrityError("Research evidence changed after final release")
        owners = {"scientist": research, "direct": direct}
        arm = next((name for name, owner in owners.items() if runner is owner.runner), None)
        if arm is None or grant["authorized"][arm].get(label) != source_hash:
            raise IntegrityError("Final grant does not authorize this runner, method label or source")
        owner = owners[arm]
        owner.store.put_json(grant, "paired-final-grant")
        if owner.store.get("study_final_release_ref") not in (None, grant_ref):
            raise IntegrityError("Arm already belongs to a different final release")
        owner.store.set("study_final_release_ref", grant_ref)
        owner.budget.activate_final_grant(grant_ref, grant["final_deadline"])
        return {"status": "BOTH_ARMS_FROZEN", "grant_digest": digest(canonical(grant)),
                "candidate_source_hash": source_hash, "seeds": [0, 1, 2],
                "final_deadline": grant["final_deadline"]}
    return verify


def release_finals(study_store, research, direct):
    """Evaluate frozen selections and append the research manuscript without model calls."""
    ref = grant_final_access(study_store, research, direct)
    verifier = final_grant_verifier(study_store, research, direct)
    for owner in (research, direct):
        owner.runner.final_grant_ref, owner.runner.final_grant_verifier = ref, verifier
    grant = study_store.json(ref)
    for name, owner in (("scientist", research), ("direct", direct)):
        # Activate both barriers before the first seed, including the idle arm.
        verifier(owner.runner, ref, "selected", grant["authorized"][name]["selected"])
    # Explicit base dispatch is the only route past the subclass's freeze barrier.
    state = ResearchController.finalize(research, research.store.get("live_frozen"))
    frozen = direct.store.json(direct.store.get("direct_frozen_ref"))
    candidate = Candidate.model_validate(direct.store.json(frozen["selected_candidate_ref"]))
    direct_final = direct.runner.final("selected", candidate)
    direct.store.set("study_direct_final_ref", direct.store.put_json(direct_final, "direct-control-final"))
    finals = research.store.get("live_final")
    for result in finals.values():
        verify_final_aggregation(research.store, result)
    verify_final_aggregation(direct.store, direct_final)
    summary = {"grant_ref": ref, "research_state": state, "reference": finals["baseline"],
               "scientist": finals.get("selected", finals["baseline"]), "direct": direct_final,
               "synthetic_agents": research.backend.synthetic or direct.backend.synthetic,
               "live_model_calls_during_final": 0}
    study_store.set("study_final_results_ref", study_store.put_json(summary, "paired-final-results"))
    return summary


def study_verdicts(finals, *, audits=None, mechanism_assessment=None):
    """Separate execution, utility, explanation and observed control-arm advantage."""
    audits = audits or {}
    ready = all(finals.get(name, {}).get("status") == "completed" for name in ("reference", "scientist", "direct"))
    workflow_complete = ready and finals.get("research_state", {}).get("status") == "complete"
    measured = all(len(finals.get(name, {}).get("per_seed_values", [])) == 3 for name in ("reference", "scientist", "direct"))
    checked = all(audits.get(name, {}).get("status") == "pass" for name in ARM_NAMES)
    output = {"workflow": {"status": "pass" if workflow_complete and checked else "unverified" if workflow_complete else "fail",
                           "autonomous_live_validation": bool(workflow_complete and checked and not finals.get("synthetic_agents", True)),
                           "reason": "Completed finals still require independent audit; synthetic agents establish orchestration only"},
              "useful_improvement": {"status": "unverified"},
              "mechanism": mechanism_assessment or {"status": "unverified", "reason": "Inspect executed ablations and the manuscript's qualified mechanism claims"},
              "added_value": {"status": "unverified"},
              "scope": "One adapted CIFAR study; no statistical-significance or general research-superiority claim"}
    if not ready or not measured:
        return output
    reference, scientist, direct = (finals[name]["per_seed_values"] for name in ("reference", "scientist", "direct"))
    differences = [left - right for left, right in zip(scientist, reference, strict=True)]
    control_differences = [left - right for left, right in zip(scientist, direct, strict=True)]
    gain = sum(differences) / 3
    useful = gain >= 0.02 - 1e-12 and all(value > 0 for value in differences)
    output["useful_improvement"] = {"status": "pass" if useful else "not_demonstrated",
                                    "mean_accuracy_gain": gain, "per_seed_differences": differences,
                                    "threshold": 0.02, "evidence_audited": checked}
    output["added_value"] = {"status": "observed_advantage" if all(value > 0 for value in control_differences) else "not_demonstrated",
                             "mean_accuracy_gain_over_direct": sum(control_differences) / 3,
                             "per_seed_differences": control_differences, "evidence_audited": checked}
    return output


def _direct_integrity(direct):
    """Audit ordinary-control evidence without inventing research-controller metadata."""
    from ..live.audit import _observation_binding, _prepared_partition

    store = direct.store
    errors = store.verify()
    if errors:
        raise IntegrityError(errors[0])
    identity = store.get("direct_identity")
    if (not identity or identity["config"] != direct.config.model_dump(mode="json")
            or identity["adapter"] != direct.adapter.identity()):
        raise IntegrityError("Direct-control configuration or source identity differs")
    for row in store.db.execute("SELECT output_ref FROM calls WHERE key LIKE 'materialization/%'"):
        materialized = store.json(row[0])
        proposed = Candidate.model_validate(store.json(materialized["proposal_ref"]))
        direct.adapter.validate(proposed)
        tree = candidate_snapshot(store, proposed)
        if (tree.model_dump(mode="json") != store.json(materialized["source_tree_ref"])
                or tree.model_dump(mode="json") != store.json(materialized["code_ref"])):
            raise IntegrityError("Direct-control materialized source differs from its bounded proposal")
    prepared = {phase: _prepared_partition(store, direct.adapter, phase) for phase in ("subset", "full")}
    checks = []
    for row in store.db.execute("SELECT * FROM calls WHERE key LIKE 'development/%'"):
        observed = Observation.model_validate(store.json(row["output_ref"]))
        _observation_binding(store, row, observed)
        if observed.status != "success":
            checks.append({"observation": observed.id, "status": observed.status, "retained_failure": True})
            continue
        phase = "subset" if observed.phase == "subset" else "full"
        seeds = [0] if phase == "subset" else [0, 1, 2]
        proof = validate_observation_receipts(store, observed, expected_seeds=seeds)
        protocol = store.json(observed.protocol_ref)
        if (protocol["adapter"] != direct.adapter.identity() or protocol["data"] != prepared[phase]["files"]
                or protocol["evaluation"] != direct.config.development.model_dump(mode="json")):
            raise IntegrityError("Direct-control experiment protocol differs from the declared study")
        if len(observed.evaluation_refs) != len(seeds) or len(observed.log_refs) != len(seeds):
            raise IntegrityError("Direct-control experiment lacks per-seed inference/scoring receipts")
        for index, seed in enumerate(seeds):
            evaluation = store.json(observed.evaluation_refs[index])
            cached = store.db.execute("SELECT output_ref FROM calls WHERE key=?",
                                      (f"pipeline/{observed.id}/{seed}",)).fetchone()
            grade = store.json(evaluation["grade_ref"])
            if (cached is None or store.json(cached[0]) != evaluation
                    or evaluation["input_bundle"] != observed.bundle_refs[index]
                    or evaluation["prediction_bundle"] != observed.prediction_refs[index]
                    or evaluation["seed"] != seed or evaluation["pipeline"] != protocol["pipeline"]
                    or evaluation["public_manifest"] != prepared[phase]["files"]
                    or grade.get("valid_submission") is not True or grade["metrics"] != evaluation["metrics"]
                    or grade["metrics"]["accuracy"] != observed.values[index]):
                raise IntegrityError("Direct-control result differs from its isolated trusted scorer")
            load_bundle(store, observed.prediction_refs[index])
            for ref in [observed.log_refs[index], *evaluation["receipts"]]:
                receipt = store.json(ref)
                if receipt["returncode"] or receipt["timed_out"] or receipt.get("limit_exceeded"):
                    raise IntegrityError("Successful control score references failed execution")
        checks.append(proof)
    if not checks:
        raise IntegrityError("Direct-control evidence contains no auditable development experiments")
    return {"status": "pass", "checks": checks, "scope": "Source/ledger/partition/training/scoring bindings"}


def _score_records(owner):
    records = []
    for row in owner.store.db.execute("SELECT * FROM calls WHERE key LIKE 'development/%'"):
        observed = Observation.model_validate(owner.store.json(row["output_ref"]))
        if observed.status != "success":
            continue
        phase = "subset" if observed.phase == "subset" else "full"
        prepared = owner.store.json(owner.store.get("prepared/" + phase))
        for index, ref in enumerate(observed.evaluation_refs):
            records.append({"store": owner.store.root, "config": owner.config, "report_ref": ref,
                            "prepared": prepared, "id": f"development/{observed.id}/{index}"})
    for database in sorted((owner.store.root / "final").glob("*/campaign.sqlite")):
        from ..store import Store
        child = Store(database.parent)
        try:
            result = child.json(child.get("workspace_result"))
            if result["status"] != "completed":
                continue
            native = WorkspaceSpec.model_validate(child.json(child.get("workspace_spec")))
            records.append({"store": database.parent,
                            "config": SimpleNamespace(native=native, development=owner.config.development),
                            "report_ref": result["official_report_ref"],
                            "prepared": {"public": str(native.public_dir), "files": native.public_manifest},
                            "id": "final/" + database.parent.name})
        finally:
            child.close()
    return records


def audit_study(study_store, research, direct, *, reexecute_evaluators=False,
                max_score_replays=100, max_audit_seconds=600):
    """Audit both arms; optional bounded CPU scoring replays never retrain/infer.

    Without scorer replay, the result remains unverified for capability acceptance.
    This helper does not infer success from a clean database or a generated PDF.
    """
    import math
    from ..live.audit import audit_live, replay_evaluator, _audit_workspace_final
    from ..store import Store

    if type(max_score_replays) is not int or max_score_replays < 1 or type(max_audit_seconds) not in (int, float) or not math.isfinite(max_audit_seconds) or max_audit_seconds <= 0:
        raise ValueError("Audit replay count and wall-time bounds must be explicit and positive")
    deadline = time.monotonic() + max_audit_seconds
    report = {"status": "unverified", "arms": {}, "live_model_calls": 0,
              "training_or_inference_reruns": 0, "scope": "Immutable evidence plus optional isolated CPU scoring replay"}
    try:
        _verify_arms(study_store, research, direct)
        finals_ref = study_store.get("study_final_results_ref")
        if not finals_ref:
            raise IntegrityError("Paired final results have not been committed")
        finals = study_store.json(finals_ref)
        for name, owner in (("scientist", research), ("direct", direct)):
            if name == "scientist":
                static = audit_live(owner.store.root, reexecute_evaluators=False)
                if static["status"] != "pass":
                    raise IntegrityError("Scientific workflow audit failed: " + str(static.get("errors")))
            else:
                static = _direct_integrity(owner)
                verification = {"required_container_replays": [], "replay_sources": {}, "container_replays": []}
                for database in sorted((owner.store.root / "final").glob("*/campaign.sqlite")):
                    child = Store(database.parent)
                    try:
                        if child.verify():
                            raise IntegrityError("Direct final child artifacts failed integrity verification")
                        _audit_workspace_final(child, owner.config, False,
                                               proof_store=study_store, verification=verification)
                    finally:
                        child.close()
            for result in ([finals["reference"], finals["scientist"]] if name == "scientist" else [finals["direct"]]):
                verify_final_aggregation(owner.store, result)
            report["arms"][name] = {"status": "unverified", "integrity": static, "score_replays": []}
        if reexecute_evaluators:
            records = {name: _score_records(owner) for name, owner in (("scientist", research), ("direct", direct))}
            if sum(len(items) for items in records.values()) > max_score_replays:
                raise ValueError("Required score replays exceed the declared audit count bound")
            for name, items in records.items():
                if not items:
                    raise IntegrityError("No score evidence available for required audit replay")
                for item in items:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Independent audit wall-time allowance exhausted")
                    source = Store(item["store"])
                    try:
                        evaluated = source.json(item["report_ref"])
                        bounded = deepcopy(evaluated)
                        scoring = bounded["pipeline"]["score"]
                        if scoring["resources"].get("gpus") is not None:
                            raise IntegrityError("Acceptance replay must use CPU-only trusted scoring")
                        scoring["resources"]["seconds"] = min(scoring["resources"]["seconds"], remaining)
                        bounded["pipeline"]["inference"] = None
                        metrics, proof = replay_evaluator(source, item["config"], bounded, item["prepared"],
                            evidence_store=study_store, proof_id="study/" + name + "/" + item["id"])
                        if (set(metrics) != set(evaluated["metrics"]) or any(not math.isclose(
                                value, evaluated["metrics"][key], rel_tol=1e-10, abs_tol=1e-10) for key, value in metrics.items())):
                            raise IntegrityError("Independent scoring replay differs from committed metrics")
                        report["arms"][name]["score_replays"].append(proof)
                    finally:
                        source.close()
                report["arms"][name]["status"] = "pass"
            report["status"] = "pass"
        report["verdicts"] = study_verdicts(finals, audits=report["arms"])
    except (IntegrityError, ValueError, OSError, KeyError, TypeError, TimeoutError) as error:
        report.update(status="fail", error=str(error))
    report["audit_ref"] = study_store.put_json(report, "paired-study-acceptance-audit")
    return report
