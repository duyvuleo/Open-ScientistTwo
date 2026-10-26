"""Independently replay recorded parent edits without executing candidate code.

Replay uses a scratch artifact store, so a proposed-but-unexecuted method does not
become an execution claim and historical campaign artifacts are never rewritten.
This establishes source fidelity and provenance, not scientific intent.
"""

from pathlib import Path
import tempfile

from ..artifacts import load_bundle
from ..models import AgentResult
from ..store import IntegrityError, Store, canonical, digest
from .models import Candidate, Observation
from .parent_edits import (
    ParentChangePlan, ParentPatch, apply_parent_patch, parent_context,
)
from .source_tree import MaterializedSourceTree, SourceSnapshot


def _call(store, key):
    row = store.db.execute("SELECT * FROM calls WHERE key=?", (key,)).fetchone()
    if row is None:
        raise IntegrityError("Parent-edit audit requires a committed call: " + key)
    return row, store.json(row["output_ref"])


def _copy_ref(store, scratch, ref):
    raw = store.read(ref)
    copied = scratch.put(raw, suffix=Path(ref).suffix.lstrip("."))
    if copied != ref:
        raise IntegrityError("Parent-edit source reference is not content-addressed")


def _copy_bundle(store, scratch, ref):
    # Source bundles only; model checkpoints and dataset labels are not copied.
    manifest = load_bundle(store, ref)
    if any(spec.role != "source" for spec in manifest.specs):
        raise IntegrityError("Parent-edit replay requires source-only bundles")
    _copy_ref(store, scratch, ref)
    for file in manifest.files.values():
        copied = scratch.put_file(store.root / file.ref, "source-audit-input")
        if copied != file.ref:
            raise IntegrityError("Parent-edit source payload differs from its registered reference")


def _copy_source_tree(store, scratch, ref):
    tree = MaterializedSourceTree.model_validate(store.json(ref))
    _copy_ref(store, scratch, ref)
    _copy_ref(store, scratch, tree.proposal_ref)
    _copy_bundle(store, scratch, tree.bundle_ref)
    if tree.base_snapshot_ref:
        snapshot = SourceSnapshot.model_validate(store.json(tree.base_snapshot_ref))
        _copy_ref(store, scratch, tree.base_snapshot_ref)
        _copy_bundle(store, scratch, snapshot.bundle_ref)
    return tree


def _parent_binding(store, adapter, proof, parent):
    parent_id = proof["parent_trial_id"]
    if parent_id == "baseline":
        observation_id = proof.get("parent_observation_id") or "baseline/full"
        if not observation_id.startswith("baseline/full"):
            raise IntegrityError("Baseline parent must reference its executed full-development baseline")
        row = store.db.execute("SELECT output_ref FROM calls WHERE key=?",
                               ("materialization/development/" + observation_id,)).fetchone()
        if row:
            materialization = store.json(row[0])
            original = Candidate.model_validate(store.json(materialization["proposal_ref"]))
        else:
            original = adapter.baseline()
        _, observation = _call(store, "development/" + observation_id)
    else:
        _, trial = _call(store, "live-trial/" + parent_id)
        if trial.get("id") != parent_id:
            raise IntegrityError("Parent trial ID differs from its committed namespace")
        original = Candidate.model_validate(trial["candidate"])
        observation = trial["result"]
    if canonical(original) != canonical(parent):
        raise IntegrityError("Parent-edit proposal differs from the committed parent candidate")
    if observation.get("phase") not in {"subset", "full", "rebuttal"}:
        raise IntegrityError("Parent-edit source cannot originate from protected final evidence")
    _, committed = _call(store, "development/" + observation["id"])
    if canonical(observation) != canonical(committed):
        raise IntegrityError("Parent trial differs from its committed development observation")
    if observation.get("source_tree_ref"):
        executed = MaterializedSourceTree.model_validate(store.json(observation["source_tree_ref"]))
        parent_tree = MaterializedSourceTree.model_validate(store.json(proof["parent_source_ref"]))
        if executed.bundle_ref != parent_tree.bundle_ref or executed.command != parent_tree.command:
            raise IntegrityError("Parent-edit source differs from the executed parent tree")
        return "bound_to_executed_parent_source"
    # Synthetic runners can omit materialization. Do not invent that proof.
    return "committed_parent_candidate_only; executed_source_unverified"


def _accepted_patch(store, key, row, proof):
    accepted_ref = store.get("accepted-agent-output/" + key)
    if not accepted_ref:
        raise IntegrityError("Parent-edit proof has no accepted output-contract record")
    accepted = store.json(accepted_ref)
    if (accepted.get("status") != "accepted" or accepted.get("invocation_key") != key
            or any(accepted.get(name) != proof.get(name)
                   for name in ("response_key", "request_ref", "response_ref"))):
        raise IntegrityError("Parent-edit proof differs from its accepted agent response")
    agent_row, raw_response = _call(store, "live-agent/" + proof["response_key"])
    if agent_row["output_ref"] != proof["response_ref"]:
        raise IntegrityError("Parent-edit response reference differs from its committed completion")
    response = AgentResult.model_validate(raw_response)
    patch = ParentPatch.model_validate(response.payload)
    if canonical(store.json(proof["patch_ref"])) != canonical(patch):
        raise IntegrityError("Parent-edit patch differs from the accepted raw response")
    request = store.json(proof["request_ref"])
    if (store.get("request/" + proof["response_key"]) != proof["request_ref"]
            or agent_row["input_hash"] != digest(canonical(request))):
        raise IntegrityError("Parent-edit response is bound to a different agent request")
    if request.get("role") not in {"implement", "engineer", "ablation_coder", "rebuttal_coder"}:
        raise IntegrityError("Parent-edit patch came from an unsupported agent role")
    if row["input_hash"] != digest(canonical({"parent_ref": proof["parent_source_ref"],
                                              "patch": patch.model_dump()})):
        raise IntegrityError("Parent-edit completion inputs differ from its parent and patch")
    if request["inputs"]["parent"]["parent_source_ref"] != proof["parent_source_ref"]:
        raise IntegrityError("Parent-edit request supplied a different parent source")
    if canonical(request["inputs"]["change_plan"]) != canonical(proof["change_plan"]):
        raise IntegrityError("Parent-edit proof changed the independently declared change plan")
    return patch, request


def _executions(store, proof, candidate, after_rowid):
    """Bind equal proposal/source bytes to committed development attempts.

    A match reports executions of this candidate, not a claim that matching
    hashes demonstrate causal independence or that a method improved.
    """
    observations = []
    for row in store.db.execute("SELECT key,output_ref FROM calls WHERE key LIKE 'materialization/development/%' AND rowid>? ORDER BY key", (after_rowid,)):
        materialization = store.json(row["output_ref"])
        if canonical(store.json(materialization["proposal_ref"])) != canonical(candidate):
            continue
        observation_id = row["key"].removeprefix("materialization/development/")
        if materialization.get("source_tree_ref") != proof["candidate_source_ref"]:
            raise IntegrityError("Parent-edited candidate differs from its materialized child source")
        completed = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("development/" + observation_id,)).fetchone()
        if not completed:
            observations.append({"id": observation_id, "status": "completion_unknown",
                                 "materialization_ref": row["output_ref"],
                                 "lineage": "not_yet_recorded",
                                 "note": "Source was materialized but no development completion was committed; no execution or non-execution claim"})
            continue
        observation = Observation.model_validate(store.json(completed[0]))
        if observation.phase == "final" or observation.id != observation_id:
            raise IntegrityError("Parent-edit execution is in the wrong evidence namespace")
        if (materialization.get("source_tree_ref") != proof["candidate_source_ref"]
                or observation.source_tree_ref != proof["candidate_source_ref"]
                or observation.code_ref != materialization["code_ref"]):
            raise IntegrityError("Parent-edited candidate differs from its executed child source")
        trial_row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("live-trial/" + observation_id,)).fetchone()
        lineage = "not_yet_recorded"
        if trial_row:
            trial = store.json(trial_row[0])
            if canonical(trial.get("result")) != canonical(observation):
                raise IntegrityError("Child trial differs from its executed observation")
            if canonical(trial.get("candidate")) != canonical(candidate):
                raise IntegrityError("Child trial differs from the parent-edited candidate")
            if proof["parent_trial_id"] not in trial.get("parents", []):
                raise IntegrityError("Parent-edited child trial omits its declared parent lineage")
            lineage = "parent_recorded"
        observations.append({"id": observation.id, "status": observation.status,
                             "observation_ref": completed[0], "lineage": lineage,
                             "source_binding": "recorded"})
    known = {item["id"] for item in observations}
    for row in store.db.execute("SELECT key,output_ref FROM calls WHERE key LIKE 'development/%' AND rowid>? ORDER BY key", (after_rowid,)):
        observation = Observation.model_validate(store.json(row["output_ref"]))
        if observation.id in known or observation.phase == "final":
            continue
        # Synthetic runners and pre-materialization failures can have a stored
        # proposal but no executed tree. Preserve that uncertainty explicitly.
        if canonical(store.json(observation.code_ref)) == canonical(candidate):
            observations.append({"id": observation.id, "status": observation.status,
                                 "observation_ref": row["output_ref"], "source_binding": "unverified",
                                 "lineage": "unverified",
                                 "note": "Matching proposal has a development observation but no bound materialized source"})
    return observations


def audit_parent_edits(store, adapter):
    """Replay all newly recorded edit proofs; absence is not a source-fidelity pass."""
    checks = []
    for row in store.db.execute("SELECT rowid,* FROM calls WHERE key LIKE 'parent-edit/%' ORDER BY key"):
        key = row["key"].removeprefix("parent-edit/")
        proof = store.json(row["output_ref"])
        if proof.get("schema_version") != 1:
            raise IntegrityError("Unsupported parent-edit proof schema")
        parent = Candidate.model_validate(store.json(proof["parent_candidate_ref"]))
        candidate = Candidate.model_validate(store.json(proof["candidate_ref"]))
        if (digest(canonical(parent)) != proof["parent_proposal_sha256"]
                or digest(canonical(candidate)) != proof["candidate_proposal_sha256"]):
            raise IntegrityError("Parent-edit proposal hashes differ from recorded candidates")
        patch, request = _accepted_patch(store, key, row, proof)
        parent_binding = _parent_binding(store, adapter, proof, parent)
        materialize = getattr(adapter, "materialize", None)
        paths = getattr(adapter, "editable_paths", None)
        paths = paths() if callable(paths) else adapter.context().get("editable_files")
        options = {"materialize": materialize, "editable_paths": paths}
        plan = ParentChangePlan.model_validate(proof["change_plan"])
        declared = plan.changed_files
        if request["role"] == "engineer":
            declared = sorted({edit.path for edit in patch.edits})
            if not set(declared) <= set(plan.changed_files):
                raise IntegrityError("Parent-edit repair exceeds the independently planned file scope")
        with tempfile.TemporaryDirectory(prefix="scientisttwo-parent-audit-") as temp:
            scratch = Store(Path(temp))
            try:
                _copy_source_tree(store, scratch, proof["parent_source_ref"])
                context = parent_context(scratch, parent, proof["parent_source_ref"], **options)
                context["trial_id"] = proof["parent_trial_id"]
                if canonical(context) != canonical(request["inputs"]["parent"]):
                    raise IntegrityError("Agent parent context differs from exact frozen parent source")
                reproduced, replay = apply_parent_patch(scratch, parent, proof["parent_source_ref"],
                    patch, declared_paths=declared, **options)
                if canonical(reproduced) != canonical(candidate):
                    raise IntegrityError("Replayed parent patch does not produce the recorded child candidate")
                if any(canonical(proof.get(name)) != canonical(value) for name, value in replay.items()):
                    raise IntegrityError("Parent-edit proof differs from independent anchored-edit replay")
                actual_tree = store.json(proof["candidate_source_ref"])
                if canonical(actual_tree) != canonical(scratch.json(replay["candidate_source_ref"])):
                    raise IntegrityError("Parent-edit child source differs from independent materialization")
            finally:
                scratch.close()
        executions = _executions(store, proof, candidate, row["rowid"])
        checks.append({"parent_edit": key, "proof_ref": row["output_ref"],
            "status": "pass", "parent_binding": parent_binding,
            "source_fidelity": "replayed_exact_parent_patch",
            "executed_child": ("recorded" if any(item.get("source_binding") == "recorded" for item in executions)
                               else "source_binding_unverified" if any("observation_ref" in item for item in executions)
                               else "completion_unknown" if executions else "proposed_not_executed"),
            "observations": executions, "changed_paths": proof["changed_paths"],
            "semantic_intent": "unverified",
            "scope": "Accepted response, exact parent bytes, anchored patch and executed source binding; not scientific intent or improvement"})
    return checks
