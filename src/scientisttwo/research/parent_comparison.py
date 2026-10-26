"""Verified historical parent controls for a declared development phase.

This is an additional critic context contract. It does not change the task's
fixed-baseline comparison, select an incumbent, or execute a new control.
"""

from collections import Counter

from ..artifacts import ArtifactFile, verified_file
from ..live.models import Observation
from ..live.source_tree import MaterializedSourceTree, compare_source_trees
from ..store import IntegrityError, canonical, digest


CONTEXT_VERSION = "parent-comparison-context-v1"


def _committed(store, offered):
    observation = Observation.model_validate(offered)
    if observation.phase not in {"subset", "full", "rebuttal"}:
        raise IntegrityError("Parent comparison cannot use protected final evidence")
    row = store.db.execute(
        "SELECT output_ref FROM calls WHERE key=?", ("development/" + observation.id,)
    ).fetchone()
    if row is None or canonical(store.json(row[0])) != canonical(observation):
        raise IntegrityError("Parent comparison requires the exact committed observation")
    if not observation.protocol_ref:
        raise IntegrityError("Parent comparison requires a frozen evaluation protocol")
    protocol = store.json(observation.protocol_ref)
    seeds = protocol.get("seeds")
    if (digest(canonical(protocol)) != observation.protocol_hash
            or protocol.get("phase") != observation.phase
            or not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or observation.status == "success" and len(observation.values) != len(seeds)):
        raise IntegrityError("Parent comparison observation/protocol binding differs")
    return observation, row[0]


def resolve_parent_comparison(store, parent_trial, parent_source_ref, result, compare):
    """Return a versioned context and an eligible existing parent Observation.

Eligibility requires a successful committed observation of the exact parent
source/argv in the child's non-final phase and exact evaluation protocol.
The earliest compatible record committed before this candidate is reused;
scores never rank reuse candidates. This fixed chronological cutoff keeps a
later observation from changing an earlier resumed critic prompt. Missing
evidence is explicit and does not launch a substitute run.
Hash corruption and contradictory provenance remain fatal.
"""
    result, result_ref = _committed(store, result)
    cutoff = store.db.execute("SELECT rowid FROM calls WHERE key=?",
                             ("development/" + result.id,)).fetchone()[0]
    parent_result, parent_result_ref = _committed(store, parent_trial["result"])
    target = MaterializedSourceTree.model_validate(store.json(parent_source_ref))
    # This reconstructs and verifies the source bundle, not merely a claimed hash.
    compare_source_trees(store, parent_source_ref, parent_source_ref)
    context = {
        "schema_version": CONTEXT_VERSION,
        "status": "unavailable",
        "parent_trial_id": parent_trial["id"],
        "parent_original_observation_ref": parent_result_ref,
        "parent_original_execution_status": parent_result.status,
        "parent_original_source_tree_ref": parent_result.source_tree_ref,
        "parent_source_ref": parent_source_ref,
        "candidate_observation_id": result.id,
        "candidate_observation_ref": result_ref,
        "required_phase": result.phase,
        "required_protocol_hash": result.protocol_hash,
        "reference": None,
        "comparison": None,
        "new_experiments": 0,
        "reuse_policy": "earliest_committed_exact_source_argv_phase_and_protocol",
        "availability_snapshot": "committed_before_candidate_observation",
        "instruction": (
            "The existing comparison field remains the task's fixed-baseline comparison. "
            "This separate parent comparison addresses a parent-relative hypothesis only when "
            "status is available. Its structured observation, protocol, source and receipts "
            "are supplied in evidence_package. Verified compatible historical evidence is "
            "reused, not a fresh execution or an independent replication. If unavailable, "
            "do not substitute baseline evidence or critique prose, and do not claim that "
            "the parent-relative question was tested. Critic methodological veto remains available."
        ),
    }
    if parent_result.status != "success" or not parent_result.source_tree_ref:
        context["reason"] = (
            "The declared parent's original observation must be successful and have a "
            "verifiable executed source-tree binding; candidate source alone is insufficient."
        )
        return context, None
    original = MaterializedSourceTree.model_validate(store.json(parent_result.source_tree_ref))
    if (original.command != target.command
            or compare_source_trees(store, parent_source_ref, parent_result.source_tree_ref)):
        raise IntegrityError("Declared parent source differs from its committed execution")
    rejected = Counter()
    rows = store.db.execute(
        "SELECT key,output_ref FROM calls WHERE key LIKE 'development/%' AND rowid < ? ORDER BY rowid",
        (cutoff,),
    ).fetchall()
    for row in rows:
        raw = store.json(row["output_ref"])
        if raw.get("id") == result.id or raw.get("phase") != result.phase:
            continue
        if row["key"] != "development/" + raw.get("id", ""):
            raise IntegrityError("Parent-reference cache key differs from its observation")
        observation, observation_ref = _committed(store, raw)
        if observation.status != "success":
            rejected["execution_not_successful"] += 1
            continue
        if not observation.source_tree_ref:
            rejected["source_unavailable"] += 1
            continue
        tree = MaterializedSourceTree.model_validate(store.json(observation.source_tree_ref))
        if (tree.command != target.command
                or compare_source_trees(store, parent_source_ref, observation.source_tree_ref)):
            rejected["source_or_command_mismatch"] += 1
            continue
        if observation.protocol_hash != result.protocol_hash:
            rejected["protocol_mismatch"] += 1
            continue
        if not observation.log_refs or len(observation.prediction_refs) != len(observation.values):
            rejected["execution_evidence_incomplete"] += 1
            continue
        for ref in [*observation.log_refs, *observation.evaluation_refs,
                    *observation.prediction_refs, *observation.timing_refs]:
            artifact = store.db.execute("SELECT sha,size FROM artifacts WHERE path=?", (ref,)).fetchone()
            if artifact is None:
                raise IntegrityError("Parent comparison evidence artifact is unregistered")
            verified_file(store, ArtifactFile(ref=ref, sha256=artifact["sha"], size=artifact["size"]))
        context.update(
            status="available",
            reference={
                "id": observation.id, "observation_ref": observation_ref,
                "source_tree_ref": observation.source_tree_ref,
                "protocol_hash": observation.protocol_hash,
                "protocol_ref": observation.protocol_ref,
                "execution_refs": observation.log_refs,
                "evaluation_refs": observation.evaluation_refs,
                "prediction_refs": observation.prediction_refs,
                "origin": "existing_committed_development_execution",
                "verification_scope": "Committed observation/protocol/source and artifact identity; no new scorer replay or scientific certification",
            },
            comparison=compare(result, observation),
        )
        return context, observation
    context.update(
        reason="No successful exact-parent observation matches this development phase and protocol; a new matched control would require separate explicit admission.",
        rejected_reference_counts=dict(rejected),
    )
    return context, None
