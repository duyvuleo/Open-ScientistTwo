"""Bounded descriptive arithmetic from committed non-final writer evidence.

This prepares a new ledger before its immutable writer snapshot. It never edits
measurements, treats metadata as numerical evidence, or evaluates model code.
"""

from copy import deepcopy

from ..store import IntegrityError, canonical, digest
from .manuscript import derive_comparison, verify_evidence_ledger
from .parent_comparison import resolve_parent_comparison


VERSION = "research-writer-comparisons-v1"
MAX_OBSERVATIONS = 256
MAX_DERIVED_COMPARISONS = 1024
_NONFINAL = {"subset", "full", "rebuttal"}


def _plain(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def include_phase_baselines(observations, baselines):
    """Include the exact declared references, including screening history's one."""
    result = list(observations)
    by_id = {}
    for value in result:
        item = _plain(value)
        previous = by_id.get(item["id"])
        if previous is not None and canonical(previous) != canonical(item):
            raise IntegrityError("Conflicting writer observation identity")
        by_id[item["id"]] = item
    for phase, value in sorted(baselines.items()):
        item = _plain(value)
        if phase not in {"subset", "full"} or item.get("phase") != phase:
            raise IntegrityError("Writer references must be declared subset/full baselines")
        if item["id"] in by_id:
            if canonical(by_id[item["id"]]) != canonical(item):
                raise IntegrityError("Writer evidence conflicts with its declared baseline")
        else:
            result.append(value)
            by_id[item["id"]] = item
    if len(result) > MAX_OBSERVATIONS:
        raise IntegrityError("Writer comparison observation limit exceeded")
    return result


def register_writer_comparisons(store, ledger, baselines):
    """Return a new verified ledger with two-way, same-protocol differences.

    Pair only declared phase references, committed matched controls and verified
    historical parent comparisons. There is no all-pairs search. Both signs
    are registered so a gain or drop can cite exact arithmetic without changing
    the task's optimization direction or certifying mechanism attribution.
    """
    if ledger.get("includes_final") is not False:
        raise IntegrityError("Writer comparisons exclude protected final evidence")
    verify_evidence_ledger(store, ledger)
    if len(ledger["observations"]) > MAX_OBSERVATIONS:
        raise IntegrityError("Writer comparison observation limit exceeded")
    by_observation = {}
    for key, entry in ledger["entries"].items():
        if entry.get("kind") == "measured":
            if entry.get("phase") not in _NONFINAL:
                raise IntegrityError("Writer comparison encountered protected final evidence")
            by_observation.setdefault(entry["observation_id"], {})[entry["metric"]] = key
    references, pairs = {}, {}

    def propose(left_id, right_id, reason):
        if left_id == right_id:
            return
        left = by_observation.get(left_id, {})
        right = by_observation.get(right_id, {})
        for metric in sorted(left.keys() & right.keys()):
            a, b = ledger["entries"][left[metric]], ledger["entries"][right[metric]]
            if (a["phase"] != b["phase"] or a["protocol_hash"] != b["protocol_hash"]
                    or [item["seed"] for item in a["per_seed"]] != [item["seed"] for item in b["per_seed"]]):
                continue
            for first, second in ((left[metric], right[metric]), (right[metric], left[metric])):
                pairs.setdefault((first, second), set()).add(reason)
                if len(pairs) > MAX_DERIVED_COMPARISONS:
                    raise IntegrityError("Writer comparison derivation limit exceeded")

    for phase, raw in sorted(baselines.items()):
        baseline = _plain(raw)
        if phase not in {"subset", "full"} or baseline.get("phase") != phase:
            raise IntegrityError("Writer references must be declared subset/full baselines")
        identifier = baseline["id"]
        record = ledger["observations"].get(identifier)
        if record is None or canonical(store.json(record["observation_ref"])) != canonical(baseline):
            raise IntegrityError("Writer baseline differs from its committed ledger observation")
        references[phase] = identifier
        for other_id, other in sorted(ledger["observations"].items()):
            if other["phase"] == phase:
                propose(other_id, identifier, "declared_phase_baseline")
    for control in ledger.get("control_context", []):
        propose(control["control_id"], control["matched_id"], "committed_matched_control")
    parent_requests = []
    for row in store.db.execute("SELECT key,input_hash FROM calls WHERE key LIKE 'live-agent/%' ORDER BY key"):
        key = row["key"].removeprefix("live-agent/")
        if "critic" not in key.split("/"):
            continue
        ref = store.get("request/" + key)
        if ref is None:
            continue
        request = store.json(ref)
        if request.get("role") != "critic":
            continue
        context = request.get("inputs", {}).get("parent_comparison")
        if not isinstance(context, dict) or context.get("status") != "available":
            continue
        candidate_id = context.get("candidate_observation_id")
        reference = context.get("reference")
        if not isinstance(reference, dict):
            raise IntegrityError("Saved parent comparison has no structured reference")
        reference_id = reference.get("id")
        if candidate_id not in ledger["observations"] or reference_id not in ledger["observations"]:
            continue
        if row["input_hash"] != digest(canonical(request)):
            raise IntegrityError("Parent comparison differs from its committed critic request")
        candidate_ref = ledger["observations"][candidate_id]["observation_ref"]
        if (context.get("candidate_observation_ref") != candidate_ref
                or context["reference"].get("observation_ref") != ledger["observations"][reference_id]["observation_ref"]):
            raise IntegrityError("Parent comparison differs from writer observation references")
        # Resolve the parent's earlier exact source/argv/protocol again. Do not
        # trust a model's prose or reuse a possibly vetoed numerical delta from
        # the original critic context; only its independently verified pair.
        try:
            expected, parent = resolve_parent_comparison(store,
                {"id": context["parent_trial_id"], "result": store.json(context["parent_original_observation_ref"])},
                context["parent_source_ref"], store.json(candidate_ref), lambda candidate, reference: None)
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError("Saved parent comparison has invalid reference provenance") from error
        if (parent is None or parent.id != reference_id
                or any(canonical(expected.get(field)) != canonical(context.get(field)) for field in (
                    "schema_version", "status", "parent_original_observation_ref", "parent_source_ref",
                    "candidate_observation_id", "candidate_observation_ref", "required_phase",
                    "required_protocol_hash", "reference", "new_experiments", "reuse_policy", "availability_snapshot"))):
            raise IntegrityError("Saved parent comparison does not resolve to the same committed reference")
        propose(candidate_id, reference_id, "verified_parent_comparison")
        parent_requests.append(ref)

    updated = deepcopy(ledger)
    entries = []
    for (left, right), reasons in sorted(pairs.items()):
        identifier = derive_comparison(updated, left, right, operation="difference")
        entries.append({"evidence_id": identifier, "input_ids": [left, right],
                        "pair_sources": sorted(reasons)})
    verify_evidence_ledger(store, updated)
    proof = {"schema_version": VERSION, "baseline_observation_ids": references,
             "input_ledger_sha256": digest(canonical(ledger)), "output_ledger_sha256": digest(canonical(updated)),
             "operation": "difference", "direction": "left_minus_right", "entries": entries,
             "parent_comparison_request_refs": sorted(set(parent_requests)),
             "max_derived_comparisons": MAX_DERIVED_COMPARISONS,
             "pair_policy": "Declared phase baseline, committed matched-control and verified parent-comparison pairs only; no all-pairs expansion",
             "scope": "Descriptive arithmetic from measured aggregates; not new measurements, significance or causal validation",
             "instruction": "For gains or drops, cite the matching derived entry and its signed left-minus-right value. Do not use research-history prose or metadata as numerical evidence. Missing derivations are not permission to invent comparisons. Control attribution restrictions remain unchanged."}
    return updated, proof
