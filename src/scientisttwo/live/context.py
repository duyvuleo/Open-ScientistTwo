"""Deterministic role-context presentation without dropping scientific evidence.

Only historical source excerpts and diagnostic log text may be shortened. Every
trial, observation, constraint, decision and lineage edge remains represented.
The exact serialized input limit is a character limit, not a token estimate.
"""

from copy import deepcopy

from ..store import canonical, digest


class ContextProtocolError(ValueError):
    """The frozen context protocol cannot safely present this role invocation."""


def input_characters(inputs):
    return len(canonical(inputs).decode("utf-8"))


def _allowances(lengths, budget):
    """Share a total content-character allowance fairly in stable input order."""
    if budget < 0:
        raise ValueError("Context excerpt budgets must be nonnegative")
    allocations = [0] * len(lengths)
    active = [i for i, length in enumerate(lengths) if length]
    remaining = budget
    while active and remaining:
        share = max(1, remaining // len(active))
        for i in active:
            count = min(lengths[i] - allocations[i], share, remaining)
            allocations[i] += count
            remaining -= count
        active = [i for i in active if allocations[i] < lengths[i]]
    return allocations


def compact_trial_context(trials, *, code_chars, archive_candidate):
    """Keep every trial intact except its repeated/full candidate source text.

    The archive callback stores the original candidate and returns an immutable,
    relative artifact reference. Observations keep their executed-code references,
    which may differ from the original proposal after trusted materialization.
    """
    output = deepcopy(trials)
    distinct = {}
    for trial in output:
        candidate = trial["candidate"]
        candidate_hash = digest(canonical(candidate))
        if candidate_hash not in distinct:
            distinct[candidate_hash] = {
                "candidate": candidate,
                "first_trial": trial["id"],
                "artifact_ref": archive_candidate(candidate),
            }
    files = [
        (candidate_hash, file)
        for candidate_hash, record in distinct.items()
        for file in record["candidate"]["files"]
    ]
    allowances = _allowances([len(file["content"]) for _, file in files], code_chars)
    excerpts = {}
    for (candidate_hash, file), count in zip(files, allowances, strict=True):
        content = file["content"]
        head = (count + 1) // 2
        tail = count // 2
        excerpts[(candidate_hash, file["path"])] = {
            "path": file["path"],
            "sha256": digest(content.encode("utf-8")),
            "characters": len(content),
            "excerpt_head": content[:head],
            "excerpt_tail": content[-tail:] if tail else "",
            "omitted_characters": len(content) - count,
        }
    emitted = set()
    for trial in output:
        candidate_hash = digest(canonical(trial["candidate"]))
        record = distinct[candidate_hash]
        candidate = record["candidate"]
        summary = {
            "proposal_artifact_ref": record["artifact_ref"],
            "proposal_sha256": candidate_hash,
        }
        if candidate_hash in emitted:
            summary.update(
                candidate_context_trial=record["first_trial"],
                context_note="Identical proposal source omitted here; see the named earlier trial.",
            )
        else:
            summary.update(
                command=deepcopy(candidate["command"]),
                description=candidate["description"],
                files=[excerpts[(candidate_hash, f["path"])] for f in candidate["files"]],
                context_note="Historical source is represented by bounded head/tail excerpts. "
                "The omitted source remains in the proposal artifact; artifact references are "
                "provenance for human inspection, not permission to use tools. "
                "Do not infer unseen implementation details.",
            )
            emitted.add(candidate_hash)
        trial["candidate"] = summary
    return output, {
        "kind": "historical_candidate_excerpts",
        "trials_preserved": len(trials),
        "distinct_proposals": len(distinct),
        "total_source_characters": sum(len(file["content"]) for _, file in files),
        "source_excerpt_characters": sum(allowances),
        "source_excerpt_limit": code_chars,
        "preserved": "Every trial, hypothesis, observation, decision and parent edge",
    }


def present_inputs(role, inputs, *, history_code_chars, log_tail_chars, archive_candidate):
    """Return the presented inputs and an auditable list of all text reductions."""
    presented = deepcopy(inputs)
    changes = []
    if role == "evolve" and "attempts" in presented:
        presented["attempts"], change = compact_trial_context(
            presented["attempts"],
            code_chars=history_code_chars,
            archive_candidate=archive_candidate,
        )
        changes.append(change)
    logs = []

    def locate(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "execution_logs":
                    if not isinstance(child, list) or any(not isinstance(v, str) for v in child):
                        raise ContextProtocolError(
                            "Execution diagnostics must be a list of text logs"
                        )
                    for i, content in enumerate(child):
                        logs.append((child, i, content, [*path, key, i]))
                else:
                    locate(child, [*path, key])
        elif isinstance(value, list):
            for i, child in enumerate(value):
                locate(child, [*path, i])

    locate(presented, [])
    allocations = _allowances([len(content) for _, _, content, _ in logs], log_tail_chars)
    for (container, index, content, path), count in zip(logs, allocations, strict=True):
        container[index] = content[-count:] if count else ""
        if count < len(content):
            changes.append(
                {
                    "kind": "execution_log_tail",
                    "input_path": path,
                    "source_sha256": digest(content.encode("utf-8")),
                    "source_characters": len(content),
                    "tail_characters": count,
                    "omitted_characters": len(content) - count,
                    "note": "Earlier diagnostic text omitted; full logs remain in observation log_refs.",
                }
            )
    if changes:
        if "context_presentation" in presented:
            raise ContextProtocolError("Reserved context_presentation field already exists")
        presented["context_presentation"] = {
            "note": "Only historical source and diagnostic text may be excerpted. "
            "All supplied numeric evidence, constraints, decisions and lineage are retained. "
            "An excerpt is not evidence about omitted source or diagnostics.",
            "changes": changes,
        }
    return presented, changes


def validate_presentation(inputs, *, max_prompt_chars, protected_paths=()):
    """Reject oversize inputs or known protected host paths without mutating evidence."""
    count = input_characters(inputs)
    if count > max_prompt_chars:
        raise ContextProtocolError(
            f"Presented role inputs contain {count} JSON characters, exceeding the frozen "
            f"max_prompt_chars={max_prompt_chars}. No model call was reserved. Create a fresh "
            "manifest with an explicit context limit/protocol appropriate to this task. "
            "This is a character limit on inputs, not a model token-context guarantee."
        )
    paths = tuple(str(p) for p in protected_paths if p is not None)

    def contains_private(value):
        if isinstance(value, str):
            return any(path and path in value for path in paths)
        if isinstance(value, dict):
            return any(contains_private(k) or contains_private(v) for k, v in value.items())
        if isinstance(value, list):
            return any(contains_private(v) for v in value)
        return False

    if contains_private(inputs):
        raise ContextProtocolError(
            "Presented context contains a protected host data/evaluator path. "
            "No model call was reserved; revise the task context and freeze a new manifest."
        )
    return count
