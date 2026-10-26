"""Role-local provenance contracts for completed, structurally valid outputs.

Bad agent references are distinct from corrupted stored evidence. This module
never repairs a response, widens a reference scope, or invokes a backend.
"""
from dataclasses import dataclass

from ..store import canonical
from .models import Draft, Hypothesis, Revision, Seeds
from .parent_edits import MethodChange


class AgentOutputError(Exception):
    """A completed response violates the invocation's declared output contract."""


@dataclass(frozen=True)
class OutputScope:
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    seed_count: int | None = None

    def prompt(self):
        return {
            "source_ids": list(self.source_ids),
            "evidence_ids": list(self.evidence_ids),
            "source_ids_meaning": "Task/literature document IDs only. Do not put experiment IDs here.",
            "evidence_ids_meaning": "Only the listed committed non-final experiments shown in this request; failed attempts may be cited. Never cite final or future results.",
        }


def applies(schema):
    return issubclass(schema, (Hypothesis, Seeds, Revision, MethodChange, Draft))


def invocation_scope(store, role, inputs, sources):
    """Offer only references to committed observations actually in role inputs.

    Candidate code and arbitrary nested dictionaries are never searched for
    citations. Context presentation preserves these observation bodies verbatim.
    Reading a registered record uses Store.json, so corruption stays fatal.
    """
    offered = set()

    def cached_record(key):
        row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", (key,)).fetchone()
        return store.json(row[0]) if row else None

    def nonfinal(observation):
        return (isinstance(observation, dict)
                and observation.get("phase") in {"subset", "full", "rebuttal"}
                and isinstance(observation.get("id"), str))

    def visit(value, depth=0):
        if depth > 32:
            raise ValueError("Evidence context nesting exceeds its bound")
        if isinstance(value, list):
            for item in value:
                visit(item, depth + 1)
        elif isinstance(value, dict):
            identifier, result = value.get("id"), value.get("result")
            if isinstance(identifier, str) and nonfinal(result):
                stored = cached_record("live-trial/" + identifier)
                if (stored and stored.get("id") == identifier
                        and canonical(stored.get("result")) == canonical(result)):
                    offered.add(identifier)
                    offered.add(result["id"])
                visit(result, depth + 1)
            if nonfinal(value):
                identifier = value["id"]
                stored = cached_record("development/" + identifier)
                if stored is not None and canonical(stored) == canonical(value):
                    offered.add(identifier)
                return
            # These are controller-owned evidence containers, not arbitrary
            # candidate fields or prose that happens to resemble an ID.
            for key in ("incumbent", "feedback", "evidence", "observation", "observations", "ablations", "matched", "control"):
                if key in value:
                    visit(value[key], depth + 1)

    if role == "evolve":
        visit(inputs.get("attempts", []))
    elif role == "refine":
        visit(inputs.get("incumbent", {}))
        visit(inputs.get("feedback", {}))
    elif role == "draft":
        visit(inputs.get("evidence", []))
    # Seeds precede experiments and may not cite experimental history.
    return OutputScope(tuple(sorted({source["id"] for source in sources})),
                       tuple(sorted(offered)), inputs.get("count") if role == "seeds" else None)


def validate_output(value, scope):
    """Validate references without modifying any part of the model's response."""
    if isinstance(value, Seeds):
        if (scope.seed_count is not None and len(value.ideas) != scope.seed_count
                or len({idea.title for idea in value.ideas}) != len(value.ideas)):
            raise AgentOutputError("Return exactly the requested count of seed ideas with distinct titles")
        ideas = value.ideas
    elif isinstance(value, (Revision, MethodChange)):
        ideas = [value.idea]
    else:
        ideas = [value]
    for idea in ideas:
        if not set(idea.source_ids) <= set(scope.source_ids):
            raise AgentOutputError(
                "source_ids must contain only document IDs in reference_contract.source_ids; "
                "experiment references belong in evidence_ids"
            )
        if isinstance(idea, Hypothesis) and not set(idea.evidence_ids) <= set(scope.evidence_ids):
            raise AgentOutputError(
                "evidence_ids must contain only the committed non-final IDs in "
                "reference_contract.evidence_ids for this invocation"
            )
    return value
