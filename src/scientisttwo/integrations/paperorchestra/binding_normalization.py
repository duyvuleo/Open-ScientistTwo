"""Pure, narrowly scoped normalization of redundant source-binding annotations.

This never edits TeX, evidence, numeric bindings or persisted model responses.
Callers must preserve the original response and explicitly record the returned
proof when using this as an externally versioned recovery step.
"""
from copy import deepcopy
import math

from ...research.manuscript import _number_matches
from ...store import canonical, digest
from .manuscript_binding import (
    PaperOrchestraBindings, _check_span, _NUMBER, _numeric_value,
    bind_paperorchestra_manuscript,
)


NORMALIZATION_VERSION = "paperorchestra-redundant-numeric-claim-normalization-v1"


def normalize_redundant_numeric_claims(
    bindings, latex, ledger, sources=(), *, synthetic_agents, concern_responses=(),
):
    """Remove only exact numeric claim wrappers redundant with an outer claim.

    Every removed measured/derived claim must be wholly inside one interpretation
    carrying the same evidence and attribution scope. A retained independent
    aggregate numeric binding must validate the same literal interval and value.
    All other overlaps and all subsequent original-binder failures remain fatal.
    Returned payload/proof are in memory only; original inputs are not mutated.
    """
    original = (bindings.model_dump(mode="json") if isinstance(bindings, PaperOrchestraBindings)
                else deepcopy(bindings))
    parsed = PaperOrchestraBindings.model_validate(original)
    source_hash = digest(latex.encode())
    if parsed.latex_sha256 != source_hash:
        raise ValueError("Normalization source hash differs from the bound TeX")
    for span in [*parsed.sections, *parsed.claims, *parsed.numeric_values]:
        _check_span(span, latex)
    identifiers = [bound.claim.id for bound in parsed.claims]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Normalization cannot repair duplicate claim identities")
    if any(bound.claim.text != bound.quote for bound in parsed.claims):
        raise ValueError("Normalization requires literal claim text")

    removed = {}
    for index, left in enumerate(parsed.claims):
        for right in parsed.claims[index + 1:]:
            if left.end <= right.start or right.end <= left.start:
                continue
            if left.start <= right.start and right.end <= left.end and (left.start, left.end) != (right.start, right.end):
                parent, child = left, right
            elif right.start <= left.start and left.end <= right.end and (left.start, left.end) != (right.start, right.end):
                parent, child = right, left
            else:
                raise ValueError("Crossing or identical claim overlaps cannot be normalized")
            if (parent.claim.kind != "interpretation" or child.claim.kind not in {"measured", "derived"}
                    or len(child.claim.evidence_ids) != 1
                    or child.claim.evidence_ids[0] not in parent.claim.evidence_ids
                    or child.claim.attribution_scope != parent.claim.attribution_scope
                    or child.claim.attribution_scope == "isolated_component"
                    or child.claim.source_ids or child.claim.passage_ids
                    or not _NUMBER.fullmatch(child.quote)):
                raise ValueError("Overlapping claim annotations are not redundant numeric wrappers")
            evidence_id = child.claim.evidence_ids[0]
            entry = ledger.get("entries", {}).get(evidence_id, {})
            if entry.get("kind") != child.claim.kind:
                raise ValueError("Nested numeric claim differs from its evidence kind")
            matches = [bound for bound in parsed.numeric_values
                if (bound.start, bound.end, bound.quote) == (child.start, child.end, child.quote)]
            if (len(matches) != 1 or matches[0].evidence_id != evidence_id
                    or matches[0].location != "value"):
                raise ValueError("Redundant numeric claim requires one identical aggregate numeric binding")
            value = _numeric_value(matches[0], ledger)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or child.claim.value is None or child.claim.value != value
                    or not _number_matches(child.quote.replace(r"\%", "%"), [value])):
                raise ValueError("Redundant numeric claim/value binding differs from recorded evidence")
            if child.claim.id in removed:
                raise ValueError("Numeric claim has multiple containing annotations")
            removed[child.claim.id] = {
                "claim_id": child.claim.id, "kind": child.claim.kind,
                "start": child.start, "end": child.end, "quote": child.quote,
                "evidence_id": evidence_id, "value": value,
                "parent_claim_id": parent.claim.id,
                "parent_start": parent.start, "parent_end": parent.end,
                "retained_numeric_binding_index": parsed.numeric_values.index(matches[0]),
            }

    normalized = deepcopy(original)
    if removed:
        normalized["claims"] = [claim for claim in original["claims"] if claim["claim"]["id"] not in removed]
    # The unchanged binder checks every retained annotation, exact value, source
    # coverage, citation and attribution. Normalization is not an acceptance bypass.
    manuscript = bind_paperorchestra_manuscript(normalized, latex, ledger, sources,
        synthetic_agents=synthetic_agents, concern_responses=concern_responses)
    unchanged = sorted(key for key in original if key != "claims")
    if any(canonical(original[key]) != canonical(normalized[key]) for key in unchanged):
        raise AssertionError("Normalization changed a non-claim field")
    return {"bindings": normalized, "proof": {
        "schema_version": NORMALIZATION_VERSION,
        "changed": bool(removed), "latex_sha256": source_hash,
        "input_bindings_sha256": digest(canonical(original)),
        "output_bindings_sha256": digest(canonical(normalized)),
        "ledger_sha256": digest(canonical(ledger)),
        "sources_sha256": digest(canonical(sources)),
        "removed_claims": sorted(removed.values(), key=lambda value: (value["start"], value["claim_id"])),
        "unchanged_fields": unchanged,
        "retained_claims_sha256": digest(canonical(normalized.get("claims", []))),
        "numeric_bindings_sha256": digest(canonical(original.get("numeric_values", []))),
        "original_binder_validation": "pass",
        "canonical_manuscript_sha256": digest(canonical(manuscript)),
        "scope": "Redundant annotation removal only; original TeX, evidence and numeric bindings unchanged; scientific validity remains unverified",
    }}
