"""Synthetic literal-binding fixtures; no models, containers or persisted runs."""
from copy import deepcopy

import pytest

from scientisttwo.integrations.paperorchestra.binding_normalization import (
    NORMALIZATION_VERSION, normalize_redundant_numeric_claims,
)
from scientisttwo.integrations.paperorchestra.manuscript_binding import (
    SourceClaimBinding, bind_paperorchestra_manuscript, fixture_bindings,
)
from scientisttwo.research.manuscript import LedgerClaim
from scientisttwo.store import canonical, digest


def example(kind="derived"):
    text = "The recorded difference was 0.6; this comparison is descriptive."
    latex = (r"\documentclass{article}\title{Synthetic normalization fixture}\begin{document}"
             + "\n" + r"\section{Recorded measurements}" + "\n" + text + "\n" + r"\end{document}")
    identifier = kind + ":synthetic"
    ledger = {"includes_final": False, "entries": {identifier: {
        "kind": kind, "value": .6, "metric": "synthetic_metric", "per_seed": []}},
        "observations": {}, "control_context": [], "tables": {}, "figures": {}}
    sidecar = fixture_bindings(latex, ledger)
    parent_start = latex.index(text)
    value_start = latex.index("0.6")
    sidecar.claims = [
        SourceClaimBinding(start=parent_start, end=parent_start + len(text), quote=text,
            claim=LedgerClaim(id="interpretation", kind="interpretation", text=text,
                evidence_ids=[identifier], attribution_scope="package_comparison")),
        SourceClaimBinding(start=value_start, end=value_start + 3, quote="0.6",
            claim=LedgerClaim(id="numeric", kind=kind, text="0.6", value=.6,
                evidence_ids=[identifier], attribution_scope="package_comparison")),
    ]
    return sidecar.model_dump(mode="json"), latex, ledger


@pytest.mark.parametrize("kind", ["measured", "derived"])
def test_redundant_numeric_wrapper_only_is_removed_and_original_binder_accepts(kind):
    bindings, latex, ledger = example(kind)
    before = canonical(bindings), latex, canonical(ledger)
    with pytest.raises(ValueError, match="Overlapping"):
        bind_paperorchestra_manuscript(bindings, latex, ledger, synthetic_agents=True)
    result = normalize_redundant_numeric_claims(bindings, latex, ledger, synthetic_agents=True)
    fixed, proof = result["bindings"], result["proof"]
    assert fixed["claims"] == bindings["claims"][:1]
    assert all(fixed[key] == value for key, value in bindings.items() if key != "claims")
    assert (canonical(bindings), latex, canonical(ledger)) == before
    assert proof["schema_version"] == NORMALIZATION_VERSION and proof["changed"]
    assert proof["removed_claims"][0]["claim_id"] == "numeric"
    assert proof["removed_claims"][0]["parent_claim_id"] == "interpretation"
    assert proof["removed_claims"][0]["retained_numeric_binding_index"] == 0
    assert proof["input_bindings_sha256"] == digest(canonical(bindings))
    assert proof["output_bindings_sha256"] == digest(canonical(fixed))
    assert proof["latex_sha256"] == digest(latex.encode())
    assert proof["original_binder_validation"] == "pass"
    assert normalize_redundant_numeric_claims(bindings, latex, ledger, synthetic_agents=True) == result
    no_op = normalize_redundant_numeric_claims(fixed, latex, ledger, synthetic_agents=True)
    assert no_op["bindings"] == fixed and no_op["proof"]["changed"] is False


@pytest.mark.parametrize("change", ["source_hash", "claim_quote", "claim_text", "parent_evidence",
    "kind", "value", "missing_value", "missing_numeric", "numeric_evidence", "numeric_location",
    "duplicate_numeric", "duplicate_claim_id", "attribution", "citation", "parent_kind"])
def test_conflicting_or_unverified_nested_annotations_cannot_be_repaired(change):
    bindings, latex, ledger = example()
    parent, child = bindings["claims"]
    if change == "source_hash":
        bindings["latex_sha256"] = "0" * 64
    elif change == "claim_quote":
        child["quote"] = "0.7"
    elif change == "claim_text":
        child["claim"]["text"] = "0.7"
    elif change == "parent_evidence":
        parent["claim"]["evidence_ids"] = ["derived:other"]
    elif change == "kind":
        child["claim"]["kind"] = "measured"
    elif change == "value":
        child["claim"]["value"] = .7
    elif change == "missing_value":
        child["claim"]["value"] = None
    elif change == "missing_numeric":
        bindings["numeric_values"] = []
    elif change == "numeric_evidence":
        bindings["numeric_values"][0]["evidence_id"] = "derived:other"
    elif change == "numeric_location":
        bindings["numeric_values"][0].update(location="per_seed_value", seed_index=0)
    elif change == "duplicate_numeric":
        bindings["numeric_values"].append(deepcopy(bindings["numeric_values"][0]))
    elif change == "duplicate_claim_id":
        child["claim"]["id"] = parent["claim"]["id"]
    elif change == "attribution":
        child["claim"]["attribution_scope"] = "isolated_component"
    elif change == "citation":
        child["claim"]["source_ids"] = ["citation-cannot-be-discarded"]
    else:
        parent["claim"]["kind"] = "method"
    before = canonical(bindings)
    with pytest.raises(ValueError):
        normalize_redundant_numeric_claims(bindings, latex, ledger, synthetic_agents=True)
    assert canonical(bindings) == before


@pytest.mark.parametrize("change", ["crossing", "identical", "nested_interpretation"])
def test_other_overlap_types_remain_fatal(change):
    bindings, latex, ledger = example()
    parent, child = bindings["claims"]
    if change == "crossing":
        parent["end"] = child["start"] + 1
    elif change == "identical":
        parent["start"], parent["end"] = child["start"], child["end"]
    else:
        child["claim"]["kind"] = "interpretation"
        child["claim"]["value"] = None
    parent["quote"] = parent["claim"]["text"] = latex[parent["start"]:parent["end"]]
    with pytest.raises(ValueError):
        normalize_redundant_numeric_claims(bindings, latex, ledger, synthetic_agents=True)


def test_normalization_never_bypasses_another_original_binder_failure():
    bindings, latex, ledger = example()
    bindings["title"] = "Title absent from the authoritative TeX"
    with pytest.raises(ValueError, match="title"):
        normalize_redundant_numeric_claims(bindings, latex, ledger, synthetic_agents=True)
