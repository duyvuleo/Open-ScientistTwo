"""Default controller normalization uses synthetic writer/PDF/measurement fixtures."""
from copy import deepcopy

import pytest

from scientisttwo.integrations.paperorchestra import controller_writer as bridge
from scientisttwo.integrations.paperorchestra.manuscript_binding import (
    SourceClaimBinding, bind_controller_manuscript, bind_paperorchestra_manuscript,
    fixture_bindings, render_bound_manuscript, verify_bound_manifest,
)
from scientisttwo.research.manuscript import LedgerClaim
from scientisttwo.store import IntegrityError, canonical
import test_paperorchestra_controller_writer as writer_tests
import test_paperorchestra_manuscript_binding as binding_tests
from test_paperorchestra_manuscript_binding import LATEX, publication_inputs
from test_paperorchestra_pdf import FakeContainer, IMAGE

context = writer_tests.context
setup = binding_tests.setup


def nested_bindings(latex, ledger):
    sidecar = fixture_bindings(latex, ledger)
    value = next(item for item in sidecar.numeric_values
        if item.location == "value" and ledger["entries"][item.evidence_id]["kind"] in {"measured", "derived"})
    start = latex.rfind("\n", 0, value.start) + 1
    end = latex.index("\n", value.end)
    quote = latex[start:end]
    entry = ledger["entries"][value.evidence_id]
    sidecar.claims = [
        SourceClaimBinding(start=start, end=end, quote=quote,
            claim=LedgerClaim(id="interpretation", kind="interpretation", text=quote,
                evidence_ids=[value.evidence_id], attribution_scope="package_comparison")),
        SourceClaimBinding(start=value.start, end=value.end, quote=value.quote,
            claim=LedgerClaim(id="numeric-wrapper", kind=entry["kind"], text=value.quote,
                value=entry["value"], evidence_ids=[value.evidence_id], attribution_scope="package_comparison")),
    ]
    return sidecar


def test_default_controller_handles_known_overlap_without_another_agent_call(context, monkeypatch):
    controller, ledger, inputs = context
    invoke = controller.backend.invoke
    raw = []
    def response(role, incoming, schema):
        result = invoke(role, incoming, schema)
        if role == "paperorchestra_bindings":
            result.payload = nested_bindings(incoming["latex"], incoming["ledger"]).model_dump(mode="json")
            raw.append(deepcopy(result.payload))
        return result
    monkeypatch.setattr(controller.backend, "invoke", response)
    pdf = bridge.ContainerPdfTools(IMAGE, executor=FakeContainer())
    result = bridge.write_controller_manuscript(controller, "normalized", inputs, ledger, pdf_tools=pdf)
    assert controller.backend.roles.count("paperorchestra_bindings") == 1
    assert len(result.manuscript.claims) == 1
    manifest = controller.store.json(result.rendered.manifest_ref)
    writer = manifest["writer"]
    assert controller.store.json(writer["bindings_ref"]) == raw[0]
    proof = controller.store.json(writer["binding_normalization_ref"])
    assert proof["removed_claims"][0]["claim_id"] == "numeric-wrapper"
    assert verify_bound_manifest(controller.store, manifest)["binding_normalization"] == "pass"
    before = controller.budget.summary()
    again = bridge.write_controller_manuscript(controller, "normalized", inputs, ledger, pdf_tools=pdf)
    assert again == result and controller.budget.summary() == before


def publication(setup, tmp_path):
    store, _, ledger = setup
    sidecar = nested_bindings(LATEX, ledger)
    manuscript, proof = bind_controller_manuscript(sidecar, LATEX, ledger, synthetic_agents=True)
    rendered = render_bound_manuscript(store, manuscript, sidecar, ledger,
        normalization_proof=proof, **publication_inputs(setup, tmp_path))
    return store, sidecar, proof, store.json(rendered.manifest_ref)


@pytest.mark.parametrize("change", ["removed", "input_hash", "output_hash", "numeric_hash", "missing_proof", "raw_binding", "proof_file"])
def test_normalization_proof_is_recomputed_and_tampering_rejected(setup, tmp_path, change):
    store, sidecar, proof, manifest = publication(setup, tmp_path)
    assert verify_bound_manifest(store, manifest)["binding_normalization"] == "pass"
    changed = deepcopy(manifest)
    if change == "missing_proof":
        changed["writer"].pop("binding_normalization_ref")
    elif change == "raw_binding":
        raw = sidecar.model_dump(mode="json")
        raw["claims"].pop()
        changed["writer"]["bindings_ref"] = store.put_json(raw)
    elif change == "proof_file":
        changed["files"]["evidence/paperorchestra-binding-normalization.json"]["ref"] = store.put_json({"forged": True})
    else:
        proof = deepcopy(proof)
        if change == "removed":
            proof["removed_claims"][0]["value"] += .01
        else:
            field = {"input_hash": "input_bindings_sha256", "output_hash": "output_bindings_sha256",
                     "numeric_hash": "numeric_bindings_sha256"}[change]
            proof[field] = "0" * 64
        changed["writer"]["binding_normalization_ref"] = store.put_json(proof)
    with pytest.raises((IntegrityError, ValueError)):
        verify_bound_manifest(store, changed)


@pytest.mark.parametrize("change", ["crossing", "conflicting"])
def test_controller_fallback_does_not_accept_other_overlaps(setup, change):
    _, _, ledger = setup
    sidecar = nested_bindings(LATEX, ledger)
    parent, child = sidecar.claims
    if change == "crossing":
        parent.end = child.start + 1
        parent.quote = parent.claim.text = LATEX[parent.start:parent.end]
    else:
        parent.claim.evidence_ids = ["unsupported-evidence"]
    with pytest.raises(ValueError):
        bind_controller_manuscript(sidecar, LATEX, ledger, synthetic_agents=True)


def test_legacy_bundle_remains_strict_and_has_no_implicit_normalization(setup, tmp_path, monkeypatch):
    from scientisttwo.integrations.paperorchestra import binding_normalization
    def forbidden(*args, **kwargs):
        pytest.fail("An already-valid legacy binding must not invoke normalization")
    monkeypatch.setattr(binding_normalization, "normalize_redundant_numeric_claims", forbidden)
    store, _, ledger = setup
    sidecar = fixture_bindings(LATEX, ledger)
    manuscript, proof = bind_controller_manuscript(sidecar, LATEX, ledger, synthetic_agents=True)
    assert proof is None
    assert manuscript == bind_paperorchestra_manuscript(sidecar, LATEX, ledger, synthetic_agents=True)
    rendered = render_bound_manuscript(store, manuscript, sidecar, ledger, **publication_inputs(setup, tmp_path))
    manifest = store.json(rendered.manifest_ref)
    assert "binding_normalization_ref" not in manifest["writer"]
    assert "binding_normalization" not in verify_bound_manifest(store, manifest)
    assert canonical(store.json(manifest["writer"]["bindings_ref"])) == canonical(sidecar)


def test_only_exact_overlap_failure_invokes_normalization(setup, monkeypatch):
    from scientisttwo.integrations.paperorchestra import binding_normalization
    def forbidden(*args, **kwargs):
        pytest.fail("A source identity failure must not invoke normalization")
    monkeypatch.setattr(binding_normalization, "normalize_redundant_numeric_claims", forbidden)
    _, _, ledger = setup
    sidecar = nested_bindings(LATEX, ledger)
    sidecar.latex_sha256 = "0" * 64
    with pytest.raises(ValueError, match="source hash"):
        bind_controller_manuscript(sidecar, LATEX, ledger, synthetic_agents=True)
