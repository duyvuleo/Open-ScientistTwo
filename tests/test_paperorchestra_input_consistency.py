"""Input consistency uses fixed upstream source and synthetic model/PDF services."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scientisttwo.integrations.paperorchestra import controller_writer as bridge
from scientisttwo.integrations.paperorchestra.runtime import compatible_template, ModelRequest, SOURCE_HASHES, verify_upstream
from scientisttwo.models import AgentResult
from scientisttwo.store import IntegrityError, digest

import test_paperorchestra_controller_writer as writer_tests
import test_paperorchestra_runtime as runtime_tests

context = writer_tests.context
upstream = runtime_tests.upstream
materials = runtime_tests.materials


def test_compatibility_template_changes_only_one_package_insertion(upstream):
    path = upstream / "templates/iclr2025/template.tex"
    source = path.read_bytes()
    staged, proof = compatible_template(source)
    addition = b"\\usepackage{booktabs}\n"
    assert source.count(addition) == 0 and staged.count(addition) == 1
    assert staged.replace(addition, b"", 1) == source
    assert staged.index(addition) < staged.index(b"\\begin{document}")
    assert proof["source_sha256"] == digest(source) == SOURCE_HASHES["templates/iclr2025/template.tex"]
    assert proof["staged_sha256"] == digest(staged)
    assert proof["added_packages"] == ["booktabs"] and proof["upstream_checkout_modified"] is False
    repeated, again = compatible_template(staged)
    assert repeated == staged and again["added_packages"] == []
    assert path.read_bytes() == source


@pytest.mark.parametrize("declaration", [r"\usepackage{booktabs}", r"\usepackage{array, booktabs}",
                                          r"\RequirePackage[2020/01/12]{booktabs}"])
def test_existing_active_booktabs_declaration_is_preserved(declaration):
    source = ("\\documentclass{article}\n" + declaration + "\n\\begin{document}\nBody\n\\end{document}\n").encode()
    staged, proof = compatible_template(source)
    assert staged == source and proof["change"] == "already_compatible"


def test_commented_package_and_document_marker_are_not_active_declarations():
    source = b"% \\usepackage{booktabs}\r\n% \\begin{document}\r\n\\documentclass{article}\r\n\\begin{document}\r\nBody\\end{document}\r\n"
    staged, proof = compatible_template(source)
    assert proof["added_packages"] == ["booktabs"]
    assert staged.replace(b"\\usepackage{booktabs}\r\n", b"", 1) != source  # First occurrence is a comment.
    active_addition = b"\\documentclass{article}\r\n\\usepackage{booktabs}\r\n\\begin{document}"
    assert active_addition in staged
    assert compatible_template(staged)[0] == staged


@pytest.mark.parametrize("source", [b"no document marker", b"\\begin{document}\\begin{document}",
    b"\\usepackage{booktabs}\n\\usepackage{booktabs}\n\\begin{document}"])
def test_ambiguous_template_fails_instead_of_rewriting_unrelated_preamble(source):
    with pytest.raises(ValueError):
        compatible_template(source)


def test_both_upstream_writers_receive_the_same_compatible_template(upstream, materials, tmp_path):
    before = verify_upstream(upstream)
    result, transport, report = runtime_tests.run(upstream, materials, tmp_path)
    assert result.status == "success", result.error
    assert [item.role for item in transport.requests[:2]] == ["outline", "section_writing"]
    assert all(r"\usepackage{booktabs}" in item.prompt for item in transport.requests[:2])
    staged = (tmp_path / "attempt/latex_writeup/template.tex").read_bytes()
    assert staged.count(b"\\usepackage{booktabs}") == 1
    assert (tmp_path / "attempt/literature_agent_output/updated_template.tex").read_bytes() == staged
    proof = json.loads((tmp_path / "attempt/template-compatibility.json").read_text())
    assert proof == report["template_compatibility"] and proof["staged_sha256"] == digest(staged)
    assert any("booktabs" in item and "unchanged" in item for item in report["adaptations"])
    assert verify_upstream(upstream) == before


def inventory_ledger():
    return {"schema_version": "evidence-ledger-v2", "includes_final": False, "entries": {
        **{f"measurement:{index}": {"kind": "measured", "metric": "mse", "value": index / 10} for index in range(10)},
        **{f"history:{index}": {"kind": "research_history", "text": "Synthetic historical decision"} for index in range(3)},
        "method:selected": {"kind": "method", "text": "Synthetic method description"},
    }}


def test_inventory_distinguishes_ten_measurements_from_four_other_records():
    ledger = inventory_ledger()
    before = deepcopy(ledger)
    value = bridge.authoritative_evidence_inventory(ledger)
    assert value["status"] == "available"
    assert value["counts_by_kind"] == {"measured": 10, "method": 1, "research_history": 3}
    assert value["measured_entry_count"] == 10 and value["total_entries"] == 14
    assert value["measured_entry_ids"] == sorted(f"measurement:{index}" for index in range(10))
    assert value["ledger_sha256"] == digest(bridge.canonical(ledger))
    assert ledger == before


@pytest.mark.parametrize("ledger", [None, {}, {"entries": {}}, {"sources": {"fixture": {}}}])
def test_unavailable_inventory_never_asserts_zero_measurements(ledger):
    value = bridge.authoritative_evidence_inventory(ledger)
    assert value["status"] == "unavailable"
    assert value["counts_by_kind"] is value["measured_entry_count"] is value["measured_entry_ids"] is None


def test_production_context_prioritizes_ledger_over_incorrect_outline_and_freezes_proof(context):
    controller, _, _ = context
    controller.backend.synthetic = False  # Only the injection below executes; no live backend exists.
    calls = []

    def invoke(role, inputs, schema):
        calls.append(deepcopy(inputs))
        inventory = inputs["evidence_inventory"]
        output = json.dumps({"record_count": inventory["measured_entry_count"], "ids": inventory["measured_entry_ids"]})
        return AgentResult(payload={"output": output}, backend=controller.backend.name,
                           model=controller.config.model, synthetic=False,
                           usage={"input_tokens": 10, "output_tokens": 10})

    controller.backend.invoke = invoke
    ledger = inventory_ledger()
    original_ledger = deepcopy(ledger)
    upstream_prompt = "Generated outline A.1 requests 12 measurement records, all with empirical values."
    request = ModelRequest("section_writing", upstream_prompt, controller.config.model, "latex")
    def make():
        return bridge.ControllerTransport(controller, "inventory", ledger, SimpleNamespace(synthetic=True))
    result = make()(request)
    parsed = json.loads(result.raw_text)
    assert parsed["record_count"] == 10 and set(parsed["ids"]) == {f"measurement:{index}" for index in range(10)}
    assert calls[0]["prompt"] == upstream_prompt and "scripted_ledger" not in calls[0]
    assert calls[0]["instruction"].startswith("Controller evidence authority takes precedence")
    assert "outline is a writing plan, not experimental evidence" in calls[0]["instruction"]
    assert ledger == original_ledger
    proof_ref = controller.store.get("paperorchestra-evidence-inventory/inventory/upstream/0001")
    proof = controller.store.json(proof_ref)
    assert proof["inventory"] == calls[0]["evidence_inventory"]
    assert proof["source_priority"] == "typed_ledger_and_frozen_log_over_generated_outline"
    assert proof["no_measurements_generated"] is True
    before = controller.budget.summary()
    assert make()(request) == result and len(calls) == 1 and controller.budget.summary() == before
    ledger["entries"].pop("measurement:9")
    with pytest.raises(IntegrityError, match="evidence inventory changed"):
        make()(request)
    assert len(calls) == 1 and controller.store.get("paperorchestra-evidence-inventory/inventory/upstream/0001") == proof_ref


def test_nonempty_malformed_entries_fail_instead_of_inventing_an_inventory():
    for ledger in ({"entries": [1]}, {"entries": {"measurement:1": {}}}):
        with pytest.raises(IntegrityError):
            bridge.authoritative_evidence_inventory(ledger)
