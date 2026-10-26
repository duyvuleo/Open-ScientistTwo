"""Synthetic binding/receipt contracts; no actual scientific or PDF validation."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scientisttwo.integrations.paperorchestra.manuscript_binding import (
    PaperOrchestraBindings, SourceClaimBinding, bind_paperorchestra_manuscript,
    binding_prompt_context,
    fixture_bindings, render_bound_manuscript, verify_bound_final_manifest,
    verify_bound_manifest, _required_numbers,
)
from scientisttwo.live.models import Candidate, Observation, SourceFile
from scientisttwo.live.runtime import Budget
from scientisttwo.research.manuscript import (
    LedgerClaim, build_evidence_ledger, final_results_appendix, validate_manuscript,
)
from scientisttwo.store import IntegrityError, Store, canonical, digest

IMAGE = "sha256:" + "a" * 64
LATEX = r"""\documentclass{article}
\usepackage{longtable}
\title{Synthetic bound document}
\begin{document}
\begin{abstract}This is a scripted source fixture.\end{abstract}
\section{Method}
The tested source remains unchanged.
\section{Recorded measurements}
\begin{longtable}{p{0.5\linewidth}rr}
Recorded mean & 0.6 & \\
Seed values & 0: 0.5 & 1: 0.7 \\
\end{longtable}
\section{Limitations}
Scientific interpretation remains unverified.
\end{document}
"""


@pytest.fixture
def setup(tmp_path):
    store = Store(tmp_path / "campaign")
    budget = Budget(store, SimpleNamespace(campaign_seconds=1000, max_calls=10,
                                          max_experiments=10, token_stop_threshold=10000))
    source = Candidate(files=[SourceFile(path="train.py", content="print('synthetic fixture')")],
                       command=["python", "train.py"], description="Synthetic test candidate")
    protocol = {"phase": "full", "seeds": [0, 1], "adapter": "synthetic_unit_test"}
    observation = Observation(id="baseline", phase="full", status="success", value=.6,
        values=[.5, .7], metric="accuracy", protocol_hash=digest(canonical(protocol)),
        protocol_ref=store.put_json(protocol), code_ref=store.put_json(source),
        prediction_refs=[store.put_json({"synthetic_prediction": True})])
    store.cache("development/baseline", {}, observation)
    ledger = build_evidence_ledger(store, [observation])
    yield store, budget, ledger
    store.close()


def binding(ledger, latex=LATEX):
    result = fixture_bindings(latex, ledger)
    return result, bind_paperorchestra_manuscript(result, latex, ledger, synthetic_agents=True)


def test_review_projection_is_full_literal_source_with_bound_values(setup):
    _, _, ledger = setup
    sidecar, manuscript = binding(ledger)
    assert validate_manuscript(manuscript, ledger).status == "pass"
    assert "The tested source remains unchanged." in manuscript.method.text
    assert "[Bound value " in manuscript.results.text
    assert "0.6" not in manuscript.results.text
    assert manuscript.synthetic_agents
    assert sidecar.latex_sha256 == digest(LATEX.encode())
    assert "natural_language_claim_completeness" in validate_manuscript(manuscript, ledger).checks


@pytest.mark.parametrize("change", ["hash", "quote", "gap", "overlap", "drop_section", "number", "drop_number", "title"])
def test_source_sidecar_cannot_rewrite_omit_or_fabricate_source(setup, change):
    _, _, ledger = setup
    sidecar, _ = binding(ledger)
    if change == "hash":
        sidecar.latex_sha256 = "0" * 64
    elif change == "quote":
        sidecar.sections[0].quote = "unrelated summary"
    elif change == "gap":
        sidecar.sections[0].end -= 1
        sidecar.sections[0].quote = sidecar.sections[0].quote[:-1]
    elif change == "overlap":
        sidecar.sections.append(sidecar.sections[0])
    elif change == "drop_section":
        sidecar.sections.pop()
    elif change == "number":
        sidecar.numeric_values[0].evidence_id = "fabricated"
    elif change == "drop_number":
        sidecar.numeric_values.pop(0)
    else:
        sidecar.title = "An unrelated generated manuscript"
    with pytest.raises(ValueError):
        bind_paperorchestra_manuscript(sidecar, LATEX, ledger, synthetic_agents=True)


def test_unbound_table_integer_and_decimal_are_rejected(setup):
    _, _, ledger = setup
    for text in (LATEX.replace("0.6", "99.9"), LATEX.replace("0.6", "99")):
        with pytest.raises(ValueError, match="absent from"):
            fixture_bindings(text, ledger)


@pytest.mark.parametrize("cell,expected", [
    (r"\multicolumn{3}{l}{Screening}", []),
    (r"\multicolumn{3}{l}{0.6}", ["0.6"]),
    (r"\multicolumn{2}{@{}l@{}}{\textbf{0.6}}", ["0.6"]),
    (r"\multicolumn{2}{|p{0.5\linewidth}|}{0.6}", ["0.6"]),
    (r"\multirow{2}{*}{0.6}", ["0.6"]),
    (r"\multirow[t]{-2.5}{2cm}{0.6}", ["0.6"]),
    (r"\multirow{2}{=}{600}", ["600"]),
    (r"\multicolumn{3}{c}{\multirow{2}{*}{0.6}}", ["0.6"]),
])
def test_only_standard_multicolumn_multirow_layout_arguments_are_exempt(cell, expected):
    latex = r"\begin{tabular}{lll}" + cell + r"\\\end{tabular}"
    assert [match[0] for match in _required_numbers(latex)] == expected


@pytest.mark.parametrize("cell,unsupported", [
    (r"\multicolumn{3}{l}{99.9}", "99.9"),
    (r"\multirow{2}{*}{\textbf{600}}", "600"),
    (r"\multicolumn{3}{>{99.9}l}{0.6}", "99.9"),
    (r"\multicolumn{3}{@{99.9}l}{0.6}", "99.9"),
    (r"\multirow{2}{99.9}{0.6}", "99.9"),
    (r"\multicolumn{3}{l}{\phantom{99.9}}", "99.9"),
])
def test_layout_syntax_cannot_hide_unsupported_cell_or_arbitrary_format_content(setup, cell, unsupported):
    _, _, ledger = setup
    latex = LATEX.replace("Recorded mean & 0.6 &", cell + " &")
    assert unsupported in [match[0] for match in _required_numbers(latex)]
    with pytest.raises(ValueError, match="absent from"):
        fixture_bindings(latex, ledger)


def test_full_binding_accepts_header_spans_and_width_without_fabricating_score_evidence(setup):
    _, _, ledger = setup
    latex = LATEX.replace("Recorded mean & 0.6 &", r"\multicolumn{3}{l}{Screening} \\" + "\n"
                         + r"\multirow{2.0}{0.5\linewidth}{Recorded mean} & 0.6 &")
    sidecar, manuscript = binding(ledger, latex)
    assert all(bound.quote not in {"3", "2.0"} for bound in sidecar.numeric_values)
    assert any(bound.quote == "0.6" for bound in sidecar.numeric_values)
    assert validate_manuscript(manuscript, ledger).status == "pass"
    assert r"\multicolumn{3}{l}{Screening}" in manuscript.results.text


def test_removing_actual_multicolumn_cell_value_binding_still_fails(setup):
    _, _, ledger = setup
    latex = LATEX.replace("Recorded mean & 0.6 &", r"\multicolumn{3}{l}{0.6}")
    sidecar = fixture_bindings(latex, ledger)
    start = latex.index(r"\multicolumn{3}{l}{0.6}") + len(r"\multicolumn{3}{l}{")
    sidecar.numeric_values = [value for value in sidecar.numeric_values if value.start != start]
    with pytest.raises(ValueError, match="Unbound rendered numerical"):
        bind_paperorchestra_manuscript(sidecar, latex, ledger, synthetic_agents=True)


def test_exact_claim_text_and_strict_original_claim_validation_are_preserved(setup):
    _, _, ledger = setup
    sidecar, _ = binding(ledger)
    quote = "The tested source remains unchanged."
    start = LATEX.index(quote)
    evidence = next(iter(ledger["entries"]))
    sidecar.claims = [SourceClaimBinding(start=start, end=start + len(quote), quote=quote,
        claim=LedgerClaim(id="method-claim", kind="interpretation", text=quote, evidence_ids=[evidence]))]
    manuscript = bind_paperorchestra_manuscript(sidecar, LATEX, ledger, synthetic_agents=True)
    assert manuscript.claims[0].text == quote
    assert manuscript.method.claim_ids == ["method-claim"]
    sidecar.claims[0].claim.text = "A rewrite unsupported by the original manuscript"
    with pytest.raises(ValueError, match="literal source"):
        bind_paperorchestra_manuscript(sidecar, LATEX, ledger, synthetic_agents=True)


def test_unknown_citations_and_final_evidence_are_refused(setup):
    _, _, ledger = setup
    latex = LATEX.replace("The tested", r"\cite{unknown} The tested")
    sidecar = fixture_bindings(latex, ledger)
    with pytest.raises(ValueError, match="unknown citation"):
        bind_paperorchestra_manuscript(sidecar, latex, ledger, synthetic_agents=True)
    sidecar = fixture_bindings(LATEX, ledger)
    ledger["includes_final"] = True
    with pytest.raises(ValueError, match="protected final"):
        bind_paperorchestra_manuscript(sidecar, LATEX, ledger, synthetic_agents=True)


def publication_inputs(setup, tmp_path):
    store, budget, ledger = setup
    project = tmp_path / "upstream-project"
    project.mkdir()
    latex_path, pdf_path = project / "final_refined_paper.tex", tmp_path / "final_paper.pdf"
    latex_path.write_text(LATEX)
    pdf = b"%PDF-synthetic receipt contract fixture only"
    pdf_path.write_bytes(pdf)
    (project / "references.bib").write_bytes(b"")
    inputs = {name: {"sha256": digest((project / name).read_bytes()), "bytes": (project / name).stat().st_size}
              for name in ("final_refined_paper.tex", "references.bib")}
    worker = b"synthetic compiler dependency fixture; never executed"
    worker_ref = store.put(worker, "py")
    receipt = {"operation": "compile", "status": "success", "image_digest": IMAGE,
        "network_disabled": True, "shell_escape_disabled": True,
        "argv": ["docker", "run", "--network=none"],
        "process": {"code": 0, "timed_out": False, "limit_exceeded": None},
        "inputs": inputs, "outputs": {"paper.pdf": {"sha256": digest(pdf), "bytes": len(pdf)}},
        "worker_sha256": digest(worker)}
    record = {"request": {"operation": "compile", "tex_name": "final_refined_paper",
                           "inputs": inputs, "identity": {"synthetic_test": True}},
              "outputs": {"paper.pdf": store.put(pdf, "pdf")}, "receipt": receipt,
              "worker_ref": worker_ref, "worker_sha256": digest(worker)}
    key = "paperorchestra/test/pdf/0001"
    budget.reserve(key, "probe", record["request"])
    budget.finish(key, "success", record, .01)
    report = {"status": "success", "controller_pdf_receipts": [{"operation_key": key, **record}],
              "authoritative_compile": {"source_tex_sha256": digest(LATEX.encode()),
                  "pdf_sha256": digest(pdf), "operation_index": 0},
              "pdf_operations": [{"operation": "compile", "source_tex_sha256": digest(LATEX.encode()),
                                  "sha256": digest(pdf)}]}
    return {"latex_path": latex_path, "pdf_path": pdf_path, "runtime_ref": store.put_json(report),
            "image_digest": IMAGE, "project_dir": project}


def test_authoritative_source_pdf_and_binding_are_sealed_without_rewriting(setup, tmp_path):
    store, budget, ledger = setup
    sidecar, manuscript = binding(ledger)
    kwargs = publication_inputs(setup, tmp_path)
    before = budget.summary()
    rendered = render_bound_manuscript(store, manuscript, sidecar, ledger, **kwargs)
    assert store.read(rendered.latex_ref) == LATEX.encode()
    assert store.read(rendered.pdf_ref) == kwargs["pdf_path"].read_bytes()
    assert budget.summary() == before
    manifest = store.json(rendered.manifest_ref)
    checks = verify_bound_manifest(store, manifest, expected_image_digest=IMAGE)
    assert checks["authoritative_pdf"] == "pass"
    assert checks["table_label_entailment"] == "unverified"
    again = render_bound_manuscript(store, manuscript, sidecar, ledger, **kwargs)
    assert again == rendered
    assert budget.summary() == before


@pytest.mark.parametrize("change", ["canonical", "source", "pdf", "worker", "supporting", "image", "receipt", "probe"])
def test_binding_and_receipt_tampering_cannot_be_accepted(setup, tmp_path, change):
    store, _, ledger = setup
    sidecar, manuscript = binding(ledger)
    kwargs = publication_inputs(setup, tmp_path)
    report = deepcopy(store.json(kwargs["runtime_ref"]))
    if change == "canonical":
        manuscript.method.text = "Unrelated rewritten canonical content"
    elif change == "source":
        kwargs["latex_path"].write_text(LATEX.replace("unchanged", "rewritten"))
    elif change == "pdf":
        kwargs["pdf_path"].write_bytes(b"%PDF-other document")
    elif change == "worker":
        record = report["controller_pdf_receipts"][0]
        (store.root / record["worker_ref"]).write_bytes(b"modified dependency")
    elif change == "supporting":
        (kwargs["project_dir"] / "references.bib").write_bytes(b"fabricated bibliography")
    elif change == "image":
        kwargs["image_digest"] = "sha256:" + "b" * 64
    elif change == "receipt":
        report["controller_pdf_receipts"][0]["receipt"]["network_disabled"] = False
        kwargs["runtime_ref"] = store.put_json(report)
    else:
        store.db.execute("UPDATE live_operations SET status='running'")
    with pytest.raises((IntegrityError, ValueError)):
        render_bound_manuscript(store, manuscript, sidecar, ledger, **kwargs)


def test_protected_final_appendix_keeps_upstream_document_without_model_calls(setup, tmp_path):
    store, budget, ledger = setup
    sidecar, manuscript = binding(ledger)
    rendered = render_bound_manuscript(store, manuscript, sidecar, ledger,
                                      **publication_inputs(setup, tmp_path))
    result = {"status": "completed", "metric": "accuracy", "score": .625}
    store.set("live_frozen", {"candidate_id": "synthetic-selected"})
    store.set("live_final", {"selected": result})
    budget.reserve("final/selected", "experiment", {"synthetic": True})
    budget.finish("final/selected", "success", result, .01)
    before = budget.summary()
    final = final_results_appendix(store, rendered, {"selected": result})
    manifest = store.json(final.manifest_ref)
    assert verify_bound_final_manifest(store, manifest)["paperorchestra_development_source_preserved"] == "pass"
    assert budget.summary() == before
    assert "0.625" not in store.read(rendered.latex_ref).decode()
    assert "0.625" in store.read(manifest["files"]["final-results.tex"]["ref"]).decode()
    changed = deepcopy(manifest)
    text = store.read(changed["files"]["main.tex"]["ref"]).replace(b"unchanged", b"modified")
    changed["files"]["main.tex"]["ref"] = store.put(text, "tex")
    with pytest.raises(IntegrityError, match="development prose"):
        verify_bound_final_manifest(store, changed)


def test_binding_schema_is_closed_and_offsets_are_explicit(setup):
    _, _, ledger = setup
    values = fixture_bindings(LATEX, ledger).model_dump()
    values["unexpected"] = "not allowed"
    with pytest.raises(ValueError):
        PaperOrchestraBindings.model_validate(values)
    values.pop("unexpected")
    values["sections"][0]["end"] += 1
    with pytest.raises(ValueError, match="exact Python character"):
        PaperOrchestraBindings.model_validate(values)


def test_live_binding_context_supplies_exact_unicode_offsets_without_semantic_approval(setup):
    _, _, ledger = setup
    latex = LATEX.replace("This is", "αβ This is")
    context = binding_prompt_context(latex, ledger)
    assert context["latex_sha256"] == digest(latex.encode())
    assert "Python Unicode character" in context["offsets"]
    spans = context["section_spans"]
    assert spans[0]["start"] == context["body_start"]
    assert spans[-1]["end"] == context["body_end"]
    for item in spans + context["passage_spans"] + context["required_numeric_spans"]:
        assert latex[item["start"]:item["end"]] == item["quote"]
    groups = context["numeric_candidate_groups"]
    assert all(groups[item["candidate_group"]]["matching_candidates"] for item in context["required_numeric_spans"])
    assert "scientific support" in " ".join(context["instructions"])
    assert not any("approved" in item for item in context)
    tampered = binding_prompt_context(latex.replace("0.6", "99.9"), ledger)
    missing = [item for item in tampered["required_numeric_spans"] if item["quote"] == "99.9"]
    assert len(missing) == 1
    assert tampered["numeric_candidate_groups"][missing[0]["candidate_group"]]["matching_candidates"] == []


def test_duplicate_heavy_binding_context_shares_all_candidates_losslessly():
    # A realistic many-trial ledger repeats seeds and common measured values.
    # The prompt keeps every match once per number, rather than truncating to 64
    # and repeating that same large list for every table occurrence.
    ledger = {"entries": {
        "measurement:" + digest(str(index).encode()): {
            "kind": "measured", "value": .6, "observation_id": "trial-" + str(index),
            "metric": "accuracy", "per_seed": [{"seed": 0, "value": .5}, {"seed": 1, "value": .7}],
        } for index in range(96)}}
    row = r"Repeated mean & 0.6 & 0: 0.5; 1: 0.7 \\" + "\n"
    latex = (r"\documentclass{article}\title{Synthetic bound document}\begin{document}"
        + r"\section{Results}\begin{longtable}{lrr}" + row * 40
        + r"\end{longtable}\end{document}")
    context = binding_prompt_context(latex, ledger)
    groups = context["numeric_candidate_groups"]
    assert len(context["required_numeric_spans"]) == 200
    assert len(groups) == 5
    assert all(len(group["matching_candidates"]) == 96 for group in groups.values())
    assert all(group["candidates_truncated"] is False for group in groups.values())
    seen = {}
    for span in context["required_numeric_spans"]:
        group = groups[span["candidate_group"]]
        assert group["numeric_text"] == span["quote"]
        candidates = group["matching_candidates"]
        assert set(candidates[0]) == {"evidence_id", "location", "seed_index"}
        expected_location = {"0.6": "value", "0": "seed", "1": "seed",
                             "0.5": "per_seed_value", "0.7": "per_seed_value"}[span["quote"]]
        assert {item["location"] for item in candidates} == {expected_location}
        assert {item["evidence_id"] for item in candidates} == ledger["entries"].keys()
        if span["quote"] in seen:
            assert seen[span["quote"]] == span["candidate_group"]
        seen[span["quote"]] = span["candidate_group"]
    # Whole request-like payload remains within the existing frozen ceiling.
    assert len(canonical({"latex": latex, "ledger": ledger, "source_binding_context": context})) < 200000
