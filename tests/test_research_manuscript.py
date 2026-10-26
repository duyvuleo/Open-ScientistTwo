"""Synthetic contracts; numerical values here are explicit unit-test fixtures."""

from pathlib import Path

import pytest

from scientisttwo.live.models import Candidate, Observation, SourceFile
from scientisttwo.research.manuscript import (
    CompilationPolicy, CompilationResult, LedgerClaim, ManuscriptRecord,
    build_evidence_ledger, derive_comparison, fixture_manuscript, legacy_draft_view,
    render_manuscript, validate_manuscript,
)
from scientisttwo.store import IntegrityError, Store, canonical, digest


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path / "campaign")
    yield result
    result.close()


def observation(store, name="baseline", values=(.5, .6, .7), *, phase="full", seeds=(0, 1, 2)):
    code = Candidate(files=[SourceFile(path="train.py", content="print('synthetic fixture')\n")],
                     command=["python", "train.py"], description="Synthetic test candidate")
    code_ref = store.put_json(code)
    protocol = {"phase": phase, "seeds": list(seeds), "adapter": "synthetic_unit_test"}
    protocol_ref = store.put_json(protocol)
    result = Observation(
        id=name, phase=phase, status="success", code_ref=code_ref,
        protocol_ref=protocol_ref, protocol_hash=digest(canonical(protocol)), metric="accuracy",
        value=sum(values) / len(values), values=list(values),
        prediction_refs=[store.put_json({"synthetic_prediction_seed": seed}) for seed in seeds],
        log_refs=[store.put_json({"synthetic_receipt_seed": seed}) for seed in seeds],
    )
    result_ref = store.put_json(result)
    with store.db:
        store.db.execute("INSERT INTO calls VALUES (?,?,?)", ("development/" + name, "fixture", result_ref))
    return result


def sources():
    text = "This is an explicitly synthetic reference passage."
    return [{"id": "paper", "title": "Synthetic research reference", "authors": ["Fixture Author"],
             "year": 2026, "metadata_status": "supplied", "content_status": "supplied_excerpt",
             "passages": [{"id": "paper:paragraph", "text": text, "sha256": digest(text.encode()),
                           "scope": "supplied_excerpt"}]}]


def test_complete_manuscript_has_required_sections_and_no_hidden_final(store):
    results = [observation(store), observation(store, "heldout", phase="final")]
    ledger = build_evidence_ledger(store, results, sources())
    assert set(ledger["observations"]) == {"baseline"}
    manuscript = fixture_manuscript(ledger)
    assert validate_manuscript(manuscript, ledger).status == "pass"
    assert validate_manuscript(manuscript, ledger).checks["citation_entailment"] == "unverified"
    fields = manuscript.model_dump()
    fields.pop("method")
    with pytest.raises(ValueError):
        ManuscriptRecord.model_validate(fields)
    with pytest.raises(ValueError, match="frozen method"):
        build_evidence_ledger(store, results, include_final=True)
    final_ledger = build_evidence_ledger(store, results, include_final=True, method_frozen=True)
    assert "heldout" in final_ledger["observations"]


@pytest.mark.parametrize("change", ["score", "protocol", "prediction", "uncommitted"])
def test_ledger_rejects_changed_or_uncommitted_evidence(store, change):
    result = observation(store)
    if change == "score":
        result.value = .9
    elif change == "protocol":
        (store.root / result.protocol_ref).write_text("{}")
    elif change == "prediction":
        (store.root / result.prediction_refs[0]).write_text("{}")
    else:
        result.id = "never-committed"
    with pytest.raises(IntegrityError):
        build_evidence_ledger(store, [result])


def test_ledger_keeps_multi_metric_and_per_seed_evidence(store):
    result = observation(store)
    result.metrics = {"accuracy": result.value, "macro_f1": .58}
    result.metric_values = {"accuracy": result.values, "macro_f1": [.48, .58, .68]}
    with store.db:
        store.db.execute("UPDATE calls SET output_ref=? WHERE key=?",
                         (store.put_json(result), "development/baseline"))
    ledger = build_evidence_ledger(store, [result])
    assert {entry["metric"] for entry in ledger["entries"].values()} == {"accuracy", "macro_f1"}
    assert len(ledger["tables"]) == 2
    assert all([item["seed"] for item in entry["per_seed"]] == [0, 1, 2]
               for entry in ledger["entries"].values())


def test_deterministic_comparisons_require_matched_protocols(store):
    ledger = build_evidence_ledger(store, [observation(store), observation(store, "candidate", (.7, .8, .9))])
    keys = {entry["observation_id"]: key for key, entry in ledger["entries"].items()}
    difference = derive_comparison(ledger, keys["candidate"], keys["baseline"])
    assert ledger["entries"][difference]["value"] == pytest.approx(.2)
    relative = derive_comparison(ledger, keys["candidate"], keys["baseline"], operation="relative_change")
    assert ledger["entries"][relative]["value"] == pytest.approx(1 / 3)
    ledger["entries"][keys["candidate"]]["protocol_hash"] = "incomparable"
    with pytest.raises(ValueError, match="matching"):
        derive_comparison(ledger, keys["candidate"], keys["baseline"])


def test_zero_relative_comparison_is_rejected(store):
    ledger = build_evidence_ledger(store, [observation(store, values=(0., 0., 0.))])
    key = next(iter(ledger["entries"]))
    with pytest.raises(ValueError, match="zero reference"):
        derive_comparison(ledger, key, key, operation="relative_change")


@pytest.mark.parametrize("change", ["number", "text_number", "section_number", "unknown_claim", "unknown_table", "wrong_kind"])
def test_unsupported_claims_cannot_render(store, change):
    ledger = build_evidence_ledger(store, [observation(store)])
    manuscript = fixture_manuscript(ledger)
    if change == "number":
        manuscript.claims[0].value = .999
    elif change == "text_number":
        manuscript.claims[0].text = "The score increased to 99%."
    elif change == "section_number":
        manuscript.results.text = "We achieved 99.9% accuracy."
    elif change == "unknown_claim":
        manuscript.results.claim_ids = ["unknown"]
    elif change == "unknown_table":
        manuscript.table_ids = ["unknown"]
    else:
        manuscript.claims[0].kind = "derived"
    assert validate_manuscript(manuscript, ledger).status == "fail"
    with pytest.raises(ValueError, match="Invalid manuscript"):
        render_manuscript(store, manuscript, ledger)


def test_source_existence_does_not_establish_support(store):
    ledger = build_evidence_ledger(store, [observation(store)], sources())
    manuscript = fixture_manuscript(ledger)
    manuscript.claims.append(LedgerClaim(
        id="literature", kind="literature", text="This passage motivates a hypothesis.",
        source_ids=["paper"], passage_ids=["paper:paragraph"],
    ))
    manuscript.related_work.claim_ids = ["literature"]
    valid = validate_manuscript(manuscript, ledger)
    assert valid.status == "pass"
    assert valid.checks["citation_id_resolution"] == "pass"
    assert valid.checks["citation_existence"] == "unverified"
    assert valid.checks["citation_entailment"] == "unverified"
    manuscript.claims[-1].passage_ids = ["unread-fulltext"]
    invalid = validate_manuscript(manuscript, ledger)
    assert invalid.status == "fail"
    assert invalid.checks["citation_existence"] == "unverified"
    assert invalid.checks["citation_id_resolution"] == "pass"
    assert invalid.checks["cited_content_identity"] == "fail"


def test_modified_passage_is_detected(store):
    changed = sources()
    changed[0]["passages"][0]["text"] = "Tampered text"
    with pytest.raises(IntegrityError, match="passage hash"):
        build_evidence_ledger(store, [], changed)


def test_method_and_interpretation_claims_remain_unverified(store):
    fact_ref = store.put_json({"effective_epochs": 5})
    ledger = build_evidence_ledger(store, [observation(store)], method_facts={
        "training": {"text": "Effective training settings", "evidence_ref": fact_ref}
    })
    manuscript = fixture_manuscript(ledger)
    manuscript.claims.append(LedgerClaim(id="method", kind="method", text="Recorded settings apply.",
                                        evidence_ids=["method:training"]))
    manuscript.method.claim_ids = ["method"]
    assert validate_manuscript(manuscript, ledger).status == "pass"
    manuscript.claims[-1].evidence_ids = []
    assert validate_manuscript(manuscript, ledger).status == "fail"


def test_diagnostic_requires_committed_record(store):
    record = {"request": {"kind": "classification"}, "status": "success", "summary": {"samples": 10},
              "tables": {"confusion": [{"true": 0, "predicted": 0, "count": 10}]}}
    ref = store.put_json(record)
    ledger = build_evidence_ledger(store, [], diagnostics=[{**record, "artifact_ref": ref}])
    assert next(iter(ledger["entries"].values()))["summary"] == {"samples": 10}
    with pytest.raises(IntegrityError, match="immutable artifact"):
        build_evidence_ledger(store, [], diagnostics=[{**record, "summary": {"samples": 999}, "artifact_ref": ref}])


def test_render_is_deterministic_complete_and_does_not_compile(store):
    ledger = build_evidence_ledger(store, [observation(store)], sources())
    manuscript = fixture_manuscript(ledger)
    manuscript.title = r"An escaped \input{/etc/passwd} <script>alert(1)</script> title"
    rendered = render_manuscript(store, manuscript, ledger)
    again = render_manuscript(store, manuscript, ledger)
    assert again.manifest_ref == rendered.manifest_ref
    assert rendered.pdf_ref is None
    assert rendered.compilation.status == "unavailable"
    latex = store.read(rendered.latex_ref).decode()
    markdown = store.read(rendered.markdown_ref).decode()
    assert r"\input{/etc/passwd}" not in latex
    assert "<script>" not in markdown
    assert "0.6" in latex and "0: 0.5; 1: 0.6; 2: 0.7" in markdown
    assert "scientific interpretation" in markdown
    manifest = store.json(rendered.manifest_ref)
    names = set(manifest["files"])
    assert {"main.tex", "references.bib", "manuscript.md", "REPRODUCE.md", "evidence/ledger.json"} <= names
    assert any(name.startswith("code/") and name.endswith("train.py") for name in names)
    assert len(rendered.figure_refs) == 1
    assert "circle" in store.read(next(iter(rendered.figure_refs.values()))).decode()
    assert not store.verify()


def test_render_detects_partial_or_modified_immutable_project(store):
    ledger = build_evidence_ledger(store, [observation(store)])
    manuscript = fixture_manuscript(ledger)
    rendered = render_manuscript(store, manuscript, ledger)
    Path(rendered.project_path, "main.tex").write_text("tampered")
    with pytest.raises(IntegrityError, match="Immutable manuscript"):
        render_manuscript(store, manuscript, ledger)


def test_explicit_template_dependencies_are_pinned(store, tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "iclr2025_conference.sty").write_text("% Fixture style and license notice\n")
    (template / "iclr2025_conference.bst").write_text("% Fixture bibliography style\n")
    (template / "LICENSE").write_text("Synthetic fixture license\n")
    ledger = build_evidence_ledger(store, [observation(store)])
    rendered = render_manuscript(store, fixture_manuscript(ledger), ledger, template_path=template)
    manifest = store.json(rendered.manifest_ref)
    assert manifest["template"]["iclr_template_used"]
    assert "iclr2025_conference.sty" in manifest["template"]["files"]
    assert "iclr2025_conference.bst" in manifest["template"]["files"]
    latex = store.read(rendered.latex_ref).decode()
    assert r"\bibliographystyle{iclr2025_conference}" in latex
    assert r"\renewcommand{\@maketitle}" in latex
    assert r"\lhead{Research workflow draft -- not submitted}" in latex
    assert r"\resizebox{\linewidth}{!}{\input{figures/" in latex
    assert r"\usepackage{longtable,url,graphicx}" in latex
    assert "template-notices/LICENSE" in manifest["files"]
    assert Path(rendered.project_path, "iclr2025_conference.sty").read_bytes() == (
        template / "iclr2025_conference.sty").read_bytes()


def test_iclr_presentation_changes_preserve_canonical_claims_and_figure_values(store):
    from scientisttwo.research import manuscript as module

    ledger = build_evidence_ledger(store, [observation(store)], sources())
    manuscript = fixture_manuscript(ledger)
    before_ledger, before_manuscript = canonical(ledger), canonical(manuscript)
    plain = render_manuscript(store, manuscript, ledger)
    iclr = render_manuscript(store, manuscript, ledger,
                             template_path=Path(module.__file__).parent / "templates/iclr2025")
    assert canonical(ledger) == before_ledger and canonical(manuscript) == before_manuscript
    assert plain.figure_refs == iclr.figure_refs
    assert plain.manuscript_ref == iclr.manuscript_ref and plain.ledger_ref == iclr.ledger_ref
    assert r"\renewcommand{\@maketitle}" not in store.read(plain.latex_ref).decode()
    assert "not submitted" in store.read(iclr.latex_ref).decode()


def test_iclr_template_requires_matching_bibliography_style(store, tmp_path):
    template = tmp_path / "incomplete-template"
    template.mkdir()
    (template / "iclr2025_conference.sty").write_text("% Incomplete synthetic style\n")
    ledger = build_evidence_ledger(store, [])
    with pytest.raises(ValueError, match="iclr2025_conference.bst"):
        render_manuscript(store, fixture_manuscript(ledger), ledger, template_path=template)


@pytest.mark.container
def test_cited_iclr_manuscript_compiles_in_pinned_offline_container(store):
    """Real TeX passes over synthetic evidence; no model or candidate execution."""
    import os
    from types import SimpleNamespace
    from scientisttwo.benchmarks.models import NativeResources
    from scientisttwo.live.runtime import Budget
    from scientisttwo.process import run_process
    from scientisttwo.research.compiler import IsolatedCompiler
    from scientisttwo.research import manuscript as module

    image = os.environ.get("SCIENTISTTWO_COMPILER_IMAGE")
    if not image:
        pytest.skip("Set SCIENTISTTWO_COMPILER_IMAGE to an existing pinned compiler image")
    config = SimpleNamespace(campaign_seconds=180, max_calls=0, max_experiments=0,
                             token_stop_threshold=None)
    budget = Budget(store, config)
    spec = SimpleNamespace(runtime="docker", context=os.environ.get("SCIENTISTTWO_TEST_CONTEXT"),
                           resources=NativeResources(cpus=1, memory_mb=1024))
    runner = SimpleNamespace(store=store, budget=budget, spec=spec, executor=run_process)
    compiler = IsolatedCompiler(runner, image)
    ledger = build_evidence_ledger(store, [observation(store)], sources())
    manuscript = fixture_manuscript(ledger)
    rendered = render_manuscript(
        store, manuscript, ledger,
        template_path=Path(module.__file__).parent / "templates/iclr2025",
        compiler=compiler, compilation_policy=CompilationPolicy(image_digest=image),
    )
    assert rendered.compilation.status == "success", rendered.compilation.diagnostics
    assert rendered.pdf_ref and store.read(rendered.pdf_ref).startswith(b"%PDF-")
    assert len(budget.summary()) == 1 and budget.summary()[0]["kind"] == "probe"
    repeated = render_manuscript(
        store, manuscript, ledger,
        template_path=Path(module.__file__).parent / "templates/iclr2025",
        compiler=compiler, compilation_policy=CompilationPolicy(image_digest=image),
    )
    assert repeated == rendered and len(budget.summary()) == 1


def test_compiler_requires_bound_policy_and_explicit_receipt(store):
    ledger = build_evidence_ledger(store, [observation(store)])
    manuscript = fixture_manuscript(ledger)
    calls = []
    policy = CompilationPolicy(image_digest="sha256:" + "a" * 64)

    def fake_compiler(request):
        calls.append(request)
        assert request.policy.network is False and request.policy.shell_escape is False
        assert "-no-shell-escape" in request.command
        pdf = Path(request.project_dir, "main.pdf")
        pdf.write_bytes(b"%PDF-synthetic-unit-test")
        return CompilationResult(status="success", diagnostics="Synthetic compiler receipt",
                                 pdf_path=str(pdf), receipt_ref=store.put_json({"synthetic": True}),
                                 image_digest=policy.image_digest, network_disabled=True,
                                 shell_escape_disabled=True)

    with pytest.raises(ValueError, match="pinned"):
        render_manuscript(store, manuscript, ledger, compiler=fake_compiler)
    rendered = render_manuscript(store, manuscript, ledger, compiler=fake_compiler, compilation_policy=policy)
    assert rendered.pdf_ref is not None
    assert len(calls) == 1


def test_compiler_failure_is_bounded_and_not_misreported(store):
    ledger = build_evidence_ledger(store, [observation(store)])
    calls = []
    def failure(request):
        calls.append(request.attempt)
        return CompilationResult(status="failed", diagnostics="Synthetic failure", timed_out=True)
    rendered = render_manuscript(store, fixture_manuscript(ledger), ledger, compiler=failure,
                                compilation_policy=CompilationPolicy(image_digest="sha256:" + "a" * 64, max_attempts=3))
    assert calls == [1, 2, 3]
    assert rendered.compilation.status == "failed" and rendered.pdf_ref is None


def test_compiler_cannot_claim_success_without_isolation(store):
    ledger = build_evidence_ledger(store, [observation(store)])
    with pytest.raises(IntegrityError, match="isolation receipt"):
        render_manuscript(store, fixture_manuscript(ledger), ledger,
                          compiler=lambda request: CompilationResult(status="success", diagnostics="bad"),
                          compilation_policy=CompilationPolicy(image_digest="sha256:" + "a" * 64))


def test_legacy_view_keeps_canonical_numerical_values(store):
    ledger = build_evidence_ledger(store, [observation(store)])
    draft = legacy_draft_view(fixture_manuscript(ledger), ledger)
    assert draft.claims[0].experiment_id == "baseline"
    assert draft.claims[0].value == pytest.approx(.6)


@pytest.mark.parametrize("changed_part", ["measurement", "derivation", "table", "figure"])
def test_render_recomputes_ledger_and_rejects_fabricated_records(store, changed_part):
    ledger = build_evidence_ledger(store, [observation(store), observation(store, "candidate", (.7, .8, .9))])
    keys = {item["observation_id"]: key for key, item in ledger["entries"].items()}
    comparison = derive_comparison(ledger, keys["candidate"], keys["baseline"])
    manuscript = fixture_manuscript(ledger)
    if changed_part == "measurement":
        ledger["entries"][keys["candidate"]]["value"] = .999
    elif changed_part == "derivation":
        ledger["entries"][comparison]["value"] = 100.
    elif changed_part == "table":
        ledger["tables"][next(iter(ledger["tables"]))]["evidence_ids"] = [keys["candidate"]]
    else:
        ledger["figures"][next(iter(ledger["figures"]))]["caption"] = "Tampered claim"
    with pytest.raises(IntegrityError, match="ledger differs"):
        render_manuscript(store, manuscript, ledger)


def test_diagnostic_tables_and_summary_are_rendered_from_exact_artifact(store):
    record = {"request": {"kind": "classification"}, "status": "success", "summary": {"accuracy": .8},
              "tables": {"confusion": [{"true": 0, "predicted": 0, "count": 8}]}}
    ref = store.put_json(record)
    ledger = build_evidence_ledger(store, [], diagnostics=[{**record, "artifact_ref": ref}])
    manuscript = fixture_manuscript(ledger)
    rendered = render_manuscript(store, manuscript, ledger)
    markdown = store.read(rendered.markdown_ref).decode()
    assert "confusion" in markdown
    assert "Recorded value: 0.8" in markdown
    assert ref in markdown
    assert ref in store.json(rendered.manifest_ref)["files"]


def test_compiler_input_change_stops_before_retry(store):
    ledger = build_evidence_ledger(store, [observation(store)])
    calls = []
    def modifies_input(request):
        calls.append(request.attempt)
        Path(request.project_dir, "main.tex").write_text("Modified compiler input")
        return CompilationResult(status="failed", diagnostics="Synthetic mutation")
    with pytest.raises(IntegrityError, match="modified an immutable"):
        render_manuscript(store, fixture_manuscript(ledger), ledger, compiler=modifies_input,
                          compilation_policy=CompilationPolicy(image_digest="sha256:" + "a" * 64, max_attempts=3))
    assert calls == [1]


def test_canonical_manuscript_preserves_concern_execution_responses(store):
    ledger = build_evidence_ledger(store, [observation(store)])
    response = {"concern_id": "concern-one", "status": "limited", "experiment_ids": ["baseline"],
                "explanation": "The requested wider evidence remains unfulfilled."}
    manuscript = fixture_manuscript(ledger, concern_responses=[response])
    draft = legacy_draft_view(manuscript, ledger)
    assert draft.concern_responses[0].model_dump() == response
    rendered = render_manuscript(store, manuscript, ledger)
    assert "distinct from scientific resolution" in store.read(rendered.markdown_ref).decode()
    assert "wider evidence remains unfulfilled" in store.read(rendered.markdown_ref).decode()


def test_legacy_supplied_citation_is_not_independently_verified_metadata(store):
    from scientisttwo.live.models import Source
    source = Source(id="local", title="Unresolved local paper", text="Supplied passage")
    ledger = build_evidence_ledger(store, [observation(store)], [source])
    result = validate_manuscript(fixture_manuscript(ledger), ledger)
    assert result.status == "pass"
    assert result.checks["citation_id_resolution"] == "pass"
    assert result.checks["citation_existence"] == "unverified"
    assert result.checks["citation_metadata_existence"] == "unverified"
    assert result.checks["cited_content_identity"] == "pass"
    assert result.checks["citation_entailment"] == "unverified"


def test_metadata_pass_requires_committed_retrieval_receipt(store):
    raw = b"Synthetic retrieval body containing scholarly metadata"
    body_ref = store.put(raw, suffix="bin")
    receipt = {"key": "research/literature/request/synthetic", "status": "retrieved", "http_status": 200,
               "body_ref": body_ref, "sha256": digest(raw)}
    source = {**sources()[0], "metadata_status": "verified", "content_ref": body_ref,
              "content_sha256": digest(raw), "provenance": [receipt]}
    with pytest.raises(IntegrityError, match="committed request"):
        build_evidence_ledger(store, [], [source])
    store.cache(receipt["key"], {"synthetic": True}, receipt)
    ledger = build_evidence_ledger(store, [], [source])
    result = validate_manuscript(fixture_manuscript(ledger), ledger)
    assert result.checks["citation_existence"] == "pass"
    assert result.checks["citation_entailment"] == "unverified"


def test_final_appendix_is_deterministic_and_never_changes_research_ledger(store):
    from scientisttwo.research.manuscript import final_results_appendix
    ledger = build_evidence_ledger(store, [observation(store)])
    original = render_manuscript(store, fixture_manuscript(ledger), ledger)
    original_bytes = store.read(original.markdown_ref)
    results = {"baseline": {"status": "completed", "score": .49, "metric": "accuracy"}}
    store.set("live_frozen", {"best": {"id": "baseline"}, "improved": False})
    store.set("live_final", results)
    store.db.execute("CREATE TABLE live_operations(key TEXT,kind TEXT,result_ref TEXT)")
    store.db.execute("INSERT INTO live_operations VALUES (?,?,?)",
                     ("final/baseline", "experiment", store.put_json(results["baseline"])))
    store.db.commit()
    final = final_results_appendix(store, original, results)
    assert final.ledger_ref == original.ledger_ref
    assert final.manuscript_ref == original.manuscript_ref
    assert store.read(original.markdown_ref) == original_bytes
    assert b"0.49" in store.read(final.markdown_ref)
    assert b"0.49" not in store.read(original.markdown_ref)
    assert store.json(final.manifest_ref)["model_calls"] == 0
    assert final_results_appendix(store, original, results).manifest_ref == final.manifest_ref
    with pytest.raises(ValueError, match="only once"):
        final_results_appendix(store, final, results)


def test_final_appendix_requires_freeze_and_committed_results(store):
    from scientisttwo.research.manuscript import final_results_appendix
    ledger = build_evidence_ledger(store, [observation(store)])
    original = render_manuscript(store, fixture_manuscript(ledger), ledger)
    with pytest.raises(IntegrityError, match="frozen method"):
        final_results_appendix(store, original, {})
    store.set("live_frozen", {"best": {"id": "baseline"}})
    store.set("live_final", {"baseline": {"score": .5}})
    with pytest.raises(IntegrityError, match="committed final results"):
        final_results_appendix(store, original, {"baseline": {"score": .9}})
