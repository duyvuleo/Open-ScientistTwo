"""Real pinned upstream control flow; explicitly synthetic language/PDF services."""

import json
import hashlib
import shutil
import sys

import pytest

from upstream_checkouts import require_upstream

from scientisttwo.integrations.paperorchestra.runtime import (
    ModelResponse,
    RuntimeConfig,
    ScriptedTransport,
    SOURCE_HASHES,
    UPSTREAM_REVISION,
    run_paperorchestra,
    verify_upstream,
)


TEX = r"\documentclass{article}\begin{document}Frozen evidence only.\end{document}"
REVISED = TEX.replace("Frozen evidence only.", "Frozen evidence with limitations.")


def review(score=6):
    result = {axis: 3 for axis in ["Originality", "Quality", "Clarity", "Significance",
                                  "Soundness", "Presentation", "Contribution", "Confidence"]}
    return json.dumps({**result, "Overall": score, "Summary": "Synthetic test review",
                       "Strengths": [], "Weaknesses": [], "Questions": [],
                       "Limitations": ["Synthetic agent response"], "Decision": "Reject"})


def trace(*, decline=False):
    return [json.dumps({"title": "Synthetic workflow test", "sections": []}), TEX,
            *[review()] * 4,
            '```json\n{"plan":"Clarify limitations only"}\n```\n```latex\n' + REVISED + '\n```',
            *[review(5 if decline else 7)] * 4,
            json.dumps({"figure_and_tables": {}, "other_issues": []})]


class SyntheticPdf:
    synthetic = True
    isolation = "synthetic-test-double"

    def __init__(self):
        self.compilations = []

    def compile(self, work_dir, pdf_path, tex_name, timeout_seconds):
        assert timeout_seconds > 0
        self.compilations.append(tex_name)
        pdf_path.write_bytes(b"%PDF-1.7\nexplicit synthetic unit-test bytes\n")

    def text(self, pdf_path, *, timeout_seconds=None):
        return "Synthetic extracted paper text for control-flow tests only."

    def render(self, pdf_path, output_dir, *, timeout_seconds=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        image = output_dir / "page.png"
        image.write_bytes(b"synthetic-unit-test-image")
        return [image]


@pytest.fixture
def upstream():
    root = require_upstream("paperorchestra")
    if not root.is_dir():
        pytest.skip("Optional pinned official PaperOrchestra checkout is not installed")
    return root


@pytest.fixture
def materials(tmp_path):
    root = tmp_path / "materials"
    root.mkdir()
    (root / "idea_sparse.md").write_text("A deterministic integration test, no scientific claim.")
    (root / "experimental_log.md").write_text("No experiments in this synthetic unit test.")
    (root / "citation_map.json").write_text(json.dumps({
        "fixture": {"title": "Explicit test fixture", "authors": ["Test author"], "year": 2026}
    }))
    (root / "references.bib").write_text("@misc{fixture,title={Explicit test fixture}}")
    return root


def run(upstream, materials, tmp_path, *, responses=None, pdf=None, **options):
    transport = ScriptedTransport(trace() if responses is None else responses)
    result = run_paperorchestra(
        upstream, materials, tmp_path / "attempt", transport, pdf or SyntheticPdf(),
        RuntimeConfig(allow_synthetic_pdf=True, **options),
    )
    return result, transport, json.loads(result.report_path.read_text())


def test_actual_upstream_agents_finish_without_sdk_imports(upstream, materials, tmp_path):
    before = {name: sys.modules.get(name) for name in ["google.genai", "utils.gemini_utils"]}
    result, transport, report = run(upstream, materials, tmp_path)
    assert result.status == "success", result.error
    assert result.final_tex.read_text() == REVISED
    assert report["upstream"]["revision"] == UPSTREAM_REVISION
    assert report["synthetic_agents"] and report["synthetic_pdf"]
    assert report["observed_tokens"] == 0
    assert report["model_calls"] == report["completed_calls"] == 12
    assert report["stages"] == ["official_outline_agent", "adapted_frozen_literature",
                                "official_section_writing_agent",
                                "official_content_refinement_agent", "official_formatting_loop"]
    assert [r.role for r in transport.requests] == ["outline", "section_writing",
        *["peer_review"] * 4, "content_refinement", *["peer_review"] * 4, "format_review"]
    assert transport.requests[6].response_kind == "json_latex"
    assert transport.requests[6].media[0].mime_type == "application/pdf"
    assert transport.requests[-1].media[0].mime_type == "image/png"
    assert all(sys.modules.get(name) is value for name, value in before.items())
    assert len(list((tmp_path / "attempt" / "model_calls").glob("*.json"))) == 12


def test_official_refinement_rejects_worse_review_and_preserves_draft(upstream, materials, tmp_path):
    result, _, report = run(upstream, materials, tmp_path, responses=trace(decline=True))
    assert result.status == "success", result.error
    assert result.final_tex.read_text() == TEX
    worklog = json.loads((tmp_path / "attempt" / "content_refinement_workdir" /
                         "content_refinement_worklog.json").read_text())
    assert worklog["v1"]["outcome"] == "REJECTED_SCORE_DECREASE"
    assert report["model_calls"] == 12


def test_call_budget_crosses_upstream_exception_handlers(upstream, materials, tmp_path):
    result, transport, report = run(upstream, materials, tmp_path, max_model_calls=2)
    assert result.status == "failed"
    assert "model-call budget exhausted" in result.error
    assert len(transport.requests) == report["model_calls"] == 2
    assert result.final_pdf is None


def test_token_threshold_stops_after_observed_usage(upstream, materials, tmp_path):
    response = ModelResponse(trace()[0], observed_tokens=11)
    result, transport, report = run(upstream, materials, tmp_path, responses=[response], max_tokens=10)
    assert result.status == "failed"
    assert "observed-token threshold" in result.error
    assert report["observed_tokens"] == 11
    assert len(transport.requests) == 1


def test_bad_model_output_is_retained_without_retry(upstream, materials, tmp_path):
    result, transport, report = run(upstream, materials, tmp_path, responses=["invalid JSON"])
    assert result.status == "failed"
    assert len(transport.requests) == report["model_calls"] == 1
    call = json.loads((tmp_path / "attempt" / "model_calls" / "001.json").read_text())
    assert call["raw_response"] == "invalid JSON"
    assert call["status"] == "failed"
    assert report["completed_calls"] == 0


def test_failure_inside_refinement_is_not_silently_accepted(upstream, materials, tmp_path):
    result, transport, report = run(upstream, materials, tmp_path, responses=trace()[:6] + ["broken"])
    assert result.status == "failed"
    assert report["model_calls"] == len(transport.requests) == 7
    assert "Content refinement requires" in result.error


def test_missing_pdf_is_explicit_failure_before_review(upstream, materials, tmp_path):
    class MissingPdf(SyntheticPdf):
        def compile(self, *args):
            pass
    result, transport, _ = run(upstream, materials, tmp_path, pdf=MissingPdf())
    assert result.status == "failed"
    assert "did not produce a bounded PDF" in result.error
    assert len(transport.requests) == 2


def test_live_default_rejected_before_any_output_or_call(tmp_path):
    class Live(ScriptedTransport):
        synthetic = False
    transport = Live([])
    with pytest.raises(ValueError, match="Live PaperOrchestra calls disabled"):
        run_paperorchestra(tmp_path, tmp_path, tmp_path / "attempt", transport, SyntheticPdf())
    assert transport.requests == []
    assert not (tmp_path / "attempt").exists()


def test_no_synthetic_pdf_in_production(upstream, materials, tmp_path):
    with pytest.raises(ValueError, match="Synthetic PDF tools are forbidden"):
        run_paperorchestra(upstream, materials, tmp_path / "attempt", ScriptedTransport([]), SyntheticPdf())


def test_no_host_compiler(upstream, materials, tmp_path):
    class HostPdf(SyntheticPdf):
        synthetic = False
        isolation = "host"
    with pytest.raises(ValueError, match="isolated container PDF tools"):
        run_paperorchestra(upstream, materials, tmp_path / "attempt", ScriptedTransport([]), HostPdf())


def test_existing_attempt_preserved_without_reexecution(upstream, materials, tmp_path):
    output = tmp_path / "attempt"
    output.mkdir()
    sentinel = output / "partial.txt"
    sentinel.write_text("Preserve unknown attempt")
    with pytest.raises(ValueError, match="output already exists"):
        run(upstream, materials, tmp_path)
    assert sentinel.read_text() == "Preserve unknown attempt"


def test_modified_upstream_is_refused_before_calls(upstream, materials, tmp_path):
    copied = tmp_path / "upstream"
    for name in SOURCE_HASHES:
        target = copied / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream / name, target)
    (copied / "methods" / "agents" / "outline_agent.py").write_text("raise RuntimeError('modified')")
    with pytest.raises(ValueError, match="Pinned PaperOrchestra source mismatch"):
        run(copied, materials, tmp_path)


def test_missing_frozen_materials_preserves_failure_report(upstream, materials, tmp_path):
    (materials / "experimental_log.md").unlink()
    result, transport, report = run(upstream, materials, tmp_path)
    assert result.status == "failed"
    assert "materials missing" in result.error
    assert report["model_calls"] == len(transport.requests) == 0


def test_symlink_material_rejected(upstream, materials, tmp_path):
    (materials / "external.txt").symlink_to(tmp_path / "does-not-exist")
    result, transport, _ = run(upstream, materials, tmp_path)
    assert result.status == "failed"
    assert "Symlinks are forbidden" in result.error
    assert transport.requests == []


def test_upstream_lock_matches_reviewed_checkout(upstream):
    assert verify_upstream(upstream) == SOURCE_HASHES


def test_evidence_fixture_runs_official_agents_with_exact_exported_values(upstream, tmp_path):
    from scientisttwo.integrations.paperorchestra.evidence import export_evidence_bundle
    from scientisttwo.integrations.paperorchestra.fixtures import EvidenceFixtureTransport
    from scientisttwo.live.models import Candidate, Observation, SourceFile
    from scientisttwo.research.manuscript import build_evidence_ledger
    from scientisttwo.store import Store, canonical, digest, file_digest

    store = Store(tmp_path / "synthetic-evidence-source")
    try:
        candidate = Candidate(files=[SourceFile(path="train.py", content="print('not executed')")],
                              command=["python", "train.py"], description="Synthetic unit-test only")
        protocol = {"phase": "full", "seeds": [0], "adapter": "synthetic_test"}
        observation = Observation(id="synthetic-baseline", phase="full", status="success",
            code_ref=store.put_json(candidate), protocol_ref=store.put_json(protocol),
            protocol_hash=digest(canonical(protocol)), metric="accuracy", value=.678901234,
            values=[.678901234], prediction_refs=[store.put_json({"synthetic": True})],
            log_refs=[store.put_json({"synthetic": True})])
        store.cache("development/" + observation.id, {"synthetic": True}, observation)
        ledger = build_evidence_ledger(store, [observation])
        materials = tmp_path / "export"
        export_evidence_bundle(store, materials, ledger=ledger, title="Synthetic unit test",
                               hypothesis="No scientific claim", upstream_revision=UPSTREAM_REVISION)
        transport = EvidenceFixtureTransport(materials)
        result = run_paperorchestra(upstream, materials, tmp_path / "attempt", transport,
            SyntheticPdf(), RuntimeConfig(allow_synthetic_pdf=True),
            expected_manifest_sha256=file_digest(materials / "manifest.json"))
        assert result.status == "success", result.error
        text = result.final_tex.read_text()
        assert "0.678901234" in text
        assert next(iter(ledger["entries"])) in text
        assert "Scripted writing demonstration" in text
        assert "Scientific validity" in text
        identities = text.split(r"\section{Evidence identities}", 1)[1].split(r"\section{", 1)[0]
        assert r"\par\smallskip" in identities
        assert not any(line.endswith("\\") for line in identities.splitlines())
        assert r"\[0.5em]" not in identities
        assert r"p{0.20\linewidth}" in text
        assert len(transport.requests) == 12
        report = json.loads(result.report_path.read_text())
        assert report["observed_tokens"] == 0
        assert report["verified_evidence_manifest_sha256"] == file_digest(materials / "manifest.json")
    finally:
        store.close()


def test_changed_expected_manifest_rejected_before_model(upstream, materials, tmp_path):
    (materials / "manifest.json").write_text("{}")
    transport = ScriptedTransport(trace())
    result = run_paperorchestra(upstream, materials, tmp_path / "attempt", transport,
        SyntheticPdf(), RuntimeConfig(allow_synthetic_pdf=True), expected_manifest_sha256="a" * 64)
    assert result.status == "failed"
    assert "differs from the authorized input identity" in result.error
    assert transport.requests == []


@pytest.mark.parametrize("options", [dict(max_reflections=4), dict(max_model_calls=0),
                                     dict(max_tokens=0), dict(timeout_seconds=0),
                                     dict(review_ensemble=0)])
def test_config_cannot_disable_bounds(options):
    with pytest.raises(ValueError, match="Invalid bounded"):
        RuntimeConfig(**options)


def run_revision(upstream, materials, tmp_path, *, responses=None, config=None):
    parent = tmp_path / "parent.tex"
    parent.write_text(TEX)
    transport = ScriptedTransport(trace()[2:] if responses is None else responses)
    result = run_paperorchestra(
        upstream, materials, tmp_path / "revision", transport, SyntheticPdf(),
        config or RuntimeConfig(allow_synthetic_pdf=True), initial_tex=parent,
        revision_feedback="Qualify the unsupported mechanism claim; preserve all measurements.",
        expected_initial_tex_sha256=hashlib.sha256(TEX.encode()).hexdigest(),
    )
    return result, transport, json.loads(result.report_path.read_text())


def test_editorial_revision_uses_exact_parent_and_explicit_feedback(upstream, materials, tmp_path):
    result, transport, report = run_revision(upstream, materials, tmp_path)
    assert result.status == "success", result.error
    assert result.final_tex.read_text() == REVISED
    assert [request.role for request in transport.requests] == [
        *["peer_review"] * 4, "content_refinement", *["peer_review"] * 4, "format_review"]
    assert report["stages"] == ["adapted_frozen_literature", "authoritative_parent_manuscript",
                                "official_content_refinement_agent", "official_formatting_loop"]
    revision = report["revision"]
    assert revision["status"] == "accepted"
    assert revision["source_changed"] is True
    assert revision["initial_tex_sha256"] == hashlib.sha256(TEX.encode()).hexdigest()
    assert (tmp_path / "revision" / revision["initial_tex"]).read_text() == TEX
    feedback = (tmp_path / "revision" / revision["feedback"]).read_text()
    assert hashlib.sha256(feedback.encode()).hexdigest() == revision["feedback_sha256"]
    assert feedback in transport.requests[4].prompt
    assert TEX in transport.requests[4].prompt
    assert feedback not in (tmp_path / "revision" / "raw_materials" /
                            "experimental_log.md").read_text()
    assert (tmp_path / "parent.tex").read_text() == TEX


def test_declined_editorial_revision_retains_parent_and_rejected_attempt(upstream, materials, tmp_path):
    result, _, report = run_revision(upstream, materials, tmp_path, responses=trace(decline=True)[2:])
    assert result.status == "success", result.error
    assert result.final_tex.read_text() == TEX
    assert report["revision"]["status"] == "rejected"
    assert report["revision"]["source_changed"] is False
    assert report["revision"]["decisions"]["v1"]["outcome"] == "REJECTED_SCORE_DECREASE"
    rejected = tmp_path / "revision" / "content_refinement_workdir" / "refined_paper_v1.tex"
    assert rejected.read_text() == REVISED
    assert (tmp_path / "parent.tex").read_text() == TEX


def test_unchanged_editorial_revision_is_not_reported_as_accepted(upstream, materials, tmp_path):
    responses = trace()[2:]
    responses[4] = '```json\n{"plan":"No supported change"}\n```\n```latex\n' + TEX + '\n```'
    result, _, report = run_revision(upstream, materials, tmp_path, responses=responses)
    assert result.status == "success", result.error
    assert report["revision"]["status"] == "unchanged"
    assert report["revision"]["source_changed"] is False


def test_failed_revision_preserves_original_and_raw_refusal(upstream, materials, tmp_path):
    responses = trace()[2:6] + ["Cannot revise the unsupported claim."]
    result, _, report = run_revision(upstream, materials, tmp_path, responses=responses)
    assert result.status == "failed"
    assert report["revision"]["status"] == "failed"
    assert (tmp_path / "parent.tex").read_text() == TEX
    assert (tmp_path / "revision" / "initial_manuscript.tex").read_text() == TEX
    call = json.loads((tmp_path / "revision" / "model_calls" / "005.json").read_text())
    assert call["raw_response"] == responses[-1]


def test_editorial_feedback_cannot_be_disabled_with_zero_reflections(upstream, materials, tmp_path):
    with pytest.raises(ValueError, match="at least one content reflection"):
        run_revision(upstream, materials, tmp_path,
                     config=RuntimeConfig(allow_synthetic_pdf=True, max_reflections=0))
    assert not (tmp_path / "revision").exists()


@pytest.mark.parametrize("options, message", [
    ({"revision_feedback": "Please revise"}, "authoritative initial_tex"),
    ({"expected_initial_tex_sha256": "a" * 64}, "authoritative initial_tex"),
    ({"revision_feedback": ""}, "nonempty revision_feedback"),
    ({"revision_feedback": "Please revise", "expected_initial_tex_sha256": "a" * 64},
     "authorized TeX identity"),
])
def test_invalid_parent_revision_contract_stops_before_calls(
        upstream, materials, tmp_path, options, message):
    parent = tmp_path / "parent.tex"
    parent.write_text(TEX)
    if message != "authoritative initial_tex":
        options = {**options, "initial_tex": parent}
    transport = ScriptedTransport([])
    with pytest.raises(ValueError, match=message):
        run_paperorchestra(upstream, materials, tmp_path / "revision", transport,
            SyntheticPdf(), RuntimeConfig(allow_synthetic_pdf=True), **options)
    assert not (tmp_path / "revision").exists()
    assert transport.requests == []


def test_final_pdf_receipt_binds_exact_selected_latex(upstream, materials, tmp_path):
    class SourceBoundPdf(SyntheticPdf):
        def compile(self, work_dir, pdf_path, tex_name, timeout_seconds):
            digest = hashlib.sha256((work_dir / (tex_name + ".tex")).read_bytes()).hexdigest()
            pdf_path.write_bytes(b"%PDF-1.7\n" + digest.encode())

    result, _, report = run(upstream, materials, tmp_path, pdf=SourceBoundPdf())
    assert result.status == "success", result.error
    receipt = report["authoritative_compile"]
    assert receipt["source_tex_sha256"] == hashlib.sha256(result.final_tex.read_bytes()).hexdigest()
    assert receipt["pdf_sha256"] == hashlib.sha256(result.final_pdf.read_bytes()).hexdigest()
    operation = report["pdf_operations"][receipt["operation_index"]]
    assert operation["source_tex_sha256"] == receipt["source_tex_sha256"]
    assert operation["sha256"] == receipt["pdf_sha256"]
    assert operation["tex_name"] == "refined_paper_v1"


def test_compiler_cannot_change_authoritative_source(upstream, materials, tmp_path):
    class MutatingPdf(SyntheticPdf):
        def compile(self, work_dir, pdf_path, tex_name, timeout_seconds):
            super().compile(work_dir, pdf_path, tex_name, timeout_seconds)
            (work_dir / (tex_name + ".tex")).write_text(REVISED)

    result, transport, _ = run(upstream, materials, tmp_path, pdf=MutatingPdf())
    assert result.status == "failed"
    assert "changed its authoritative LaTeX source" in result.error
    assert len(transport.requests) == 2
