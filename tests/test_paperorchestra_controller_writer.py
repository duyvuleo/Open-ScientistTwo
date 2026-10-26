"""Controller-accounted upstream writing; experiments/model/PDF data are fixtures."""

from types import SimpleNamespace

import pytest

from upstream_checkouts import require_upstream

from scientisttwo.integrations.paperorchestra import controller_writer as bridge
from scientisttwo.integrations.paperorchestra.manuscript_binding import fixture_bindings
from scientisttwo.live.output_contracts import AgentOutputError
from scientisttwo.live.runtime import AmbiguousCall
from scientisttwo.models import AgentResult
from scientisttwo.research.controller import ResearchController
from scientisttwo.research.manuscript import build_evidence_ledger
from scientisttwo.store import IntegrityError, Store, digest

from test_paperorchestra_runtime import SyntheticPdf
from test_paperorchestra_pdf import FakeContainer, IMAGE
from test_research_manuscript import observation, sources


class ScriptedWriterBackend:
    synthetic = True
    name = "scripted-controller-writer-test"
    model = "fixture-model"

    def __init__(self):
        self.roles = []

    def invoke(self, role, inputs, schema):
        self.roles.append(role)
        if role == "paperorchestra_model":
            payload = bridge.fixture_model_response(inputs)
        elif role == "paperorchestra_bindings":
            payload = fixture_bindings(inputs["latex"], inputs["ledger"]).model_dump()
        else:
            raise AssertionError("Unexpected writer role: " + role)
        return AgentResult(payload=payload, backend=self.name, model=self.model,
                           synthetic=True, usage={"input_tokens": 0, "output_tokens": 0})


@pytest.fixture
def context(tmp_path, monkeypatch):
    from scientisttwo.research.controller import PROMPTS
    monkeypatch.setitem(PROMPTS, "paperorchestra_model", "Run the pinned upstream writing role.")
    monkeypatch.setitem(PROMPTS, "paperorchestra_bindings", "Bind exact source to frozen evidence.")
    upstream = require_upstream("paperorchestra")
    if not upstream.is_dir():
        pytest.skip("Optional pinned PaperOrchestra checkout unavailable")
    store = Store(tmp_path / "campaign")
    config = SimpleNamespace(model="fixture-model", engine_version="research-v2",
        campaign_seconds=1800, max_calls=120, max_experiments=20, token_stop_threshold=100000,
        call_seconds=120, model_retries=0, agent_output_retries=1,
        history_code_chars=5000, log_tail_chars=5000, max_prompt_chars=2_000_000,
        native=SimpleNamespace(task_id="synthetic-writer", runtime="docker", context=None),
        sources=[], workflow=SimpleNamespace(network_literature=False,
            paperorchestra_root=str(upstream), compiler_image="sha256:" + "a" * 64,
            paperorchestra_reflections=1, paperorchestra_ensemble=1, paperorchestra_max_calls=24,
            paperorchestra_swap_policy="deny"))
    backend = ScriptedWriterBackend()
    controller = ResearchController(store, config, SimpleNamespace(context=lambda: {}),
                                    backend, SimpleNamespace())
    values = [observation(store), observation(store, "candidate", (.7, .8, .9))]
    ledger = build_evidence_ledger(store, values, sources())
    inputs = {"sources": sources(), "incumbent": {"idea": {"title": "Synthetic writer example",
              "hypothesis": "Report frozen synthetic evidence without empirical claims."}},
              "supplementary_coverage": []}
    yield controller, ledger, inputs
    store.close()


def test_official_controller_writer_is_fully_accounted_and_completed_resume_is_noop(context):
    controller, ledger, inputs = context
    pdf = bridge.ContainerPdfTools(IMAGE, executor=FakeContainer())
    result = bridge.write_controller_manuscript(controller, "draft", inputs, ledger, pdf_tools=pdf)
    assert result.manuscript.synthetic_agents
    assert controller.store.read(result.rendered.pdf_ref).startswith(b"%PDF-")
    operations = controller.budget.summary()
    assert any(item["kind"] == "model" for item in operations)
    assert any(item["kind"] == "probe" for item in operations)
    assert all(item["kind"] != "experiment" and item["status"] == "success" for item in operations)
    assert len([item for item in operations if item["kind"] == "model"]) == len(controller.backend.roles)
    before = (len(operations), len(controller.backend.roles), len(pdf.receipts))
    again = bridge.write_controller_manuscript(controller, "draft", inputs, ledger, pdf_tools=pdf)
    assert again == result
    assert (len(controller.budget.summary()), len(controller.backend.roles), len(pdf.receipts)) == before
    report = controller.store.json(result.evidence_ref)
    assert report["synthetic_agents"] and not report["synthetic_pdf"]
    assert report["authoritative_compile"]["source_tex_sha256"] == digest(
        controller.store.read(result.rendered.latex_ref))


def test_editorial_revision_uses_exact_previous_source_and_no_experiments(context):
    controller, ledger, inputs = context
    first = bridge.write_controller_manuscript(controller, "draft", inputs, ledger,
                                               pdf_tools=bridge.ContainerPdfTools(IMAGE, executor=FakeContainer()))
    second = bridge.write_controller_manuscript(controller, "editorial", inputs, ledger,
        previous_manuscript=first.manuscript.model_dump(), feedback={"reason": "Clarify limits"},
        pdf_tools=bridge.ContainerPdfTools(IMAGE, executor=FakeContainer()))
    report = controller.store.json(second.evidence_ref)
    assert report["revision"]["status"] == "accepted"
    assert report["revision"]["initial_tex_sha256"] == digest(controller.store.read(first.rendered.latex_ref))
    assert second.rendered.latex_ref != first.rendered.latex_ref
    assert not any(item["kind"] == "experiment" for item in controller.budget.summary())
    assert controller.store.read(first.rendered.latex_ref)


def test_writer_admission_rejects_configuration_drift_on_resume(context):
    controller, ledger, inputs = context
    bridge.write_controller_manuscript(controller, "draft", inputs, ledger, pdf_tools=bridge.ContainerPdfTools(IMAGE, executor=FakeContainer()))
    controller.config.workflow.paperorchestra_ensemble = 2
    before = len(controller.backend.roles)
    with pytest.raises(IntegrityError, match="inputs changed"):
        bridge.write_controller_manuscript(controller, "draft", inputs, ledger, pdf_tools=bridge.ContainerPdfTools(IMAGE, executor=FakeContainer()))
    assert len(controller.backend.roles) == before


def test_shared_pdf_probe_replays_committed_output_and_blocks_unknown_completion(context, tmp_path):
    controller, _, _ = context
    source = tmp_path / "project"
    source.mkdir()
    (source / "main.tex").write_text("Synthetic TeX fixture")
    target = source / "main.pdf"
    delegate = SyntheticPdf()
    pdf = bridge.BudgetedPdfTools(controller, "replay", delegate, {"synthetic": True})
    pdf.compile(source, target, "main", 60)
    target.unlink()
    again = bridge.BudgetedPdfTools(controller, "replay", delegate, {"synthetic": True})
    again.compile(source, target, "main", 60)
    assert len(delegate.compilations) == 1
    target.unlink()
    with controller.store.db:
        controller.store.db.execute("UPDATE live_operations SET result_ref=NULL,status='running'")
    unknown = bridge.BudgetedPdfTools(controller, "replay", delegate, {"synthetic": True})
    with pytest.raises(AmbiguousCall, match="outcome is unknown"):
        unknown.compile(source, target, "main", 60)
    assert len(delegate.compilations) == 1


def test_failed_pdf_probe_consumes_budget_and_cannot_silently_retry(context, tmp_path):
    controller, _, _ = context
    source = tmp_path / "project"
    source.mkdir()
    (source / "main.tex").write_text("Synthetic TeX fixture")

    class FailedPdf(SyntheticPdf):
        def compile(self, *args):
            raise ValueError("Synthetic compiler failure")

    pdf = bridge.BudgetedPdfTools(controller, "failed", FailedPdf(), {"synthetic": True})
    with pytest.raises(ValueError, match="Synthetic compiler failure"):
        pdf.compile(source, source / "main.pdf", "main", 60)
    operations = controller.budget.summary()
    assert len(operations) == 1 and operations[0]["status"] == "failed"
    replay = bridge.BudgetedPdfTools(controller, "failed", SyntheticPdf(), {"synthetic": True})
    with pytest.raises(AgentOutputError, match="Prior PaperOrchestra PDF operation failed"):
        replay.compile(source, source / "main.pdf", "main", 60)


def test_protected_final_evidence_never_reaches_writer(context):
    controller, ledger, inputs = context
    ledger = {**ledger, "includes_final": True}
    with pytest.raises(IntegrityError, match="non-final"):
        bridge.write_controller_manuscript(controller, "final-leak", inputs, ledger, pdf_tools=bridge.ContainerPdfTools(IMAGE, executor=FakeContainer()))
    assert not controller.backend.roles


def test_live_backend_cannot_accept_synthetic_pdf(context):
    controller, ledger, inputs = context
    controller.backend.synthetic = False
    with pytest.raises(ValueError, match="Live writing cannot"):
        bridge.write_controller_manuscript(controller, "live", inputs, ledger, pdf_tools=SyntheticPdf())
    assert not controller.backend.roles


def test_resume_after_runtime_completion_reuses_all_inner_operations(context, monkeypatch):
    controller, ledger, inputs = context
    original = controller.ask
    fake = FakeContainer()
    pdf = bridge.ContainerPdfTools(IMAGE, executor=fake)

    def interrupted(key, role, *args, **kwargs):
        if role == "paperorchestra_bindings":
            raise KeyboardInterrupt("Synthetic interruption before source indexing")
        return original(key, role, *args, **kwargs)

    monkeypatch.setattr(controller, "ask", interrupted)
    with pytest.raises(KeyboardInterrupt, match="Synthetic interruption"):
        bridge.write_controller_manuscript(controller, "interrupted", inputs, ledger, pdf_tools=pdf)
    before = (len(controller.budget.summary()), len(controller.backend.roles), len(fake.calls))
    monkeypatch.setattr(controller, "ask", original)
    bridge.write_controller_manuscript(controller, "interrupted", inputs, ledger, pdf_tools=pdf)
    assert len(controller.budget.summary()) == before[0] + 1
    assert len(controller.backend.roles) == before[1] + 1
    assert len(fake.calls) == before[2]
    assert len(list((controller.store.root / "paperorchestra").glob("*/attempt-*"))) == 1


def test_unknown_model_completion_is_preserved_and_never_reissued(context, monkeypatch):
    controller, ledger, inputs = context
    original = controller.backend.invoke
    calls = []

    def interrupted(role, incoming, schema):
        calls.append(role)
        if len(calls) == 2:
            raise KeyboardInterrupt("Synthetic lost model completion")
        return original(role, incoming, schema)

    monkeypatch.setattr(controller.backend, "invoke", interrupted)
    with pytest.raises(KeyboardInterrupt, match="lost model completion"):
        bridge.write_controller_manuscript(controller, "unknown", inputs, ledger,
            pdf_tools=bridge.ContainerPdfTools(IMAGE, executor=FakeContainer()))
    assert len(calls) == 2
    with pytest.raises(AmbiguousCall, match="possibly billed"):
        bridge.write_controller_manuscript(controller, "unknown", inputs, ledger,
            pdf_tools=bridge.ContainerPdfTools(IMAGE, executor=FakeContainer()))
    assert len(calls) == 2
    assert controller.store.db.execute(
        "SELECT count(*) FROM live_operations WHERE status='running'").fetchone()[0] == 1
    assert controller.store.db.execute(
        "SELECT count(*) FROM metadata WHERE key LIKE 'paperorchestra-interrupted/%'").fetchone()[0] == 1
