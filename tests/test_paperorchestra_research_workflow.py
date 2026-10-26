"""Full controller integration with explicit synthetic agents/container transports.

These tests execute the real research state machine, pinned official writer,
evidence binding and trusted fixture arithmetic. Model responses, predictions
and PDF bytes are synthetic contracts, not scientific or live-writer validation.
No container daemon, compiler, GPU or research model is invoked.
"""

from copy import deepcopy
import json

import pytest

from upstream_checkouts import require_upstream
from typer.testing import CliRunner

from scientisttwo.research.controller import ResearchController
from scientisttwo.research.policy import ResearchPolicy
from scientisttwo.store import digest

from test_paperorchestra_pdf import FakeContainer, IMAGE
from test_research_compiler import FakeContainers
from test_research_workflow import campaign


class WorkflowContainerTransport:
    """Dispatch solely to the existing synthetic scientific/PDF transport doubles."""

    def __init__(self, scientific):
        self.scientific = scientific
        self.writer = FakeContainer()
        self.appendix = FakeContainers()

    def __call__(self, argv, timeout, **kwargs):
        if "inspect" in argv and IMAGE in argv:
            return self.writer(argv, timeout, **kwargs)
        if any("target=/worker.py" in arg for arg in argv):
            return self.writer(argv, timeout, **kwargs)
        if any(arg.endswith(":/work:rw") for arg in argv):
            return self.appendix(argv, timeout, **kwargs)
        return self.scientific(argv, timeout, **kwargs)


def paper_campaign(tmp_path, **options):
    upstream = require_upstream("paperorchestra")
    if not upstream.is_dir():
        pytest.skip("Optional pinned official PaperOrchestra checkout is not installed")
    controller, store, runner, agents, scientific = campaign(tmp_path, **options)
    controller.config.workflow = ResearchPolicy.model_validate({
        **controller.config.workflow.model_dump(), "writer_backend": "paperorchestra",
        "paperorchestra_root": str(upstream), "compiler_image": IMAGE,
        "paperorchestra_max_calls": 12,
    })
    execution = WorkflowContainerTransport(scientific)
    runner.raw_executor = execution
    return controller, store, runner, agents, execution


def metadata(store, prefix):
    return [(row["key"], json.loads(row["value"])) for row in store.db.execute(
        "SELECT key,value FROM metadata ORDER BY key") if row["key"].startswith(prefix)]


def cli_creation(tmp_path, monkeypatch):
    from scientisttwo.live import cli as live_cli

    controller, store, _, _, _ = campaign(tmp_path)
    native, development = tmp_path / "native.json", tmp_path / "development.json"
    native.write_text(controller.config.native.model_dump_json())
    development.write_text(controller.config.development.model_dump_json())
    store.close()
    preflights = []

    def preflight(spec):
        preflights.append(spec)
        return {"ready": True, "errors": [], "synthetic_test": True}

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("Manifest creation cannot construct a live backend")

    monkeypatch.setattr(live_cli, "preflight", preflight)
    monkeypatch.setattr(live_cli, "load_development", lambda *args: controller.adapter)
    monkeypatch.setattr(live_cli, "CodexBackend", forbidden_backend)
    output = tmp_path / "live.json"
    args = ["create", str(native), str(output), "--development", str(development),
            "--model", "configured-model-not-invoked", "--max-calls", "200", "--seconds", "600"]
    return live_cli.app, args, output, preflights


@pytest.mark.parametrize("mode, alternative, expected", [
    ("research", None, "paperorchestra"),
    ("research", "canonical", "canonical"),
    ("smoke", None, "canonical"),
])
def test_new_manifest_writer_defaults_preserve_explicit_alternative_and_smoke(
        tmp_path, monkeypatch, mode, alternative, expected):
    from scientisttwo.live.models import LiveSpec

    app, args, output, preflights = cli_creation(tmp_path, monkeypatch)
    args += ["--mode", mode]
    if alternative:
        args += ["--writer-backend", alternative]
    if expected == "paperorchestra":
        args += ["--compiler-image", IMAGE]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.exception or result.output
    frozen = LiveSpec.model_validate_json(output.read_text())
    assert frozen.workflow.writer_backend == expected
    assert frozen.workflow.require_pdf == (expected == "paperorchestra")
    if expected == "paperorchestra":
        from scientisttwo.integrations.paperorchestra.runtime import bundled_upstream_root
        assert frozen.workflow.paperorchestra_root == str(bundled_upstream_root())
    assert len(preflights) == 1
    assert json.loads(result.stdout)["live_model_calls"] == 0


@pytest.mark.parametrize("missing", ["image", "invalid_checkout"])
def test_default_research_writer_missing_prerequisite_fails_before_preflight_or_model(
        tmp_path, monkeypatch, missing):
    app, args, output, preflights = cli_creation(tmp_path, monkeypatch)
    args += ["--mode", "research"]
    args += ["--paperorchestra-root", str(tmp_path / "absent-checkout")]
    if missing != "image":
        args += ["--compiler-image", IMAGE]
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert not output.exists()
    assert preflights == []
    if missing == "invalid_checkout":
        assert "Pinned PaperOrchestra source mismatch" in str(result.exception)
    else:
        assert "PaperOrchestra requires a pinned PDF image" in str(result.exception)


def test_old_policy_default_stays_canonical_for_frozen_manifest_compatibility():
    assert ResearchPolicy().writer_backend == "canonical"
    assert ResearchPolicy.model_validate({"version": 1}).writer_backend == "canonical"


def assert_authoritative_reviews(store, agents):
    reviewers = [(role, inputs) for role, inputs in agents.seen
                 if role in {"writing_critic", "scientific_review", "meta"}]
    assert {role for role, _ in reviewers} == {"writing_critic", "scientific_review", "meta"}
    for _, inputs in reviewers:
        source = inputs["authoritative_manuscript"]
        assert source["writer"] == "paperorchestra"
        assert source["latex"] == store.read(source["latex_ref"]).decode()
        manifest = store.json(source["manifest_ref"])
        assert manifest["files"]["main.tex"]["sha256"] == digest(source["latex"].encode())
        assert store.read(source["pdf_ref"]).startswith(b"%PDF-")


def test_paperorchestra_controller_finishes_rebuttal_editorial_and_final(tmp_path):
    controller, store, _, agents, execution = paper_campaign(tmp_path, rebuttal=True, editorial=True)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        assert state["outcome"] == "development_improvement"
        roles = [role for role, _ in agents.seen]
        assert {"paperorchestra_model", "paperorchestra_bindings", "writing_critic",
                "scientific_review", "rebuttal_planner", "rebuttal_coder", "meta"} <= set(roles)
        assert not {"initial_drafter", "draft_enhancer", "draft", "review"} & set(roles)
        assert_authoritative_reviews(store, agents)
        revisions = metadata(store, "manuscript-revision/")
        assert revisions
        assert all(store.json(ref)["no_new_experiments"] for _, ref in revisions)
        assert store.get("live_final")["selected"]["status"] == "completed"
        assert store.get("research_final_bundle")
        final = store.json(store.get("research_final_bundle"))
        assert final["compilation"]["status"] == "success"
        assert r"\input{final-results.tex}" in store.read(final["latex_ref"]).decode()
        development = store.json(store.get("research_manuscript_bundle"))
        assert r"\input{final-results.tex}" not in store.read(development["latex_ref"]).decode()
        assert "FINAL_STAGED_LABELS_MUST_NOT_ENTER_REFINEMENT" not in str(agents.seen)
        from scientisttwo.research.audit import _manuscripts
        manuscript_checks = _manuscripts(store, None, controller.config)
        assert manuscript_checks and all(check["pdf"] == "pass" for check in manuscript_checks)
        assert all(check["scientific_validity"] == "unverified" for check in manuscript_checks)
        assert any(check["authoritative_writer"].get("paperorchestra_development_source_preserved") == "pass"
                   for check in manuscript_checks)
        assert not store.verify()
        snapshot = (controller.budget.summary(), len(agents.seen), len(execution.writer.calls),
                    len(execution.appendix.calls), len(execution.scientific.stages))
        assert controller.run() == state
        assert snapshot == (controller.budget.summary(), len(agents.seen), len(execution.writer.calls),
                            len(execution.appendix.calls), len(execution.scientific.stages))
    finally:
        store.close()


def test_failed_method_revision_preserves_paperorchestra_incumbent(tmp_path):
    controller, store, _, agents, _ = paper_campaign(tmp_path, writing_refinements=0)
    controller.config.meta_refinements = 1
    previous_response = agents.responder
    submitted = []

    def request_worse_method(role, inputs):
        value = previous_response(role, inputs)
        if role == "meta":
            submitted.append(deepcopy(inputs))
            return {"decision": "ReviseMethod", "reason": "Synthetic unsuccessful revision branch"}
        if role == "implement" and "parent" in inputs:
            value["edits"][0]["after"] = value["edits"][0]["after"].replace(
                "SYNTHETIC_TEST_OFFSET = 0.25", "SYNTHETIC_TEST_OFFSET = 1.5")
        return value

    agents.responder = request_worse_method
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        assert len(submitted) == 1
        original = submitted[0]["authoritative_manuscript"]
        final_development = store.json(store.get("research_manuscript_bundle"))
        assert final_development["latex_ref"] == original["latex_ref"]
        assert final_development["pdf_ref"] == original["pdf_ref"]
        assert store.get("live_draft")
        assert store.json(store.get("live_draft"))["manuscript_ref"] == submitted[0]["draft"]["manuscript_ref"]
        assert store.get("live_frozen")["best"]["id"] == store.get("live_best")["id"]
        assert any(row["kind"] == "experiment" and "meta/0" in row["key"]
                   for row in controller.budget.summary())
        assert len([1 for role, _ in agents.seen if role == "paperorchestra_bindings"]) == 1
        assert not store.verify()
    finally:
        store.close()


def test_unfulfilled_editorial_revision_preserves_submitted_source(tmp_path):
    controller, store, _, agents, execution = paper_campaign(
        tmp_path, writing_refinements=0, editorial=True)
    previous_response = agents.responder
    submitted = []

    def unchanged_editorial(role, inputs):
        value = previous_response(role, inputs)
        if role == "meta":
            submitted.append(deepcopy(inputs))
        if (role == "paperorchestra_model" and inputs["upstream_role"] == "content_refinement"
                and inputs.get("scripted_previous_latex")):
            return {"output": '```json\n{"unfulfilled":"No supported change"}\n```\n```latex\n'
                    + inputs["scripted_previous_latex"] + '\n```'}
        return value

    agents.responder = unchanged_editorial
    try:
        state = controller.run()
        assert state["status"] == "blocked", state
        assert state["outcome"] == "invalid_agent_output"
        assert "prior source retained" in state["reason"]
        original = submitted[0]["authoritative_manuscript"]
        preserved = store.json(store.get("research_manuscript_bundle"))
        assert preserved["latex_ref"] == original["latex_ref"]
        assert preserved["pdf_ref"] == original["pdf_ref"]
        assert store.get("live_frozen") is None
        revisions = metadata(store, "manuscript-revision/")
        assert revisions and all(store.json(ref)["no_new_experiments"] for _, ref in revisions)
        attempts = [store.json(ref) for _, ref in metadata(store, "paperorchestra-attempt/")]
        assert any(attempt.get("revision", {}).get("status") == "unchanged" for attempt in attempts)
        assert execution.writer.calls
        assert not store.verify()
    finally:
        store.close()


@pytest.mark.parametrize("interruption", ["model", "binding", "pdf"])
def test_paperorchestra_controller_replays_committed_work_without_duplicates(tmp_path, interruption):
    controller, store, runner, agents, execution = paper_campaign(tmp_path, writing_refinements=0)
    interrupted = False
    original_ask = controller.ask
    original_finish = controller.budget.finish

    def stop_after_model(key, role, *args, **kwargs):
        nonlocal interrupted
        result = original_ask(key, role, *args, **kwargs)
        target = "paperorchestra_model" if interruption == "model" else "paperorchestra_bindings"
        if interruption != "pdf" and role == target and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return result

    def stop_after_pdf(key, *args, **kwargs):
        nonlocal interrupted
        result = original_finish(key, *args, **kwargs)
        if interruption == "pdf" and key.startswith("paperorchestra/") and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return result

    controller.ask = stop_after_model
    controller.budget.finish = stop_after_pdf
    try:
        with pytest.raises(KeyboardInterrupt):
            controller.run()
        calls_before, operations_before = deepcopy(agents.seen), controller.budget.summary()
        controller.budget.finish = original_finish
        resumed = ResearchController(store, controller.config, controller.adapter, agents, runner,
                                     budget=controller.budget)
        state = resumed.run()
        assert state["status"] == "complete", state
        assert agents.seen[:len(calls_before)] == calls_before
        assert resumed.budget.summary()[:len(operations_before)] == operations_before
        # An uninterrupted equivalent run gives the exact number of external
        # model/PDF requests. Restoration must add neither hidden retries nor
        # a second execution of a committed scientific experiment.
        reference, reference_store, _, reference_agents, reference_execution = paper_campaign(
            tmp_path / "reference", writing_refinements=0)
        try:
            assert reference.run()["status"] == "complete"
            assert [role for role, _ in agents.seen] == [role for role, _ in reference_agents.seen]
            # Resume rechecks the pinned image; only actual PDF execution must
            # remain exactly once for each committed operation.
            assert sum("run" in call[0] for call in execution.writer.calls) == sum(
                "run" in call[0] for call in reference_execution.writer.calls)
            assert sum("run" in call[0] for call in execution.appendix.calls) == sum(
                "run" in call[0] for call in reference_execution.appendix.calls)
            assert len(execution.scientific.stages) == len(reference_execution.scientific.stages)
        finally:
            reference_store.close()
        snapshot = resumed.budget.summary(), len(agents.seen), len(execution.writer.calls)
        assert resumed.run() == state
        assert snapshot == (resumed.budget.summary(), len(agents.seen), len(execution.writer.calls))
        assert_authoritative_reviews(store, agents)
    finally:
        controller.budget.finish = original_finish
        store.close()
