"""CLI contracts with fake writer/container execution and synthetic observations."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest
from typer.testing import CliRunner

from scientisttwo.integrations.paperorchestra import cli, runtime
from scientisttwo.integrations.paperorchestra.evidence import export_evidence_bundle
from scientisttwo.live.models import Candidate, Observation, SourceFile
from scientisttwo.research.manuscript import build_evidence_ledger
from scientisttwo.store import Store, canonical, digest, file_digest

RUNNER = CliRunner()
IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "synthetic-campaign"
    store = Store(root)
    candidate = Candidate(files=[SourceFile(path="train.py", content="pass\n")],
                          command=["python", "train.py"], description="Synthetic test, never executed")
    for identifier, phase in (("baseline", "full"), ("heldout", "final")):
        protocol = {"phase": phase, "seeds": [0], "adapter": "synthetic_cli_test"}
        observation = Observation(id=identifier, phase=phase, status="success", metric="accuracy",
            value=.5, values=[.5], code_ref=store.put_json(candidate),
            protocol_ref=store.put_json(protocol), protocol_hash=digest(canonical(protocol)),
            prediction_refs=[store.put_json({"synthetic_prediction_payload": True})])
        store.cache("development/" + identifier, {"synthetic": True}, observation)
    yield store
    store.close()


@pytest.fixture
def materials(campaign, tmp_path):
    observation = campaign.json(campaign.db.execute(
        "SELECT output_ref FROM calls WHERE key='development/baseline'"
    ).fetchone()[0])
    destination = tmp_path / "materials"
    export_evidence_bundle(campaign, destination,
        ledger=build_evidence_ledger(campaign, [observation]), title="Synthetic CLI test",
        hypothesis="No scientific claim", upstream_revision=runtime.UPSTREAM_REVISION)
    return destination


@pytest.fixture
def fake_execution(monkeypatch, tmp_path):
    from scientisttwo.integrations.paperorchestra import codex_transport, pdf_tools

    state = SimpleNamespace(calls=[], constructions=[], status="success", unknown_citation=False,
                            mutate_materials=False)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "locked-source").write_text("unchanged")
    state.upstream = upstream

    def verify(root):
        if (Path(root) / "locked-source").read_text() != "unchanged":
            raise ValueError("Pinned PaperOrchestra source mismatch: locked-source")
        return {"locked-source": digest(b"unchanged")}

    def pdf(**kwargs):
        state.constructions.append(("pdf", kwargs))
        return SimpleNamespace(synthetic=False, isolation="container", receipts=[])

    def fixture(materials):
        state.constructions.append(("fixture", str(materials)))
        return SimpleNamespace(synthetic=True, name="synthetic-cli-fixture")

    def codex(**kwargs):
        state.constructions.append(("codex", kwargs))
        return SimpleNamespace(synthetic=False, name="fake-codex-never-called")

    # Keep these CLI tests independent of the separately tested fixture writer.
    module = ModuleType("scientisttwo.integrations.paperorchestra.fixtures")
    module.EvidenceFixtureTransport = fixture
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runtime, "verify_upstream", verify)
    monkeypatch.setattr(pdf_tools, "ContainerPdfTools", pdf)
    monkeypatch.setattr(codex_transport, "CodexTransport", codex)

    def run(*, upstream_root, materials_dir, output_dir, transport, pdf_tools, config,
            expected_manifest_sha256=None):
        verify(upstream_root)
        if expected_manifest_sha256 is not None:
            assert file_digest(Path(materials_dir) / "manifest.json") == expected_manifest_sha256
        state.calls.append(asdict(config))
        output = Path(output_dir)
        output.mkdir()
        shutil.copytree(materials_dir, output / "raw_materials")
        report = output / "runtime-report.json"
        report.write_text(json.dumps({"status": state.status, "config": asdict(config)}))
        if state.status != "success":
            return runtime.RuntimeResult("failed", None, None, report, "Synthetic forced writer failure")
        latex = output / "final_paper.tex"
        text = r"\documentclass{article}\begin{document}Synthetic test."
        if state.unknown_citation:
            text += r"\cite{invented_source}"
        latex.write_text(text + r"\end{document}")
        final_pdf = output / "final_paper.pdf"
        final_pdf.write_bytes(b"%PDF-synthetic-test-double-not-a-real-document")
        if state.mutate_materials:
            idea = Path(materials_dir) / "idea_sparse.md"
            idea.write_text("Changed valid material bundle during execution")
            manifest_path = Path(materials_dir) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["idea_sparse.md"] = {"sha256": file_digest(idea), "bytes": idea.stat().st_size}
            manifest_path.write_text(json.dumps(manifest))
        return runtime.RuntimeResult("success", latex, final_pdf, report)

    monkeypatch.setattr(runtime, "run_paperorchestra", run)
    return state


def run_args(materials, state, output, *extra):
    return ["run", "--materials", str(materials), "--upstream", str(state.upstream),
            "--output", str(output), "--compiler-image", IMAGE, *extra]


def test_export_command_reads_original_campaign_without_changing_commitments(campaign, tmp_path):
    before_calls = [tuple(row) for row in campaign.db.execute("SELECT * FROM calls ORDER BY key")]
    before_metadata = [tuple(row) for row in campaign.db.execute("SELECT * FROM metadata ORDER BY key")]
    before_artifacts = {path.name: file_digest(path) for path in (campaign.root / "artifacts").iterdir()}
    output = tmp_path / "exported"
    response = RUNNER.invoke(cli.app, ["export", "--campaign", str(campaign.root),
        "--output", str(output), "--title", "Synthetic", "--hypothesis", "No scientific claim"])
    assert response.exit_code == 0, response.output + repr(response.exception)
    assert json.loads(response.output)["model_calls"] == 0
    ledger = json.loads((output / "evidence_ledger.json").read_text())
    assert set(ledger["observations"]) == {"baseline"}
    assert before_calls == [tuple(row) for row in campaign.db.execute("SELECT * FROM calls ORDER BY key")]
    assert before_metadata == [tuple(row) for row in campaign.db.execute("SELECT * FROM metadata ORDER BY key")]
    assert before_artifacts == {path.name: file_digest(path) for path in (campaign.root / "artifacts").iterdir()}


@pytest.mark.parametrize("options", [
    ["--backend", "codex", "--model", "gpt-6-astra"],
    ["--backend", "codex", "--allow-live"],
    ["--backend", "fixture", "--allow-live"],
    ["--backend", "fixture", "--model", "gpt-6-astra"],
    ["--backend", "unknown"],
])
def test_live_opt_in_and_backend_configuration_checked_before_any_execution(
        materials, fake_execution, tmp_path, options):
    output = tmp_path / "blocked"
    response = RUNNER.invoke(cli.app, run_args(materials, fake_execution, output, *options))
    assert response.exit_code == 2, response.output
    assert not output.exists() and not fake_execution.calls and not fake_execution.constructions


def test_completed_resume_has_no_new_operations(materials, fake_execution, tmp_path):
    output = tmp_path / "result"
    args = run_args(materials, fake_execution, output)
    first = RUNNER.invoke(cli.app, args)
    assert first.exit_code == 0, first.output + repr(first.exception)
    constructions = len(fake_execution.constructions)
    second = RUNNER.invoke(cli.app, args)
    assert second.exit_code == 0, second.output + repr(second.exception)
    assert json.loads(second.output)["reused"] is True
    assert len(fake_execution.calls) == 1 and len(fake_execution.constructions) == constructions
    assert json.loads(second.output)["new_model_calls"] == 0
    assert json.loads(second.output)["new_experiments"] == 0


@pytest.mark.parametrize("option,value", [
    ("--compiler-image", "sha256:" + "b" * 64), ("--context", "another-context"),
    ("--max-calls", "20"), ("--max-tokens", "1000"),
    ("--timeout-seconds", "30"), ("--reflections", "2"),
])
def test_completed_reuse_rejects_changed_execution_contract(
        materials, fake_execution, tmp_path, option, value):
    output = tmp_path / "result"
    args = run_args(materials, fake_execution, output)
    first = RUNNER.invoke(cli.app, args)
    assert first.exit_code == 0, first.output + repr(first.exception)
    if option in args:
        args[args.index(option) + 1] = value
    else:
        args.extend([option, value])
    second = RUNNER.invoke(cli.app, args)
    assert second.exit_code == 2, "Changed request must not reuse another validated contract: " + second.output
    assert len(fake_execution.calls) == 1


def test_completed_reuse_rejects_changed_backend_and_model(materials, fake_execution, tmp_path):
    output = tmp_path / "result"
    args = run_args(materials, fake_execution, output)
    assert RUNNER.invoke(cli.app, args).exit_code == 0
    changed = RUNNER.invoke(cli.app, args + ["--backend", "codex", "--model", "gpt-6-astra", "--allow-live"])
    assert changed.exit_code == 2, changed.output
    assert len(fake_execution.calls) == 1


def test_completed_reuse_rechecks_upstream_source_identity(materials, fake_execution, tmp_path):
    output = tmp_path / "result"
    args = run_args(materials, fake_execution, output)
    assert RUNNER.invoke(cli.app, args).exit_code == 0
    (fake_execution.upstream / "locked-source").write_text("tampered")
    changed = RUNNER.invoke(cli.app, args)
    assert changed.exit_code == 2, changed.output
    assert len(fake_execution.calls) == 1


@pytest.mark.parametrize("tamper", ["edited", "extra", "missing"])
def test_audit_and_reuse_detect_modified_output(materials, fake_execution, tmp_path, tamper):
    output = tmp_path / "result"
    args = run_args(materials, fake_execution, output)
    assert RUNNER.invoke(cli.app, args).exit_code == 0
    if tamper == "edited":
        (output / "final_paper.tex").write_text("tampered")
    elif tamper == "extra":
        (output / "unexpected.txt").write_text("extra")
    else:
        (output / "final_paper.pdf").unlink()
    assert RUNNER.invoke(cli.app, ["audit", str(output)]).exit_code == 2
    assert RUNNER.invoke(cli.app, args).exit_code == 2
    assert len(fake_execution.calls) == 1


def test_failed_runtime_is_preserved_and_not_reexecuted(materials, fake_execution, tmp_path):
    fake_execution.status = "failed"
    output = tmp_path / "failed-result"
    args = run_args(materials, fake_execution, output)
    result = RUNNER.invoke(cli.app, args)
    assert result.exit_code == 2, result.output
    assert json.loads((output / "integration-result.json").read_text())["status"] != "success"
    assert RUNNER.invoke(cli.app, args).exit_code == 2
    assert len(fake_execution.calls) == 1


def test_external_validation_failure_cannot_report_success(materials, fake_execution, tmp_path):
    fake_execution.unknown_citation = True
    output = tmp_path / "invalid-manuscript"
    result = RUNNER.invoke(cli.app, run_args(materials, fake_execution, output))
    assert result.exit_code == 2, result.output
    report = json.loads((output / "integration-result.json").read_text())
    assert report["status"] != "success"
    assert report["external_manuscript"]["validation"]["status"] == "fail"
    assert report["external_manuscript"]["controller_accepted"] is False
    assert (output / "final_paper.tex").is_file()


def test_changed_frozen_materials_cannot_reuse_completed_output(materials, fake_execution, tmp_path):
    output = tmp_path / "result"
    args = run_args(materials, fake_execution, output)
    assert RUNNER.invoke(cli.app, args).exit_code == 0
    other = tmp_path / "other-materials"
    shutil.copytree(materials, other)
    (other / "experimental_log.md").write_text("changed without valid hash")
    args[args.index("--materials") + 1] = str(other)
    assert RUNNER.invoke(cli.app, args).exit_code == 2
    assert len(fake_execution.calls) == 1


def test_materials_replaced_during_execution_cannot_be_imported_as_original(materials, fake_execution, tmp_path):
    fake_execution.mutate_materials = True
    output = tmp_path / "changed-input"
    result = RUNNER.invoke(cli.app, run_args(materials, fake_execution, output))
    assert result.exit_code == 2, result.output
    assert (output / "final_paper.tex").is_file()
    report_path = output / "integration-result.json"
    if report_path.exists():
        assert json.loads(report_path.read_text())["status"] != "success"
