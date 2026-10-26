"""Synthetic isolated transport tests; no Docker daemon, TeX or GPU is used."""

import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from scientisttwo.benchmarks.models import NativeResources
from scientisttwo.live.runtime import AmbiguousCall, Budget
from scientisttwo.process import ProcessResult
from scientisttwo.research.compiler import IsolatedCompiler
from scientisttwo.research.compiler_worker import COMMANDS, run as worker_run
from scientisttwo.research.manuscript import (
    CompilationPolicy, CompilationRequest, build_evidence_ledger, fixture_manuscript, render_manuscript,
)
from scientisttwo.store import IntegrityError, Store, digest

IMAGE = "sha256:" + "a" * 64


class FakeContainers:
    def __init__(self, behavior="success"):
        self.behavior, self.calls = behavior, []

    def __call__(self, argv, timeout, **kwargs):
        self.calls.append((argv, timeout, kwargs))
        if "inspect" in argv:
            return ProcessResult(1 if self.behavior == "missing-image" else 0,
                                 ("sha256:" + "b" * 64) if self.behavior == "wrong-image" else IMAGE,
                                 "", .01)
        if "rm" in argv:
            return ProcessResult(0, "removed", "", .01)
        assert "run" in argv
        assert "--pull=never" in argv and "--network=none" in argv and "--read-only" in argv
        assert "--cap-drop=ALL" in argv and "no-new-privileges" in argv
        assert "--gpus" not in argv and "--device" not in argv
        assert "--user" in argv and "65534:65534" in argv
        assert "--pids-limit=64" in argv and "--memory=1024m" in argv and "--cpus=1.0" in argv
        assert argv[-3:] == [IMAGE, "python", "/bridge/bridge.py"] or argv[-3:] == ["python", "/bridge/bridge.py"]
        work = Path(next(value.removesuffix(":/work:rw") for value in argv if value.endswith(":/work:rw")))
        control = Path(next(value.removesuffix(":/control:ro") for value in argv if value.endswith(":/control:ro")))
        job = json.loads((control / "job.json").read_text())
        assert job["commands"] == COMMANDS
        assert kwargs["output_limits"].directories == (work,)
        if self.behavior == "unknown":
            raise OSError("Synthetic disconnected transport after admission")
        if self.behavior == "timeout":
            return ProcessResult(-9, "", "Synthetic timeout", timeout, timed_out=True)
        if self.behavior == "no-receipt":
            return ProcessResult(0, "", "", .05)
        pdf = b"%PDF-synthetic-container-output"
        if self.behavior != "no-pdf":
            (work / "main.pdf").write_bytes(pdf)
        report = {"status": "success", "timed_out": False,
                  "project_manifest_sha256": job["project_manifest_sha256"],
                  "commands": [{"command": command, "returncode": 0} for command in COMMANDS],
                  "pdf_bytes": len(pdf), "pdf_sha256": digest(pdf)}
        (work / "compile-result.json").write_text(json.dumps(report))
        if self.behavior == "modified-input":
            (work / "main.tex").write_text("Modified copied input")
        if self.behavior == "missing-input":
            (work / "main.tex").unlink()
        return ProcessResult(0, "", "", .05)


@pytest.fixture
def setup(tmp_path):
    store = Store(tmp_path / "campaign")
    config = SimpleNamespace(campaign_seconds=600, max_calls=10, max_experiments=10, token_stop_threshold=1000)
    budget = Budget(store, config)
    ledger = build_evidence_ledger(store, [])
    rendered = render_manuscript(store, fixture_manuscript(ledger), ledger)
    manifest = store.json(rendered.manifest_ref)
    request = CompilationRequest(project_dir=rendered.project_path,
                                 project_manifest_sha256=manifest["project_sha256"],
                                 policy=CompilationPolicy(image_digest=IMAGE), attempt=1)
    spec = SimpleNamespace(runtime="docker", context=None, resources=NativeResources())
    yield store, budget, spec, request
    store.close()


def make_compiler(setup, behavior="success", *, native_fallback=False):
    store, budget, spec, request = setup
    transport = FakeContainers(behavior)
    runner = SimpleNamespace(store=store, budget=budget, executor=transport)
    if native_fallback:
        runner.config = SimpleNamespace(native=spec)
    else:
        runner.spec = spec
    return IsolatedCompiler(runner, IMAGE), transport, request


def test_compilation_is_isolated_probe_and_cached_resume_adds_no_operations(setup):
    compiler, transport, request = make_compiler(setup, native_fallback=True)
    original = Path(request.project_dir, "main.tex").read_bytes()
    result = compiler(request)
    assert result.status == "success"
    assert result.network_disabled and result.shell_escape_disabled
    assert Path(result.pdf_path).read_bytes().startswith(b"%PDF-")
    assert Path(request.project_dir, "main.tex").read_bytes() == original
    store, budget, _, _ = setup
    operations = budget.summary()
    assert len(operations) == 1 and operations[0]["kind"] == "probe"
    receipt = store.json(result.receipt_ref)
    assert receipt["image_id"] == IMAGE and receipt["resources"]["gpus"] is None
    calls = len(transport.calls)
    assert compiler(request) == result
    assert len(transport.calls) == calls and budget.summary() == operations
    Path(result.pdf_path).unlink()
    assert compiler(request).status == "success"
    assert Path(result.pdf_path).exists() and len(transport.calls) == calls


@pytest.mark.parametrize("behavior", ["no-pdf", "no-receipt", "modified-input", "missing-input"])
def test_quiet_or_partial_execution_cannot_report_a_pdf_success(setup, behavior):
    compiler, _, request = make_compiler(setup, behavior)
    result = compiler(request)
    assert result.status == "failed" and result.pdf_path is None
    assert not Path(request.project_dir, "main.pdf").exists()


@pytest.mark.parametrize("behavior", ["missing-image", "wrong-image"])
def test_unavailable_image_never_falls_back_to_host_or_pulls(setup, behavior):
    compiler, transport, request = make_compiler(setup, behavior)
    assert compiler(request).status == "unavailable"
    assert len(transport.calls) == 1
    assert all(call[0][0] == "docker" for call in transport.calls)
    assert not any("pull" in call[0] for call in transport.calls)


def test_timeout_uses_container_cleanup_and_persists_failure(setup):
    compiler, transport, request = make_compiler(setup, "timeout")
    result = compiler(request)
    assert result.status == "failed" and result.timed_out
    assert any("rm" in argv and "--force" in argv for argv, _, _ in transport.calls)
    count = len(transport.calls)
    assert compiler(request) == result
    assert len(transport.calls) == count


def test_unknown_completion_refuses_duplicate_execution(setup):
    compiler, transport, request = make_compiler(setup, "unknown")
    with pytest.raises(AmbiguousCall):
        compiler(request)
    count = len(transport.calls)
    with pytest.raises(AmbiguousCall, match="unknown"):
        compiler(request)
    assert len(transport.calls) == count
    assert setup[1].summary()[0]["status"] == "running"


def test_original_input_modification_is_detected_before_launch(setup):
    compiler, transport, request = make_compiler(setup)
    Path(request.project_dir, "main.tex").write_text("Tampered manuscript")
    with pytest.raises(IntegrityError, match="immutable manifest"):
        compiler(request)
    assert not transport.calls and not setup[1].summary()


def test_compiler_rejects_arbitrary_command_and_unpinned_image(setup):
    compiler, transport, request = make_compiler(setup)
    request.commands = [["sh", "-c", "unexpected"]]
    with pytest.raises(ValueError, match="fixed"):
        compiler(request)
    assert not transport.calls
    with pytest.raises(ValueError, match="digest"):
        IsolatedCompiler(compiler.runner, "latest")


def test_compilation_limits_cannot_exceed_campaign_remaining(setup):
    compiler, transport, request = make_compiler(setup)
    compiler.budget.remaining = lambda: 1.5
    assert compiler(request).status == "success"
    assert all(timeout <= 1.5 for _, timeout, _ in transport.calls)


def worker_job(tmp_path):
    path = tmp_path / "main.tex"
    path.write_text("Synthetic TeX fixture")
    return {"schema_version": "manuscript-compile-job-v1", "commands": COMMANDS,
            "timeout_seconds": 30, "project_manifest_sha256": "a" * 64,
            "files": {"main.tex": {"sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}}}


def test_worker_runs_fixed_bibtex_sequence_without_host_test_execution(tmp_path):
    calls = []
    def synthetic_execute(argv, **kwargs):
        calls.append(argv)
        assert kwargs["env"]["shell_escape"] == "0"
        assert kwargs["env"]["openin_any"] == "p" and kwargs["env"]["openout_any"] == "p"
        assert "shell" not in kwargs
        (tmp_path / "main.pdf").write_bytes(b"%PDF-unit-test")
        return SimpleNamespace(returncode=0)
    result = worker_run(worker_job(tmp_path), tmp_path, synthetic_execute)
    assert result["status"] == "success" and calls == COMMANDS


def test_worker_deadline_and_missing_tools_are_explicit(tmp_path):
    job = worker_job(tmp_path)
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
    result = worker_run(job, tmp_path, timeout)
    assert result["status"] == "failed" and result["timed_out"]
    second = tmp_path / "second"
    second.mkdir()
    def missing(argv, **kwargs):
        raise FileNotFoundError("synthetic absent TeX")
    result = worker_run(worker_job(second), second, missing)
    assert result["status"] == "unavailable"
