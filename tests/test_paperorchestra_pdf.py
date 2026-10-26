"""Synthetic transport tests for isolated PDF execution, without a TeX host fallback."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scientisttwo.integrations.paperorchestra.pdf_tools import ContainerPdfTools, PdfToolsError
from scientisttwo.integrations.paperorchestra.pdf_worker import run as worker_run
from scientisttwo.process import ProcessResult


IMAGE = "sha256:" + "a" * 64


def metadata(payload):
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class FakeContainer:
    def __init__(self, mode="success"):
        self.mode, self.calls = mode, []

    def __call__(self, argv, timeout, **kwargs):
        self.calls.append((argv, timeout, kwargs))
        assert argv[0] in {"docker", "podman"}, "Generated TeX cannot execute on the host"
        if "inspect" in argv:
            return ProcessResult(0, "sha256:" + "b" * 64 if self.mode == "wrong-image" else IMAGE, "", 0.01)
        if "rm" in argv:
            return ProcessResult(0, "", "", 0.01)
        assert "run" in argv
        assert {"--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
                "--memory=768m", "--memory-swap=768m", "--pids-limit=64", "--cpus=1",
                "--ipc=none", "no-new-privileges", "65534:65534"} <= set(argv)
        assert "--gpus" not in argv and "--device" not in argv
        assert argv[-3:] == ["python", IMAGE, "/worker.py"]
        mounts = {item.split("target=")[1].split(",")[0]: Path(item.split("source=")[1].split(",")[0])
                  for item in argv if item.startswith("type=bind,")}
        assert set(mounts) == {"/input", "/output", "/control", "/worker.py"}
        assert kwargs["output_limits"].directories == (mounts["/output"],)
        assert kwargs["max_output_bytes"] == 65536
        if self.mode == "timeout":
            return ProcessResult(-9, "", "", timeout, timed_out=True)
        if self.mode == "disconnected":
            raise OSError("Synthetic unavailable transport")
        if self.mode == "quota":
            return ProcessResult(-9, "", "", 0.01, limit_exceeded="Synthetic quota failure")
        if self.mode == "failed":
            return ProcessResult(1, "", "Synthetic TeX error", 0.01)
        job = json.loads((mounts["/control"] / "job.json").read_text())
        output = mounts["/output"]
        payloads = {"compile": {"paper.pdf": b"%PDF-synthetic"},
                    "text": {"paper.txt": b"Real PDF tools are container-only.\n"},
                    "render": {"page-01.png": b"\x89PNG\r\n\x1a\nsynthetic1",
                               "page-02.png": b"\x89PNG\r\n\x1a\nsynthetic2"}}[job["operation"]]
        if self.mode == "bad-pdf":
            payloads = {"paper.pdf": b"This is not a PDF"}
        for name, payload in payloads.items():
            (output / name).write_bytes(payload)
        report = {"status": "success", "operation": job["operation"],
                  "outputs": {name: metadata(value) for name, value in payloads.items()}}
        if self.mode == "tampered":
            (output / next(iter(payloads))).write_bytes(b"tampered")
        if self.mode == "symlink":
            (output / "escape").symlink_to(mounts["/input"])
        if self.mode != "no-receipt":
            (output / "result.json").write_text(json.dumps(report))
        return ProcessResult(0, "", "", 0.01)


def project(tmp_path):
    root = tmp_path / "paper"
    root.mkdir()
    (root / "main.tex").write_text(r"\documentclass{article}\begin{document}test\end{document}")
    return root


def test_compile_text_and_render_have_pinned_isolation_and_receipts(tmp_path):
    fake = FakeContainer()
    adapter = ContainerPdfTools(IMAGE, context="colima-scientisttwo", executor=fake)
    root = project(tmp_path)
    (root / "private.json").write_text("not a paper asset")
    pdf = root / "main.pdf"
    adapter.compile(root, pdf, "main.tex", 10)
    assert pdf.read_bytes() == b"%PDF-synthetic"
    assert adapter.text(pdf).startswith("Real PDF tools")
    rendered = adapter.render(pdf, root / "rendered")
    assert [path.name for path in rendered] == ["page-01.png", "page-02.png"]
    assert adapter.synthetic is False and adapter.isolation == "container"
    assert len(adapter.receipts) == 3
    assert all(item["status"] == "success" for item in adapter.receipts)
    assert "private.json" not in adapter.receipts[0]["inputs"]
    assert all(call[0][:3] == ["docker", "--context", "colima-scientisttwo"] for call in fake.calls)


def test_compile_accepts_the_official_upstream_filename_stem(tmp_path):
    root = project(tmp_path)
    (root / "main.tex").rename(root / "initial_draft.tex")
    adapter = ContainerPdfTools(IMAGE, executor=FakeContainer())
    adapter.compile(root, root / "initial_draft.pdf", "initial_draft", 10)
    assert (root / "initial_draft.pdf").read_bytes() == b"%PDF-synthetic"
    assert "initial_draft.tex" in adapter.receipts[0]["inputs"]


def test_explicit_staging_stays_inside_existing_workspace_and_is_removed(tmp_path):
    staging = tmp_path / "run"
    fake = FakeContainer()
    adapter = ContainerPdfTools(IMAGE, executor=fake, staging_root=staging)
    assert not staging.exists(), "Construction must not create the runtime output directory"
    staging.mkdir()
    root = project(tmp_path)
    adapter.compile(root, root / "main.pdf", "main", 10)
    run = next(argv for argv, _, _ in fake.calls if "run" in argv)
    source_mounts = [Path(item.split("source=")[1].split(",")[0])
                     for item in run if item.startswith("type=bind,") and "target=/worker.py" not in item]
    assert all(path.is_relative_to(staging) for path in source_mounts)
    assert not list(staging.iterdir()), "Temporary staging must be removed after the operation"


@pytest.mark.parametrize("kind", ["missing", "symlink", "symlink-parent", "regular-file"])
def test_invalid_staging_fails_before_container_admission(tmp_path, kind):
    staging = tmp_path / "staging"
    if kind == "symlink":
        staging.symlink_to(tmp_path, target_is_directory=True)
    elif kind == "symlink-parent":
        staging.symlink_to(tmp_path, target_is_directory=True)
        staging = staging / "nested"
        (tmp_path / "nested").mkdir()
    elif kind == "regular-file":
        staging.write_text("not a directory")
    fake = FakeContainer()
    adapter = ContainerPdfTools(IMAGE, executor=fake, staging_root=staging)
    with pytest.raises(PdfToolsError, match="staging root"):
        adapter.compile(project(tmp_path), tmp_path / "out.pdf", "main", 5)
    assert not fake.calls


@pytest.mark.parametrize("mode", ["wrong-image", "timeout", "quota", "failed", "disconnected",
                                  "no-receipt", "tampered", "symlink", "bad-pdf"])
def test_failures_have_no_host_fallback_or_published_pdf(tmp_path, mode):
    fake = FakeContainer(mode)
    adapter = ContainerPdfTools(IMAGE, executor=fake)
    root = project(tmp_path)
    target = tmp_path / "final.pdf"
    with pytest.raises(PdfToolsError):
        adapter.compile(root, target, "main.tex", 10)
    assert not target.exists()
    assert adapter.receipts[-1]["status"] == "failed"
    if mode in {"timeout", "quota", "disconnected"}:
        assert any("rm" in call[0] for call in fake.calls)
    assert all(call[0][0] == "docker" for call in fake.calls)


@pytest.mark.parametrize("options", [{"image_digest": "mutable:tag"},
                                      {"image_digest": IMAGE, "runtime": "host"},
                                      {"image_digest": IMAGE, "timeout_seconds": float("nan")},
                                      {"image_digest": IMAGE, "max_pages": 101},
                                      {"image_digest": IMAGE, "context": "x,y=/home"}])
def test_bad_runtime_configuration_fails_before_execution(options):
    with pytest.raises(ValueError):
        ContainerPdfTools(**options)


@pytest.mark.parametrize("kind", ["bytes", "files", "symlink", "traversal"])
def test_project_input_limits_prevent_container_admission(tmp_path, kind):
    fake = FakeContainer()
    root = project(tmp_path)
    options = {"max_input_bytes": 1} if kind == "bytes" else {"max_files": 1} if kind == "files" else {}
    if kind == "files":
        (root / "extra.tex").write_text("extra")
    if kind == "symlink":
        (root / "escape.tex").symlink_to(tmp_path / "elsewhere")
    adapter = ContainerPdfTools(IMAGE, executor=fake, **options)
    with pytest.raises(PdfToolsError):
        adapter.compile(root, tmp_path / "out.pdf", "../main.tex" if kind == "traversal" else "main.tex", 5)
    assert not fake.calls


def test_expired_budget_never_launches(tmp_path):
    fake = FakeContainer()
    adapter = ContainerPdfTools(IMAGE, executor=fake)
    adapter.deadline = 0
    with pytest.raises(PdfToolsError, match="deadline"):
        adapter.compile(project(tmp_path), tmp_path / "out.pdf", "main.tex", 5)
    assert not fake.calls


def test_output_page_count_is_checked_independently_of_worker(tmp_path):
    adapter = ContainerPdfTools(IMAGE, executor=FakeContainer(), max_pages=1)
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"%PDF-synthetic")
    with pytest.raises(PdfToolsError, match="page output contract"):
        adapter.render(pdf, tmp_path / "pages")
    assert not (tmp_path / "pages").exists()


def test_output_byte_limit_is_checked_independently_of_worker(tmp_path):
    adapter = ContainerPdfTools(IMAGE, executor=FakeContainer(), max_output_bytes=10)
    with pytest.raises(PdfToolsError, match="byte limit"):
        adapter.compile(project(tmp_path), tmp_path / "out.pdf", "main.tex", 5)
    assert not (tmp_path / "out.pdf").exists()


def test_failed_revision_preserves_existing_pdf(tmp_path):
    target = tmp_path / "out.pdf"
    target.write_bytes(b"%PDF-previous-best")
    adapter = ContainerPdfTools(IMAGE, executor=FakeContainer("failed"))
    with pytest.raises(PdfToolsError):
        adapter.compile(project(tmp_path), target, "main.tex", 5)
    assert target.read_bytes() == b"%PDF-previous-best"


def test_text_and_render_use_remaining_timeout(tmp_path):
    fake = FakeContainer()
    adapter = ContainerPdfTools(IMAGE, executor=fake)
    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"%PDF-synthetic")
    adapter.text(pdf, timeout_seconds=2)
    adapter.render(pdf, tmp_path / "pages", timeout_seconds=3)
    runs = [(argv, timeout) for argv, timeout, _ in fake.calls if "run" in argv]
    assert runs[0][1] <= 2 and runs[1][1] <= 3


def test_worker_fixed_commands_disable_shell_escape_and_preserve_inputs(tmp_path):
    inputs, outputs, work = [tmp_path / item for item in ("input", "output", "work")]
    inputs.mkdir()
    outputs.mkdir()
    tex = b"synthetic source"
    (inputs / "main.tex").write_bytes(tex)
    job = {"schema_version": "paperorchestra-pdf-job-v1", "operation": "compile",
           "tex_name": "main.tex", "timeout_seconds": 10, "max_pages": 24,
           "max_output_bytes": 100000, "inputs": {"main.tex": metadata(tex)}}
    calls = []

    def execute(argv, **kwargs):
        calls.append(argv)
        assert kwargs["env"]["shell_escape"] == "0"
        assert kwargs["env"]["openin_any"] == "p"
        assert kwargs["env"]["openout_any"] == "p"
        assert kwargs["env"]["HOME"] == "/tmp"
        assert "-no-shell-escape" in argv or argv == ["bibtex", "main"]
        (work / "main.aux").write_text(r"\bibdata{references}")
        (work / "main.pdf").write_bytes(b"%PDF-synthetic")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    result = worker_run(job, inputs, outputs, work, execute=execute)
    assert result["status"] == "success"
    assert len(calls) == 4 and calls[1] == ["bibtex", "main"]
    assert (inputs / "main.tex").read_bytes() == tex
    assert json.loads((outputs / "result.json").read_text())["outputs"]["paper.pdf"] == metadata(b"%PDF-synthetic")
