"""Pinned, offline manuscript compilation through the existing isolated runner.

Only image inspection and container commands reach the host process executor.
Compilation uses probe accounting, so manuscript-only work does not become a
scientific experiment or bypass the campaign wall-clock budget.
"""

import json
from pathlib import Path
import re
import shutil
import time

from ..artifacts import safe_path
from ..benchmarks.models import NativeResources
from ..benchmarks.runner import NativeRunner
from ..live.runtime import AmbiguousCall
from ..runtime_contracts import execution_identity
from ..store import IntegrityError, canonical, digest, file_digest
from .compiler_worker import COMMANDS, MAX_PDF_BYTES
from .manuscript import CompilationRequest, CompilationResult


class IsolatedCompiler:
    _image = NativeRunner._image
    _launch = NativeRunner._launch

    def __init__(self, runner, image_digest):
        if not isinstance(image_digest, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", image_digest):
            raise ValueError("Manuscript compiler requires an explicit image digest")
        self.runner, self.image_digest = runner, image_digest
        self.store, self.budget = runner.store, runner.budget
        self.spec = getattr(runner, "spec", None) or runner.config.native
        if self.spec.runtime not in {"docker", "podman"}:
            raise ValueError("Manuscript compilation requires an isolated Docker or Podman runtime")
        self.prefix = [self.spec.runtime] + (["--context", self.spec.context] if self.spec.context else [])
        self.raw_executor = getattr(runner, "raw_executor", None) or runner.executor

    def executor(self, argv, timeout, **kwargs):
        if argv[:len(self.prefix)] != self.prefix:
            raise IntegrityError("Manuscript compiler attempted a non-container host command")
        if argv[len(self.prefix):len(self.prefix) + 2] != ["rm", "--force"]:
            timeout = min(timeout, self.budget.remaining())
        return self.raw_executor(argv, timeout, **kwargs)

    def _inputs(self, request):
        project = Path(request.project_dir)
        expected = self.store.root / "manuscripts" / request.project_manifest_sha256
        if project.is_symlink() or project.resolve() != expected.resolve() or not project.is_dir():
            raise IntegrityError("Compiler input must be the owned immutable manuscript project")
        declaration = project / "inputs-manifest.json"
        if (declaration.is_symlink() or not declaration.is_file()
                or declaration.stat().st_size > 5_000_000):
            raise IntegrityError("Compiler input manifest is missing or oversized")
        raw = declaration.read_bytes()
        manifest = json.loads(raw)
        if manifest.get("schema_version") != "manuscript-inputs-v1":
            raise IntegrityError("Unknown manuscript input manifest")
        files = dict(manifest["files"])
        files["inputs-manifest.json"] = {"sha256": digest(raw), "bytes": len(raw)}
        if not {"main.tex", "references.bib"} <= files.keys() or len(files) > 11000:
            raise IntegrityError("Compiler input file contract is incomplete or oversized")
        total = 0
        for name, item in files.items():
            safe_path(name)
            path = project / name
            if (path.is_symlink() or not path.resolve().is_relative_to(project.resolve())
                    or not path.is_file() or path.stat().st_size != item["bytes"]
                    or file_digest(path) != item["sha256"]):
                raise IntegrityError("Compiler input differs from its immutable manifest")
            total += item["bytes"]
            if total > 100_000_000:
                raise IntegrityError("Compiler inputs exceed their aggregate byte limit")
        if digest(canonical({name: item["sha256"] for name, item in sorted(files.items())})) != request.project_manifest_sha256:
            raise IntegrityError("Compiler project hash does not bind its input manifest")
        return project, files

    def _publish_pdf(self, project, ref):
        # Restore only a verified PDF artifact; never copy container symlinks.
        source = self.store.root / ref
        registered = self.store.db.execute("SELECT sha,size FROM artifacts WHERE path=?", (ref,)).fetchone()
        if (registered is None or not source.resolve().is_relative_to((self.store.root / "artifacts").resolve())
                or source.is_symlink() or not source.is_file()
                or not 5 <= source.stat().st_size <= MAX_PDF_BYTES
                or source.stat().st_size != registered["size"] or file_digest(source) != registered["sha"]):
            raise IntegrityError("Cached compiler PDF failed integrity verification")
        with source.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise IntegrityError("Cached compiler artifact is not a PDF")
        target = project / "main.pdf"
        if target.exists():
            if target.is_symlink() or not target.is_file() or file_digest(target) != registered["sha"]:
                raise IntegrityError("Published manuscript PDF differs from its sealed artifact")
        else:
            temporary = project / ("main.pdf." + registered["sha"] + ".tmp")
            with source.open("rb") as incoming, temporary.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            if file_digest(temporary) != registered["sha"]:
                raise IntegrityError("PDF changed during restoration")
            temporary.replace(target)
        return str(target)

    def __call__(self, request):
        request = CompilationRequest.model_validate(request)
        if (request.policy.image_digest != self.image_digest or request.command != COMMANDS[0]
                or request.commands != COMMANDS or request.main_file != "main.tex"
                or request.attempt > request.policy.max_attempts):
            raise ValueError("Compiler image and fixed TeX/BibTeX command contract must match")
        if request.policy.memory_mib < 256:
            raise ValueError("The isolated runtime requires at least 256 MiB")
        project, files = self._inputs(request)
        bridge = Path(__file__).with_name("compiler_worker.py")
        worker_ref = self.store.put_file(bridge, "manuscript_compiler_worker")
        worker_sha = file_digest(bridge)
        key = f"artifact/compile/{request.project_manifest_sha256}/{request.attempt}"
        admission = {"request": request.model_dump(mode="json"), "worker_sha256": worker_sha,
                     "runtime": self.spec.runtime, "context": self.spec.context,
                     "execution_policy": execution_identity(self.spec.runtime)}
        prior = self.budget.reserve(key, "probe", admission)
        if prior:
            if not prior["result_ref"]:
                raise AmbiguousCall("Compiler outcome is unknown; refusing duplicate execution: " + key)
            saved = self.store.json(prior["result_ref"])
            result = CompilationResult.model_validate(saved["result"])
            if result.receipt_ref:
                self.store.read(result.receipt_ref)
            if result.status == "success":
                result.pdf_path = self._publish_pdf(project, saved["pdf_ref"])
            return result
        started = time.monotonic()
        try:
            image = self._image(self.image_digest)
        except ValueError as error:
            result = CompilationResult(status="unavailable", diagnostics=str(error), image_digest=self.image_digest)
            self.budget.finish(key, "unavailable", {"result": result.model_dump(mode="json"), "pdf_ref": None}, time.monotonic() - started)
            return result
        if image != self.image_digest:
            result = CompilationResult(status="unavailable", diagnostics="Inspected compiler image differs from its pinned digest", image_digest=image)
            self.budget.finish(key, "unavailable", {"result": result.model_dump(mode="json"), "pdf_ref": None}, time.monotonic() - started)
            return result
        work = self.store.root / "compilation" / request.project_manifest_sha256 / str(request.attempt)
        try:
            work.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise AmbiguousCall("Compiler work exists without a committed outcome") from error
        for name in sorted(files):
            target = work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)
        for path in [work, *work.rglob("*")]:
            if path.is_dir():
                path.chmod(0o777)
        timeout = min(request.policy.timeout_seconds, self.budget.remaining())
        resources = NativeResources(
            cpus=min(request.policy.cpus, self.spec.resources.cpus),
            memory_mb=min(request.policy.memory_mib, self.spec.resources.memory_mb), seconds=timeout,
            grading_seconds=timeout, gpus=None, shm_mb=16, tmp_mb=256, pids_limit=64,
            threads=min(request.policy.cpus, self.spec.resources.threads), output_max_bytes=134_217_728,
            output_max_files=12000, log_max_bytes=1_048_576,
        )
        job = {"schema_version": "manuscript-compile-job-v1", "commands": COMMANDS,
               "project_manifest_sha256": request.project_manifest_sha256,
               "files": files, "timeout_seconds": timeout, "worker_sha256": worker_sha}
        stage = "compile-" + digest(key.encode())[:24]
        # Exceptions during process admission retain the reservation. Even after
        # cleanup, an unrecorded completion cannot safely become a fresh attempt.
        try:
            process, receipt_ref = self._launch(
                stage, image, job, [(work, "/work", "rw")], timeout,
                gpus=None, bridge=bridge, resources=resources,
            )
        except (OSError, ValueError) as error:
            raise AmbiguousCall("Compiler execution lacks a committed outcome") from error
        self._inputs(request)
        result_file = work / "compile-result.json"
        report = None
        if result_file.is_file() and not result_file.is_symlink() and result_file.stat().st_size <= 100_000:
            try:
                report = json.loads(result_file.read_text())
            except (ValueError, OSError):
                pass
        pdf_ref = None
        reason = "Isolated compiler did not return a complete successful receipt"
        status = "failed"
        timed_out = bool(process.timed_out or report and report.get("timed_out"))
        if process.limit_exceeded:
            reason = "Compiler output or process resource limit exceeded"
        elif timed_out:
            reason = "Isolated compiler timed out; container cleanup requested"
        elif (not process.code and report and report.get("status") == "success"
              and report.get("project_manifest_sha256") == request.project_manifest_sha256
              and report.get("commands") == [{"command": command, "returncode": 0} for command in COMMANDS]):
            pdf = work / "main.pdf"
            if (pdf.is_file() and not pdf.is_symlink() and 5 <= pdf.stat().st_size <= MAX_PDF_BYTES
                    and pdf.stat().st_size == report.get("pdf_bytes")
                    and file_digest(pdf) == report.get("pdf_sha256")):
                with pdf.open("rb") as stream:
                    if stream.read(5) == b"%PDF-":
                        pdf_ref = self.store.put_file(pdf, "compiled_manuscript_pdf")
                        status, reason = "success", "Pinned offline TeX/BibTeX compilation completed"
        elif report and report.get("status") == "unavailable":
            status, reason = "unavailable", "Pinned image lacks required TeX/BibTeX tools"
        # Verify copied inputs too: the worker must not erase scientific evidence.
        try:
            working_inputs_intact = all(
                not (work / name).is_symlink() and (work / name).is_file()
                and file_digest(work / name) == item["sha256"] for name, item in files.items()
            )
        except OSError:
            working_inputs_intact = False
        if not working_inputs_intact:
            status, reason, pdf_ref = "failed", "Compiler modified a frozen working input", None
        result = CompilationResult(
            status=status, diagnostics=reason, receipt_ref=receipt_ref, image_digest=image,
            network_disabled=True, shell_escape_disabled=True, timed_out=timed_out,
        )
        if pdf_ref:
            result.pdf_path = self._publish_pdf(project, pdf_ref)
        saved = {"result": result.model_dump(mode="json"), "pdf_ref": pdf_ref,
                 "execution_receipt_ref": receipt_ref, "worker_ref": worker_ref,
                 "worker_sha256": worker_sha, "report_ref": self.store.put_json(report, "compiler_report") if report else None,
                 "input_manifest": files, "probe_only": True}
        self.budget.finish(key, status, saved, time.monotonic() - started)
        return result
