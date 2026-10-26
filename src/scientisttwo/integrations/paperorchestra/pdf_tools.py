"""Standalone PDF operations for PaperOrchestra, confined to a pinned container.

Only the container CLI is executed on the host. Neither generated TeX nor PDF
parsers run on the host, and unavailable isolation never falls back to host tools.
"""

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
import uuid

from ...process import OutputLimits, run_process


class PdfToolsError(RuntimeError):
    """An isolated PDF operation failed or violated its artifact contract."""


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _relative(name):
    path = Path(name)
    if (not isinstance(name, str) or not name or path.is_absolute()
            or ".." in path.parts or str(path) != name or "\\" in name):
        raise PdfToolsError("Unsafe relative PDF artifact path")
    return path


class ContainerPdfTools:
    """The synchronous PDF interface injected into the official writer runtime.

    ``image_digest`` is the locally built image's config digest, not a mutable
    tag. Every invocation verifies that exact image. ``receipts`` captures input
    and output hashes, worker identity, container policy, and process outcomes.
    The executor injection is for transport tests, never a production fallback.
    """

    synthetic = False
    isolation = "container"

    def __init__(self, image_digest, *, runtime="docker", context=None,
                 timeout_seconds=120, budget_seconds=None, executor=run_process,
                 max_input_bytes=100_000_000, max_output_bytes=100_000_000,
                 max_files=1000, max_pages=24, staging_root=None, swap_policy="deny"):
        if not isinstance(image_digest, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", image_digest):
            raise ValueError("PDF tools require a pinned local image sha256 digest")
        if runtime not in {"docker", "podman"}:
            raise ValueError("PDF tools require Docker or Podman isolation")
        if swap_policy not in {"deny", "inherit"} or (swap_policy == "inherit" and runtime != "podman"):
            raise ValueError("Inherited PDF swap requires an explicit Podman profile")
        if context is not None and (runtime != "docker" or not re.fullmatch(r"[A-Za-z0-9_.-]+", context)):
            raise ValueError("Only an explicit Docker context name is supported")
        if not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 300:
            raise ValueError("PDF operation timeout must be in (0, 300] seconds")
        for name, value, ceiling in (("max_input_bytes", max_input_bytes, 500_000_000),
                                     ("max_output_bytes", max_output_bytes, 500_000_000),
                                     ("max_files", max_files, 10000), ("max_pages", max_pages, 100)):
            if type(value) is not int or not 0 < value <= ceiling:
                raise ValueError(f"Invalid PDF artifact limit: {name}")
        if budget_seconds is not None and (not math.isfinite(budget_seconds) or budget_seconds <= 0):
            raise ValueError("PDF operation budget must be positive and finite")
        self.image_digest, self.timeout_seconds = image_digest, timeout_seconds
        self.swap_policy = swap_policy
        self.prefix = [runtime] + (["--context", context] if context else [])
        self.executor = executor
        self.max_input_bytes, self.max_output_bytes = max_input_bytes, max_output_bytes
        self.max_files, self.max_pages = max_files, max_pages
        # The runtime creates its owned output directory after this adapter is
        # constructed. Defer validation and never create that directory here.
        self.staging_root = Path(staging_root).absolute() if staging_root is not None else None
        self.deadline = time.monotonic() + budget_seconds if budget_seconds is not None else None
        self.receipts = []

    def _timeout(self, requested=None):
        timeout = self.timeout_seconds
        if requested is not None:
            if not math.isfinite(requested) or requested <= 0:
                raise PdfToolsError("PDF operation deadline exhausted")
            timeout = min(timeout, requested)
        if self.deadline is not None:
            timeout = min(timeout, self.deadline - time.monotonic())
        if timeout <= 0:
            raise PdfToolsError("PDF operation deadline exhausted")
        return timeout

    def _manifest(self, root, *, input_tree=False):
        if root.is_symlink() or not root.is_dir():
            raise PdfToolsError("PDF artifact root must be a regular directory")
        files, size, entries = {}, 0, 0
        for current, directories, names in os.walk(root, followlinks=False):
            for name in directories + names:
                entries += 1
                if entries > self.max_files:
                    raise PdfToolsError("PDF artifact file-count limit exceeded")
                path = Path(current) / name
                info = path.lstat()
                if stat.S_ISDIR(info.st_mode):
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise PdfToolsError("PDF artifacts must not contain symlinks or special files")
                rel = path.relative_to(root).as_posix()
                _relative(rel)
                # Include only the paper project, never runtime configs or code.
                if input_tree and path.suffix.lower() not in {
                    ".tex", ".bib", ".bst", ".sty", ".cls", ".pdf", ".png",
                    ".jpg", ".jpeg", ".eps", ".svg", ".def", ".clo", ".cfg",
                }:
                    continue
                size += info.st_size
                if size > (self.max_input_bytes if input_tree else self.max_output_bytes):
                    raise PdfToolsError("PDF artifact byte limit exceeded")
                files[rel] = {"bytes": info.st_size, "sha256": _sha(path)}
        return files

    def _regular(self, path, *, header=None):
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= self.max_input_bytes:
            raise PdfToolsError("Missing, oversized, or nonregular PDF input")
        if header:
            with path.open("rb") as stream:
                if stream.read(len(header)) != header:
                    raise PdfToolsError("Input is not a PDF")

    def _operation(self, kind, source, *, tex_name=None, timeout_seconds=None):
        timeout = self._timeout(timeout_seconds)
        started = time.monotonic()
        if self.staging_root is not None:
            if (self.staging_root.is_symlink() or not self.staging_root.is_dir()
                    or self.staging_root.resolve() != self.staging_root
                    or any(path.is_symlink() for path in self.staging_root.parents)):
                raise PdfToolsError("PDF staging root must be an existing regular directory without symlink traversal")
        source = Path(source)
        if kind == "compile":
            _relative(tex_name)
            if not tex_name.endswith(".tex"):
                raise PdfToolsError("Compiler main file must be a .tex source")
            manifest = self._manifest(source, input_tree=True)
            if tex_name not in manifest:
                raise PdfToolsError("Compiler main source is missing")
        else:
            self._regular(source, header=b"%PDF-")
            manifest = {"paper.pdf": {"bytes": source.stat().st_size, "sha256": _sha(source)}}
        worker = Path(__file__).with_name("pdf_worker.py")
        receipt = {"schema_version": "paperorchestra-pdf-receipt-v1", "operation": kind,
                   "image_digest": self.image_digest, "worker_sha256": _sha(worker),
                   "inputs": manifest, "network_disabled": True, "shell_escape_disabled": True,
                   "swap_policy": self.swap_policy,
                   "memory_budget_scope": "physical_ram" if self.swap_policy == "inherit" else "memory_and_swap",
                   "status": "started", "max_output_bytes": self.max_output_bytes,
                   "max_files": self.max_files, "max_pages": self.max_pages}
        self.receipts.append(receipt)
        try:
            inspected = self.executor(self.prefix + ["image", "inspect", self.image_digest,
                                                      "--format", "{{.Id}}"], min(15, timeout),
                                      max_output_bytes=4096)
            image = inspected.stdout.strip()
            if re.fullmatch(r"[a-f0-9]{64}", image):
                image = "sha256:" + image
            if (inspected.code or inspected.timed_out or inspected.limit_exceeded
                    or inspected.stdout_truncated or image != self.image_digest):
                raise PdfToolsError("Pinned PDF image is unavailable or differs from its digest")
            with tempfile.TemporaryDirectory(prefix="scientisttwo-pdf-",
                                             dir=self.staging_root) as temporary:
                root = Path(temporary)
                inputs, outputs, control = (root / name for name in ("input", "output", "control"))
                for directory in (inputs, outputs, control):
                    directory.mkdir(mode=0o755)
                outputs.chmod(0o777)
                for name in manifest:
                    destination = inputs / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / name if kind == "compile" else source, destination)
                    destination.chmod(0o644)
                    if _sha(destination) != manifest[name]["sha256"]:
                        raise PdfToolsError("PDF input changed during snapshot")
                for directory, _, _ in os.walk(inputs):
                    Path(directory).chmod(0o755)
                remaining = self._timeout(timeout - (time.monotonic() - started))
                job = {"schema_version": "paperorchestra-pdf-job-v1", "operation": kind,
                       "tex_name": tex_name, "inputs": manifest, "timeout_seconds": remaining,
                       "max_pages": self.max_pages, "max_output_bytes": self.max_output_bytes}
                (control / "job.json").write_text(json.dumps(job, sort_keys=True))
                (control / "job.json").chmod(0o644)
                name = "scientisttwo-pdf-" + uuid.uuid4().hex[:20]
                # Keep one explicit timeout contract for the native runtime and
                # the Slurm shim. Podman's guarded vocabulary uses -v; Docker's
                # existing bind-mount invocation remains unchanged.
                mounts = []
                for path, target, readonly in (
                    (inputs, "/input", True), (outputs, "/output", False),
                    (control, "/control", True), (worker, "/worker.py", True),
                ):
                    if self.prefix[0] == "podman":
                        absolute = path.absolute()
                        if ":" in str(absolute) or "\n" in str(absolute):
                            raise PdfToolsError("Podman bind paths cannot contain delimiters")
                        mounts += ["-v", f"{absolute}:{target}:{'ro' if readonly else 'rw'}"]
                    else:
                        mounts += ["--mount", f"type=bind,source={path},target={target}"
                                   + (",readonly" if readonly else "")]
                argv = self.prefix + [
                    "run", "--rm", "--pull=never", "--name", name,
                    "--network=none", "--read-only", "--cap-drop=ALL",
                    "--security-opt", "no-new-privileges", "--memory=768m",
                    *(["--memory-swap=768m"] if self.swap_policy == "deny" else []),
                    "--cpus=1", "--pids-limit=64",
                    # Podman rejects --shm-size together with --ipc=none.
                    # Keep IPC disabled; PDF tools use the bounded /tmp mount.
                    *(["--shm-size=16m"] if self.prefix[0] == "docker" else []), "--ipc=none",
                    "--user", "65534:65534", "--ulimit", "nofile=256:256",
                    "--ulimit", f"fsize={self.max_output_bytes}:{self.max_output_bytes}",
                    "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=256m,mode=1777", "-w", "/tmp",
                    "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
                    "-e", "SCIENTISTTWO_PDF_SWAP_POLICY=" + self.swap_policy,
                    "--label", f"scientisttwo.timeout-seconds={remaining}",
                    *mounts,
                    "--entrypoint", "python", self.image_digest, "/worker.py",
                ]
                receipt["argv"] = argv
                result = None
                try:
                    result = self.executor(argv, remaining, max_output_bytes=65536,
                                           output_limits=OutputLimits((outputs,), self.max_output_bytes,
                                                                      self.max_files))
                    receipt["process"] = asdict(result)
                finally:
                    if result is None or result.timed_out or result.limit_exceeded:
                        cleanup = self.executor(self.prefix + ["rm", "--force", name], 15,
                                                max_output_bytes=4096)
                        receipt["cleanup"] = asdict(cleanup)
                        if cleanup.timed_out or (cleanup.code and "No such container" not in cleanup.stderr):
                            raise PdfToolsError("PDF container cleanup could not be confirmed")
                if result.timed_out:
                    raise PdfToolsError("PDF operation timed out; owned container cleanup requested")
                if result.limit_exceeded:
                    raise PdfToolsError("PDF output resource limit exceeded")
                if result.code:
                    raise PdfToolsError("PDF container operation failed: " + result.stderr[-4000:])
                actual = self._manifest(outputs)
                declaration = outputs / "result.json"
                if declaration.is_symlink() or not declaration.is_file() or declaration.stat().st_size > 100_000:
                    raise PdfToolsError("Missing or oversized PDF output receipt")
                report = json.loads(declaration.read_text())
                if report.get("status") != "success" or report.get("operation") != kind:
                    raise PdfToolsError("PDF output receipt does not confirm the requested operation")
                declared = report.get("outputs")
                if not isinstance(declared, dict) or not declared or actual != {**declared, "result.json": actual["result.json"]}:
                    raise PdfToolsError("PDF output artifacts differ from the declared receipt")
                for path in declared:
                    _relative(path)
                if kind == "compile" and set(declared) != {"paper.pdf"}:
                    raise PdfToolsError("Compiler output contract requires one PDF")
                if kind == "text" and set(declared) != {"paper.txt"}:
                    raise PdfToolsError("Text output contract requires one text file")
                if kind == "render" and (len(declared) > self.max_pages or any(
                        not re.fullmatch(r"page-[0-9]+\.png", item) for item in declared)):
                    raise PdfToolsError("Rendered page output contract exceeded")
                if kind == "compile":
                    self._regular(outputs / "paper.pdf", header=b"%PDF-")
                if kind == "render":
                    for item in declared:
                        with (outputs / item).open("rb") as stream:
                            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                                raise PdfToolsError("Renderer output is not a PNG")
                # Return bytes from verified, bounded artifacts before removing staging.
                payload = {item: (outputs / item).read_bytes() for item in sorted(declared)}
                receipt.update(status="success", outputs=declared, worker_report=report)
                return payload
        except (OSError, ValueError, KeyError, TypeError, PdfToolsError) as error:
            receipt.update(status="failed", error=str(error))
            if isinstance(error, PdfToolsError):
                raise
            raise PdfToolsError("Isolated PDF operation failed: " + str(error)) from error
        finally:
            receipt["seconds"] = time.monotonic() - started

    @staticmethod
    def _publish(path, payload):
        path = Path(path)
        if path.is_symlink() or path.parent.is_symlink():
            raise PdfToolsError("PDF output destination must not be a symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def compile(self, work_dir: Path, pdf_path: Path, tex_name: str,
                timeout_seconds: float = 120) -> None:
        # The upstream compile_latex API passes a filename stem; direct callers
        # may instead provide the complete .tex name. Both identify the same file.
        if not Path(tex_name).suffix:
            tex_name += ".tex"
        payload = self._operation("compile", work_dir, tex_name=tex_name,
                                  timeout_seconds=timeout_seconds)
        self._publish(pdf_path, payload["paper.pdf"])

    def text(self, pdf_path: Path, *, timeout_seconds=None) -> str:
        return self._operation("text", pdf_path, timeout_seconds=timeout_seconds)["paper.txt"].decode("utf-8")

    def render(self, pdf_path: Path, output_dir: Path, *, timeout_seconds=None) -> list[Path]:
        payload = self._operation("render", pdf_path, timeout_seconds=timeout_seconds)
        output_dir = Path(output_dir)
        paths = []
        for name in sorted(payload, key=lambda value: int(value.removeprefix("page-").removesuffix(".png"))):
            path = output_dir / name
            self._publish(path, payload[name])
            paths.append(path)
        return paths
