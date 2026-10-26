"""Fixed offline TeX worker, executed only inside the declared Linux container.

The controller mounts this module as /bridge/bridge.py. No candidate Python or
model-supplied command is imported or executed. The TeX package/image closure is
an explicitly built and pinned runtime prerequisite.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

COMMANDS = [
    ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ["bibtex", "main"],
    ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
]
MAX_PDF_BYTES = 50_000_000


def _sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def run(job, work, execute=subprocess.run):
    """The injected executor exists only to unit-test the worker without TeX."""
    work = Path(work)
    if job.get("schema_version") != "manuscript-compile-job-v1" or job.get("commands") != COMMANDS:
        raise ValueError("Unsupported manuscript compiler job")
    timeout = job["timeout_seconds"]
    if type(timeout) not in (int, float) or not 0 < timeout <= 120:
        raise ValueError("Compiler timeout is outside its bound")
    expected = job["files"]
    for name, value in expected.items():
        path = work / name
        if (path.is_symlink() or not path.resolve().is_relative_to(work.resolve())
                or not path.is_file() or path.stat().st_size != value["bytes"]
                or _sha(path) != value["sha256"]):
            raise ValueError("Compiler input differs from frozen project")
    started = time.monotonic()
    attempts = []
    status, reason, timed_out = "success", "Fixed TeX and bibliography passes completed", False
    # TeX's restrictive file settings complement container mounts and read-only root.
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp",
        "TEXMFVAR": "/tmp/texmf-var", "TEXMFCONFIG": "/tmp/texmf-config",
        "openin_any": "p", "openout_any": "p", "shell_escape": "0",
        "SOURCE_DATE_EPOCH": "0", "FORCE_SOURCE_DATE": "1",
        "TZ": "UTC", "LANG": "C.UTF-8",
    }
    for index, command in enumerate(COMMANDS):
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            status, reason, timed_out = "failed", "Compiler deadline exhausted", True
            break
        output = work / f"compiler-pass-{index}.log"
        try:
            with output.open("xb") as stream:
                result = execute(command, cwd=work, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, timeout=remaining, check=False)
            attempts.append({"command": command, "returncode": result.returncode})
            if result.returncode:
                status, reason = "failed", "A fixed TeX/BibTeX pass failed"
                break
        except subprocess.TimeoutExpired:
            status, reason, timed_out = "failed", "Compiler pass timed out", True
            break
        except FileNotFoundError:
            status, reason = "unavailable", "Pinned image lacks the required TeX/BibTeX executable"
            break
    pdf = work / "main.pdf"
    pdf_hash, pdf_size = None, None
    if status == "success":
        if pdf.is_symlink() or not pdf.is_file() or not 5 <= pdf.stat().st_size <= MAX_PDF_BYTES:
            status, reason = "failed", "Compiler did not produce a bounded regular PDF"
        else:
            with pdf.open("rb") as stream:
                if stream.read(5) != b"%PDF-":
                    status, reason = "failed", "Compiler output is not a PDF"
            if status == "success":
                pdf_hash, pdf_size = _sha(pdf), pdf.stat().st_size
    # Compilation may create auxiliary files but may not edit its frozen inputs.
    for name, value in expected.items():
        if _sha(work / name) != value["sha256"]:
            status, reason = "failed", "Compiler modified a frozen input"
            break
    result = {"status": status, "reason": reason, "timed_out": timed_out,
              "commands": attempts, "pdf_sha256": pdf_hash, "pdf_bytes": pdf_size,
              "seconds": time.monotonic() - started,
              "project_manifest_sha256": job["project_manifest_sha256"]}
    (work / "compile-result.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    declaration = json.loads(Path("/control/job.json").read_text())
    outcome = run(declaration, Path("/work"))
    print(json.dumps({"status": outcome["status"], "reason": outcome["reason"]}))
    os._exit(0 if outcome["status"] == "success" else 1)
