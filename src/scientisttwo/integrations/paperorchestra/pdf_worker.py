"""Trusted fixed PDF worker. Invoked only inside the pinned offline container."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def run(job, inputs, output, work, execute=subprocess.run):
    if job.get("schema_version") != "paperorchestra-pdf-job-v1":
        raise ValueError("Unsupported PDF job")
    operation = job["operation"]
    if operation not in {"compile", "text", "render"}:
        raise ValueError("Unsupported PDF operation")
    timeout = job["timeout_seconds"]
    if not 0 < timeout <= 300 or not 0 < job["max_pages"] <= 100:
        raise ValueError("PDF job bounds exceeded")
    started = time.monotonic()
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp",
           "TEXMFVAR": "/tmp/texmf-var", "TEXMFCONFIG": "/tmp/texmf-config",
           "openin_any": "p", "openout_any": "p", "shell_escape": "0",
           "SOURCE_DATE_EPOCH": "0", "FORCE_SOURCE_DATE": "1", "TZ": "UTC", "LANG": "C.UTF-8"}
    work.mkdir(parents=True, exist_ok=True)
    commands = []
    for name, metadata in job["inputs"].items():
        path = inputs / name
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(inputs.resolve())
                or path.stat().st_size != metadata["bytes"] or sha(path) != metadata["sha256"]):
            raise ValueError("PDF input snapshot verification failed")
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    def command(argv):
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("PDF operation deadline exhausted")
        result = execute(argv, cwd=work, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         timeout=remaining, check=False)
        commands.append({"argv": argv, "returncode": result.returncode})
        if result.returncode:
            # Logs are capped; generated source cannot cause unlimited response text.
            raise RuntimeError("PDF tool failed: " + (result.stderr + result.stdout)[-8192:].decode("utf-8", errors="replace"))
        return result

    if operation == "compile":
        main = Path(job["tex_name"])
        if main.is_absolute() or ".." in main.parts or main.suffix != ".tex":
            raise ValueError("Unsafe TeX main source")
        tex_command = ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", str(main)]
        command(tex_command)
        stem = main.stem
        auxiliary = work / (stem + ".aux")
        if auxiliary.is_file() and b"\\bibdata{" in auxiliary.read_bytes():
            command(["bibtex", stem])
        command(tex_command)
        command(tex_command)
        pdf = work / (stem + ".pdf")
        if pdf.is_symlink() or not pdf.is_file() or pdf.stat().st_size > job["max_output_bytes"]:
            raise ValueError("Compiler did not produce a bounded regular PDF")
        shutil.copyfile(pdf, output / "paper.pdf")
    elif operation == "text":
        command(["pdftotext", "-enc", "UTF-8", "/input/paper.pdf", "/output/paper.txt"])
    else:
        info = command(["pdfinfo", "/input/paper.pdf"]).stdout.decode("utf-8", errors="strict")
        match = re.search(r"^Pages:\s+(\d+)\s*$", info, re.MULTILINE)
        if not match or not 0 < int(match[1]) <= job["max_pages"]:
            raise ValueError("PDF page count exceeds the rendering limit")
        command(["pdftoppm", "-png", "-r", "96", "-scale-to", "1600",
                 "/input/paper.pdf", "/output/page"])
    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("PDF worker produced a nonregular output")
        artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    if sum(item["bytes"] for item in artifacts.values()) > job["max_output_bytes"]:
        raise ValueError("PDF output byte quota exceeded")
    report = {"status": "success", "operation": operation, "commands": commands,
              "outputs": artifacts, "seconds": time.monotonic() - started}
    (output / "result.json").write_text(json.dumps(report, sort_keys=True))
    return report


if __name__ == "__main__":
    try:
        job = json.loads(Path("/control/job.json").read_text())
        run(job, Path("/input"), Path("/output"), Path("/tmp/work"))
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=__import__("sys").stderr)
        os._exit(1)
