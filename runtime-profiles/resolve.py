"""Resolve reproducible transitive locks explicitly on the Linux worker.

This is a preparation operation (network access), never an experiment operation.
No lock/image is marked verified by this script; the actual probe is mandatory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o644)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=["cpu-tabular", "torch-cuda", "tensorflow-cuda", "jax-clrs"])
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--platform", default="x86_64-manylinux_2_28")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / args.profile
    version = {"jax-clrs": "3.10", "cpu-tabular": "3.12"}.get(args.profile, "3.11")
    command = [args.uv, "pip", "compile", str(root / "requirements.in"),
               "--python-version", version, "--python-platform", args.platform, "--generate-hashes",
               "--output-file", str(root / "requirements.lock")]
    if args.profile == "jax-clrs":
        command += ["--find-links", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html", "--emit-find-links"]
    else:
        command += ["--no-build"]
    subprocess.run(command, check=True)
    manifest = {"schema_version": 1, "status": "resolved_not_runtime_validated",
                "resolver_host": platform.platform(), "target_platform": args.platform, "python": version,
                "requested_sha256": hashlib.sha256((root / "requirements.in").read_bytes()).hexdigest(),
                "lock_sha256": hashlib.sha256((root / "requirements.lock").read_bytes()).hexdigest(),
                "package_versions": {match.group(1): match.group(2) for match in re.finditer(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", (root / "requirements.lock").read_text(), re.MULTILINE)},
                "build_status": "unverified", "gpu_probe_status": "unverified"}
    manifest["package_count"] = len(manifest["package_versions"])
    atomic_json(root / "resolution.json", manifest)
    packaged = root.parents[1] / "src/scientisttwo/runtime_locks.json"
    receipts = json.loads(packaged.read_text())
    receipts[args.profile] = manifest
    atomic_json(packaged, receipts)


if __name__ == "__main__":
    main()
