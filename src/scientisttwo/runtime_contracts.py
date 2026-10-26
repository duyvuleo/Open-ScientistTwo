"""Frozen runtime assets and resource identities shared by task families."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import time
import tomllib
from typing import Literal

from pydantic import Field, model_validator

from .models import Record
from .benchmarks.files import file_hash, verify_tree


EXECUTION_CONFIG_ENV = "SCIENTISTTWO_EXECUTION_CONFIG"
EXECUTION_CONFIG_SHA_ENV = "SCIENTISTTWO_EXECUTION_CONFIG_SHA256"
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_CONTAINER_NAME = re.compile(r"scientisttwo-(?:native|timing)-[a-f0-9]{18}\Z")


def effective_execution_timeout(requested, identity=None, *, now=None):
    """Bound actual work by the approved window, including Slurm's rounding.

    The worker reserves 15 seconds for cleanup and rounds to whole minutes;
    admission reserves another 10 seconds. Leave five more seconds for handing
    the request to the shim instead of admitting at an exact clock boundary.
    """
    requested = float(requested)
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError("Execution timeout must be finite and positive")
    if identity is None:
        return requested
    remaining = identity["allocation"]["expires_at"] - (time.time() if now is None else now)
    cap = math.floor((remaining - 15) / 60) * 60 - 15
    if cap <= 0:
        raise ValueError("Insufficient time remains in the approved shared-step window")
    return min(requested, float(cap))


def _regular_file(path, *, max_bytes=None):
    path = Path(path)
    if not path.is_absolute() or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Execution evidence requires real absolute paths without symlinks")
    value = path.stat()
    if not stat.S_ISREG(value.st_mode) or (max_bytes is not None and value.st_size > max_bytes):
        raise ValueError("Execution evidence file is not regular or exceeds its byte limit")
    return path


def execution_identity(runtime):
    """Hash an optional external execution policy without importing or invoking it.

    The execution shim validates its live Slurm context and admission controls.
    Here only frozen bytes are checked, so historical audit remains possible
    after an approved allocation expires.
    """
    config_path = os.environ.get(EXECUTION_CONFIG_ENV)
    expected = os.environ.get(EXECUTION_CONFIG_SHA_ENV)
    if config_path is None and expected is None:
        return None
    if not config_path or not expected or not _SHA256.fullmatch(expected):
        raise ValueError("Execution config path and complete SHA256 must be declared together")
    if runtime != "podman":
        raise ValueError("The declared Slurm execution policy requires the Podman adapter")
    path = _regular_file(config_path, max_bytes=1024 * 1024)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Execution configuration changed from its declared SHA256")
    config = json.loads(raw)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported execution configuration schema")
    bindings = {}
    for name, target in (
        ("podman", config.get("podman")),
        ("guard", Path(__file__).with_name("slurm_execution.py")),
        ("negative_control", config.get("negative_control")),
        ("nvidia_container_runtime", config.get("nvidia_container_runtime")),
        ("nvidia_config", config.get("nvidia_config")),
        ("cdi_spec", config.get("cdi_spec")),
        ("nvidia_ctk", config.get("nvidia_ctk")),
        ("nvidia_cdi_hook", config.get("nvidia_cdi_hook")),
    ):
        digest = config.get(name + "_sha256")
        if target is None and digest is None and name.startswith(("nvidia_", "cdi_")):
            continue
        if not target or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("Execution policy dependency is not pinned: " + name)
        target = _regular_file(target)
        if file_hash(target) != digest:
            raise ValueError("Execution policy dependency changed: " + name)
        bindings[name] = {"path": str(target), "sha256": digest}
    if "cdi_spec" in bindings:
        cdi = json.loads(_regular_file(config["cdi_spec"], max_bytes=1024 * 1024).read_bytes())
        declared_hooks = [*cdi.get("containerEdits", {}).get("hooks", [])]
        for device in cdi.get("devices", []):
            declared_hooks.extend(device.get("containerEdits", {}).get("hooks", []))
        pinned_binaries = {entry["path"] for key, entry in bindings.items()
                           if key in {"nvidia_ctk", "nvidia_cdi_hook"}}
        if any(hook.get("path") not in pinned_binaries for hook in declared_hooks):
            raise ValueError("Every CDI hook executable must be explicitly hash-pinned in the execution policy")
        nvidia = tomllib.loads(
            _regular_file(config["nvidia_config"], max_bytes=1024 * 1024).read_text())
        ctk = nvidia.get("nvidia-ctk", {}).get("path")
        if ctk is not None and ctk not in pinned_binaries:
            raise ValueError("Configured NVIDIA CTK executable must be explicitly hash-pinned")
    wrapper = shutil.which(runtime)
    if wrapper is None:
        raise ValueError("Configured execution runtime is not on PATH")
    wrapper = _regular_file(Path(wrapper).absolute())
    binding_fields = ("job_id", "node", "uid", "gpu_uuid", "gpu_gres", "max_cpus",
                      "max_memory_mb", "controller_memory_mb", "max_seconds", "expires_at")
    if any(name not in config for name in binding_fields):
        raise ValueError("Execution configuration omits its allocation or resource scope")
    receipts = Path(config.get("receipt_dir", ""))
    if (not receipts.is_absolute() or not receipts.is_dir()
            or any(parent.is_symlink() for parent in (receipts, *receipts.parents))):
        raise ValueError("Execution receipt directory must be a real absolute directory")
    metadata = receipts.stat()
    if metadata.st_uid != config["uid"] or metadata.st_mode & 0o022:
        raise ValueError("Execution receipt directory must be private to its declared account")
    return {
        "schema_version": 1, "kind": "slurm-step-resource-envelope",
        "config_path": str(path), "config_sha256": expected,
        "runtime_wrapper": {"path": str(wrapper), "sha256": file_hash(wrapper)},
        "dependencies": bindings, "allocation": {name: config[name] for name in binding_fields},
        "receipt_dir": str(receipts),
    }


def seal_execution_receipt(store, name, identity, *, started_at, expected_gpu_uuid=None,
                           wait_seconds=5):
    """Retain fresh trusted host evidence; candidate log markers are not trusted."""
    if identity is None:
        return None
    if (not _CONTAINER_NAME.fullmatch(name) or not math.isfinite(started_at)
            or not 0 <= wait_seconds <= 5):
        raise ValueError("Invalid execution receipt identity or wait bound")
    if execution_identity("podman") != identity:
        raise ValueError("Execution policy changed while the stage was running")
    path = Path(identity["receipt_dir"]) / (name + ".json")
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            raw = _regular_file(path, max_bytes=1024 * 1024).read_bytes()
            receipt = json.loads(raw)
        except FileNotFoundError:
            receipt = {}
        started, completed = receipt.get("started_at"), receipt.get("completed_at")
        fresh = (isinstance(started, (int, float)) and math.isfinite(started)
                 and started >= started_at and isinstance(completed, (int, float))
                 and math.isfinite(completed) and completed >= started)
        if fresh:
            break
        if time.monotonic() >= deadline:
            raise ValueError("Fresh completed execution enforcement receipt is unavailable")
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    allocation = identity["allocation"]
    if (receipt.get("schema_version") != 1 or receipt.get("container_name") != name
            or receipt.get("config_sha256") != identity["config_sha256"]
            or receipt.get("guard_sha256") != identity["dependencies"]["guard"]["sha256"]
            or any(receipt.get(key) != allocation[key] for key in ("job_id", "node", "uid"))
            or not re.fullmatch(r"[0-9a-f]{32}", str(receipt.get("admission_id", "")))):
        raise ValueError("Execution receipt does not match the frozen policy and container")
    if path.stat().st_uid != allocation["uid"] or path.stat().st_mode & 0o022:
        raise ValueError("Execution receipt is writable by another account")
    ref = store.put(raw, "json", "execution-enforcement")
    guard = receipt.get("guard") or {}
    description = receipt.get("description") or {}
    description_hash = hashlib.sha256(
        json.dumps(description, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (receipt.get("status") != "completed" or guard.get("status") != "verified"
            or guard.get("description_sha256") != description_hash
            or receipt.get("parser_errors") or not receipt.get("cleanup", {}).get("confirmed_absent")):
        raise ValueError("Execution enforcement or cleanup failed; retained evidence: " + ref)
    # A resident timing worker can acknowledge its final request just before the
    # host deadline kills it. Correct output and successful cleanup do not turn
    # that budget failure into a successful execution. Intentional timing-worker
    # shutdown (normally code 137 with timed_out=False) remains admissible.
    if receipt.get("timed_out") is True:
        raise ValueError("Execution exceeded its trusted host timeout; retained evidence: " + ref)
    if (description.get("gpu_uuid") != expected_gpu_uuid
            or guard.get("gpu_uuid") != expected_gpu_uuid
            or (expected_gpu_uuid is not None and expected_gpu_uuid != allocation["gpu_uuid"])):
        raise ValueError("Execution receipt GPU differs from the requested stage; retained evidence: " + ref)
    return {"status": "verified", "evidence_ref": ref,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "config_sha256": identity["config_sha256"],
            "container_name": name, "step_id": receipt.get("step_id"),
            "admission_id": receipt["admission_id"], "gpu_uuid": expected_gpu_uuid,
            "returncode": receipt.get("returncode"), "timed_out": receipt.get("timed_out")}


class RuntimeAsset(Record):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    path: Path
    role: Literal["model", "tokenizer", "runtime", "public_data"]
    manifest: dict
    stages: list[Literal["execute", "inference", "score"]] = ["execute", "inference"]

    @model_validator(mode="after")
    def valid(self):
        if not self.path.is_absolute() or not self.manifest:
            raise ValueError("Runtime assets require absolute paths and a frozen manifest")
        if not self.stages or len(self.stages) != len(set(self.stages)):
            raise ValueError("Declare distinct asset stages")
        return self


def asset_mounts(spec, stage, *, protected=()):
    mounts = []
    names = set()
    for asset in getattr(spec, "assets", []):
        if asset.name in names:
            raise ValueError("Duplicate runtime asset name")
        names.add(asset.name)
        if stage not in asset.stages:
            continue
        root = asset.path.resolve()
        for value in protected:
            if value is None:
                continue
            other = Path(value).resolve()
            if root.is_relative_to(other) or other.is_relative_to(root):
                raise ValueError("Runtime asset overlaps a protected input tree")
        verify_tree(asset.path, asset.manifest)
        mounts.append((asset.path, "/assets/" + asset.name, "ro"))
    return mounts


def runtime_identity(spec, images, hardware=None):
    result = {
        "images": images,
        "resources": spec.resources.model_dump(mode="json"),
        "runtime": spec.runtime,
        "assets": [a.model_dump(mode="json", exclude={"path"}) for a in getattr(spec, "assets", [])],
        "hardware": hardware or {"status": "unverified"},
    }
    enforcement = execution_identity(spec.runtime)
    if enforcement is not None:
        result["execution_enforcement"] = enforcement
    return result
