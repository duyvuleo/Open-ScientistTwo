"""Fail-closed Slurm resource envelope for a task-owned rootless Podman runtime.

This module is also mounted as a standalone, standard-library-only container
entry guard. It never modifies Slurm, cgroups, user mappings, or system settings.
CPU and aggregate memory are enforced by a fresh Slurm step. Linux RLIMIT_NPROC
limits threads for the mapped real UID, potentially more strictly than a private
PID cgroup when another container shares that UID. That distinction is recorded.
Actual node negative controls are required before the adapter can be enabled.
"""

import argparse
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid


RECEIPT_PREFIX = "SCIENTISTTWO_ENFORCEMENT_V1 "
GPU_PATTERN = r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
PDF_NAME_PATTERN = r"scientisttwo-pdf-[0-9a-f]{20}"
NAME_PATTERN = rf"(?:scientisttwo-(?:native|timing)-[0-9a-f]{{18}}|{PDF_NAME_PATTERN})"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _encode(value):
    return base64.urlsafe_b64encode(_json_bytes(value)).decode()


def _decode(value):
    if len(value) > 1024 * 1024:
        raise ValueError("Oversized execution description")
    return json.loads(base64.urlsafe_b64decode(value))


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_json_bytes(value) + b"\n")
    os.replace(temporary, path)


def _process_identity(pid):
    """PID plus Linux start ticks avoids treating a reused PID as its old owner."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]
    except FileNotFoundError:
        return None
    except (OSError, IndexError, ValueError):
        return "unknown"


@contextmanager
def admission_ledger(config):
    """Short, process-safe transactions on the task-local admission ledger."""
    root = Path(config["receipt_dir"])
    descriptor = os.open(root / ".admission.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    deadline = time.monotonic() + 5
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValueError("Timed out acquiring resource admission ledger")
                time.sleep(0.05)
        path = root / "admission.json"
        if path.is_symlink():
            raise ValueError("Admission ledger cannot be a symlink")
        ledger = json.loads(path.read_bytes()) if path.exists() else {
            "schema_version": 1, "config_sha256": config["_sha256"], "active": {},
        }
        if ledger.get("schema_version") != 1 or not isinstance(ledger.get("active"), dict):
            raise ValueError("Invalid resource admission ledger")
        if ledger.get("config_sha256") != config["_sha256"]:
            if ledger["active"]:
                raise ValueError("Previous configuration still has charged resource leases")
            ledger["config_sha256"] = config["_sha256"]
        yield ledger
        _write_json(path, ledger)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _completion_proves_cleanup(config, name, token):
    path = Path(config["receipt_dir"]) / (name + ".json")
    if path.is_symlink():
        return False
    try:
        receipt = json.loads(path.read_bytes())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        receipt.get("admission_id") == token
        and receipt.get("config_sha256") == config["_sha256"]
        and receipt.get("cleanup", {}).get("confirmed_absent") is True
        and isinstance(receipt.get("completed_at"), (int, float))
    )


def acquire_admission(plan, config):
    """Admit bounded sibling steps before srun; never guess unknown work has ended."""
    description, name = plan["description"], plan["name"]
    with admission_ledger(config) as ledger:
        active = ledger["active"]
        for previous_name, lease in list(active.items()):
            observed = _process_identity(lease["owner_pid"])
            owner_dead = observed is None or (
                observed != "unknown" and observed != lease["owner_start_ticks"]
            )
            if owner_dead and _completion_proves_cleanup(
                config, previous_name, lease["admission_id"]
            ):
                del active[previous_name]
        if name in active:
            raise ValueError("Experiment already has an active or unresolved resource lease")
        totals = {
            "cpus": sum(lease["cpus"] for lease in active.values()),
            "memory_mb": sum(lease["memory_mb"] for lease in active.values()),
            "gpu_runs": sum(bool(lease["gpu_uuid"]) for lease in active.values()),
        }
        if totals["cpus"] + description["cpus"] > config["max_cpus"]:
            raise ValueError("Aggregate experiment CPU admission budget exhausted")
        available_memory = config["max_memory_mb"] - config["controller_memory_mb"]
        if totals["memory_mb"] + description["memory_mb"] > available_memory:
            raise ValueError("Aggregate experiment memory admission budget exhausted")
        if description["gpu_uuid"] and totals["gpu_runs"]:
            raise ValueError("Another GPU experiment is active or unresolved")
        token = uuid.uuid4().hex
        lease = {
            "admission_id": token, "owner_pid": os.getpid(),
            "owner_start_ticks": _process_identity(os.getpid()),
            "cpus": description["cpus"], "memory_mb": description["memory_mb"],
            "gpu_uuid": description["gpu_uuid"], "admitted_at": time.time(),
        }
        active[name] = lease
        plan["admission_id"] = token
        plan["admission_snapshot"] = {
            "previously_charged": totals,
            "controller_memory_mb": config["controller_memory_mb"],
            "max_cpus": config["max_cpus"], "max_memory_mb": config["max_memory_mb"],
        }
    return token


def release_admission(plan, config, *, never_launched=False):
    token, name = plan.get("admission_id"), plan["name"]
    if token is None:
        return
    if not never_launched and not _completion_proves_cleanup(config, name, token):
        return
    with admission_ledger(config) as ledger:
        lease = ledger["active"].get(name)
        if lease and lease["admission_id"] == token:
            del ledger["active"][name]


def verify_admission(plan, config):
    with admission_ledger(config) as ledger:
        lease = ledger["active"].get(plan["name"])
        if lease is None or lease["admission_id"] != plan.get("admission_id"):
            raise ValueError("Slurm worker has no matching admitted resource lease")
        if any(
            lease[key] != plan["description"][key]
            for key in ("cpus", "memory_mb", "gpu_uuid")
        ):
            raise ValueError("Slurm worker resource request differs from admitted lease")


def cpu_set(value):
    result = set()
    for item in value.strip().split(","):
        if not item:
            continue
        if not re.fullmatch(r"\d+(?:-\d+)?", item):
            raise ValueError("Invalid kernel CPU set")
        numbers = list(map(int, item.split("-")))
        start, end = numbers[0], numbers[-1]
        if end < start or end > 1_000_000:
            raise ValueError("Invalid kernel CPU range")
        result.update(range(start, end + 1))
    return result


def _unescape_mount(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def cgroup_locations(proc=Path("/proc/self")):
    """Resolve process memberships against its own mount namespace."""
    memberships = {}
    for line in (proc / "cgroup").read_text().splitlines():
        _, controllers, member = line.split(":", 2)
        for controller in controllers.split(",") if controllers else ["unified"]:
            memberships[controller] = member
    found = {}
    for line in (proc / "mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        left, right = before.split(), after.split()
        if right[0] not in {"cgroup", "cgroup2"}:
            continue
        root, mount = _unescape_mount(left[3]), Path(_unescape_mount(left[4]))
        controllers = ["unified"] if right[0] == "cgroup2" else right[2].split(",")
        for controller in controllers:
            member = memberships.get(controller)
            if member is None:
                continue
            if root == "/":
                relative = member.lstrip("/")
            elif member == root:
                relative = ""
            elif member.startswith(root.rstrip("/") + "/"):
                relative = member[len(root.rstrip("/")) + 1:]
            else:
                raise ValueError("Cgroup mount root does not contain process membership")
            if ".." in Path(relative).parts:
                raise ValueError("Invalid cgroup membership")
            found[controller] = (mount / relative, mount)
    return found


def _ancestors(leaf, root):
    if not leaf.is_relative_to(root):
        raise ValueError("Cgroup path escapes mount")
    while True:
        yield leaf
        if leaf == root:
            break
        leaf = leaf.parent


def swap_observation(memory, *, unified):
    """Report inherited swap policy without claiming or imposing a swap budget."""
    sources, finite, aggregate_unbounded = [], [], False
    hierarchy = memory[0] / "memory.use_hierarchy"
    hierarchical = unified or (hierarchy.is_file() and hierarchy.read_text().strip() == "1")
    for path in _ancestors(*memory) if hierarchical else [memory[0]]:
        file = path / ("memory.swap.max" if unified else "memory.memsw.limit_in_bytes")
        if not file.is_file():
            continue
        raw = file.read_text().strip()
        sources.append({"path": str(file), "value": raw})
        if raw != "max":
            try:
                value = int(raw)
            except ValueError:
                continue
            if 0 <= value < 2**60:
                finite.append(value)
    if not unified:
        statistics = memory[0] / "memory.stat"
        if hierarchical and statistics.is_file():
            entries = [line.split() for line in statistics.read_text().splitlines()
                       if line.startswith("hierarchical_memsw_limit ")]
            if len(entries) == 1 and len(entries[0]) == 2:
                raw = entries[0][1]
                sources.append({"path": str(statistics), "field": "hierarchical_memsw_limit",
                                "value": raw})
                try:
                    value = int(raw)
                except ValueError:
                    value = -1
                if 0 <= value < 2**60:
                    finite.append(value)
                aggregate_unbounded = value >= 2**60
    return {
        "scope": "swap_only" if unified else "memory_and_swap_combined",
        "status": "bounded" if finite else "unbounded" if aggregate_unbounded else "unknown",
        "limit_bytes": min(finite) if finite else None,
        "sources": sources,
        "policy": "inherited_not_enforced_by_adapter",
    }


def resource_observation(proc=Path("/proc/self")):
    """Read effective hierarchical limits, never OS FreeMem or process RSS."""
    locations = cgroup_locations(proc)
    unified = "unified" in locations
    memory = locations.get("unified" if unified else "memory")
    cpuset = locations.get("unified" if unified else "cpuset")
    if memory is None or cpuset is None:
        raise ValueError("Memory and CPU cgroup hierarchies must be visible")
    memory_limits, cpu_limits = [], []
    for path in _ancestors(*memory):
        file = path / ("memory.max" if unified else "memory.limit_in_bytes")
        if file.is_file():
            value = file.read_text().strip()
            if value != "max" and int(value) < 2**60:
                memory_limits.append({"path": str(file), "bytes": int(value)})
    if not unified:
        # An ancestor limit is only hierarchical when memory.use_hierarchy is on.
        hierarchy = memory[0] / "memory.use_hierarchy"
        if not hierarchy.is_file() or hierarchy.read_text().strip() != "1":
            memory_limits = [
                entry for entry in memory_limits
                if Path(entry["path"]).parent == memory[0]
            ]
        else:
            # A private cgroup mount may hide its limiting Slurm ancestor.
            # The kernel exposes the effective ancestor-aware hard limit in
            # memory.stat even when the visible leaf's own limit is unlimited.
            # This is a limit, never RSS, usage, FreeMem, or a workload report.
            statistics = memory[0] / "memory.stat"
            if statistics.is_file():
                entries = [
                    line.split() for line in statistics.read_text().splitlines()
                    if line.startswith("hierarchical_memory_limit ")
                ]
                if len(entries) > 1 or (entries and len(entries[0]) != 2):
                    raise ValueError("Malformed kernel hierarchical memory limit")
                if entries:
                    effective = int(entries[0][1])
                    if effective <= 0:
                        raise ValueError("Invalid kernel hierarchical memory limit")
                    if effective < 2**60:
                        memory_limits.append({
                            "path": str(statistics),
                            "field": "hierarchical_memory_limit",
                            "bytes": effective,
                        })
    for path in _ancestors(*cpuset):
        for filename in (
            ("cpuset.cpus.effective", "cpuset.cpus") if unified
            else ("cpuset.effective_cpus", "cpuset.cpus")
        ):
            file = path / filename
            if file.is_file():
                values = cpu_set(file.read_text())
                if values:
                    cpu_limits.append(values)
                    break
    if not memory_limits or not cpu_limits:
        raise ValueError("No finite hierarchical memory or CPU set")
    allowed = set.intersection(*cpu_limits)
    return {
        "cgroup_version": 2 if unified else 1,
        "memory_limit_bytes": min(item["bytes"] for item in memory_limits),
        "memory_budget_scope": "physical_ram",
        "memory_limits": memory_limits,
        "swap": swap_observation(memory, unified=unified),
        "cpuset": sorted(allowed),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cgroup_membership": (proc / "cgroup").read_text(),
    }


def verify_resources(observation, cpus, memory_mb):
    cpuset = set(observation["cpuset"])
    affinity = set(observation["affinity"])
    if not cpuset or len(cpuset) > cpus or not affinity or not affinity <= cpuset:
        raise ValueError("Effective kernel cpuset exceeds requested CPU limit")
    memory = observation["memory_limit_bytes"]
    if not 0 < memory <= memory_mb * 1024**2:
        raise ValueError("Hierarchical memory limit exceeds requested budget")


def security_observation(proc=Path("/proc/self")):
    fields = dict(line.split(":", 1) for line in (proc / "status").read_text().splitlines()
                  if ":" in line)
    return {
        "uid": os.getuid(),
        "euid": os.geteuid(),
        "cap_effective": int(fields["CapEff"].strip(), 16),
        "cap_permitted": int(fields["CapPrm"].strip(), 16),
        "no_new_privileges": int(fields["NoNewPrivs"].strip()),
        "nproc_soft": resource.getrlimit(resource.RLIMIT_NPROC)[0],
        "nproc_hard": resource.getrlimit(resource.RLIMIT_NPROC)[1],
        "uid_map": (proc / "uid_map").read_text(),
        "pid_limit_method": "rlimit_nproc_shared_uid",
    }


def verify_security(observation, pids):
    if observation["uid"] == 0 or observation["euid"] == 0:
        raise ValueError("RLIMIT_NPROC requires a non-root real and effective UID")
    if observation["cap_effective"] or observation["cap_permitted"]:
        raise ValueError("Workloads must have no permitted or effective capabilities")
    if observation["no_new_privileges"] != 1:
        raise ValueError("Workloads require no-new-privileges")
    if not 0 < observation["nproc_soft"] <= observation["nproc_hard"] <= pids:
        raise ValueError("Kernel process/thread hard limit exceeds requested budget")
    mappings = [list(map(int, line.split())) for line in observation["uid_map"].splitlines()]
    if not mappings or mappings == [[0, 0, 4294967295]]:
        raise ValueError("Private user namespace mapping is required")


def container_guard(description, command):
    """Emit trusted pre-exec evidence before candidate code can write any output."""
    if not command or command != description["command"]:
        raise ValueError("Original workload command differs from frozen guard description")
    observation = resource_observation()
    verify_resources(observation, description["cpus"], description["memory_mb"])
    security = security_observation()
    verify_security(security, description["pids"])
    pdf_limits = None
    if description.get("profile") == "paperorchestra-pdf-v1":
        pdf_limits = verify_pdf_resources(description, observation,
                                          resource.getrlimit(resource.RLIMIT_NOFILE),
                                          resource.getrlimit(resource.RLIMIT_FSIZE))
    expected = description.get("gpu_uuid")
    if expected:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != expected:
            raise ValueError("GPU UUID binding missing from the guarded environment")
        # CUDA driver identity only; framework/device memory allocation is not needed.
        import ctypes
        cuda = ctypes.CDLL("libcuda.so.1")
        count, device = ctypes.c_int(), ctypes.c_int()
        identifier = (ctypes.c_ubyte * 16)()
        for name, args in [
            ("cuInit", [0]), ("cuDeviceGetCount", [ctypes.byref(count)]),
            ("cuDeviceGet", [ctypes.byref(device), 0]),
            ("cuDeviceGetUuid", [ctypes.byref(identifier), device.value]),
        ]:
            if getattr(cuda, name)(*args) != 0:
                raise ValueError("CUDA identity guard failed: " + name)
        actual = "GPU-" + str(uuid.UUID(bytes=bytes(identifier)))
        if count.value != 1 or actual != expected:
            raise ValueError("Container CUDA visibility differs from the sole authorized GPU")
    receipt = {"schema_version": 1, "status": "verified", "resources": observation,
               "security": security, "gpu_uuid": expected,
               "description_sha256": hashlib.sha256(_json_bytes(description)).hexdigest()}
    if pdf_limits is not None:
        receipt["pdf_limits"] = pdf_limits
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True), file=sys.stderr, flush=True)
    os.execvp(command[0], command)


def load_config(path, expected_hash, *, check_identity=True, now=None, allow_expired_cleanup=False):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or sha256(path) != expected_hash:
        raise ValueError("Slurm adapter configuration identity mismatch")
    config = json.loads(path.read_bytes())
    required = {
        "schema_version", "job_id", "node", "uid", "podman", "srun", "python",
        "receipt_dir", "max_cpus", "max_memory_mb", "max_seconds", "expires_at",
        "gpu_uuid", "gpu_gres", "negative_control", "negative_control_sha256",
        "podman_sha256", "guard_sha256", "controller_memory_mb",
    }
    nvidia_fields = {
        "nvidia_container_runtime", "nvidia_container_runtime_sha256",
        "nvidia_config", "nvidia_config_sha256", "cdi_spec", "cdi_spec_sha256",
    }
    helper_fields = {
        "nvidia_cdi_hook", "nvidia_cdi_hook_sha256", "nvidia_ctk", "nvidia_ctk_sha256",
    }
    optional = set(config) & nvidia_fields
    if optional and optional != nvidia_fields:
        raise ValueError("NVIDIA runtime, configuration and CDI specification must all be pinned")
    for helper in ("nvidia_cdi_hook", "nvidia_ctk"):
        supplied = set(config) & {helper, helper + "_sha256"}
        if supplied and (not optional or supplied != {helper, helper + "_sha256"}):
            raise ValueError("NVIDIA helper paths and hashes require the pinned runtime")
    if set(config) - nvidia_fields - helper_fields != required or config["schema_version"] != 1:
        raise ValueError("Unknown or missing adapter configuration fields")
    if not re.fullmatch(r"[1-9][0-9]*", str(config["job_id"])):
        raise ValueError("Expected numeric Slurm allocation")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config["node"]):
        raise ValueError("Invalid Slurm node")
    for name in ("uid", "max_cpus", "max_memory_mb", "max_seconds", "controller_memory_mb"):
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] <= 0:
            raise ValueError("Invalid positive resource ceiling: " + name)
    if config["controller_memory_mb"] >= config["max_memory_mb"]:
        raise ValueError("Controller reservation consumes the whole memory budget")
    if not math.isfinite(config["expires_at"]) or (
        not allow_expired_cleanup and config["expires_at"] <= (now or time.time())
    ):
        raise ValueError("Approved shared-step window has expired")
    if config["gpu_uuid"] is not None and not re.fullmatch(GPU_PATTERN, config["gpu_uuid"]):
        raise ValueError("GPU requires its complete stable UUID")
    if config["gpu_uuid"] is None:
        if config["gpu_gres"] != "none":
            raise ValueError("CPU-only execution requires gpu_gres=none")
    elif (not isinstance(config["gpu_gres"], str)
          or not re.fullmatch(r"gpu:a100:[1-9][0-9]*", config["gpu_gres"])):
        raise ValueError("Expected explicitly authorized typed GPU declaration")
    for name in ("podman", "srun", "python", "negative_control"):
        target = Path(config[name])
        if not target.is_absolute() or not target.is_file():
            raise ValueError("Missing absolute adapter path: " + name)
    if sha256(config["podman"]) != config["podman_sha256"]:
        raise ValueError("Task-owned runtime wrapper changed")
    if sha256(__file__) != config["guard_sha256"]:
        raise ValueError("Trusted execution guard changed")
    if optional:
        for name in (
            "nvidia_container_runtime", "nvidia_config", "cdi_spec",
            *(helper for helper in ("nvidia_cdi_hook", "nvidia_ctk") if helper in config),
        ):
            target = Path(config[name])
            if not target.is_absolute() or not target.is_file() or target.is_symlink():
                raise ValueError("Missing real absolute NVIDIA runtime path: " + name)
            if sha256(target) != config[name + "_sha256"]:
                raise ValueError("NVIDIA runtime dependency changed: " + name)
    evidence = Path(config["negative_control"])
    if sha256(evidence) != config["negative_control_sha256"]:
        raise ValueError("Resource negative-control evidence changed")
    validation = json.loads(evidence.read_bytes())
    if any(validation.get(key) != config[key] for key in ("node", "uid", "podman_sha256")):
        raise ValueError("Negative controls belong to another runtime or worker")
    if any(validation.get(key) is not True for key in (
        "cpu_enforced", "memory_enforced", "nproc_enforced", "timeout_cleanup_verified"
    )):
        raise ValueError("Real resource negative controls have not all passed")
    receipts = Path(config["receipt_dir"])
    if not receipts.is_absolute() or receipts.is_symlink() or not receipts.is_dir():
        raise ValueError("Missing task-owned external receipt directory")
    if check_identity:
        if os.getuid() != config["uid"] or socket.gethostname().split(".")[0] != config["node"]:
            raise ValueError("Slurm execution adapter called outside its approved worker")
        if receipts.stat().st_uid != os.getuid() or receipts.stat().st_mode & 0o022:
            raise ValueError("External execution receipt directory must be private")
    config["_path"], config["_sha256"] = str(path), expected_hash
    return config


def verify_pdf_resources(description, observation, nofile, fsize):
    """A requested no-swap envelope must be demonstrable, never downgraded."""
    for actual, bound, name in ((nofile, 256, "open-file"),
                                (fsize, description["file_size_limit"], "file-size")):
        if not 0 < actual[0] <= actual[1] <= bound:
            raise ValueError("PDF kernel " + name + " limit exceeds its declared bound")
    if description["swap_policy"] == "deny":
        swap = observation.get("swap", {})
        bounded = swap.get("status") == "bounded" and type(swap.get("limit_bytes")) is int
        valid = bounded and (
            (swap.get("scope") == "swap_only" and swap["limit_bytes"] == 0)
            or (swap.get("scope") == "memory_and_swap_combined"
                and 0 <= swap["limit_bytes"] <= description["memory_mb"] * 1024**2)
        )
        if not valid:
            raise ValueError("PDF no-swap request is not enforced by the inherited kernel envelope")
    elif description["swap_policy"] != "inherit":
        raise ValueError("Unknown PDF swap policy")
    return {"status": "verified", "swap_policy": description["swap_policy"],
            "swap": observation.get("swap", {"status": "unknown"}),
            "nofile": list(nofile), "fsize": list(fsize)}


def _pdf_profile(values, command):
    """Admit the fixed PDF tool profile, not new options for generated workloads."""
    expected = {"--memory": ["768m"], "--cpus": ["1"], "--pids-limit": ["64"],
                "--shm-size": None, "--ipc": ["none"], "--entrypoint": ["python"],
                "--tmpfs": ["/tmp:rw,nosuid,nodev,noexec,size=256m,mode=1777"], "-w": ["/tmp"]}
    if command != ["/worker.py"] or any(values.get(k) != v for k, v in expected.items()):
        raise ValueError("PDF command or fixed execution profile changed")
    if values.get("--device") or values.get("-i"):
        raise ValueError("PDF operations cannot request GPU devices or interactive input")
    environment = values.get("-e", [])
    policy = next((item.removeprefix("SCIENTISTTWO_PDF_SWAP_POLICY=") for item in environment
                   if item.startswith("SCIENTISTTWO_PDF_SWAP_POLICY=")), None)
    if policy not in {"deny", "inherit"} or sorted(environment) != sorted([
        "HOME=/tmp", "PYTHONDONTWRITEBYTECODE=1", "SCIENTISTTWO_PDF_SWAP_POLICY=" + policy,
    ]):
        raise ValueError("PDF environment or explicit swap policy changed")
    if values.get("--memory-swap", []) != (["768m"] if policy == "deny" else []):
        raise ValueError("PDF memory-swap option differs from its explicit policy")
    limits = values.get("--ulimit", [])
    if len(limits) != 2 or limits[0] != "nofile=256:256":
        raise ValueError("PDF resource limits differ from the fixed profile")
    match = re.fullmatch(r"fsize=([1-9][0-9]*):\1", limits[1])
    if not match or int(match[1]) > 500_000_000:
        raise ValueError("PDF file-size limit is missing or oversized")
    mounts = {}
    for value in values.get("-v", []):
        pieces = value.split(":")
        if len(pieces) != 3 or pieces[1] in mounts:
            raise ValueError("PDF mounts must be distinct fixed destinations")
        mounts[pieces[1]] = (Path(pieces[0]), pieces[2])
    modes = {"/input": "ro", "/output": "rw", "/control": "ro", "/worker.py": "ro"}
    if set(mounts) != set(modes) or any(mounts[k][1] != v for k, v in modes.items()):
        raise ValueError("PDF mount scope differs from the fixed profile")
    directories = [mounts[k][0] for k in ("/input", "/output", "/control")]
    if (len({p.parent for p in directories}) != 1 or
            [p.name for p in directories] != ["input", "output", "control"] or
            not directories[0].parent.name.startswith("scientisttwo-pdf-")):
        raise ValueError("PDF mounts must share an owned staging directory")
    expected_worker = Path(__file__).resolve().parent / "integrations/paperorchestra/pdf_worker.py"
    if mounts["/worker.py"][0] != expected_worker:
        raise ValueError("PDF worker must be the frozen integration source")
    for path, _ in mounts.values():
        if not path.is_absolute() or path.resolve() != path or path.is_symlink():
            raise ValueError("PDF mount paths must be normalized without symlink traversal")
    if any(not path.is_dir() for path in directories) or not expected_worker.is_file():
        raise ValueError("PDF mounts must refer to existing staging directories and worker")
    return {"profile": "paperorchestra-pdf-v1", "swap_policy": policy,
            "memory_budget_scope": "physical_ram" if policy == "inherit" else "memory_and_swap",
            "file_size_limit": int(match[1])}


def parse_run(argv, config, *, now=None):
    """Accept only the runner's restricted Podman vocabulary; never a shell command."""
    if not argv or argv[0] != "run":
        raise ValueError("Expected a container run")
    switches = {"--rm", "-i", "--read-only"}
    valued = {
        "--pull", "--name", "--network", "--cap-drop", "--security-opt", "--memory",
        "--cpus", "--pids-limit", "--shm-size", "--user", "--tmpfs", "-w", "-e", "-v",
        "--device", "--label",
    }
    pdf_only = {"--memory-swap", "--ipc", "--ulimit", "--entrypoint"}
    values, retained, index = {}, [], 1
    while index < len(argv) and argv[index].startswith("-"):
        raw = argv[index]
        option, equal, inline = raw.partition("=")
        if option in switches and not equal:
            retained.append(raw)
            values.setdefault(option, []).append(True)
            index += 1
            continue
        if option not in valued | pdf_only:
            raise ValueError("Unsupported container option: " + option)
        if equal:
            value = inline
            index += 1
        else:
            if index + 1 >= len(argv):
                raise ValueError("Missing container option value")
            value = argv[index + 1]
            index += 2
        values.setdefault(option, []).append(value)
        removed = {"--memory", "--cpus", "--pids-limit", "--label", "--memory-swap"}
        if config.get("nvidia_container_runtime"):
            removed.add("--device")
        if option not in removed:
            retained.extend([option, value])
    if len(argv[index:]) < 2:
        raise ValueError("Container image and workload command required")
    image, command = argv[index], argv[index + 1:]
    if not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", image):
        raise ValueError("An immutable local image identity is required")

    def one(option):
        entries = values.get(option, [])
        if len(entries) != 1:
            raise ValueError("Expected exactly one " + option)
        return entries[0]

    for option, expected in [
        ("--rm", True), ("--read-only", True), ("--pull", "never"),
        ("--network", "none"), ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"), ("--user", "65534:65534"),
    ]:
        if one(option) != expected:
            raise ValueError("Restricted container option changed: " + option)
    name = one("--name")
    if not re.fullmatch(NAME_PATTERN, name):
        raise ValueError("Container name is outside the owned ScientistTwo scope")
    pdf = bool(re.fullmatch(PDF_NAME_PATTERN, name))
    if not pdf and set(values) & pdf_only:
        raise ValueError("PDF-only container options cannot change ordinary experiment execution")
    profile = _pdf_profile(values, command) if pdf else {}
    if pdf:
        command = ["python", *command]
    cpu_value = float(one("--cpus"))
    if not math.isfinite(cpu_value) or cpu_value != int(cpu_value):
        raise ValueError("Slurm adapter requires whole CPU allocations")
    cpus, pids = int(cpu_value), int(one("--pids-limit"))
    memory = one("--memory")
    if not re.fullmatch(r"[1-9][0-9]*m", memory):
        raise ValueError("Memory must be an explicit positive MiB value")
    memory_mb = int(memory[:-1])
    label = one("--label")
    if not label.startswith("scientisttwo.timeout-seconds="):
        raise ValueError("Explicit experiment wall-time label required")
    seconds = float(label.split("=", 1)[1])
    if not math.isfinite(seconds) or not 0 < seconds <= config["max_seconds"]:
        raise ValueError("Experiment wall-time exceeds approved ceiling")
    if not 1 <= cpus <= config["max_cpus"] or not 1 <= memory_mb <= config["max_memory_mb"]:
        raise ValueError("Experiment resources exceed approved ceiling")
    if not 16 <= pids <= 65536:
        raise ValueError("Invalid kernel process/thread limit")
    if math.ceil((seconds + 15) / 60) * 60 + 10 > config["expires_at"] - (now or time.time()):
        raise ValueError("Insufficient time remains in the approved shared-step window")
    devices = values.get("--device", [])
    gpu_uuid = config["gpu_uuid"] if devices else None
    if devices and (not gpu_uuid or devices != ["nvidia.com/gpu=" + gpu_uuid]):
        raise ValueError("Container device selection exceeds the single authorized GPU")
    for mount in values.get("-v", []):
        fields = mount.split(":")
        if len(fields) != 3 or fields[2] not in {"ro", "rw"}:
            raise ValueError("Expected explicit read-only or bounded-output mount")
        if fields[1].startswith("/scientisttwo-guard") or fields[1] in {"/sys", "/proc", "/dev"}:
            raise ValueError("Container mount masks a trusted enforcement path")
        host, receipt_dir = Path(fields[0]), Path(config["receipt_dir"])
        if not host.is_absolute() or ".." in host.parts:
            raise ValueError("Container mounts require absolute normalized host paths")
        if fields[2] == "rw" and (
            receipt_dir.is_relative_to(host) or host.is_relative_to(receipt_dir)
        ):
            raise ValueError("Candidate cannot mount external enforcement receipts writable")
    for environment in values.get("-e", []):
        if environment.split("=", 1)[0] in {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}:
            raise ValueError("GPU environment belongs to the trusted adapter")
    description = {"cpus": cpus, "memory_mb": memory_mb, "pids": pids,
                   "seconds": seconds, "gpu_uuid": gpu_uuid, "command": command, **profile}
    guard = str(Path(__file__).resolve())
    retained += ["--userns=keep-id:uid=65534,gid=65534",
                 f"--ulimit=nproc={pids}:{pids}",
                 "-v", f"{guard}:/scientisttwo-guard.py:ro"]
    if gpu_uuid:
        retained += ["-e", "CUDA_VISIBLE_DEVICES=" + gpu_uuid,
                     "-e", "NVIDIA_VISIBLE_DEVICES=" + (
                         "nvidia.com/gpu=" if config.get("nvidia_container_runtime") else ""
                     ) + gpu_uuid]
    else:
        retained += ["-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void"]
    runtime = [config["podman"]]
    if gpu_uuid and config.get("nvidia_container_runtime"):
        runtime += ["--runtime", config["nvidia_container_runtime"]]
    # The PDF profile explicitly pins Python as the entrypoint; avoid passing
    # a second interpreter as its script while retaining the trusted guard.
    container = [*runtime, "run", *retained, image, *([] if pdf else ["python"]),
                 "/scientisttwo-guard.py", "guard", _encode(description), *command]
    return {"name": name, "description": description, "container_argv": container}


def slurm_argv(plan, config):
    description = plan["description"]
    # Slurm's minute-rounded walltime is a fail-safe. The worker independently
    # enforces the exact seconds budget and removes only its own container.
    minutes = math.ceil((description["seconds"] + 15) / 60)
    return [
        config["srun"], "--jobid=" + str(config["job_id"]), "--overlap", "--exact",
        "--nodelist=" + config["node"], "--nodes=1", "--ntasks=1",
        "--cpus-per-task=" + str(description["cpus"]),
        "--mem=" + str(description["memory_mb"]) + "M",
        "--gres=" + (config["gpu_gres"] if description["gpu_uuid"] else "none"),
        "--time=" + str(minutes), "--immediate=10", "--kill-on-bad-exit=1",
        "--cpu-bind=threads", "--export=NONE",
        config["python"], str(Path(__file__).resolve()), "worker",
        "--config", config["_path"], "--config-sha256", config["_sha256"],
        "--", _encode(plan),
    ]


def _remove_container(config, name):
    return subprocess.run([config["podman"], "rm", "--force", name],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          timeout=15, check=False).returncode


def cleanup_container(config, name):
    removed = _remove_container(config, name)
    exists = subprocess.run([config["podman"], "container", "exists", name],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=10, check=False).returncode
    return {"remove_returncode": removed, "exists_returncode": exists,
            "confirmed_absent": exists == 1}


def worker(plan, config):
    """Run inside the new Slurm step, with external, candidate-inaccessible receipts."""
    verify_admission(plan, config)
    if os.environ.get("SLURM_JOB_ID") != str(config["job_id"]):
        raise ValueError("Worker is outside the expected Slurm allocation")
    step = os.environ.get("SLURM_STEP_ID", "")
    if not re.fullmatch(r"\d+", step):
        raise ValueError("Worker requires a real numbered Slurm step")
    description, name = plan["description"], plan["name"]
    if not re.fullmatch(NAME_PATTERN, name):
        raise ValueError("Invalid owned container name")
    if description["seconds"] + 15 > config["expires_at"] - time.time():
        raise ValueError("Experiment cannot finish within the approved window")
    observation = resource_observation()
    if f"job_{config['job_id']}/step_{step}" not in observation["cgroup_membership"]:
        raise ValueError("Worker process is not in its declared Slurm step cgroup")
    verify_resources(observation, description["cpus"], description["memory_mb"])
    if description["gpu_uuid"]:
        if not os.environ.get("SLURM_STEP_GPUS"):
            raise ValueError("GPU worker has no step-level assignment")
    receipt_path = Path(config["receipt_dir"]) / (name + ".json")
    receipt = {
        "schema_version": 1, "status": "starting", "config_sha256": config["_sha256"],
        "guard_sha256": sha256(__file__), "node": config["node"], "uid": os.getuid(),
        "job_id": config["job_id"], "step_id": step, "container_name": name,
        "description": description, "host_resources": observation,
        "started_at": time.time(), "guard": None,
        "admission_id": plan.get("admission_id"),
        "admission_snapshot": plan.get("admission_snapshot"),
    }
    _write_json(receipt_path, receipt)
    process, reader = None, None
    code = 125
    parser_errors = []
    old_handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError("Slurm worker interrupted by signal " + str(signum))

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        old_handlers[signum] = signal.signal(signum, interrupted)

    def forward_stderr(stream):
        prefix = RECEIPT_PREFIX.encode()
        pending = bytearray()
        awaiting = True
        while block := os.read(stream.fileno(), 8192):
            if awaiting:
                pending.extend(block)
                while b"\n" in pending:
                    line, _, tail = pending.partition(b"\n")
                    pending = bytearray(tail)
                    if line.startswith(prefix):
                        try:
                            evidence = json.loads(line[len(prefix):])
                            expected = hashlib.sha256(_json_bytes(description)).hexdigest()
                            if evidence.get("status") != "verified" or evidence.get("description_sha256") != expected:
                                raise ValueError("Guard receipt does not match workload")
                            receipt["guard"] = evidence
                            receipt["status"] = "running"
                            _write_json(receipt_path, receipt)
                            awaiting = False
                        except Exception as error:
                            parser_errors.append(str(error))
                    else:
                        sys.stderr.buffer.write(line + b"\n")
                        sys.stderr.buffer.flush()
                    if not awaiting:
                        break
                if len(pending) > 65536:
                    parser_errors.append("Guard receipt preamble exceeded limit")
                    awaiting = False
                if not awaiting and pending:
                    sys.stderr.buffer.write(pending)
                    sys.stderr.buffer.flush()
                    pending.clear()
            else:
                sys.stderr.buffer.write(block)
                sys.stderr.buffer.flush()
        if pending:
            sys.stderr.buffer.write(pending)
            sys.stderr.buffer.flush()

    try:
        environment = os.environ.copy()
        if description["gpu_uuid"] and config.get("nvidia_config"):
            environment["NVIDIA_CTK_CONFIG_FILE_PATH"] = config["nvidia_config"]
        process = subprocess.Popen(plan["container_argv"], stderr=subprocess.PIPE,
                                   env=environment, start_new_session=True)
        reader = threading.Thread(target=forward_stderr, args=(process.stderr,), daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=description["seconds"])
            receipt["timed_out"] = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            code = 124
            receipt["timed_out"] = True
    except (InterruptedError, OSError) as error:
        receipt["execution_error"] = str(error)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        try:
            receipt["cleanup"] = cleanup_container(config, name)
        except (OSError, subprocess.TimeoutExpired) as error:
            receipt["cleanup"] = {"confirmed_absent": False, "error": str(error)}
        if reader is not None:
            reader.join(timeout=5)
        if process is not None:
            process.stderr.close()
    receipt.update({"completed_at": time.time(), "returncode": code,
                    "status": "completed" if receipt["guard"] and not parser_errors
                    and receipt["cleanup"]["confirmed_absent"] else "enforcement_failed",
                    "parser_errors": parser_errors})
    _write_json(receipt_path, receipt)
    archive_path = receipt_path.with_name(
        name + "." + plan["admission_id"] + ".json"
    )
    descriptor = os.open(archive_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_json_bytes(receipt) + b"\n")
    release_admission(plan, config)
    print("SCIENTISTTWO_ENFORCEMENT_RECEIPT " + json.dumps({
        "path": str(archive_path), "sha256": sha256(archive_path),
        "status": receipt["status"],
    }, sort_keys=True), file=sys.stderr, flush=True)
    return code if receipt["status"] == "completed" else 125


def launch_admitted(plan, config):
    acquire_admission(plan, config)
    process = None
    try:
        process = subprocess.Popen(slurm_argv(plan, config))
        return process.wait()
    finally:
        # A terminated client does not prove a remote step/container ended.
        # The worker also releases its own lease after recording confirmed cleanup.
        release_admission(plan, config, never_launched=process is None)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "guard":
        return container_guard(_decode(argv[1]), argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["shim", "worker", "plan"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    if "--" not in argv:
        raise ValueError("Separate adapter options and runtime arguments with --")
    separator = argv.index("--")
    args = parser.parse_args(argv[:separator])
    arguments = argv[separator + 1:]
    cleanup = args.mode == "shim" and len(arguments) == 3 and arguments[:2] == ["rm", "--force"]
    config = load_config(args.config, args.config_sha256, allow_expired_cleanup=cleanup)
    if args.mode == "worker":
        return worker(_decode(arguments[0]), config)
    if arguments[:2] == ["image", "inspect"]:
        if args.mode == "plan":
            raise ValueError("Plan mode requires run")
        os.execv(config["podman"], [config["podman"], *arguments])
    if len(arguments) == 3 and arguments[:2] == ["rm", "--force"]:
        if not re.fullmatch(NAME_PATTERN, arguments[2]):
            raise ValueError("Refusing cleanup outside the task's container names")
        return _remove_container(config, arguments[2])
    plan = parse_run(arguments, config)
    launch = slurm_argv(plan, config)
    if args.mode == "plan":
        print(json.dumps({"plan": plan, "slurm_argv": launch}, indent=2))
        return 0
    return launch_admitted(plan, config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print("ScientistTwo execution enforcement rejected: " + str(error), file=sys.stderr)
        raise SystemExit(125) from error
