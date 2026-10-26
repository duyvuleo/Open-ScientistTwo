"""Plan or run sequential offline framework smoke and separate workload calibration.

Only --mode run accesses the container runtime/GPU. It requires an existing Slurm
allocation on its compute node; this tool never submits jobs or downloads assets.
Image manifest: {"torch-cuda": "sha256:<64 hex>", ...}.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import threading
import time
from types import SimpleNamespace

from scientisttwo.benchmarks.files import FORBIDDEN
from scientisttwo.benchmarks.models import NativeResources
from scientisttwo.benchmarks.runner import NativeRunner
from scientisttwo.process import run_process
from scientisttwo.runtime_profiles import runtime_profile, runtime_profiles, probe_source, slurm_gpu_allocation
from scientisttwo.store import Store, canonical, file_digest


def image_manifest(path):
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Image manifest must map profile names to immutable image IDs")
    known = {p.name for p in runtime_profiles()}
    for name, image in values.items():
        if name not in known or not isinstance(image, str) or not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image):
            raise ValueError("Image manifest contains an unknown profile or mutable/unresolved image")
    return values


def verify_worker(gpu_index, memory_mb, cpus, *, gpu_uuid=None):
    if platform.system() != "Linux":
        raise ValueError("Run on the allocated Linux worker")
    node = os.environ.get("SLURMD_NODENAME", "")
    if not node or node.split(".")[0] != platform.node().split(".")[0]:
        raise ValueError("Run inside an existing Slurm step on its allocated compute node")
    allocation = slurm_gpu_allocation(gpu_index=gpu_index, gpu_uuid=gpu_uuid)
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if allocated_cpus < cpus:
        raise ValueError("Slurm CPUs per task are missing or below requested container CPUs")
    memory = os.environ.get("SLURM_MEM_PER_NODE")
    if memory is None and os.environ.get("SLURM_MEM_PER_CPU"):
        memory = str(int(os.environ["SLURM_MEM_PER_CPU"]) * allocated_cpus)
    if memory is None or (int(memory) and int(memory) < memory_mb):
        raise ValueError("Slurm host memory is missing or below the requested container limit")
    return allocation


def _query(arguments):
    process = subprocess.run(["nvidia-smi", *arguments], check=True, capture_output=True,
                             text=True, timeout=10)
    return list(csv.reader(process.stdout.splitlines(), skipinitialspace=True))


def gpu_sample(device):
    rows = _query(["--id=" + device, "--query-gpu=uuid,name,memory.total,memory.used,memory.free",
                   "--format=csv,noheader,nounits"])
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError("Expected one allocated GPU")
    uuid, name, total, used, free = [v.strip() for v in rows[0]]
    if device.startswith("GPU-") and uuid != device:
        raise ValueError("GPU monitor returned a different UUID from the selected device")
    processes = []
    for row in _query(["--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"]):
        if len(row) == 3 and row[0].strip() == uuid:
            processes.append({"pid": int(row[1]), "used_mb": int(row[2])})
    return {"uuid": uuid, "name": name, "total_mb": int(total), "used_mb": int(used),
            "free_mb": int(free), "processes": processes}


class GpuMonitor:
    def __init__(self, device, interval=0.2):
        self.device, self.interval = device, interval
        self.samples, self.errors = [], []
        self.stopped = threading.Event()
        self.thread = None

    def __enter__(self):
        self.started = time.monotonic()
        baseline = gpu_sample(self.device)
        if baseline["processes"]:
            raise ValueError("Allocated GPU has active compute processes; refusing to disturb or mix measurements")
        self.samples.append({"elapsed_seconds": 0.0, **baseline})

        def sample():
            while not self.stopped.wait(self.interval):
                try:
                    self.samples.append({"elapsed_seconds": time.monotonic() - self.started,
                                         **gpu_sample(self.device)})
                except Exception as exc:
                    self.errors.append(type(exc).__name__)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join()

    def report(self):
        samples = self.samples
        observed = [row for row in samples if row["processes"]]
        process_peaks = {}
        for row in observed:
            for process in row["processes"]:
                pid = str(process["pid"])
                process_peaks[pid] = max(process_peaks.get(pid, 0), process["used_mb"])
        return {"schema_version": 1, "status": "measured" if observed and not self.errors else "unverified",
                "scope": "allocated physical GPU from before container startup through process exit",
                "sample_interval_seconds": self.interval,
                "sample_count": len(samples), "errors": self.errors,
                "device_peak_used_mb": max((row["used_mb"] for row in samples), default=None),
                "minimum_free_mb": min((row["free_mb"] for row in samples), default=None),
                "sampled_compute_process_peak_mb": max((sum(p["used_mb"] for p in row["processes"]) for row in observed), default=None),
                "per_pid_sampled_peak_mb": process_peaks,
                "peak_semantics": "sampled lower bounds; no exact allocator-peak or full-task-fit claim",
                "samples": samples}


class ProbeRunner(NativeRunner):
    """Reuse the restricted generic launch path without a benchmark task lookup."""
    def __init__(self, root, runtime, resources):
        self.store = Store(root)
        self.spec = SimpleNamespace(runtime=runtime, resources=resources)
        self.prefix = [runtime]
        self.executor = run_process


def calibration_spec(path, workspace):
    value = json.loads(path.read_text())
    required = {"name", "profile", "task_id", "protocol_id", "source_dir", "command", "resources"}
    if not isinstance(value, dict) or set(value) - (required | {"assets", "expected_phases", "environment"}) or not required <= set(value):
        raise ValueError("Calibration requires name/profile/task_id/protocol_id/source_dir/command/resources")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value["name"]):
        raise ValueError("Invalid calibration name")
    runtime_profile(value["profile"])
    for key in ["task_id", "protocol_id"]:
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError("Task and protocol identities are required")
    if not isinstance(value["command"], list) or not value["command"] or not all(isinstance(x, str) and x for x in value["command"]):
        raise ValueError("Calibration command must be an argv list")
    for source in [value["source_dir"], *value.get("assets", {}).values()]:
        source = Path(source)
        if not source.is_absolute() or not source.resolve().is_relative_to(workspace.resolve()) or not source.is_dir() or source.is_symlink():
            raise ValueError("Calibration inputs must be existing real directories inside the worker workspace")
    for name in value.get("assets", {}):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Invalid asset mount name")
    allowed = {"XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "TF_FORCE_GPU_ALLOW_GROWTH", "PYTORCH_CUDA_ALLOC_CONF"}
    if any(k not in allowed or not isinstance(v, str) for k, v in value.get("environment", {}).items()):
        raise ValueError("Only declared framework allocator settings may override the environment")
    phases = value.get("expected_phases", ["loading", "warmup", "measurement"])
    if not isinstance(phases, list) or not phases or not all(isinstance(p, str) and p for p in phases):
        raise ValueError("Expected phases must be nonempty strings")
    value["expected_phases"] = phases
    resources = NativeResources.model_validate(value["resources"])
    if resources.seconds > 7200:
        raise ValueError("Calibration is bounded to two hours per workload")
    return value


def input_manifest(root):
    """Hash all mounted bytes, including files ignored by normal source snapshots."""
    result = {}
    total = 0
    forbidden = FORBIDDEN | {".ssh", ".aws", ".codex", ".netrc", ".kaggle"}
    if any(part.lower() in forbidden for part in root.parts):
        raise ValueError("Credential-like path must not enter a calibration mount")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.lower() in forbidden or part.startswith(".env.") for part in relative.parts):
            raise ValueError("Credential-like path must not enter a calibration mount")
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("Calibration input links and special files are forbidden")
        if stat.S_ISREG(mode):
            size = path.stat().st_size
            total += size
            if total > 256 * 1024**3 or len(result) >= 100000:
                raise ValueError("Calibration input exceeds bounded manifest limits")
            result[relative.as_posix()] = {"size": size, "sha256": file_digest(path)}
    return result


def plan(images, profiles, calibration=None):
    entries = []
    for name in profiles:
        profile = runtime_profile(name)
        entries.append({"name": name, "kind": "tiny_framework_smoke", "profile": name,
                        "image": images.get(name), "lock_sha256": profile.dependency_lock_sha256,
                        "status": "planned" if name in images else "blocked_missing_image_digest",
                        "task_peak_memory": "unverified", "task_readiness": "unverified"})
    if calibration:
        entries.append({"name": calibration["name"], "kind": "declared_workload_calibration",
                        "profile": calibration["profile"], "image": images.get(calibration["profile"]),
                        "task_id": calibration["task_id"], "protocol_id": calibration["protocol_id"],
                        "status": "planned" if calibration["profile"] in images else "blocked_missing_image_digest",
                        "task_readiness": "unverified"})
    return {"schema_version": 1, "mode": "plan", "live_model_calls": 0,
            "sequential": True, "entries": entries,
            "full_suite_readiness": "unverified"}


def phase_evidence(path, expected):
    if not path.is_file():
        return {"status": "unverified", "missing": expected, "events": []}
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Phase evidence exceeds one MiB")
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(not isinstance(row, dict) or not isinstance(row.get("phase"), str) for row in events):
        raise ValueError("Phase events must be JSON objects with phase names")
    missing = [name for name in expected if name not in {event["phase"] for event in events}]
    return {"status": "reported_by_workload" if not missing else "unverified", "missing": missing, "events": events}


def calibration_memory_evidence(path, gpu_report, cuda):
    """Require observed process/device usage before declaring a calibration complete."""
    failures = []

    def number(value, *, positive=False):
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and (value > 0 if positive else value >= 0))

    try:
        if not path.is_file() or path.stat().st_size > 64 * 1024**2:
            raise ValueError("missing or oversized memory report")
        memory = json.loads(path.read_text())
        samples = memory.get("samples", [])
        if (memory.get("schema_version") != 1 or memory.get("returncode") != 0
                or not isinstance(samples, list) or not samples
                or memory.get("sample_count") != len(samples)
                or not number(memory.get("elapsed_seconds"), positive=True)
                or not number(memory.get("sample_interval_seconds"), positive=True)
                or not number(memory.get("direct_children_high_water_rss_bytes"), positive=True)):
            raise ValueError("incomplete process memory measurements")
        for sample in samples:
            if (not isinstance(sample, dict) or not number(sample.get("rss_bytes"))
                    or not number(sample.get("elapsed_seconds"))
                    or not number(sample.get("process_count"))):
                raise ValueError("invalid memory sample")
        peak = max(sample["rss_bytes"] for sample in samples)
        if peak <= 0 or peak != memory.get("sampled_process_tree_peak_rss_bytes"):
            raise ValueError("no observed workload RSS or inconsistent peak")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        failures.append("process_memory: " + str(exc))
    if cuda:
        if (not isinstance(gpu_report, dict) or gpu_report.get("status") != "measured"
                or gpu_report.get("errors")
                or not number(gpu_report.get("sampled_compute_process_peak_mb"), positive=True)
                or not number(gpu_report.get("device_peak_used_mb"), positive=True)
                or gpu_report.get("sample_count", 0) < 2):
            failures.append("gpu_memory: no complete observed allocated-device/process measurements")
    return {"status": "measured" if not failures else "unverified", "failures": failures,
            "scope": "observed overall workload peaks; sampling may miss short spikes, phase names are workload-reported"}


def run_one(entry, args, allocation, calibration=None):
    profile = runtime_profile(entry["profile"])
    requirement = profile.requirements
    resources = (NativeResources.model_validate(calibration["resources"]) if calibration else
                 NativeResources(cpus=args.cpus, memory_mb=args.memory_mb, seconds=args.timeout,
                                 shm_mb=requirement.min_shm_mb, tmp_mb=requirement.min_tmp_mb,
                                 threads=int(args.cpus), output_max_bytes=64 * 1024**2,
                                 output_max_files=1000))
    resources.gpus = allocation if requirement.cuda else None
    actual = verify_worker(args.gpu_index, resources.memory_mb, resources.cpus,
                           gpu_uuid=getattr(args, "gpu_uuid", None))
    if actual != allocation:
        raise ValueError("GPU allocation changed between planning and execution")
    root = args.output / entry["name"]
    if root.exists():
        raise ValueError("An attempt directory already exists; inspect it and choose a fresh output, never implicitly rerun")
    runner = ProbeRunner(root, args.runtime, resources)
    report = {**entry, "status": "failed", "live_model_calls": 0, "task_readiness": "unverified"}
    try:
        inspected = runner._image(entry["image"])
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", inspected):
            raise ValueError("Runtime did not resolve one immutable image ID")
        if entry["image"].startswith("sha256:") and entry["image"] != inspected:
            raise ValueError("Inspected image differs from requested image ID")
        report["resolved_image_id"] = inspected
        control, output = root / "definition", root / "output"
        control.mkdir()
        control.chmod(0o755)
        output.mkdir()
        output.chmod(0o777)
        bridge = Path(__file__).with_name("gpu_smoke_monitor.py")
        source_refs = {"profiler": runner.store.put_file(bridge, "calibration-source"),
                       "launcher": runner.store.put_file(Path(__file__), "calibration-source")}
        mounts = [(control, "/runtime", "ro"), (output, "/output", "rw")]
        if calibration:
            command = calibration["command"]
            cwd = "/calibration"
            source = Path(calibration["source_dir"])
            mounts.append((source, "/calibration", "ro"))
            inputs = {"workload_manifest": source}
            source_refs["workload_manifest"] = runner.store.put_json(input_manifest(source))
            for name, path in calibration.get("assets", {}).items():
                mounts.append((Path(path), "/assets/" + name, "ro"))
                inputs["asset:" + name] = Path(path)
                source_refs["asset:" + name] = runner.store.put_json(input_manifest(Path(path)))
        else:
            shutil.copyfile(probe_source(), control / "probe.py")
            shutil.copyfile(Path(__file__).with_name("gpu_smoke_payload.py"), control / "payload.py")
            (control / "requirements.json").write_bytes(canonical(requirement))
            for path in control.iterdir():
                path.chmod(0o644)
            source_refs["probe"] = runner.store.put_file(control / "probe.py", "calibration-source")
            source_refs["payload"] = runner.store.put_file(control / "payload.py", "calibration-source")
            command = ["python", "/runtime/payload.py"]
            cwd = "/output"
        job = {"command": command, "cwd": cwd, "seed": 0,
               "memory_limit_bytes": resources.memory_mb * 1024**2,
               "sample_interval_seconds": args.sample_interval,
               "environment": calibration.get("environment", {}) if calibration else {}}
        # NativeRunner creates control/job.json under the caller's umask. Create
        # its mount root in advance so the restricted UID can read this job only.
        launch_control = root / "control" / "profile"
        launch_control.mkdir(parents=True)
        launch_control.chmod(0o755)
        (launch_control / "job.json").write_bytes(canonical(job))
        (launch_control / "job.json").chmod(0o644)
        runner.store.set("pending", True)
        runner.store.set("request", {"entry": entry, "job": job, "resources": resources.model_dump(mode="json"),
                                     "source_refs": source_refs, "allocation": allocation})
        gpu_report = None
        if requirement.cuda:
            with GpuMonitor(allocation.removeprefix("device="), args.sample_interval) as gpu:
                process, execution_ref = runner._launch("profile", inspected, job, mounts, resources.seconds,
                                                       gpus=resources.gpus, bridge=bridge)
            gpu_report = gpu.report()
            report["gpu_memory_ref"] = runner.store.put_json(gpu_report, "gpu-memory")
            report["gpu_memory_status"] = gpu_report["status"]
        else:
            process, execution_ref = runner._launch("profile", inspected, job, mounts, resources.seconds, bridge=bridge)
            report["gpu_memory_status"] = "not_applicable"
        report.update(execution_ref=execution_ref, resources=resources.model_dump(mode="json"),
                      source_refs=source_refs, timed_out=process.timed_out, limit_exceeded=process.limit_exceeded)
        memory = output / "process-memory.json"
        report["process_memory_ref"] = runner.store.put_file(memory, "process-memory") if memory.is_file() else None
        framework_memory = output / "framework-memory.json"
        report["framework_memory_ref"] = runner.store.put_file(framework_memory, "framework-memory") if framework_memory.is_file() else None
        execution_pass = process.code == 0 and not process.timed_out and not process.limit_exceeded
        report["execution_status"] = "pass" if execution_pass else "fail"
        if not calibration:
            try:
                verification = json.loads(process.stdout.splitlines()[-1])
            except (IndexError, ValueError):
                verification = {"status": "fail", "reason": "missing structured probe output"}
            report["probe_ref"] = runner.store.put_json(verification, "runtime-probe")
            valid = verification.get("status") == "pass"
            report["peak_calibration_status"] = "not_performed_tiny_framework_smoke"
        else:
            report["phases"] = phase_evidence(output / "phases.jsonl", calibration["expected_phases"])
            report["memory_evidence"] = calibration_memory_evidence(memory, gpu_report, requirement.cuda)
            for key, path in inputs.items():
                if input_manifest(path) != runner.store.json(source_refs[key]):
                    raise ValueError("Calibration source or asset changed during execution")
            valid = (report["phases"]["status"] == "reported_by_workload"
                     and report["memory_evidence"]["status"] == "measured")
            report["peak_calibration_status"] = "complete" if execution_pass and valid else "incomplete"
            if not valid:
                report["reason"] = "Calibration requires all expected phase markers and valid observed memory evidence"
        report["status"] = "pass" if execution_pass and valid else "fail"
        report["measurement_scope"] = "declared workload only; no full-task fit guarantee" if calibration else "tiny framework workloads; real-task peaks remain unverified"
        runner.store.set("pending", False)
    except Exception as exc:
        report.update(status="blocked", error_type=type(exc).__name__, reason=str(exc))
    finally:
        ref = runner.store.put_json(report, "calibration-result")
        runner.store.set("result_ref", ref)
        (root / "result.json").write_bytes(canonical(report))
        runner.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", choices=["podman", "docker"], default="podman")
    parser.add_argument("--profile", action="append", choices=[p.name for p in runtime_profiles()])
    parser.add_argument("--mode", choices=["plan", "run"], default="plan")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--gpu-index", type=int, default=0,
                           help="Ordinal within the Slurm step's GPU list; mapped to a verified UUID")
    selection.add_argument("--gpu-uuid", help="Complete GPU UUID assigned to this Slurm step")
    parser.add_argument("--memory-mb", type=int, default=8192)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    if not args.workspace.is_absolute() or not args.workspace.is_dir() or not args.output.is_absolute():
        parser.error("Workspace and output must be absolute worker-local paths; workspace must exist")
    if not 0.05 <= args.sample_interval <= 5 or not 1 <= args.timeout <= 7200:
        parser.error("Use sample interval 0.05..5 seconds and timeout 1..7200 seconds")
    images = image_manifest(args.images)
    calibration = calibration_spec(args.calibration, args.workspace) if args.calibration else None
    if calibration:
        for source in [calibration["source_dir"], *calibration.get("assets", {}).values()]:
            if args.output.resolve().is_relative_to(Path(source).resolve()) or Path(source).resolve().is_relative_to(args.output.resolve()):
                parser.error("Calibration inputs and output directory must not overlap")
    profiles = list(dict.fromkeys(args.profile or [p.name for p in runtime_profiles()]))
    if calibration and calibration["name"] in profiles:
        parser.error("Calibration name must differ from selected framework profile names")
    result = plan(images, profiles, calibration)
    result["launcher_sha256"] = file_digest(Path(__file__))
    if args.mode == "plan":
        print(json.dumps(result, indent=2))
        return 0
    if any(entry["status"] != "planned" for entry in result["entries"]):
        parser.error("Every selected profile requires a resolved image digest; build images first")
    allocation = verify_worker(args.gpu_index, args.memory_mb, args.cpus, gpu_uuid=args.gpu_uuid)
    os.environ["SCIENTISTTWO_GPU_ALLOCATION"] = allocation
    if args.output.exists():
        parser.error("Output already exists; inspect prior evidence and choose a fresh run directory")
    args.output.mkdir(parents=True)
    result.update(mode="run", allocation=allocation, slurm_job_id=os.environ["SLURM_JOB_ID"],
                  hostname=platform.node(), status="running")
    (args.output / "request.json").write_bytes(canonical(result))
    results = []
    for entry in result["entries"]:
        results.append(run_one(entry, args, allocation,
                               calibration if entry["kind"] == "declared_workload_calibration" else None))
        result["results"] = results
        result["completed_entries"] = len(results)
        if len(results) == len(result["entries"]):
            result["status"] = "pass" if all(row["status"] == "pass" for row in results) else "fail"
        (args.output / "result.json").write_bytes(canonical(result))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
