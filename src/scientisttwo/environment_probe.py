"""Probe the actual container before experiments; never infer readiness from a tag."""

import json
from pathlib import Path
import shutil
import time

from .runtime_profiles import runtime_profile, probe_argv, probe_source
from .runtime_contracts import asset_mounts
from .store import IntegrityError, canonical
from .benchmarks.files import file_hash


def probe_runtime(runner, profile_name):
    """Account for each physical probe separately from research experiments."""
    budget = getattr(runner, "budget", None)
    if budget is None:
        return _run_probe(runner, profile_name)
    sequence = runner.store.db.execute("SELECT count(*) FROM live_operations WHERE kind='probe'").fetchone()[0]
    key = f"probe/runtime/{sequence:04d}"
    request = {"profile": profile_name, "image": runner.image,
               "resources": runner.spec.resources.model_dump(mode="json"),
               "settings": runner.config.development.settings,
               "source_sha256": file_hash(probe_source())}
    budget.reserve(key, "probe", request)
    started = time.monotonic()
    try:
        result = _run_probe(runner, profile_name)
    except Exception as error:
        budget.finish(key, "failed", {"error_type": type(error).__name__}, time.monotonic() - started)
        raise
    budget.finish(key, "success", {"probe_ref": runner.store.get("runtime_probe_ref"),
                                   "hardware": result}, time.monotonic() - started)
    return result


def _run_probe(runner, profile_name):
    profile = runtime_profile(profile_name)
    requirements = profile.requirements.model_dump(mode="json")
    minimum = runner.config.development.settings.get("required_free_disk_mb", 0)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        raise ValueError("required_free_disk_mb must be a nonnegative integer")
    requirements["min_disk_mb"] = max(requirements["min_disk_mb"], minimum)
    minimum_vram = runner.config.development.settings.get("minimum_free_vram_mb", 0)
    if isinstance(minimum_vram, bool) or not isinstance(minimum_vram, int) or minimum_vram < 0:
        raise ValueError("minimum_free_vram_mb must be a nonnegative integer")
    requirements["min_vram_mb"] = max(requirements["min_vram_mb"], minimum_vram)
    requirements["assets"] = [
        {"path": asset.name + "/" + name, **details}
        for asset in runner.spec.assets if "execute" in asset.stages
        for name, details in asset.manifest.items()
    ]
    root = runner.store.root / "runtime-probe"
    control, workspace = root / "definition", root / "workspace"
    control.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(exist_ok=True)
    workspace.chmod(0o777)
    shutil.copyfile(probe_source(), control / "probe.py")
    (control / "requirements.json").write_bytes(canonical(requirements))
    mounts = [(control, "/runtime", "ro"), (workspace, "/workspace", "rw")]
    mounts += asset_mounts(runner.spec, "execute", protected=[runner.spec.private_dir, getattr(runner.spec, "evaluator_dir", None)])
    process, receipt = runner._launch(
        "runtime-probe", runner.image,
        {"command": probe_argv(), "cwd": "/workspace", "seed": 0}, mounts,
        min(300, runner.spec.resources.seconds,
            runner.budget.remaining() if getattr(runner, "budget", None) else 300), gpus=runner.spec.resources.gpus,
        bridge=Path(__file__).with_name("live") / "workspace_worker.py",
    )
    try:
        report = json.loads(process.stdout.splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise ValueError("Runtime probe omitted its structured report") from error
    runner.store.set("runtime_probe_ref", runner.store.put_json({"profile": profile.model_dump(mode="json"), "report": report, "execution_ref": receipt}, "runtime-probe"))
    if process.code or process.timed_out or report.get("status") != "pass":
        failures = [item["name"] for item in report.get("checks", []) if item.get("status") != "pass"]
        raise ValueError("Runtime prerequisite probe failed: " + ", ".join(failures))
    stable = {"platform": report["platform"], "python": report["python"], "status": "verified", "checks": {}}
    for item in report["checks"]:
        name = item["name"]
        if name.startswith("version:") or name in {"actual_compute", "cpu_identity", "python_contract"}:
            stable["checks"][name] = item["value"]
        elif name == "gpu_resources":
            stable["checks"][name] = [{k: v for k, v in gpu.items() if k != "free_mb"} for gpu in item["value"]]
    prior = runner.store.get("runtime_hardware")
    if prior is not None and prior != stable:
        raise IntegrityError("Execution hardware/software changed from the frozen protocol")
    runner.store.set("runtime_hardware", stable)
    return stable
