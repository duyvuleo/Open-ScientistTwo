"""Restore a pinned worker context and reuse its completed runtime probe.

This sidecar may be loaded outside an unchanged package by an authorized
continuation operator. It changes only this process's environment and one
runner instance. It never changes a policy, image, allocation or old receipt.
"""

import os
from pathlib import Path
import re
from types import MethodType

from . import runtime_contracts, runtime_profiles, slurm_execution as guard
from .benchmarks.files import file_hash
from .live.runtime import DevelopmentRunner, validate_prepared
from .live.source_tree import freeze_source_tree
from .live.workspace_task import preflight
from .store import IntegrityError, canonical, digest


def _historical_runtime(store):
    row = store.db.execute(
        "SELECT * FROM live_operations WHERE key='experiment/baseline/full'").fetchone()
    if not row or row["kind"] != "experiment" or row["status"] != "success" or not row["result_ref"]:
        raise IntegrityError("A successful immutable full-development baseline is required")
    observation = store.json(row["result_ref"])
    if (observation.get("id") != "baseline/full" or observation.get("status") != "success"
            or observation.get("phase") != "full" or not observation.get("protocol_ref")):
        raise IntegrityError("Original baseline observation identity differs")
    protocol = store.json(observation["protocol_ref"])
    if (digest(canonical(protocol)) != observation.get("protocol_hash")
            or protocol.get("phase") != "full" or not isinstance(protocol.get("runtime"), dict)):
        raise IntegrityError("Original baseline protocol identity differs")
    return protocol["runtime"]


def restore_environment(store, execution_config_path, execution_config_sha256):
    """Verify the original live Slurm envelope before restoring process variables."""
    runtime = _historical_runtime(store)
    expected = runtime.get("execution_enforcement")
    if (not isinstance(expected, dict) or runtime.get("runtime") != "podman"
            or str(execution_config_path) != expected.get("config_path")
            or execution_config_sha256 != expected.get("config_sha256")):
        raise IntegrityError("Continuation must use the original exact execution policy")
    config = guard.load_config(execution_config_path, execution_config_sha256)
    if os.environ.get("SLURM_JOB_ID") != str(config["job_id"]):
        raise IntegrityError("Continuation is outside its original Slurm allocation")
    step = os.environ.get("SLURM_STEP_ID", "")
    if not re.fullmatch(r"[0-9]+", step):
        raise IntegrityError("Continuation requires a numbered Slurm step")
    resources = guard.resource_observation()
    if f"job_{config['job_id']}/step_{step}" not in resources["cgroup_membership"]:
        raise IntegrityError("Continuation is outside its declared kernel step cgroup")
    guard.verify_resources(resources, config["max_cpus"], config["max_memory_mb"])
    gpu_ids = guard.cpu_set(os.environ.get("SLURM_STEP_GPUS", ""))
    expected_gpu_count = 0 if config["gpu_gres"] == "none" else int(config["gpu_gres"].rsplit(":", 1)[1])
    if len(gpu_ids) != expected_gpu_count:
        raise IntegrityError("Slurm step GPU assignment differs from the original declaration")
    wrapper = expected.get("runtime_wrapper", {})
    path = Path(wrapper.get("path", ""))
    if (not path.is_absolute() or any(item.is_symlink() for item in (path, *path.parents))
            or not path.is_file() or file_hash(path) != wrapper.get("sha256")):
        raise IntegrityError("Original runtime shim path or content changed")
    values = {
        runtime_contracts.EXECUTION_CONFIG_ENV: str(execution_config_path),
        runtime_contracts.EXECUTION_CONFIG_SHA_ENV: execution_config_sha256,
        "PATH": str(path.parent) + ":/usr/local/bin:/usr/bin:/bin",
        "CUDA_VISIBLE_DEVICES": config["gpu_uuid"] or "",
        "SCIENTISTTWO_GPU_UUID": config["gpu_uuid"] or "",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled",
    }
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        if runtime_contracts.execution_identity("podman") != expected:
            raise IntegrityError("Restored execution identity differs from the baseline protocol")
    except Exception:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        raise
    return {"status": "verified", "execution_enforcement": expected,
            "host_resources": resources, "physical_probes": 0}


def _cached_hardware(runner, profile):
    store = runner.store
    hardware, probe_ref = store.get("runtime_hardware"), store.get("runtime_probe_ref")
    if not hardware or not probe_ref:
        raise IntegrityError("A completed recorded runtime probe is required")
    request = {"profile": profile, "image": runner.image,
               "resources": runner.spec.resources.model_dump(mode="json"),
               "settings": runner.config.development.settings,
               "source_sha256": file_hash(runtime_profiles.probe_source())}
    matches = []
    for row in store.db.execute(
            "SELECT * FROM live_operations WHERE kind='probe' AND key LIKE 'probe/runtime/%'"):
        if row["status"] != "success" or row["request_hash"] != digest(canonical(request)) or not row["result_ref"]:
            continue
        result = store.json(row["result_ref"])
        if result.get("probe_ref") == probe_ref and result.get("hardware") == hardware:
            matches.append(row["key"])
    if not matches:
        raise IntegrityError("Recorded runtime probe does not match exact settings and source")
    evidence = store.json(probe_ref)
    report = evidence.get("report", {})
    if (evidence.get("profile") != runtime_profiles.runtime_profile(profile).model_dump(mode="json")
            or report.get("status") != "pass" or not report.get("checks")
            or any(check.get("status") != "pass" for check in report["checks"])
            or not evidence.get("execution_ref")):
        raise IntegrityError("Recorded runtime probe report did not pass its frozen profile")
    store.read(evidence["execution_ref"])
    stable = {"platform": report["platform"], "python": report["python"],
              "status": "verified", "checks": {}}
    for check in report["checks"]:
        name = check["name"]
        if name.startswith("version:") or name in {"actual_compute", "cpu_identity", "python_contract"}:
            stable["checks"][name] = check["value"]
        elif name == "gpu_resources":
            stable["checks"][name] = [{key: value for key, value in gpu.items() if key != "free_mb"}
                                      for gpu in check["value"]]
    if stable != hardware:
        raise IntegrityError("Stored hardware differs from its successful probe report")
    return hardware


def _cached_preflight(runner):
    expected_runtime = _historical_runtime(runner.store)
    checks = preflight(runner.spec)
    if not checks["ready"]:
        raise IntegrityError("Native prerequisites unavailable: " + "; ".join(checks["errors"]))
    images = {"execution": runner._image(runner.spec.image),
              "grading": runner._image(runner.spec.grader_image)}
    if runner.store.get("live_images") != images or expected_runtime.get("images") != images:
        raise IntegrityError("Original runtime images changed")
    runner.image = images["execution"]
    declaration = getattr(runner.adapter, "source_tree", lambda: None)()
    if declaration:
        ref = freeze_source_tree(runner.store, Path(declaration["path"]), declaration["editable_paths"])
        if runner.store.get("authorized_source_snapshot") != ref:
            raise IntegrityError("Authorized original source tree changed")
        runner.adapter.bind_source(ref)
    elif runner.store.get("authorized_source_snapshot") is not None:
        raise IntegrityError("Original adapter source-tree declaration disappeared")
    profile = runner.config.development.settings.get("runtime_profile")
    if not profile:
        raise IntegrityError("Cached probe continuation requires a declared runtime profile")
    runner.hardware = _cached_hardware(runner, profile)
    runner.adapter.validate(runner.adapter.baseline())
    for phase in ("subset", "full"):
        ref = runner.store.get("prepared/" + phase)
        if not ref:
            raise IntegrityError("Original prepared development partition is missing")
        validate_prepared(runner.store.json(ref), runner.store.root / "development" / phase)
    actual = runtime_contracts.runtime_identity(runner.spec, images, runner.hardware)
    if actual != expected_runtime:
        raise IntegrityError("Complete current runtime identity differs from the original baseline")


def install_cached_preflight(controller):
    """Reuse only a verified probe on this runner; no global patch or new operation."""
    runner = controller.runner
    if (not isinstance(runner, DevelopmentRunner) or runner.store is not controller.store
            or runner.config is not controller.config or runner.budget is not controller.budget):
        raise IntegrityError("Cached preflight requires the controller's original development runner")
    runner.preflight = MethodType(_cached_preflight, runner)
    return {"adapter": "verified-cached-preflight-v1", "physical_probes": 0}
