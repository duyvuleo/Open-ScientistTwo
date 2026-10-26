"""Synthetic worker/probe contracts; no real GPU, Slurm or framework execution."""

from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import time

import pytest

from scientisttwo import continuation_environment as environment
from scientisttwo import runtime_contracts, runtime_profiles, slurm_execution as guard
from scientisttwo.store import IntegrityError, Store, canonical, digest

from test_research_workflow import campaign


@pytest.fixture
def ready_campaign(tmp_path, monkeypatch):
    monkeypatch.delenv(runtime_contracts.EXECUTION_CONFIG_ENV, raising=False)
    monkeypatch.delenv(runtime_contracts.EXECUTION_CONFIG_SHA_ENV, raising=False)
    controller, store, runner, _, transport = campaign(tmp_path)
    runner.preflight()
    controller.config.development.settings["runtime_profile"] = "cpu-tabular"
    report = {"status": "pass", "platform": "synthetic-worker", "python": "3.12.0",
              "checks": [{"name": "actual_compute", "status": "pass", "value": "synthetic-only"}]}
    hardware = {key: report[key] for key in ("platform", "python")}
    hardware.update(status="verified", checks={"actual_compute": "synthetic-only"})
    receipt = store.put_json({"synthetic_test": True}, "synthetic-probe-receipt")
    probe_ref = store.put_json({"profile": runtime_profiles.runtime_profile("cpu-tabular").model_dump(mode="json"),
                               "report": report, "execution_ref": receipt})
    request = {"profile": "cpu-tabular", "image": runner.image,
               "resources": runner.spec.resources.model_dump(mode="json"),
               "settings": controller.config.development.settings,
               "source_sha256": environment.file_hash(runtime_profiles.probe_source())}
    runner.budget.reserve("probe/runtime/0000", "probe", request)
    runner.budget.finish("probe/runtime/0000", "success", {"probe_ref": probe_ref, "hardware": hardware}, .1)
    store.set("runtime_probe_ref", probe_ref)
    store.set("runtime_hardware", hardware)
    runner.hardware = hardware
    assert runner.execute("baseline/full", controller.adapter.baseline(), "full").status == "success"
    try:
        yield controller, store, runner, transport
    finally:
        store.close()


def test_cached_preflight_preserves_operations_and_runs_no_physical_stage(ready_campaign):
    controller, store, runner, transport = ready_campaign
    rows, stages = runner.budget.summary(), deepcopy(transport.stages)
    metadata = list(store.db.execute("SELECT * FROM metadata ORDER BY key"))
    environment.install_cached_preflight(controller)
    runner.preflight()
    runner.preflight()
    assert runner.budget.summary() == rows
    assert transport.stages == stages
    assert list(store.db.execute("SELECT * FROM metadata ORDER BY key")) == metadata
    assert runner.hardware == store.get("runtime_hardware")


@pytest.mark.parametrize("changed", ["image", "settings", "probe", "probe_source", "source", "hardware", "runtime_wrapper"])
def test_cached_preflight_rejects_changed_protocol_without_probe(ready_campaign, monkeypatch, tmp_path, changed):
    controller, store, runner, transport = ready_campaign
    if changed == "image":
        monkeypatch.setattr(runner, "_image", lambda _: "sha256:changed")
    elif changed == "settings":
        controller.config.development.settings["synthetic_changed_setting"] = True
    elif changed == "probe":
        ref = store.get("runtime_probe_ref")
        report = store.json(ref)
        report["report"]["status"] = "fail"
        store.set("runtime_probe_ref", store.put_json(report))
    elif changed == "probe_source":
        source = tmp_path / "different_probe.py"
        source.write_text("# Explicit synthetic changed probe\n")
        monkeypatch.setattr(runtime_profiles, "probe_source", lambda: source)
    elif changed == "source":
        source = runner.spec.candidate_dir / "train.py"
        source.write_text(source.read_text() + "\n# Synthetic source change\n")
    elif changed == "hardware":
        hardware = store.get("runtime_hardware")
        hardware["checks"]["actual_compute"] = "changed"
        store.set("runtime_hardware", hardware)
    else:
        previous = runtime_contracts.runtime_identity
        monkeypatch.setattr(runtime_contracts, "runtime_identity", lambda *args, **kwargs:
                            {**previous(*args, **kwargs), "execution_enforcement": {"runtime_wrapper": "changed"}})
    rows, stages = runner.budget.summary(), deepcopy(transport.stages)
    environment.install_cached_preflight(controller)
    with pytest.raises((IntegrityError, ValueError)):
        runner.preflight()
    assert runner.budget.summary() == rows
    assert transport.stages == stages


@pytest.fixture
def worker_context(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    binary = root / "bin"
    binary.mkdir()
    shim = binary / "podman"
    shim.write_text("#!/bin/sh\n# Synthetic wrapper, never invoked\n")
    shim.chmod(0o700)
    podman = root / "podman"
    podman.write_text("Synthetic underlying runtime, never invoked\n")
    receipts = root / "receipts"
    receipts.mkdir(mode=0o700)
    node, uid = socket.gethostname().split(".")[0], os.getuid()
    negative = root / "negative.json"
    negative.write_bytes(canonical({"node": node, "uid": uid, "podman_sha256": guard.sha256(podman),
                                   "cpu_enforced": True, "memory_enforced": True, "nproc_enforced": True,
                                   "timeout_cleanup_verified": True, "synthetic_test": True}))
    config = {"schema_version": 1, "job_id": "3920", "node": node, "uid": uid,
              "podman": str(podman), "podman_sha256": guard.sha256(podman),
              "guard_sha256": guard.sha256(guard.__file__), "srun": str(shim), "python": str(shim),
              "receipt_dir": str(receipts), "max_cpus": 8, "max_memory_mb": 65536,
              "controller_memory_mb": 512, "max_seconds": 3600, "expires_at": time.time() + 3600,
              "gpu_uuid": "GPU-4f81c4b7-3011-b470-5205-7afcfb47f569", "gpu_gres": "gpu:a100:8",
              "negative_control": str(negative), "negative_control_sha256": guard.sha256(negative)}
    path = root / "config.json"
    path.write_text(json.dumps(config))
    sha = guard.sha256(path)
    monkeypatch.setenv("PATH", str(binary))
    monkeypatch.setenv(runtime_contracts.EXECUTION_CONFIG_ENV, str(path))
    monkeypatch.setenv(runtime_contracts.EXECUTION_CONFIG_SHA_ENV, sha)
    expected = runtime_contracts.execution_identity("podman")
    monkeypatch.setenv("SLURM_JOB_ID", "3920")
    monkeypatch.setenv("SLURM_STEP_ID", "123")
    monkeypatch.setenv("SLURM_STEP_GPUS", "0,1,2,3,4,5,6,7")
    resource = {"cpuset": list(range(8)), "affinity": list(range(8)),
                "memory_limit_bytes": 65536 * 1024**2, "cgroup_membership": "0::/job_3920/step_123"}
    monkeypatch.setattr(guard, "resource_observation", lambda: resource)
    store = Store(root / "campaign")
    store.db.execute("CREATE TABLE live_operations(key TEXT PRIMARY KEY,kind TEXT,status TEXT,result_ref TEXT)")
    protocol = {"phase": "full", "runtime": {"runtime": "podman", "execution_enforcement": expected}}
    observation = {"id": "baseline/full", "phase": "full", "status": "success",
                   "protocol_ref": store.put_json(protocol), "protocol_hash": digest(canonical(protocol))}
    store.db.execute("INSERT INTO live_operations VALUES(?,?,?,?)",
                     ("experiment/baseline/full", "experiment", "success", store.put_json(observation)))
    store.db.commit()
    try:
        yield store, path, sha, expected, resource, config
    finally:
        store.close()


def test_environment_restores_original_shim_and_offline_gpu_scope(worker_context, monkeypatch):
    store, path, sha, expected, _, config = worker_context
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv(runtime_contracts.EXECUTION_CONFIG_ENV)
    monkeypatch.delenv(runtime_contracts.EXECUTION_CONFIG_SHA_ENV)
    proof = environment.restore_environment(store, path, sha)
    assert proof["physical_probes"] == 0
    assert runtime_contracts.execution_identity("podman") == expected
    assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["SCIENTISTTWO_GPU_UUID"] == config["gpu_uuid"]
    assert os.environ["HF_HUB_OFFLINE"] == os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["PATH"] == str(Path(expected["runtime_wrapper"]["path"]).parent) + ":/usr/local/bin:/usr/bin:/bin"


@pytest.mark.parametrize("changed", ["policy_hash", "shim", "job", "step", "gpu_count", "cpus", "memory", "identity"])
def test_environment_refuses_changed_envelope(worker_context, monkeypatch, changed):
    store, path, sha, expected, resource, _ = worker_context
    if changed == "policy_hash":
        sha = "0" * 64
    elif changed == "shim":
        Path(expected["runtime_wrapper"]["path"]).write_text("changed")
    elif changed == "job":
        monkeypatch.setenv("SLURM_JOB_ID", "9999")
    elif changed == "step":
        monkeypatch.setenv("SLURM_STEP_ID", "extern")
    elif changed == "gpu_count":
        monkeypatch.setenv("SLURM_STEP_GPUS", "6")
    elif changed == "cpus":
        resource["cpuset"] = list(range(9))
    elif changed == "memory":
        resource["memory_limit_bytes"] += 1
    else:
        monkeypatch.setattr(runtime_contracts, "execution_identity", lambda _: {"changed": True})
    before = dict(os.environ)
    with pytest.raises((IntegrityError, ValueError)):
        environment.restore_environment(store, path, sha)
    assert dict(os.environ) == before
