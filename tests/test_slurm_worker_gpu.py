"""Synthetic launcher contracts; no Slurm jobs, container execution, or GPU probes."""

import importlib.util
from pathlib import Path
import sys

import pytest

import scientisttwo.runtime_profiles as profiles
from scientisttwo.runtime_profiles import GpuDevice


GPU_A = "GPU-00000000-0000-0000-0000-000000000001"
GPU_B = "GPU-00000000-0000-0000-0000-000000000002"


@pytest.fixture
def worker(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "runtime-profiles/slurm_worker.py"
    spec = importlib.util.spec_from_file_location("slurm_worker_gpu_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "node", lambda: "worker-01")
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(module.os, "chdir", lambda _path: None)
    for key, value in {
        "SLURMD_NODENAME": "worker-01", "SLURM_JOB_ID": "123",
        "SLURM_STEP_ID": "1", "SLURM_STEP_GPUS": "0-7", "SLURM_MEM_PER_NODE": "65536",
        "SCIENTISTTWO_GPU_UUID": GPU_B,
    }.items():
        monkeypatch.setenv(key, value)
    # Inventory lists only known test devices; step selectors must map completely.
    monkeypatch.setenv("SLURM_STEP_GPUS", "3,6")
    monkeypatch.delenv("SCIENTISTTWO_GPU_ALLOCATION", raising=False)
    monkeypatch.setattr(profiles, "gpu_inventory", lambda: [
        GpuDevice(index=0, minor_number=3, uuid=GPU_A, pci_bus_id="00000000:0A:00.0"),
        GpuDevice(index=4, minor_number=6, uuid=GPU_B, pci_bus_id="00000000:8D:00.0"),
    ])
    actions = []
    def inspect(argv, **_):
        assert argv[:3] == ["podman", "image", "inspect"]
        actions.append(("inspect", argv))
    monkeypatch.setattr(module.subprocess, "run", inspect)
    def execute(program, argv, env):
        actions.append(("exec", program, argv, env))
    monkeypatch.setattr(module.os, "execvpe", execute)
    base = ["slurm_worker.py", "--workspace", str(tmp_path), "--image", "sha256:" + "a" * 64,
            "--min-memory-mb", "65536"]
    return module, base, actions


@pytest.mark.parametrize("selection", [[], ["--gpus", "device=4"], ["--gpus=device=4"],
                                       ["--gpus", "device=" + GPU_B]])
def test_worker_canonicalizes_verified_gpu_selection_and_freezes_bound_uuid(worker, monkeypatch, selection):
    module, base, actions = worker
    monkeypatch.setattr(sys, "argv", base + ["--gpu-uuid", GPU_B, "--", "scientisttwo",
                                          "benchmark", "validate", *selection])
    module.main()
    execution = actions[-1]
    assert execution[0] == "exec"
    command, env = execution[2:]
    assert env["SCIENTISTTWO_GPU_ALLOCATION"] == "device=" + GPU_B
    assert env["SCIENTISTTWO_GPU_UUID"] == GPU_B
    assert any(value == "device=" + GPU_B or value == "--gpus=device=" + GPU_B for value in command)
    assert "device=4" not in command


def test_worker_cpu_step_never_uses_parent_gpu_allocation(worker, monkeypatch):
    module, base, actions = worker
    monkeypatch.delenv("SLURM_STEP_GPUS")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0-7")
    monkeypatch.setattr(sys, "argv", base + ["--", "scientisttwo", "benchmark", "validate"])
    with pytest.raises(SystemExit, match="parent-job"):
        module.main()
    assert not actions


def test_worker_rejects_ambiguous_multiple_gpu_options(worker, monkeypatch):
    module, base, actions = worker
    monkeypatch.setattr(sys, "argv", base + ["--", "scientisttwo", "benchmark", "validate",
                                          "--gpus", "device=4", "--gpus=device=0"])
    with pytest.raises(SystemExit, match="exactly once"):
        module.main()
    assert all(action[0] != "exec" for action in actions)


def test_worker_rejects_other_allocated_uuid_when_bound_to_one_gpu(worker, monkeypatch):
    module, base, actions = worker
    monkeypatch.setattr(sys, "argv", base + ["--gpu-uuid", GPU_A, "--", "scientisttwo",
                                          "benchmark", "validate"])
    with pytest.raises(SystemExit, match="bound GPU UUID"):
        module.main()
    assert not actions


def test_worker_no_gpu_query_for_help(worker, monkeypatch):
    module, _, actions = worker
    monkeypatch.setattr(profiles, "gpu_inventory", lambda: pytest.fail("help queried GPU inventory"))
    monkeypatch.setattr(sys, "argv", ["slurm_worker.py", "--help"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 0
    assert not actions
