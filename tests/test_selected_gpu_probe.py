import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE = (Path(__file__).resolve().parents[1] / "src/scientisttwo/task_families/templates/runtime_probe.py")
GPU_A = "GPU-4f81c4b7-3011-b470-5205-7afcfb47f569"
GPU_B = "GPU-11111111-2222-3333-4444-555555555555"


@pytest.fixture
def probe(monkeypatch):
    spec = importlib.util.spec_from_file_location("selected_gpu_probe", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return module


def row(gpu, free):
    return f"A100, 40960, {free}, 575.57.08, {gpu}\n"


def test_busy_selected_gpu_cannot_borrow_an_idle_gpu_memory(probe, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_A)
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        # Unscoped NVML view contains an idle unallocated GPU.
        return SimpleNamespace(stdout=row(GPU_A, 100) if "--id=" + GPU_A in command
                               else row(GPU_A, 100) + row(GPU_B, 40000))
    monkeypatch.setattr(probe.subprocess, "run", query)
    with pytest.raises(ValueError, match="selected GPU"):
        probe.gpu_resources(32000)
    assert calls[0][-1] == "--id=" + GPU_A


@pytest.mark.parametrize("output", [
    row(GPU_B, 40000), row(GPU_A, 40000) + row(GPU_B, 40000),
    row(GPU_A, 40000) * 2, "",
])
def test_driver_must_return_exact_selected_identity(probe, monkeypatch, output):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_A)
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output))
    with pytest.raises(ValueError, match="exactly the selected UUIDs"):
        probe.gpu_resources(100)


def test_each_selected_gpu_must_have_required_memory(probe, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_A + "," + GPU_B)
    monkeypatch.setattr(probe.subprocess, "run", lambda command, **kwargs: SimpleNamespace(
        stdout=row(GPU_A, 40000) + row(GPU_B, 100)))
    with pytest.raises(ValueError, match="selected GPU"):
        probe.gpu_resources(32000)


def test_selected_gpu_success_records_only_its_identity(probe, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_A)
    monkeypatch.setattr(probe.subprocess, "run", lambda command, **kwargs: SimpleNamespace(stdout=row(GPU_A, 40000)))
    assert probe.gpu_resources(32000)[0]["uuid"] == GPU_A


def test_unscoped_probe_preserves_existing_single_sufficient_device_policy(probe, monkeypatch):
    def query(command, **kwargs):
        assert not any(item.startswith("--id=") for item in command)
        return SimpleNamespace(stdout=row(GPU_A, 100) + row(GPU_B, 40000))
    monkeypatch.setattr(probe.subprocess, "run", query)
    assert len(probe.gpu_resources(32000)) == 2


@pytest.mark.parametrize("value", [GPU_A + "," + GPU_A, GPU_A + ",0", "GPU-4f81c4b7"])
def test_malformed_explicit_uuid_selection_fails_before_query(probe, monkeypatch, value):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: pytest.fail("No driver query"))
    with pytest.raises(ValueError, match="complete UUIDs"):
        probe.gpu_resources(1)
