import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import scientisttwo.runtime_profiles as profiles_module
from scientisttwo.runtime_profiles import GpuDevice


ROOT = Path(__file__).parents[1] / "runtime-profiles"
GPU_UUID = "GPU-00000000-0000-0000-0000-000000000005"


def load(name="gpu_smoke"):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_never_calls_runtime_or_gpu(tmp_path, monkeypatch, capsys):
    module = load()
    monkeypatch.setattr(module.subprocess, "run", lambda *_a, **_k: pytest.fail("plan invoked a process"))
    images = tmp_path / "images.json"
    images.write_text(json.dumps({"torch-cuda": "sha256:" + "a" * 64}))
    output = tmp_path / "unused"
    monkeypatch.setattr(sys, "argv", ["gpu_smoke.py", "--images", str(images), "--workspace", str(tmp_path), "--output", str(output)])
    assert module.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["entries"]) == 4
    assert report["live_model_calls"] == 0
    assert report["full_suite_readiness"] == "unverified"
    assert sum(e["status"] == "planned" for e in report["entries"]) == 1
    assert not output.exists()


@pytest.mark.parametrize("images", [{"torch-cuda": "mutable:tag"}, {"wrong": "sha256:" + "a" * 64}, []])
def test_image_manifest_rejects_mutable_or_unknown(tmp_path, images):
    path = tmp_path / "images.json"
    path.write_text(json.dumps(images))
    with pytest.raises(ValueError):
        load().image_manifest(path)


def test_worker_requires_actual_allocation_and_resources(monkeypatch):
    # A real workstation's selected GPU must not leak into this synthetic step.
    monkeypatch.delenv("SCIENTISTTWO_GPU_UUID", raising=False)
    module = load()
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "node", lambda: "worker-01.local")
    for key, value in {"SLURMD_NODENAME": "worker-01", "SLURM_JOB_ID": "123", "SLURM_STEP_GPUS": "5",
                       "SLURM_CPUS_PER_TASK": "4", "SLURM_MEM_PER_NODE": "65536"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(profiles_module, "gpu_inventory", lambda: [
        GpuDevice(index=2, minor_number=5, uuid=GPU_UUID, pci_bus_id="00000000:8D:00.0"),
    ])
    assert module.verify_worker(0, 65536, 4) == "device=" + GPU_UUID
    assert module.verify_worker(0, 65536, 4, gpu_uuid=GPU_UUID) == "device=" + GPU_UUID
    with pytest.raises(ValueError, match="CPUs"):
        module.verify_worker(0, 8192, 8)
    with pytest.raises(ValueError, match="memory"):
        module.verify_worker(0, 131072, 4)
    monkeypatch.setenv("SLURMD_NODENAME", "other")
    with pytest.raises(ValueError, match="compute node"):
        module.verify_worker(0, 8192, 4)


def test_gpu_monitor_starts_before_work_and_records_process_lower_bound(monkeypatch):
    module = load()
    counter = iter([0, 1, 2, 3, 4, 5])
    def sample(device):
        assert device == GPU_UUID
        value = next(counter, 5)
        return {"uuid": "GPU-sample", "name": "synthetic-test-only", "total_mb": 40000,
                "used_mb": 500 + value * 100, "free_mb": 39500 - value * 100,
                "processes": [{"pid": 10, "used_mb": value * 100}] if value else []}
    monkeypatch.setattr(module, "gpu_sample", sample)
    with module.GpuMonitor(GPU_UUID, interval=0.005) as monitor:
        time.sleep(0.03)
    result = monitor.report()
    assert result["status"] == "measured"
    assert result["samples"][0]["processes"] == []
    assert result["sampled_compute_process_peak_mb"] > 0
    assert "lower bounds" in result["peak_semantics"]


def test_gpu_sampler_pins_uuid_and_detects_wrong_device(monkeypatch):
    module = load()
    calls = []
    def query(argv):
        calls.append(argv)
        if argv[0].startswith("--id="):
            return [[GPU_UUID, "Synthetic A100", "40000", "10", "39990"]]
        return [[GPU_UUID, "123", "10"],
                ["GPU-00000000-0000-0000-0000-000000000006", "456", "1000"]]
    monkeypatch.setattr(module, "_query", query)
    report = module.gpu_sample(GPU_UUID)
    assert calls[0][0] == "--id=" + GPU_UUID
    assert report["processes"] == [{"pid": 123, "used_mb": 10}]
    with pytest.raises(ValueError, match="different UUID"):
        module.gpu_sample("GPU-00000000-0000-0000-0000-000000000006")


def test_gpu_monitor_refuses_busy_allocated_device(monkeypatch):
    module = load()
    monkeypatch.setattr(module, "gpu_sample", lambda _: {"processes": [{"pid": 12, "used_mb": 100}]})
    with pytest.raises(ValueError, match="active compute"):
        with module.GpuMonitor("0"):
            pytest.fail("entered busy GPU monitor")


def test_no_observed_workload_memory_is_unverified(monkeypatch):
    module = load()
    monkeypatch.setattr(module, "gpu_sample", lambda _: {"processes": [], "used_mb": 10, "free_mb": 40000})
    with module.GpuMonitor("0", interval=0.05) as monitor:
        pass
    assert monitor.report()["status"] == "unverified"
    assert monitor.report()["sampled_compute_process_peak_mb"] is None


def test_process_tree_sampler_ignores_other_processes(tmp_path):
    module = load("gpu_smoke_monitor")
    for pid, parent, rss in [(10, 1, 5), (11, 10, 8), (12, 11, 2), (13, 1, 900)]:
        root = tmp_path / str(pid)
        root.mkdir()
        fields = ["S", str(parent)] + ["0"] * 19 + [str(rss)]
        (root / "stat").write_text(f"{pid} (name with spaces) " + " ".join(fields))
    rss, count = module.process_tree_rss(10, tmp_path)
    assert count == 3
    assert rss == 15 * module.os.sysconf("SC_PAGE_SIZE")


def test_cgroup_peak_requires_matching_container_limit(tmp_path):
    module = load("gpu_smoke_monitor")
    (tmp_path / "memory.max").write_text("1024")
    (tmp_path / "memory.peak").write_text("512")
    assert module.cgroup_peak(1024, tmp_path) == 512
    assert module.cgroup_peak(2048, tmp_path) is None


def test_phase_marks_are_workload_reports_not_verified_peaks(tmp_path):
    module = load()
    path = tmp_path / "phases.jsonl"
    expected = ["loading", "warmup", "measurement"]
    assert module.phase_evidence(path, expected)["status"] == "unverified"
    path.write_text('\n'.join(json.dumps({"phase": name}) for name in expected))
    assert module.phase_evidence(path, expected)["status"] == "reported_by_workload"


def test_calibration_is_explicit_and_only_mounts_local_assets(tmp_path):
    module = load()
    source = tmp_path / "workload"
    source.mkdir()
    path = tmp_path / "calibration.json"
    value = {"name": "llama-paired", "profile": "torch-cuda", "task_id": "llama-inference",
             "protocol_id": "example-not-executed", "source_dir": str(source),
             "command": ["python", "/calibration/calibrate.py"], "resources": {"memory_mb": 65536}}
    path.write_text(json.dumps(value))
    result = module.calibration_spec(path, tmp_path)
    assert result["expected_phases"] == ["loading", "warmup", "measurement"]
    value["source_dir"] = "/tmp"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="inside"):
        module.calibration_spec(path, tmp_path)


def test_inputs_hash_all_bytes_and_reject_credentials_or_links(tmp_path):
    module = load()
    (tmp_path / "weights.bin").write_bytes(b"test weights")
    manifest = module.input_manifest(tmp_path)
    assert manifest["weights.bin"]["size"] == 12
    (tmp_path / "link").symlink_to(tmp_path / "weights.bin")
    with pytest.raises(ValueError, match="links"):
        module.input_manifest(tmp_path)
    (tmp_path / "link").unlink()
    (tmp_path / ".ssh").mkdir()
    with pytest.raises(ValueError, match="Credential"):
        module.input_manifest(tmp_path)


def test_allocator_peak_collection_does_not_import_host_frameworks(monkeypatch):
    module = load("gpu_smoke_payload")
    calls = []
    torch = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: calls.append("sync"), max_memory_allocated=lambda: 123,
        max_memory_reserved=lambda: 256))
    monkeypatch.setitem(sys.modules, "torch", torch)
    report = module.allocator_peaks("torch", True)
    assert calls == ["sync"]
    assert report["peak_allocated_bytes"] == 123
    assert report["peak_reserved_bytes"] == 256
    assert "real task" in report["exclusions"]
    assert module.allocator_peaks("numpy", False)["status"] == "not_applicable"


def test_isolated_launch_binds_sources_and_preserves_attempt_without_reexecution(tmp_path, monkeypatch):
    module = load()
    launches = []
    monkeypatch.setattr(module, "verify_worker", lambda *_a, **_k: "device=1")
    monkeypatch.setattr(module.ProbeRunner, "_image", lambda *_: "sha256:" + "a" * 64)
    def launch(runner, stage, image, job, mounts, timeout, **kwargs):
        # Explicit synthetic process receipt tests the controller contract only.
        launches.append(job)
        assert kwargs.get("gpus") is None
        assert job["command"] == ["python", "/runtime/payload.py"]
        assert all(mode == "ro" for _, target, mode in mounts if target != "/output")
        output = next(path for path, target, _ in mounts if target == "/output")
        (output / "process-memory.json").write_text('{"synthetic_test":true}')
        receipt = runner.store.put_json({"synthetic_test": True})
        process = SimpleNamespace(code=0, timed_out=False, limit_exceeded=None,
                                  stdout='{"status":"pass","checks":[]}\n')
        return process, receipt
    monkeypatch.setattr(module.ProbeRunner, "_launch", launch)
    args = SimpleNamespace(output=tmp_path, cpus=4, memory_mb=8192, timeout=300,
                           gpu_index=0, runtime="podman", sample_interval=.2)
    entry = module.plan({"cpu-tabular": "sha256:" + "a" * 64}, ["cpu-tabular"])["entries"][0]
    report = module.run_one(entry, args, "device=1")
    assert report["status"] == "pass"
    assert report["task_readiness"] == "unverified"
    assert report["task_peak_memory"] == "unverified"
    assert report["process_memory_ref"]
    with pytest.raises(ValueError, match="never implicitly rerun"):
        module.run_one(entry, args, "device=1")
    assert len(launches) == 1


@pytest.mark.skipif(sys.platform != "linux", reason="process sampler targets Linux /proc")
def test_real_process_rss_measurement_without_frameworks(tmp_path):
    module = load("gpu_smoke_monitor")
    job = {"command": [sys.executable, "-c", "import time; x=bytearray(20*1024*1024); time.sleep(.2)"],
           "cwd": str(tmp_path), "seed": 0, "sample_interval_seconds": .01,
           "memory_limit_bytes": 1024**3}
    assert module.run(job, tmp_path) == 0
    report = json.loads((tmp_path / "process-memory.json").read_text())
    assert report["sampled_process_tree_peak_rss_bytes"] >= 20 * 1024**2
    assert report["direct_children_high_water_rss_bytes"] > 0


def synthetic_process_memory():
    # Synthetic contract evidence is confined to this test; no empirical claim.
    return {"schema_version": 1, "returncode": 0, "elapsed_seconds": 1.0,
            "sample_interval_seconds": .2, "sample_count": 1,
            "direct_children_high_water_rss_bytes": 16384,
            "sampled_process_tree_peak_rss_bytes": 8192,
            "samples": [{"elapsed_seconds": .1, "rss_bytes": 8192, "process_count": 1}]}


@pytest.mark.parametrize("phases,memory,expected", [(False, False, "fail"),
                                                   (False, True, "fail"),
                                                   (True, False, "fail"),
                                                   (True, True, "pass")])
def test_zero_exit_calibration_requires_phases_and_memory(tmp_path, monkeypatch, phases, memory, expected):
    module = load()
    monkeypatch.setattr(module, "verify_worker", lambda *_a, **_k: "device=1")
    monkeypatch.setattr(module.ProbeRunner, "_image", lambda *_: "sha256:" + "a" * 64)
    source = tmp_path / "source"
    source.mkdir()
    (source / "calibrate.py").write_text("pass\n")
    output = tmp_path / "run"
    output.mkdir()
    def launch(runner, stage, image, job, mounts, timeout, **kwargs):
        destination = next(path for path, target, _ in mounts if target == "/output")
        if phases:
            (destination / "phases.jsonl").write_text('\n'.join(json.dumps({"phase": name}) for name in ["loading", "warmup", "measurement"]))
        if memory:
            (destination / "process-memory.json").write_text(json.dumps(synthetic_process_memory()))
        receipt = runner.store.put_json({"synthetic_test": True})
        return SimpleNamespace(code=0, timed_out=False, limit_exceeded=None, stdout=""), receipt
    monkeypatch.setattr(module.ProbeRunner, "_launch", launch)
    args = SimpleNamespace(output=output, cpus=4, memory_mb=8192, timeout=300,
                           gpu_index=0, runtime="podman", sample_interval=.2)
    calibration = {"name": "no-op-fixture", "profile": "cpu-tabular", "source_dir": str(source),
                   "command": ["python", "/calibration/calibrate.py"], "resources": {},
                   "expected_phases": ["loading", "warmup", "measurement"]}
    entry = {"name": "no-op-fixture", "profile": "cpu-tabular", "image": "sha256:" + "a" * 64,
             "kind": "declared_workload_calibration"}
    report = module.run_one(entry, args, "device=1", calibration)
    assert report["execution_status"] == "pass"
    assert report["status"] == expected
    assert report["peak_calibration_status"] == ("complete" if expected == "pass" else "incomplete")
    assert report["task_readiness"] == "unverified"


def test_calibration_rejects_unobserved_gpu_and_invalid_peak(tmp_path):
    module = load()
    path = tmp_path / "process-memory.json"
    memory = synthetic_process_memory()
    path.write_text(json.dumps(memory))
    gpu = {"status": "unverified", "sample_count": 1, "errors": [],
           "sampled_compute_process_peak_mb": None, "device_peak_used_mb": 10}
    assert module.calibration_memory_evidence(path, gpu, True)["status"] == "unverified"
    memory["sampled_process_tree_peak_rss_bytes"] = float("nan")
    path.write_text(json.dumps(memory))
    assert module.calibration_memory_evidence(path, None, False)["status"] == "unverified"
