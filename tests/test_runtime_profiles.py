import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

from scientisttwo.benchmarks.models import NativeResources
import scientisttwo.runtime_profiles as profiles_module
from scientisttwo.runtime_profiles import (
    GpuDevice, RuntimeAsset, enforce_gpu_allocation, gpu_inventory, probe_source, runtime_profile,
    runtime_profiles, slurm_gpu_allocation, validate_gpu_selector,
)


def probe_module():
    spec = importlib.util.spec_from_file_location("runtime_probe_test", probe_source())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profiles_separate_frameworks_without_loading_them():
    profiles = runtime_profiles()
    assert {profile.name for profile in profiles} == {"cpu-tabular", "torch-cuda", "tensorflow-cuda", "jax-clrs"}
    assert all(profile.image_status == "unverified" for profile in profiles if profile.name != "cpu-tabular")
    original = runtime_profile("torch-cuda")
    original.requirements.imports.append("fake")
    assert "fake" not in runtime_profile("torch-cuda").requirements.imports
    with pytest.raises(ValueError):
        runtime_profile("auto")


@pytest.mark.parametrize("path", ["/absolute", "../secret", "nested/../../secret", "a\\b"])
def test_assets_reject_escape(path):
    with pytest.raises(ValueError):
        RuntimeAsset(path=path, sha256="a" * 64, size=3)


def test_probe_real_cpu_compute_and_streamed_asset_integrity(tmp_path, monkeypatch):
    for key in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
        monkeypatch.setenv(key, "1")
    data = b"frozen model asset" * 10000
    (tmp_path / "weights").write_bytes(data)
    requirements = {"imports": ["numpy"], "framework": "numpy", "min_shm_mb": 0,
                    "assets": [{"path": "weights", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}]}
    # A macOS host has no /dev/shm; this unit test substitutes only the disk probe path.
    module = probe_module()
    real_usage = module.shutil.disk_usage
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: real_usage(tmp_path))
    report = module.probe(requirements, tmp_path, tmp_path)
    assert report["status"] == "pass"
    (tmp_path / "weights").write_bytes(b"x" * len(data))
    failed = module.probe(requirements, tmp_path, tmp_path)
    assert failed["status"] == "fail"
    assert any(check["name"] == "asset:weights" and check["status"] == "fail" for check in failed["checks"])


def test_probe_missing_cuda_does_not_pass(tmp_path, monkeypatch):
    module = probe_module()
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    report = module.probe({"cuda": True, "framework": "numpy"}, tmp_path, tmp_path)
    assert report["status"] == "fail"
    assert any(check["name"] == "gpu_resources" and check["status"] == "fail" for check in report["checks"])


def test_runtime_lock_files_are_hash_pinned():
    root = Path(__file__).parents[1] / "runtime-profiles"
    for name in ["cpu-tabular", "torch-cuda", "tensorflow-cuda", "jax-clrs"]:
        lock = root / name / "requirements.lock"
        assert lock.exists(), f"Missing resolved lock: {name}"
        lines = lock.read_text().splitlines()
        entries = [line for line in lines if line and line[0].isalnum() and not line.startswith("    ")]
        assert entries and all("==" in line for line in entries)
        assert "--hash=sha256:" in lock.read_text()
        assert "--require-hashes" in (root / name / "Dockerfile").read_text()
        expected = {match.group(1): match.group(2) for match in re.finditer(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", lock.read_text(), re.MULTILINE)}
        profile = runtime_profile(name)
        assert profile.requirements.package_versions == expected
        assert profile.dependency_lock_sha256 == hashlib.sha256(lock.read_bytes()).hexdigest()


GPU_A = "GPU-00000000-0000-0000-0000-000000000001"
GPU_B = "GPU-00000000-0000-0000-0000-000000000002"
GPU_C = "GPU-00000000-0000-0000-0000-000000000003"


@pytest.fixture
def synthetic_gpu_inventory(monkeypatch):
    # Deliberately shuffled Linux minors and NVML indices: unit-test identities
    # only. No live hardware validation or runtime-ready claim is made here.
    result = [
        GpuDevice(index=0, minor_number=3, uuid=GPU_A, pci_bus_id="00000000:0A:00.0"),
        GpuDevice(index=4, minor_number=6, uuid=GPU_B, pci_bus_id="00000000:8D:00.0"),
        GpuDevice(index=6, minor_number=2, uuid=GPU_C, pci_bus_id="00000000:7A:00.0"),
    ]
    monkeypatch.setattr(profiles_module, "gpu_inventory", lambda: result)
    return result


def test_slurm_uses_device_minors_and_returns_uuid_not_cuda_ordinals(synthetic_gpu_inventory):
    env = {"SLURM_JOB_ID": "123", "SLURM_JOB_GPUS": "0-7", "SLURM_STEP_GPUS": "3,6",
           "CUDA_VISIBLE_DEVICES": "0,1"}
    assert slurm_gpu_allocation(env) == "device=" + GPU_A
    assert slurm_gpu_allocation(env, gpu_index=1) == "device=" + GPU_B
    assert slurm_gpu_allocation(env, gpu_uuid=GPU_B) == "device=" + GPU_B
    assert enforce_gpu_allocation("device=4", env) == "device=" + GPU_B
    assert enforce_gpu_allocation("device=0,4", env) == "device=" + GPU_A + "," + GPU_B
    assert enforce_gpu_allocation("device=" + GPU_B, env) == "device=" + GPU_B
    # Numeric 6 denotes the current NVIDIA index (minor 2), not Slurm minor 6.
    for requested in ["all", "device=6", "device=" + GPU_C, "device=3,3"]:
        with pytest.raises(ValueError):
            enforce_gpu_allocation(requested, env)


def test_slurm_bound_uuid_overrides_default_ordinal_and_remains_restricted(synthetic_gpu_inventory):
    env = {"SLURM_JOB_ID": "123", "SLURM_STEP_GPUS": "3,6", "SCIENTISTTWO_GPU_UUID": GPU_B}
    assert slurm_gpu_allocation(env) == "device=" + GPU_B
    assert enforce_gpu_allocation("device=4", env) == "device=" + GPU_B
    with pytest.raises(ValueError, match="bound GPU UUID"):
        slurm_gpu_allocation(env, gpu_uuid=GPU_A)
    with pytest.raises(ValueError, match="bound GPU UUID"):
        enforce_gpu_allocation("device=" + GPU_A, env)


def test_slurm_explicit_uuid_step_and_missing_device(synthetic_gpu_inventory):
    env = {"SLURM_JOB_ID": "123", "SLURM_STEP_GPUS": GPU_B}
    assert slurm_gpu_allocation(env) == "device=" + GPU_B
    with pytest.raises(ValueError, match="outside"):
        slurm_gpu_allocation(env, gpu_uuid=GPU_A)
    env["SLURM_STEP_GPUS"] = "99"
    with pytest.raises(ValueError, match="absent"):
        slurm_gpu_allocation(env)
    env["SLURM_STEP_GPUS"] = "6," + GPU_B
    with pytest.raises(ValueError, match="duplicate identities"):
        slurm_gpu_allocation(env)


@pytest.mark.parametrize("step", [None, ""])
def test_cpu_step_never_inherits_parent_job_gpu_authorization(monkeypatch, step):
    monkeypatch.setattr(profiles_module, "gpu_inventory",
                        lambda: pytest.fail("CPU step must reject before querying GPUs"))
    env = {"SLURM_JOB_ID": "123", "SLURM_JOB_GPUS": "0-7"}
    if step is not None:
        env["SLURM_STEP_GPUS"] = step
    for operation in [lambda: slurm_gpu_allocation(env),
                      lambda: enforce_gpu_allocation("device=" + GPU_B, env)]:
        with pytest.raises(ValueError, match="parent-job"):
            operation()
    assert enforce_gpu_allocation(None, env) is None


@pytest.mark.parametrize("selector", ["", "GPU-unknown", "0,0", "4-2", "-1", "0-2048"])
def test_slurm_rejects_ambiguous_or_invalid_gpu_ids(selector):
    with pytest.raises(ValueError):
        slurm_gpu_allocation({"SLURM_JOB_ID": "123", "SLURM_STEP_GPUS": selector})


def test_slurm_launcher_selection_and_personal_host_scope(synthetic_gpu_inventory):
    assert enforce_gpu_allocation("all", {}) == "all"
    env = {"SLURM_JOB_ID": "123", "SLURM_STEP_GPUS": "3,6",
           "SCIENTISTTWO_GPU_ALLOCATION": "device=" + GPU_A}
    with pytest.raises(ValueError, match="launcher"):
        enforce_gpu_allocation("device=4", env)


@pytest.mark.parametrize("selector", [
    "device=", "device=,0", "device=0,", "device=00", "device=0,0",
    "device=GPU-4f81c4b7", "device=GPU-00000000-0000-0000-0000-00000000000z",
    "device=" + GPU_A + ",0", "device=" + GPU_A + "," + GPU_A,
    "device=MIG-" + GPU_A, "device=" + GPU_A + "\n",
])
def test_native_resources_rejects_malformed_or_duplicate_gpu_selectors(selector):
    with pytest.raises(ValueError):
        NativeResources(gpus=selector)
    with pytest.raises(ValueError):
        validate_gpu_selector(selector)


@pytest.mark.parametrize("selector", [None, "all", "device=0", "device=0,4",
                                     "device=" + GPU_A, "device=" + GPU_A + "," + GPU_B])
def test_native_resources_accepts_strict_numeric_and_full_uuid_selectors(selector):
    assert NativeResources(gpus=selector).gpus == selector


def inventory_outputs(*, duplicate=False, different_bus=False):
    xml = f"""<nvidia_smi_log><gpu><uuid>{GPU_B}</uuid><minor_number>6</minor_number>
    <pci><pci_bus_id>00000000:8D:00.0</pci_bus_id></pci></gpu></nvidia_smi_log>"""
    if duplicate:
        xml = xml.replace("</nvidia_smi_log>", xml.split("<nvidia_smi_log>")[1])
    bus = "00000000:0A:00.0" if different_bus else "00000000:8D:00.0"
    return xml, f"4, {GPU_B}, {bus}\n"


def test_real_inventory_parser_joins_xml_minor_by_uuid(monkeypatch):
    xml, csv = inventory_outputs()
    def invoke(argv, **kwargs):
        assert kwargs["check"] and kwargs["timeout"] == 10
        assert "minor_number" not in " ".join(argv[1:])
        return SimpleNamespace(stdout=xml if argv[1:] == ["-q", "-x"] else csv)
    monkeypatch.setattr(profiles_module.subprocess, "run", invoke)
    assert gpu_inventory() == [
        GpuDevice(index=4, minor_number=6, uuid=GPU_B, pci_bus_id="00000000:8D:00.0")
    ]


@pytest.mark.parametrize("case", ["mismatch", "duplicate", "malformed", "missing", "unknown_uuid"])
def test_inventory_parser_fails_closed_on_missing_or_inconsistent_driver_evidence(monkeypatch, case):
    xml, csv = inventory_outputs(duplicate=case == "duplicate", different_bus=case == "mismatch")
    if case == "malformed":
        xml = "<broken"
    if case == "missing":
        xml = xml.replace("<minor_number>6</minor_number>", "")
    if case == "unknown_uuid":
        csv = csv.replace(GPU_B, GPU_C)
    monkeypatch.setattr(profiles_module.subprocess, "run",
                        lambda argv, **_: SimpleNamespace(stdout=xml if argv[1:] == ["-q", "-x"] else csv))
    with pytest.raises(ValueError, match="Cannot verify"):
        gpu_inventory()


def test_gpu_mapping_unavailable_fails_instead_of_using_unverified_numbers(monkeypatch):
    monkeypatch.setattr(profiles_module.subprocess, "run",
                        lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi unavailable")))
    env = {"SLURM_JOB_ID": "123", "SLURM_STEP_GPUS": "6"}
    with pytest.raises(ValueError, match="Cannot verify"):
        slurm_gpu_allocation(env)
    with pytest.raises(ValueError, match="Cannot verify"):
        enforce_gpu_allocation("device=4", env)


def test_probe_python_minor_and_explicit_cpu_platform_adaptation(tmp_path, monkeypatch):
    module = probe_module()
    real_usage = module.shutil.disk_usage
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: real_usage(tmp_path))
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(module.platform, "python_version", lambda: "3.12.15")
    for key in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
        monkeypatch.setenv(key, "1")
    requirements = {"python_minor": "3.12", "resolution_platform": "x86_64-manylinux_2_28",
                    "allow_adapted_cpu_platform": True}
    report = module.probe(requirements, tmp_path, tmp_path)
    assert report["status"] == "pass"
    contract = next(check["value"] for check in report["checks"] if check["name"] == "python_contract")
    assert contract["execution_variant"] == "adapted-cpu-platform"
    requirements["python_minor"] = "3.11"
    assert module.probe(requirements, tmp_path, tmp_path)["status"] == "fail"
    requirements.update(python_minor="3.12", allow_adapted_cpu_platform=False)
    assert module.probe(requirements, tmp_path, tmp_path)["status"] == "fail"


def test_resolver_refresh_updates_packaged_receipt_without_stale_hash(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / "runtime-profiles/resolve.py"
    specification = importlib.util.spec_from_file_location("runtime_resolver_test", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    root = tmp_path / "runtime-profiles/torch-cuda"
    root.mkdir(parents=True)
    (root / "requirements.in").write_text("numpy==1.26.4\n")
    packaged = tmp_path / "src/scientisttwo/runtime_locks.json"
    packaged.parent.mkdir(parents=True)
    packaged.write_text(json.dumps({"torch-cuda": {"old": True}, "untouched": {"retain": True}}))
    module.__file__ = str(tmp_path / "runtime-profiles/resolve.py")
    def compile_only(command, **kwargs):
        assert command[1:3] == ["pip", "compile"]
        destination = Path(command[command.index("--output-file") + 1])
        destination.write_text("numpy==1.26.4 \\\n    --hash=sha256:" + "a" * 64 + "\n")
    monkeypatch.setattr(module.subprocess, "run", compile_only)
    monkeypatch.setattr(sys, "argv", ["resolve.py", "torch-cuda", "--uv", "fixture-uv"])
    module.main()
    updated = json.loads(packaged.read_text())
    assert updated["untouched"] == {"retain": True}
    assert updated["torch-cuda"]["package_versions"] == {"numpy": "1.26.4"}
    assert updated["torch-cuda"]["lock_sha256"] == hashlib.sha256((root / "requirements.lock").read_bytes()).hexdigest()
    assert updated["torch-cuda"] == json.loads((root / "resolution.json").read_text())
    assert not list(packaged.parent.glob("*.tmp"))
