"""Standalone probe: run inside the actual restricted experiment container."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


def gpu_resources(minimum):
    """Measure only explicitly selected UUIDs when CUDA uses stable identities."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected = None
    if visible is not None and "GPU-" in visible:
        selected = visible.split(",")
        pattern = r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
        if (any(re.fullmatch(pattern, value) is None for value in selected)
                or len(selected) != len(set(selected))):
            raise ValueError("CUDA GPU selection requires distinct complete UUIDs")
    command = [
        "nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,uuid",
        "--format=csv,noheader,nounits",
    ]
    if selected is not None:
        command.append("--id=" + ",".join(selected))
    proc = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    devices = []
    for line in proc.stdout.splitlines():
        name, total, free, driver, identifier = [part.strip() for part in line.split(",")]
        devices.append({"name": name, "total_mb": int(total), "free_mb": int(free),
                        "driver": driver, "uuid": identifier})
    if selected is not None:
        observed = [device["uuid"] for device in devices]
        if len(observed) != len(selected) or set(observed) != set(selected):
            raise ValueError("GPU memory query did not return exactly the selected UUIDs")
        if any(device["free_mb"] < minimum for device in devices):
            raise ValueError("Insufficient available memory on a selected GPU")
    elif not devices or not any(device["free_mb"] >= minimum for device in devices):
        raise ValueError("Insufficient available GPU memory")
    return devices


def probe(requirements, assets, workspace):
    checks = []

    def check(name, function):
        try:
            value = function()
            checks.append({"name": name, "status": "pass", "value": value})
        except Exception as exc:
            checks.append({"name": name, "status": "fail", "error_type": type(exc).__name__})

    def cpu_identity():
        source = Path("/proc/cpuinfo")
        models = set()
        if source.is_file():
            for line in source.read_text().splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip() in {"model name", "Hardware", "CPU implementer", "CPU part"}:
                    models.add(key.strip() + ": " + value.strip())
        return {"architecture": platform.machine(), "models": sorted(models),
                "processor": platform.processor()}
    check("cpu_identity", cpu_identity)
    if requirements.get("python_minor"):
        def python_contract():
            actual = ".".join(platform.python_version().split(".")[:2])
            if actual != requirements["python_minor"]:
                raise ValueError("Worker Python differs from the resolved profile")
            architecture = platform.machine()
            adapted = architecture not in {"x86_64", "AMD64"}
            if adapted and not (requirements.get("allow_adapted_cpu_platform") and architecture in {"aarch64", "arm64"}):
                raise ValueError("Worker architecture differs from the resolved profile")
            return {"python_minor": actual, "architecture": architecture,
                    "resolution_platform": requirements.get("resolution_platform"),
                    "execution_variant": "adapted-cpu-platform" if adapted else "resolved-platform"}
        check("python_contract", python_contract)

    for module in requirements.get("imports", []):
        check("import:" + module, lambda module=module: bool(importlib.import_module(module)))
    for package, expected in requirements.get("package_versions", {}).items():
        def version(package=package, expected=expected):
            actual = importlib.metadata.version(package)
            if actual != expected and ("+" in expected or actual.split("+")[0] != expected):
                raise ValueError("Pinned package version mismatch")
            return actual
        check("version:" + package, version)

    for path, minimum, label in (
        (workspace, requirements.get("min_disk_mb", 0), "workspace"),
        (Path("/dev/shm"), requirements.get("min_shm_mb", 0), "shared_memory"),
        (Path("/tmp"), requirements.get("min_tmp_mb", 0), "temporary_storage"),
    ):
        def disk(path=path, minimum=minimum):
            free = shutil.disk_usage(path).free // (1024 * 1024)
            if free < minimum:
                raise ValueError("Insufficient free storage")
            return {"free_mb": free, "required_mb": minimum}
        check(label, disk)

    check("offline_environment", lambda: _offline())
    for asset in requirements.get("assets", []):
        def verify(asset=asset):
            relative = Path(asset["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid asset path")
            path = assets / relative
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != assets.parent):
                raise ValueError("Symlink asset")
            if not path.is_file() or path.stat().st_size != asset["size"]:
                raise ValueError("Missing or wrong-sized asset")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != asset["sha256"]:
                raise ValueError("Asset checksum mismatch")
            return {"sha256": digest.hexdigest(), "size": path.stat().st_size}
        check("asset:" + asset["path"], verify)

    if requirements.get("cuda"):
        check("gpu_resources", lambda: gpu_resources(requirements.get("min_vram_mb", 0)))

    def compute():
        framework = requirements.get("framework", "numpy")
        cuda = requirements.get("cuda", False)
        if framework == "torch":
            import torch
            if cuda and not torch.cuda.is_available():
                raise ValueError("CUDA unavailable")
            device = "cuda" if cuda else "cpu"
            output = torch.ones((64, 64), device=device) @ torch.ones((64, 64), device=device)
            if cuda:
                torch.cuda.synchronize()
            assert float(output[0, 0]) == 64
            workloads = ["matrix_multiply"]
            if "torch_geometric" in requirements.get("imports", []):
                from torch_geometric.nn import SAGEConv
                graph = SAGEConv(4, 2).to(device)
                edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], device=device)
                assert graph(torch.ones((3, 4), device=device), edge_index).shape == (3, 2)
                workloads.append("graphsage_forward")
            if "transformers" in requirements.get("imports", []):
                from transformers import GPT2Config, GPT2LMHeadModel
                model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=8, n_embd=8,
                                                  n_layer=1, n_head=2)).to(device).eval()
                assert model(torch.ones((1, 4), dtype=torch.long, device=device)).logits.shape == (1, 4, 16)
                workloads.append("gpt2_forward")
            if "torchvision" in requirements.get("imports", []):
                from torchvision.models import resnet18
                image_model = resnet18(weights=None).to(device).eval()
                assert image_model(torch.zeros((1, 3, 32, 32), device=device)).shape == (1, 1000)
                workloads.append("resnet_forward")
            if cuda:
                torch.cuda.synchronize()
            return {"device": device, "cuda": torch.version.cuda,
                    "workloads": workloads}
        if framework == "tensorflow":
            import tensorflow as tf
            devices = tf.config.list_physical_devices("GPU")
            if cuda and not devices:
                raise ValueError("CUDA unavailable")
            with tf.device("/GPU:0" if cuda else "/CPU:0"):
                result = tf.matmul(tf.ones((64, 64)), tf.ones((64, 64))).numpy()
            assert float(result[0, 0]) == 64
            return {"devices": [device.name for device in devices]}
        if framework == "jax":
            import jax
            import jax.numpy as jnp
            if cuda and not any(device.platform == "gpu" for device in jax.devices()):
                raise ValueError("CUDA unavailable")
            result = (jnp.ones((64, 64)) @ jnp.ones((64, 64))).block_until_ready()
            assert float(result[0, 0]) == 64
            return {"devices": [str(device) for device in jax.devices()]}
        import numpy as np
        assert (np.ones((64, 64)) @ np.ones((64, 64)))[0, 0] == 64
        return {"device": "cpu"}
    check("actual_compute", compute)
    return {"schema_version": 1, "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
            "platform": platform.platform(), "python": platform.python_version(), "checks": checks}


def _offline():
    required = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise ValueError("Offline flags are required")
    return required


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    report = probe(json.loads(args.requirements.read_text()), args.assets, args.workspace)
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)
