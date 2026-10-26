"""Container-only probe wrapper; allocator peaks cover this tiny process only."""

import importlib.util
import json
from pathlib import Path
import sys


def allocator_peaks(framework, cuda):
    if not cuda:
        return {"status": "not_applicable", "framework": framework}
    try:
        if framework == "torch" and "torch" in sys.modules:
            torch = sys.modules["torch"]
            torch.cuda.synchronize()
            return {"status": "measured", "framework": framework,
                    "scope": "PyTorch allocator since this process initialized CUDA; includes tiny model loading/compute",
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "exclusions": "driver and other libraries outside PyTorch allocator; real task/model peaks"}
        if framework == "tensorflow" and "tensorflow" in sys.modules:
            memory = sys.modules["tensorflow"].config.experimental.get_memory_info("GPU:0")
            return {"status": "measured", "framework": framework, "scope": "TensorFlow allocator",
                    "peak_allocated_bytes": memory["peak"], "current_bytes": memory["current"]}
        if framework == "jax" and "jax" in sys.modules:
            jax = sys.modules["jax"]
            records = [{"device": str(device), "stats": device.memory_stats()} for device in jax.devices() if device.platform == "gpu"]
            return {"status": "measured" if records and all(record["stats"] for record in records) else "unverified",
                    "framework": framework, "scope": "JAX device allocator", "devices": records}
    except Exception as exc:
        return {"status": "unverified", "framework": framework, "error_type": type(exc).__name__}
    return {"status": "unverified", "framework": framework}


def main():
    specification = importlib.util.spec_from_file_location("frozen_runtime_probe", "/runtime/probe.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    requirements = json.loads(Path("/runtime/requirements.json").read_text())
    report = module.probe(requirements, Path("/runtime"), Path("/output"))
    memory = allocator_peaks(requirements["framework"], requirements["cuda"])
    Path("/output/framework-memory.json").write_text(json.dumps(memory, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
