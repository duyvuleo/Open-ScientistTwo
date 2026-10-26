"""Trusted source strings executed only inside development containers."""

VECTORIZATION_DRIVER = """import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

job = json.loads(Path("/control/job.json").read_text())
parameters = job["execution"]["parameters"]
sys.path.insert(0, "/workspace")
spec = importlib.util.spec_from_file_location("candidate", "/workspace/train.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
inputs = np.load("/data/public/inputs.npz", allow_pickle=False)
outputs = {}
for i, case in enumerate(parameters["cases"]):
    x, k, b = inputs[f"x{i}"], inputs[f"k{i}"], inputs[f"b{i}"]
    layer = module.Conv2DLayer(
        k.shape[2], k.shape[3], k.shape[0], case["stride"], case["padding"], case["activation"]
    )
    layer.kernel_matrices, layer.biases = k.copy(), b.copy()
    outputs[str(i)] = layer.forward(x.copy())
np.savez("predictions.npz", **outputs)
"""
