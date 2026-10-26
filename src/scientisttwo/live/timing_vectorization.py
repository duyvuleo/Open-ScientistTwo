"""Vectorization-specific workload and independent oracle for generic paired timing."""

from pathlib import Path
import time
import zipfile

import numpy as np

from ..artifacts import ArtifactSpec, freeze_bundle, load_bundle, safe_path, verified_file
from ..benchmarks.files import verify_tree
from ..store import digest
from .timing import run_paired_timing
from .timing_runtime import ContainerTimingTransport


VECTORIZATION_TIMING_HANDLER = r"""import importlib.util
from pathlib import Path
import sys
import numpy as np

def initialize(parameters):
    global candidate, cases, arrays, counter
    np.random.seed(parameters.get("seed", 0))
    sys.path.insert(0, "/workspace")
    spec = importlib.util.spec_from_file_location("candidate", "/workspace/train.py")
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    cases = parameters["cases"]
    with np.load("/data/public/inputs.npz", allow_pickle=False) as inputs:
        arrays = {name: inputs[name].copy() for name in inputs.files}
    counter = 0

def invoke(payload):
    global counter
    if payload != {"workload": "all_cases"}:
        raise ValueError("Unknown fixed vectorization workload")
    counter += 1
    results = {}
    for index, case in enumerate(cases):
        x, k, b = arrays[f"x{index}"], arrays[f"k{index}"], arrays[f"b{index}"]
        layer = candidate.Conv2DLayer(k.shape[2], k.shape[3], k.shape[0], case["stride"], case["padding"], case["activation"])
        layer.kernel_matrices, layer.biases = k.copy(), b.copy()
        results[str(index)] = layer.forward(x.copy())
    path = f"predictions-{counter:04d}.npz"
    np.savez(Path("/outputs") / path, **results)
    return {"predictions": path}
"""


def expected_outputs(public: Path, cases: list[dict]) -> dict[str, np.ndarray]:
    """Trusted NumPy reference, never imports candidate source or worker helpers."""
    expected = {}
    with np.load(public / "inputs.npz", allow_pickle=False) as arrays:
        for index, case in enumerate(cases):
            x, k, bias = arrays[f"x{index}"], arrays[f"k{index}"], arrays[f"b{index}"]
            padding, stride = case["padding"], case["stride"]
            x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
            windows = np.lib.stride_tricks.sliding_window_view(x, k.shape[:2], axis=(1, 2))[
                :, ::stride, ::stride
            ]
            result = np.einsum("nhwcij,ijco->nhwo", windows, k) + bias
            expected[str(index)] = np.maximum(result, 0) if case["activation"] == "relu" else result
    return expected


def vectorization_oracle(store, outputs: dict[str, Path], expected: dict[str, np.ndarray]):
    seen = {"baseline": set(), "candidate": set()}

    def check(arm, payload, response):
        if set(response) != {"predictions"} or payload != {"workload": "all_cases"}:
            raise ValueError("Unexpected vectorization timing response")
        relative = safe_path(response["predictions"])
        if relative in seen[arm] or not relative.endswith(".npz"):
            raise ValueError("Timing workers must produce a fresh prediction artifact per request")
        seen[arm].add(relative)
        ref = freeze_bundle(
            store,
            outputs[arm],
            [
                ArtifactSpec(
                    path=relative,
                    role="timing_predictions",
                    max_bytes=128_000_000,
                    max_files=1,
                )
            ],
        )
        artifact = load_bundle(store, ref).files[relative]
        path = verified_file(store, artifact)
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 128_000_000:
                raise ValueError("Oversized decompressed timing predictions")
        maximum = 0.0
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != set(expected):
                raise ValueError("Missing convolution output arrays")
            for name, truth in expected.items():
                result = values[name]
                if (
                    result.shape != truth.shape
                    or not np.isfinite(result).all()
                    or not np.allclose(result, truth, rtol=1e-7, atol=1e-8)
                ):
                    raise ValueError("Independent convolution correctness check failed")
                maximum = max(maximum, float(np.max(np.abs(result - truth))))
        return {
            "valid": True,
            "max_abs_error": maximum,
            "bundle_ref": ref,
            "prediction_ref": artifact.ref,
        }

    return check


def run_vectorization_pair(
    store,
    spec,
    image: str,
    baseline_dir: Path,
    candidate_dir: Path,
    prepared: dict,
    output_root: Path,
    *,
    timeout: float,
    seed: int = 0,
):
    """CPU-only evaluated wrapper; retain all outputs and return report plus worker receipts."""
    if spec.resources.gpus:
        raise ValueError(
            "Vectorization timing adapter is CPU-only; use a CUDA-specific synchronized adapter"
        )
    started = time.monotonic()

    def remaining():
        value = timeout - (time.monotonic() - started)
        if value <= 0:
            raise TimeoutError("Paired timing exhausted its total budget including startup")
        return value

    public = Path(prepared["public"])
    verify_tree(public, prepared["files"])
    expected = expected_outputs(public, prepared["cases"])
    prefix = [spec.runtime] + (["--context", spec.context] if spec.context else [])
    transports = {}
    receipts = {}
    failure = None
    outputs = {arm: output_root / arm / "outputs" for arm in ("baseline", "candidate")}
    try:
        for arm, source in (("baseline", baseline_dir), ("candidate", candidate_dir)):
            transports[arm] = ContainerTimingTransport(
                prefix=prefix,
                image=image,
                source=source,
                public=public,
                control=output_root / arm / "control",
                outputs=outputs[arm],
                resources=spec.resources,
                handler_source=VECTORIZATION_TIMING_HANDLER,
                parameters={"cases": prepared["cases"], "seed": seed},
                startup_timeout=min(60, remaining()),
                execution_timeout=remaining(),
            )
        report = run_paired_timing(
            transports,
            [{"workload": "all_cases"}],
            vectorization_oracle(store, outputs, expected),
            request_timeout=min(spec.resources.seconds, remaining()),
            total_timeout=remaining(),
        )
        verify_tree(public, prepared["files"])
    except BaseException as error:
        failure = type(error).__name__
        raise
    finally:
        retention_errors = {}
        for arm, transport in transports.items():
            try:
                transport.close()
            except Exception as error:
                retention_errors[arm + "/close"] = f"{type(error).__name__}: {error}"
        for arm in outputs:
            transport = transports.get(arm)
            log = output_root / arm / "control" / "worker.log"
            if transport is not None:
                try:
                    receipt = transport.receipt(store)
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    retention_errors[arm + "/receipt"] = message
                    receipt = {"execution_enforcement_error": message,
                               "closed": transport.closed}
            elif log.exists():
                receipt = {"startup_failed": True}
            else:
                continue
            if log.exists():
                receipt["log_ref"] = store.put_file(log, "timing_worker_log")
            else:
                retention_errors[arm + "/log"] = "Worker log is missing"
            receipts[arm] = store.put_json(receipt, "timing_worker_execution")
        attempt = store.put_json(
            {
                "receipts": receipts,
                "status": "failed" if failure or retention_errors else "completed",
                "failure_category": failure or ("ExecutionEvidenceError" if retention_errors else None),
                "retention_errors": retention_errors,
            },
            "paired_timing_attempt",
        )
        store.set("paired_timing/" + digest(str(output_root).encode()), attempt)
        if retention_errors and failure is None:
            raise RuntimeError("Paired timing enforcement or cleanup failed; "
                               "both worker logs and the failed attempt were retained")
    return report, receipts


def audit_vectorization_report(store, prepared, report):
    """Replay every warmup and measured correctness check from sealed output bytes."""
    expected = expected_outputs(Path(prepared["public"]), prepared["cases"])
    seen = {"baseline": set(), "candidate": set()}
    for sample in report.samples:
        evidence = sample.correctness
        relative = safe_path(sample.response["predictions"])
        if relative in seen[sample.arm]:
            raise ValueError("Timing report reuses a prediction filename")
        seen[sample.arm].add(relative)
        manifest = load_bundle(store, evidence["bundle_ref"])
        if (
            set(manifest.files) != {relative}
            or manifest.files[relative].ref != evidence["prediction_ref"]
        ):
            raise ValueError("Timing output differs from its declared artifact manifest")
        path = verified_file(store, manifest.files[relative])
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 128_000_000:
                raise ValueError("Oversized decompressed timing predictions")
        maximum = 0.0
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != set(expected):
                raise ValueError("Timing prediction keys differ from the reference workload")
            for name, truth in expected.items():
                result = values[name]
                if (
                    result.shape != truth.shape
                    or not np.isfinite(result).all()
                    or not np.allclose(result, truth, rtol=1e-7, atol=1e-8)
                ):
                    raise ValueError("Independent convolution correctness replay failed")
                maximum = max(maximum, float(np.max(np.abs(result - truth))))
        if maximum != evidence["max_abs_error"] or evidence["valid"] is not True:
            raise ValueError("Timing correctness diagnostic differs from sealed predictions")
    return {"predictions": "pass", "sample_count": len(report.samples)}
