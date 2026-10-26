"""Fixed-weight Llama inference through generic paired, GPU-synchronized timing.

Raw request latency is measured by the parent. Expected token IDs never enter the
candidate container; generated code is never imported by the oracle or auditor.
"""
from pathlib import Path
import json
import time
import zipfile

import numpy as np

from ..artifacts import ArtifactSpec, freeze_bundle, load_bundle, safe_path, verified_file
from ..benchmarks.files import verify_tree
from ..runtime_contracts import asset_mounts
from ..store import digest
from ..task_families.array_data import archive_path, open_arrays
from .timing import run_paired_timing
from .timing_runtime import ContainerTimingTransport


LLAMA_TIMING_HANDLER = r'''import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

def initialize(parameters):
    global model, arrays, ids, batch_size, counter
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA allocation is required")
    torch.manual_seed(parameters["seed"])
    torch.cuda.manual_seed_all(parameters["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = json.loads(Path("/workspace/training.json").read_text())
    config["seed"] = parameters["seed"]
    batch_size = int(config.get("batch_size", 4))
    if batch_size != 4:
        raise ValueError("Audited v1 freezes Llama batch size at four")
    architecture = json.loads(Path("/assets/model/config.json").read_text())
    if architecture.get("hidden_size") != 4096 or architecture.get("num_hidden_layers") != 32:
        raise ValueError("Expected a declared compatible 7B model")
    import sys
    sys.path.insert(0, "/workspace")
    from llama_model import load_optimized_model
    model, identity = load_optimized_model(Path("/assets"), Path("/workspace"))
    Path("/outputs/model-identity.json").write_text(json.dumps(identity, sort_keys=True))
    from array_data import archive_path, open_arrays
    with open_arrays(archive_path(Path("/data/public"), "evaluation")) as data:
        if set(data.files) != {"ids", "input_ids", "attention_mask"}:
            raise ValueError("Unexpected evaluation input fields")
        ids = data["ids"].astype(str)
        if data["input_ids"].ndim != 2 or data["input_ids"].shape[1] != 128:
            raise ValueError("Audited v1 requires fixed 128-token requests")
        arrays = {key: torch.as_tensor(data[key].astype(np.int64), device="cuda")
                  for key in ("input_ids", "attention_mask")}
    counter = 0

def synchronize():
    torch.cuda.synchronize()

def invoke(payload):
    global counter
    if payload != {"workload": "fixed_one_token_generation"}:
        raise ValueError("Unknown fixed workload")
    outputs = []
    with torch.inference_mode():
        for offset in range(0, len(ids), batch_size):
            batch = {key: value[offset:offset+batch_size] for key, value in arrays.items()}
            tokens = model.generate(**batch, max_new_tokens=1, do_sample=False, pad_token_id=0)
            outputs.append(tokens[:, -1:].detach().cpu().numpy())
    synchronize()
    counter += 1
    relative = f"predictions-{counter:04d}.npz"
    np.savez(Path("/outputs") / relative, ids=ids, predictions=np.concatenate(outputs))
    return {"predictions": relative}
'''


def expected_tokens(prepared):
    private = Path(prepared["private"])
    verify_tree(private, prepared["private_files"])
    with open_arrays(archive_path(private, "labels")) as archive:
        if set(archive.files) != {"ids", "y"}:
            raise ValueError("Frozen Llama reference must contain IDs and expected tokens only")
        ids, tokens = archive["ids"].astype(str), archive["y"]
        if ids.ndim != 1 or not len(ids) or len(set(ids)) != len(ids):
            raise ValueError("Expected token IDs must be nonempty and unique")
        if tokens.shape != (len(ids), 1) or tokens.dtype.kind not in "iu" or np.any(tokens < 0):
            raise ValueError("Expected one nonnegative generated token per request")
        return ids.copy(), tokens.copy()


def _check_tokens(path, expected):
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 32_000_000:
            raise ValueError("Oversized decompressed timing predictions")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"ids", "predictions"}:
            raise ValueError("Unexpected generated-token artifacts")
        ids, values = data["ids"].astype(str), data["predictions"]
        truth_ids, truth = expected
        if ids.ndim != 1 or len(set(ids)) != len(ids) or set(ids) != set(truth_ids):
            raise ValueError("Generated-token IDs differ from the frozen workload")
        lookup = {value: index for index, value in enumerate(ids)}
        if values.shape != truth.shape or values.dtype.kind not in "iu":
            raise ValueError("Invalid generated-token shape or type")
        aligned = values[[lookup[value] for value in truth_ids]]
        if not np.array_equal(aligned, truth):
            raise ValueError("Generated tokens differ from the frozen reference")


def llama_oracle(store, outputs, expected):
    seen = {"baseline": set(), "candidate": set()}
    identities = {}

    def check(arm, payload, response):
        if payload != {"workload": "fixed_one_token_generation"} or set(response) != {"predictions"}:
            raise ValueError("Unexpected Llama timing response")
        relative = safe_path(response["predictions"])
        if relative in seen[arm] or not relative.endswith(".npz"):
            raise ValueError("Each request must produce a fresh prediction artifact")
        seen[arm].add(relative)
        bundle = freeze_bundle(store, outputs[arm], [ArtifactSpec(
            path=relative, role="timing_predictions", max_bytes=32_000_000, max_files=1)])
        artifact = load_bundle(store, bundle).files[relative]
        _check_tokens(verified_file(store, artifact), expected)
        identity_path = outputs[arm] / "model-identity.json"
        if identity_path.is_symlink() or not identity_path.is_file() or identity_path.stat().st_size > 1000000:
            raise ValueError("Missing bounded model-state identity receipt")
        identity = json.loads(identity_path.read_text())
        if identity.get("status") != "identical_declared_state" or identity.get("before") != identity.get("after"):
            raise ValueError("Model-state fingerprint check failed")
        if arm in identities and identities[arm] != identity:
            raise ValueError("Resident worker changed its model-state identity receipt")
        identities[arm] = identity
        if len(identities) == 2 and identities["baseline"]["before"] != identities["candidate"]["before"]:
            raise ValueError("Timing arms did not start from the same frozen model state")
        identity_ref = store.put_file(identity_path, "timing_model_state")
        return {"valid": True, "exact_token_equivalence": True,
                "bundle_ref": bundle, "prediction_ref": artifact.ref, "model_identity_ref": identity_ref}
    return check


def run_llama_pair(store, spec, image, baseline_dir, candidate_dir, prepared, output_root,
                   *, timeout, seed=0):
    if not spec.resources.gpus:
        raise ValueError("Llama timing requires an explicitly allocated GPU")
    started = time.monotonic()

    def remaining():
        value = timeout - (time.monotonic() - started)
        if value <= 0:
            raise TimeoutError("Paired GPU timing exhausted its total budget including model startup")
        return value

    public = Path(prepared["public"])
    verify_tree(public, prepared["files"])
    expected = expected_tokens(prepared)
    with open_arrays(archive_path(public, "evaluation")) as values:
        if set(values["ids"].astype(str)) != set(expected[0]):
            raise ValueError("Expected tokens do not match public workload IDs")
    mounts = asset_mounts(spec, "inference", protected=[Path(prepared["private"])])
    if not {target for _, target, _ in mounts} >= {"/assets/model", "/assets/tokenizer"}:
        raise ValueError("Timing requires explicit frozen model and tokenizer asset mounts")
    prefix = [spec.runtime] + (["--context", spec.context] if spec.context else [])
    transports, receipts = {}, {}
    outputs = {arm: output_root / arm / "outputs" for arm in ("baseline", "candidate")}
    failure = None
    try:
        for arm, source in (("baseline", baseline_dir), ("candidate", candidate_dir)):
            transports[arm] = ContainerTimingTransport(
                prefix=prefix, image=image, source=source, public=public,
                control=output_root / arm / "control", outputs=outputs[arm],
                resources=spec.resources, handler_source=LLAMA_TIMING_HANDLER,
                parameters={"seed": seed}, device="cuda", extra_mounts=mounts,
                startup_timeout=min(300, remaining()),
                execution_timeout=remaining(),
            )
        report = run_paired_timing(
            transports, [{"workload": "fixed_one_token_generation"}],
            llama_oracle(store, outputs, expected),
            request_timeout=min(spec.resources.seconds, remaining()), total_timeout=remaining(),
            device="cuda", gpu_synchronization=True,
        )
        verify_tree(public, prepared["files"])
        verify_tree(prepared["private"], prepared["private_files"])
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
            log = output_root / arm / "control/worker.log"
            if transport is None and not log.exists():
                continue
            if transport is not None:
                try:
                    receipt = transport.receipt(store)
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    retention_errors[arm + "/receipt"] = message
                    receipt = {"execution_enforcement_error": message,
                               "closed": transport.closed}
            else:
                receipt = {"startup_failed": True}
            if log.exists():
                receipt["log_ref"] = store.put_file(log, "timing_worker_log")
            else:
                retention_errors[arm + "/log"] = "Worker log is missing"
            receipts[arm] = store.put_json(receipt, "timing_worker_execution")
        attempt = store.put_json({"status": "failed" if failure or retention_errors else "completed",
                                  "failure_category": failure or ("ExecutionEvidenceError" if retention_errors else None),
                                  "retention_errors": retention_errors, "receipts": receipts}, "paired_timing_attempt")
        store.set("paired_timing/" + digest(str(output_root).encode()), attempt)
        if retention_errors and failure is None:
            raise RuntimeError("Paired timing enforcement or cleanup failed; "
                               "both worker logs and the failed attempt were retained")
    return report, receipts


def audit_llama_report(store, prepared, report):
    expected = expected_tokens(prepared)
    seen = {"baseline": set(), "candidate": set()}
    identities = {}
    if report.device != "cuda" or report.gpu_synchronization != "enforced":
        raise ValueError("Llama timing requires GPU synchronization evidence")
    for sample in report.samples:
        evidence = sample.correctness
        relative = safe_path(sample.response["predictions"])
        if relative in seen[sample.arm]:
            raise ValueError("Timing report reuses prediction bytes")
        seen[sample.arm].add(relative)
        manifest = load_bundle(store, evidence["bundle_ref"])
        if set(manifest.files) != {relative} or manifest.files[relative].ref != evidence["prediction_ref"]:
            raise ValueError("Timing artifact does not match its sealed manifest")
        _check_tokens(verified_file(store, manifest.files[relative]), expected)
        identity = store.json(evidence["model_identity_ref"])
        if identity.get("status") != "identical_declared_state" or identity.get("before") != identity.get("after"):
            raise ValueError("Model-state identity does not match fixed-weight timing")
        if sample.arm in identities and identities[sample.arm] != identity:
            raise ValueError("Model identity changed within one timing arm")
        identities[sample.arm] = identity
        if evidence.get("valid") is not True or evidence.get("exact_token_equivalence") is not True:
            raise ValueError("Timing correctness claims differ from sealed evidence")
    if identities.get("baseline") != identities.get("candidate"):
        raise ValueError("Timing arms used different frozen model states")
    return {"predictions": "pass", "sample_count": len(report.samples)}
