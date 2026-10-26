"""Container-only generation of CLRS data and fixed licensed-model workloads."""

from pathlib import Path
import os
import shutil
from types import SimpleNamespace

from ..benchmarks.files import file_hash, tree_manifest
from ..benchmarks.models import NativeResources, relative_path
from ..benchmarks.runner import NativeRunner
from ..process import run_process
from ..store import Store, canonical, digest
from .upstream_helpers import CLRS_HELPERS


class _GeneratorRuntime:
    _launch = NativeRunner._launch
    _image = NativeRunner._image
    executor = staticmethod(run_process)


def _copy_inputs(inputs, request, data, store):
    """Resume immutable input copying without overwriting completed different files."""
    pending = store.get("generation_pending_copy")
    if pending:
        pending = relative_path(pending)
        if not pending.startswith(("assets/", "source_helpers/")) or not pending.endswith(".copy-pending"):
            raise ValueError("Invalid reserved partial-copy path")
        path = data / pending
        if path.is_symlink():
            raise ValueError("Unsafe partial input copy")
        if path.exists():
            path.unlink()  # This exact temporary file was reserved by this operation.
        store.set("generation_pending_copy", None)
    for name, directory_name in (("helpers", "source_helpers"), ("assets", "assets")):
        if name not in inputs:
            continue
        for relative, expected in request["input_manifests"][name].items():
            source, target = inputs[name] / relative, data / directory_name / relative
            if target.exists():
                if target.is_symlink() or target.stat().st_size != expected["size"] or file_hash(target) != expected["sha256"]:
                    raise ValueError("Completed copied input differs from its immutable source")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name("." + target.name + ".copy-pending")
            if temporary.is_symlink():
                raise ValueError("Unsafe partial input copy")
            store.set("generation_pending_copy", temporary.relative_to(data).as_posix())
            shutil.copyfile(source, temporary)
            if temporary.stat().st_size != expected["size"] or file_hash(temporary) != expected["sha256"]:
                raise ValueError("Input changed while copying into the generated source bundle")
            os.replace(temporary, target)
            store.set("generation_pending_copy", None)


def _export_receipt(data, evidence, receipt):
    """Publishing a known completed result is repairable and never reruns its process."""
    controls = data / "provider-control"
    if controls.is_symlink():
        raise ValueError("Unsafe generation evidence export directory")
    controls.mkdir(exist_ok=True)
    tree_manifest(controls)  # Reject symlinks before any idempotent export repair.
    path = data / "source_receipt.json"
    expected = canonical(receipt)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != expected:
            raise ValueError("Completed source receipt changed")
        if tree_manifest(controls / "evidence") != tree_manifest(evidence):
            raise ValueError("Published generation evidence changed")
        return
    shutil.copytree(evidence, controls / "evidence", dirs_exist_ok=True)
    if tree_manifest(controls / "evidence") != tree_manifest(evidence):
        raise ValueError("Incomplete generation export contains unregistered evidence files")
    pending = controls / "receipt.pending"
    pending.write_bytes(expected)
    os.replace(pending, path)


def generate_bundle(task_id, output, *, image, helpers=None, assets=None, source=None,
                    runtime="docker", context=None, gpus=None, seconds=3600,
                    memory_mb=8192, seed=1729, variant=None, terms_accepted=False):
    """Emit a raw prepared bundle plus reproducible generation evidence, without agents."""
    from ..cli import locked

    if task_id not in {"CLRS", "llama-inference"}:
        raise ValueError("Container data generation supports CLRS and llama-inference")
    if runtime not in {"docker", "podman"} or (runtime != "docker" and context):
        raise ValueError("Choose Docker with an optional context or Podman without a context")
    if task_id == "llama-inference" and (not terms_accepted or gpus is None or not variant):
        raise ValueError("Llama workload generation requires accepted model terms, an explicit GPU allocation and an explicit variant name")
    if task_id == "llama-inference" and not variant.startswith("adapted-"):
        raise ValueError("Explicit local compatible checkpoints use an adapted-* variant; original-weight certification requires a verified original provider snapshot")
    variant = variant or "audited-v1-generated-clrs"
    inputs = {name: Path(value).resolve() for name, value in {"helpers": helpers, "assets": assets, "source": source}.items() if value is not None}
    output = Path(output).resolve()
    for path in inputs.values():
        if not path.is_dir() or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("Generation input directories must exist and remain separate from output")
        tree_manifest(path)
    if task_id == "CLRS":
        if "helpers" not in inputs:
            raise ValueError("CLRS generation requires the exact pinned upstream helper tree")
        actual = {name: file_hash(inputs["helpers"] / name) for name in CLRS_HELPERS}
        if actual != CLRS_HELPERS:
            raise ValueError("CLRS source helpers differ from the pinned dependency closure")
    elif not {"assets", "source"} <= set(inputs):
        raise ValueError("Llama generation requires local model/tokenizer assets and a local WikiText source")
    package = Path(__file__).parents[1]
    script = package / "task_families/templates" / ("prepare_clrs.py" if task_id == "CLRS" else "prepare_llama.py")
    if not script.is_file():
        raise ValueError("The trusted task generator is not installed")
    resources = NativeResources(seconds=seconds, memory_mb=max(memory_mb, 32768) if task_id == "llama-inference" else memory_mb,
                                cpus=4, gpus=gpus, shm_mb=2048, tmp_mb=4096,
                                output_max_bytes=256 * 1024**3, output_max_files=250000)
    input_manifests = {name: tree_manifest(path) for name, path in inputs.items()}
    if "helpers" in inputs:
        input_manifests["helpers"] = {name: input_manifests["helpers"][name] for name in CLRS_HELPERS}
    request = {"task_id": task_id, "variant": variant, "image": image, "runtime": runtime,
               "context": context, "resources": resources.model_dump(mode="json"), "seed": seed,
               "input_manifests": input_manifests,
               "implementation": {"generator": file_hash(script), "orchestrator": file_hash(Path(__file__)),
                                  "runner": file_hash(package / "benchmarks/runner.py"),
                                  "bridge": file_hash(package / "live/workspace_worker.py")}}
    with locked(output):
        evidence = output / "evidence"
        store = Store(evidence)
        data = output / "source"
        cached = False
        try:
            if store.verify():
                raise ValueError("Generation evidence was modified")
            old = store.get("generation_request")
            if old is not None:
                if store.json(old) != request:
                    raise ValueError("Generation inputs or implementation changed on resume")
                request_ref = old
            else:
                request_ref = store.put_json(request, "generation-request")
                store.set("generation_request", request_ref)
            complete = store.get("generation_complete")
            if complete:
                result_ref, result = complete, store.json(complete)
                image_id = result["image_id"]
                data_manifest = tree_manifest(data)
                for name in list(data_manifest):
                    if name.startswith("provider-control/") or name == "source_receipt.json":
                        data_manifest.pop(name)
                if data_manifest != result["output_manifest"]:
                    raise ValueError("Completed generated data changed")
                cached = True
            else:
                completed_execution = store.get("generation_execution")
                cached = completed_execution is not None
                if completed_execution is None:
                    if store.get("generation_started"):
                        raise ValueError("Generation outcome was not committed; refusing an automatic duplicate execution")
                    worker = _GeneratorRuntime()
                    worker.store = store
                    worker.spec = SimpleNamespace(runtime=runtime, context=context, resources=resources)
                    worker.prefix = [runtime] + (["--context", context] if context else [])
                    image_id = worker._image(image)
                    control = output / "generator"
                    control.mkdir(exist_ok=True)
                    shutil.copyfile(script, control / script.name)
                    (control / "helper_manifest.json").write_bytes(canonical(CLRS_HELPERS))
                    mount_inputs = dict(inputs)
                    if "helpers" in inputs:
                        selected_helpers = output / "pinned-helpers"
                        selected_helpers.mkdir(exist_ok=True)
                        for name in CLRS_HELPERS:
                            shutil.copyfile(inputs["helpers"] / name, selected_helpers / name)
                        mount_inputs["helpers"] = selected_helpers
                    data.mkdir(exist_ok=False)
                    data.chmod(0o777)
                    empty = output / "empty-input"
                    empty.mkdir(exist_ok=True)
                    command = ["python", "/generator/" + script.name, "--output", "/output", "--helpers", "/helpers",
                               "--assets", "/assets", "--source", "/source", "--seed", str(seed), "--variant", variant,
                               "--helper-manifest", "/generator/helper_manifest.json"]
                    mounts = [(control, "/generator", "ro"), (data, "/output", "rw")]
                    mounts += [(mount_inputs.get(name, empty), "/" + name, "ro") for name in ("helpers", "assets", "source")]
                    store.set("generation_started", True)
                    process, execution_ref = worker._launch("generate", image_id, {"command": command, "cwd": "/generator", "seed": seed},
                                                             mounts, seconds, gpus=gpus,
                                                             bridge=package / "live/workspace_worker.py")
                    raw_manifest_ref = store.put_json(tree_manifest(data), "generation-raw-outputs")
                    # One atomic metadata value commits the known execution and
                    # the exact output bytes together, before host postprocessing.
                    completed_execution = {"execution_ref": execution_ref, "image_id": image_id,
                                           "raw_manifest_ref": raw_manifest_ref,
                                           "returncode": process.code, "timed_out": process.timed_out}
                    store.set("generation_execution", completed_execution)
                execution_ref, image_id = completed_execution["execution_ref"], completed_execution["image_id"]
                if completed_execution["returncode"] or completed_execution["timed_out"]:
                    raise ValueError("Trusted data generation failed or timed out; retained execution evidence records the failure")
                raw_manifest = store.json(completed_execution["raw_manifest_ref"])
                actual_raw = tree_manifest(data)
                for name in list(actual_raw):
                    if name.startswith(("source_helpers/", "assets/", "provider-control/")) or name == "source_receipt.json":
                        actual_raw.pop(name)
                if actual_raw != raw_manifest:
                    raise ValueError("Generated raw outputs changed before postprocessing completed")
                for name, directory in inputs.items():
                    actual = tree_manifest(directory)
                    if name == "helpers":
                        actual = {key: actual[key] for key in CLRS_HELPERS}
                    if actual != request["input_manifests"][name]:
                        raise ValueError("Generation inputs changed during execution")
                if not (data / "split_metadata.json").is_file():
                    raise ValueError("Trusted generator omitted split provenance")
                _copy_inputs(inputs, request, data, store)
                source_refs = {name: store.put_file(path, "generation-source") for name, path in {
                    "generator": script, "orchestrator": Path(__file__),
                    "runner": package / "benchmarks/runner.py", "bridge": package / "live/workspace_worker.py"}.items()}
                result = {"request_ref": request_ref, "execution_ref": execution_ref,
                          "image_id": image_id, "source_refs": source_refs,
                          "output_manifest": tree_manifest(data), "live_model_calls": 0}
                result_ref = store.put_json(result, "generation-result")
                store.set("generation_complete", result_ref)
        finally:
            store.close()
        receipt = {"schema_version": 1, "task_id": task_id, "status": "completed", "live_model_calls": 0,
                   "provider": {"kind": "isolated-generator", "variant": variant},
                   "generation_store": "provider-control/evidence", "generation_result_ref": result_ref,
                   "request_sha256": digest(canonical(request)), "image_id": image_id,
                   "verification_scope": "Frozen trusted generator in an isolated local runtime; explicitly named generated/adapted protocol"}
        _export_receipt(data, evidence, receipt)
        return {"source": str(data), "receipt": str(data / "source_receipt.json"),
                "status": "completed", "cached": cached, "live_model_calls": 0}
