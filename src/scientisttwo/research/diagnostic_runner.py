"""Optional isolated diagnostics over frozen predictions and protected development labels."""

import inspect
from pathlib import Path, PurePosixPath
import time
from typing import Literal

from pydantic import Field, model_validator

from ..artifacts import ArtifactSpec, restore_bundle, load_bundle, freeze_bundle
from ..benchmarks.files import tree_manifest, verify_tree
from ..live.pipeline import EvaluationPipeline, PipelineInterrupted
from ..models import Record
from ..store import IntegrityError, canonical, digest
from . import diagnostics
from .diagnostics import DiagnosticRequest, committed_observation


class ScoringDiagnosticContract(Record):
    schema_version: Literal[1] = 1
    kind: Literal["classification", "regression_residuals"]
    format: Literal["class_scores_npz", "cpu_frames"]
    prediction_path: str
    label_path: str
    id_column: str = "id"
    targets: list[str] = Field(default_factory=list)
    split: Literal["development"] = "development"

    @model_validator(mode="after")
    def paths(self):
        for path in (self.prediction_path, self.label_path):
            parsed = PurePosixPath(path)
            if not path or parsed.is_absolute() or any(p in {"..", "."} for p in parsed.parts) or "\\" in path:
                raise ValueError("Scoring diagnostic paths must stay in their declared mounts")
        if self.format == "class_scores_npz" and self.kind != "classification":
            raise ValueError("Class-score archives support only classification diagnostics")
        if self.format == "cpu_frames" and (not self.targets or len(set(self.targets)) != len(self.targets)):
            raise ValueError("Frame diagnostics require distinct target columns")
        return self


def _source_files():
    functions = [diagnostics._finite, diagnostics._histogram, diagnostics.classification_diagnostics, diagnostics.regression_residuals]
    math = "import numpy as np\n\n" + "\n\n".join(inspect.getsource(fn) for fn in functions)
    return {"diagnostic_math.py": math,
            "diagnostic_worker.py": Path(__file__).with_name("diagnostic_worker.py").read_text()}


def run_scoring_diagnostic(runner, request: DiagnosticRequest, observation_ref: str,
                           pipeline: EvaluationPipeline, contract: ScoringDiagnosticContract):
    """Run without candidate mounts; unknown completion never repeats automatically.

    The caller supplies an adapter-owned contract and the original frozen scoring
    pipeline for this observation. This is intentionally not an agent interface.
    The returned reference is consumable by ``run_diagnostic``. Each attempt is
    charged as an experiment, including failures.
    """
    store = runner.store
    request = DiagnosticRequest.model_validate(request.model_dump())
    contract = ScoringDiagnosticContract.model_validate(contract.model_dump())
    pipeline = EvaluationPipeline.model_validate(pipeline.model_dump(mode="python"))
    if request.kind != contract.kind or request.evaluation_split != "development" or request.parameters:
        raise ValueError("Scorer requires the exact declared development diagnostic without extra parameters")
    supported_metrics = ({"accuracy", "macro_f1", "precision", "recall", "f1"} if request.kind == "classification"
                         else {"mae", "mean_rmse", "rmse", "bias"})
    if request.metric is not None and request.metric not in supported_metrics:
        raise ValueError("Requested metric is not produced by this diagnostic scorer")
    observation, protocol = committed_observation(store, observation_ref)
    if request.expected_seeds != protocol["seeds"] or not request.expected_seeds:
        raise ValueError("Scorer seed coverage differs from the exact diagnostic request")
    legacy_frames = contract.format == "cpu_frames" and not observation.evaluation_refs
    if not pipeline.private_dir or not legacy_frames and len(observation.evaluation_refs) != len(request.expected_seeds):
        raise ValueError("Isolated diagnostics require frozen protected labels and complete scoring receipts")
    if pipeline.score.resources.gpus is not None:
        raise ValueError("Trusted diagnostic scoring requires a CPU-only resource profile")
    # Revalidate the complete source/data/output closure before cache lookup too.
    image = runner._image(pipeline.score.image)
    candidate_identity = pipeline.model_dump(mode="json")
    candidate_identity["score"]["image"] = image
    if pipeline.inference:
        candidate_identity["inference"]["image"] = runner._image(pipeline.inference.image)
        verify_tree(pipeline.inference.source_dir, pipeline.inference.source_manifest)
    verify_tree(pipeline.score.source_dir, pipeline.score.source_manifest)
    verify_tree(pipeline.private_dir, pipeline.private_manifest)
    evaluated_inputs = []
    if legacy_frames:
        prepared_ref = store.get("prepared/" + ("subset" if observation.phase == "subset" else "full"))
        if not prepared_ref or len(observation.bundle_refs) != len(request.expected_seeds):
            raise ValueError("Frame diagnostics require committed phase preparation and predictions")
        prepared = store.json(prepared_ref)
        declared = runner.adapter.diagnostics_contract(request.kind, observation.phase, prepared=prepared)
        if (declared["contract"] != contract.model_dump(mode="json")
                or declared["pipeline"] != pipeline.model_dump(mode="json")
                or protocol.get("adapter") != runner.adapter.identity()
                or protocol.get("data") != prepared["files"]):
            raise IntegrityError("Frame diagnostic contract differs from the frozen task and development inputs")
        verify_tree(prepared["public"], prepared["files"])
        for seed, bundle_ref in zip(request.expected_seeds, observation.bundle_refs, strict=True):
            bundle = load_bundle(store, bundle_ref)
            if any(spec.role != "prediction" for spec in bundle.specs):
                raise IntegrityError("Frame diagnostics accept prediction artifacts only")
            evaluated_inputs.append((seed, {"prediction_bundle": bundle_ref}, bundle))
    for seed, evaluation_ref in ([] if legacy_frames else zip(request.expected_seeds, observation.evaluation_refs, strict=True)):
        evaluated = store.json(evaluation_ref)
        committed = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("pipeline/" + observation.id + "/" + str(seed),)).fetchone()
        if not committed or store.json(committed[0]) != evaluated or evaluated["seed"] != seed:
            raise IntegrityError("Diagnostic input is not its committed scoring pipeline")
        if evaluated["pipeline"] != candidate_identity:
            raise IntegrityError("Diagnostic labels or original evaluator differ from the frozen pipeline")
        bundle = load_bundle(store, evaluated["prediction_bundle"])
        if any(spec.role != "prediction" for spec in bundle.specs):
            raise IntegrityError("Diagnostic scorer accepts prediction artifacts only")
        evaluated_inputs.append((seed, evaluated, bundle))
    # Existing evaluator identity is bound in each sealed evaluation below.
    sources = _source_files()
    envelope = {"request": request.model_dump(), "observation_ref": observation_ref,
                "pipeline": pipeline.model_dump(mode="json"), "contract": contract.model_dump(),
                "source": {name: digest(value.encode()) for name, value in sources.items()}}
    key = "trusted-diagnostic/" + request.id
    cached = store.cached(key, envelope)
    if cached is not None:
        for ref in cached["result_refs"] + cached["receipt_refs"]:
            store.read(ref)
        load_bundle(store, cached["source_bundle_ref"])
        return store.db.execute("SELECT output_ref FROM calls WHERE key=?", (key,)).fetchone()[0]
    operation = "diagnostic/" + request.id
    prior = runner.budget.reserve(operation, "experiment", envelope)
    if prior:
        raise PipelineInterrupted("Diagnostic has an unresolved or failed prior operation; refusing repeat")
    started = time.monotonic()
    receipt_refs, result_refs, per_seed = [], [], []
    try:
        root = store.root / "diagnostics" / digest(request.id.encode())[:24]
        root.mkdir(parents=True, exist_ok=False)
        source = root / "source"
        source.mkdir()
        for name, text in sources.items():
            (source / name).write_text(text)
        (source / "contract.json").write_bytes(canonical(contract))
        source_manifest = tree_manifest(source)
        source_bundle_ref = freeze_bundle(store, source, [ArtifactSpec(path=name, role="evaluator")
                                                        for name in sorted(source_manifest)])
        for seed, evaluated, bundle in evaluated_inputs:
            bundle_ref = evaluated["prediction_bundle"]
            inputs, output = root / f"inputs-{seed}", root / f"output-{seed}"
            restore_bundle(store, bundle_ref, inputs)
            output.mkdir()
            output.chmod(0o777)
            command = ["python", "/evaluator/diagnostic_worker.py", "--contract", "/evaluator/contract.json",
                       "--predictions", "/predictions/" + contract.prediction_path,
                       "--labels", "/data/private/" + contract.label_path,
                       "--output", "/grade-output/diagnostic.json"]
            store.set(key + "/started/" + str(seed), True)
            process, receipt_ref = runner._launch(
                "diagnostic-" + digest((request.id + "/" + str(seed)).encode())[:20], image,
                {"command": command, "cwd": "/evaluator", "seed": seed},
                [(source, "/evaluator", "ro"), (inputs, "/predictions", "ro"),
                 (pipeline.private_dir, "/data/private", "ro"), (output, "/grade-output", "rw")],
                min(pipeline.score.resources.seconds, runner.budget.remaining()),
                resources=pipeline.score.resources, gpus=None,
                bridge=Path(__file__).parents[1] / "live" / "workspace_worker.py")
            receipt_refs.append(receipt_ref)
            if process.code or process.timed_out or process.limit_exceeded:
                raise ValueError("Isolated diagnostic failed or exceeded its limits")
            verify_tree(source, source_manifest)
            verify_tree(pipeline.private_dir, pipeline.private_manifest)
            verify_tree(pipeline.score.source_dir, pipeline.score.source_manifest)
            if freeze_bundle(store, inputs, bundle.specs) != bundle_ref:
                raise IntegrityError("Frozen diagnostic predictions changed during scoring")
            path = output / "diagnostic.json"
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 4_000_000:
                raise ValueError("Diagnostic scorer omitted its bounded JSON output")
            result_ref = store.put_file(path, "trusted-diagnostic-output")
            result = store.json(result_ref)
            canonical(result)
            if set(result) != {"tables", "arrays", "summary", "limitations"}:
                raise ValueError("Diagnostic scorer output has undeclared fields")
            result_refs.append(result_ref)
            per_seed.append((seed, result))
        output = {"tables": {}, "arrays": {}, "summary": {"seeds": request.expected_seeds}, "limitations": []}
        for seed, result in per_seed:
            for name in ("tables", "arrays"):
                output[name].update({f"seed_{seed}/" + key: value for key, value in result[name].items()})
            output["tables"].setdefault("per_seed", []).append({"seed": seed, **result["summary"]})
            output["limitations"].extend(result["limitations"])
        output["limitations"] = list(dict.fromkeys(output["limitations"]))
        result = {"request": request.model_dump(), "observation_refs": [observation_ref], "output": output,
                  "receipt_refs": receipt_refs, "result_refs": result_refs, "source_manifest": source_manifest,
                  "source_bundle_ref": source_bundle_ref,
                  "contract": contract.model_dump(), "label_commitment": digest(canonical(pipeline.private_manifest))}
        # Commit the closure in the ledger before the convenience cache. A crash
        # between them is deliberately unresolved rather than an automatic rerun.
        runner.budget.finish(operation, "success", result, time.monotonic() - started)
        store.cache(key, envelope, result)
        return store.db.execute("SELECT output_ref FROM calls WHERE key=?", (key,)).fetchone()[0]
    except (ValueError, IntegrityError, OSError) as error:
        runner.budget.finish(operation, "failed", {"error": str(error), "receipt_refs": receipt_refs,
                              "result_refs": result_refs}, time.monotonic() - started)
        raise
