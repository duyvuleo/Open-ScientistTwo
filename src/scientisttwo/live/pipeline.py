"""Isolated inference followed by trusted scoring; candidate code never sees labels."""

import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..artifacts import ArtifactSpec, freeze_bundle, load_bundle, restore_bundle
from ..benchmarks.files import tree_manifest, verify_tree
from ..benchmarks.models import NativeResources
from ..models import Record
from ..runtime_contracts import asset_mounts
from ..store import IntegrityError, digest


class EvaluationStage(Record):
    kind: Literal["inference", "score"]
    image: str
    source_dir: Path
    source_manifest: dict
    command: list[str] = Field(min_length=1, max_length=64)
    resources: NativeResources = Field(default_factory=NativeResources)
    outputs: list[ArtifactSpec] = []

    @model_validator(mode="after")
    def valid(self):
        if not self.source_dir.is_absolute() or not self.source_manifest:
            raise ValueError("Evaluation stages require an absolute frozen trusted source tree")
        if any(not a or "\x00" in a for a in self.command):
            raise ValueError("Evaluation stages require explicit argv")
        if self.kind == "inference" and not self.outputs:
            raise ValueError("Inference must declare prediction outputs")
        if self.kind == "inference" and any(
            s.role not in {"prediction", "diagnostic"} for s in self.outputs
        ):
            raise ValueError("Inference can export only predictions and diagnostics")
        return self


class EvaluationPipeline(Record):
    inference: EvaluationStage | None = None
    score: EvaluationStage
    private_dir: Path | None = None
    private_manifest: dict = {}

    @model_validator(mode="after")
    def valid(self):
        if self.score.kind != "score" or (self.inference and self.inference.kind != "inference"):
            raise ValueError("Evaluation stage roles are inconsistent")
        if self.private_dir is None:
            if self.private_manifest:
                raise ValueError("Private manifest has no input directory")
        elif not self.private_dir.is_absolute() or not self.private_manifest:
            raise ValueError("Protected scoring data require an absolute frozen directory")
        for stage in [self.score, *([self.inference] if self.inference else [])]:
            if self.private_dir and (
                stage.source_dir.resolve().is_relative_to(self.private_dir.resolve())
                or self.private_dir.resolve().is_relative_to(stage.source_dir.resolve())
            ):
                raise ValueError("Trusted source and protected labels must be separate")
        return self


class PipelineInterrupted(IntegrityError):
    pass


def declared_pipeline(adapter, prepared):
    callback = getattr(adapter, "evaluation", None)
    if callback is None:
        return None
    value = callback(prepared)
    return EvaluationPipeline.model_validate(value) if value is not None else None


def pipeline_identity(runner, pipeline):
    pipeline = EvaluationPipeline.model_validate(pipeline.model_dump(mode="python"))
    result = pipeline.model_dump(mode="json")
    for name in ("inference", "score"):
        stage = getattr(pipeline, name)
        if stage:
            verify_tree(stage.source_dir, stage.source_manifest)
            result[name]["image"] = runner._image(stage.image)
    if pipeline.private_dir:
        verify_tree(pipeline.private_dir, pipeline.private_manifest)
    return result


def _overlap(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    return left.is_relative_to(right) or right.is_relative_to(left)


def _validate_mount_boundaries(runner, pipeline, public, candidate_source):
    """Check all intended mounts before starting any evaluator process."""
    exposed = [Path(public)] + ([Path(candidate_source)] if candidate_source is not None else [])
    protected = [stage.source_dir for stage in (pipeline.inference, pipeline.score) if stage]
    if pipeline.private_dir:
        protected.append(pipeline.private_dir)
    for path in exposed:
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Evaluation public/source inputs require real directories")
        if any(_overlap(path, other) for other in protected):
            raise ValueError(
                "Public or candidate source mount overlaps protected evaluation data/source"
            )
    assets = {}
    for name in ("inference", "score"):
        stage = getattr(pipeline, name)
        if stage:
            assets[name] = asset_mounts(
                runner.spec,
                name,
                protected=[pipeline.private_dir, *protected, candidate_source],
            )
    return assets


def _grade(store, grade_ref):
    grade = store.json(grade_ref)
    if not isinstance(grade, dict):
        raise ValueError("Scorer grade must be a structured object")
    metrics = grade.get("metrics")
    if grade.get("valid_submission") is not True or not isinstance(metrics, dict) or not metrics:
        raise ValueError("Scorer rejected the submission or omitted metrics")
    if any(
        not isinstance(k, str)
        or not k
        or isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        for k, v in metrics.items()
    ):
        raise ValueError("Scorer metrics must be finite numeric values with named metrics")
    return metrics


def evaluate_pipeline(runner, key, pipeline, bundle_ref, public, *, candidate_source=None, seed=0):
    """Execute each stage once and return scores plus sealed, replayable evidence."""
    pipeline = EvaluationPipeline.model_validate(pipeline.model_dump(mode="python"))
    store = runner.store
    assets = _validate_mount_boundaries(runner, pipeline, public, candidate_source)
    identity = pipeline_identity(runner, pipeline)
    load_bundle(store, bundle_ref)
    request = {
        "pipeline": identity,
        "bundle": bundle_ref,
        "public": tree_manifest(public),
        "seed": seed,
    }
    if candidate_source is not None:
        request["source"] = tree_manifest(candidate_source)
    cached = store.cached("pipeline/" + key, request)
    if cached:
        load_bundle(store, cached["prediction_bundle"])
        if _grade(store, cached["grade_ref"]) != cached["metrics"]:
            raise IntegrityError("Cached score differs from its sealed grader response")
        if cached.get("diagnostics_bundle"):
            load_bundle(store, cached["diagnostics_bundle"])
        return cached
    stage_root = store.root / "evaluation" / digest(key.encode())[:24]
    stage_root.mkdir(parents=True, exist_ok=True)
    receipts = []
    prediction_bundle = bundle_ref
    grade_ref = None
    diagnostics_bundle = None
    for name in ("inference", "score"):
        stage = getattr(pipeline, name)
        if stage is None:
            continue
        stage_key = "pipeline-stage/" + key + "/" + name
        stage_request = {**request, "input_bundle": prediction_bundle, "stage": name}
        saved = store.cached(stage_key, stage_request)
        if saved:
            receipts.append(saved["receipt"])
            if name == "inference":
                prediction_bundle = saved["output_bundle"]
                load_bundle(store, prediction_bundle)
                continue
            scores = saved["metrics"]
            grade_ref = saved["grade_ref"]
            diagnostics_bundle = saved.get("diagnostics_bundle")
            if _grade(store, grade_ref) != scores:
                raise IntegrityError("Cached scoring stage differs from its sealed grade")
            break
        if store.get(stage_key + "/started"):
            raise PipelineInterrupted(
                "Evaluation stage outcome is unknown; refusing automatic repeat: " + name
            )
        budget = getattr(runner, "budget", None)
        if budget is not None:
            budget.remaining()
        work = stage_root / name
        work.mkdir(exist_ok=False)
        inputs, output = work / "inputs", work / "output"
        restored = restore_bundle(store, prediction_bundle, inputs)
        if name == "score" and any(
            s.role not in {"prediction", "diagnostic"} for s in restored.specs
        ):
            raise ValueError(
                "Trusted scoring cannot receive checkpoint, source or tokenizer artifacts"
            )
        output.mkdir()
        output.chmod(0o777)
        mounts = [
            (stage.source_dir, "/evaluator", "ro"),
            (output, "/grade-output" if name == "score" else "/output", "rw"),
        ]
        if name == "inference":
            if candidate_source is None:
                raise ValueError("Checkpoint inference requires a frozen candidate source tree")
            mounts += [
                (inputs, "/artifacts", "ro"),
                (candidate_source, "/candidate", "ro"),
                (Path(public), "/data/public", "ro"),
            ]
        else:
            mounts.append((inputs, "/predictions", "ro"))
            if pipeline.private_dir:
                mounts.append((pipeline.private_dir, "/data/private", "ro"))
        mounts += assets[name]
        timeout = stage.resources.seconds
        if budget is not None:
            timeout = min(timeout, budget.remaining())
        store.set(stage_key + "/started", True)
        process, receipt = runner._launch(
            "evaluation-" + digest(stage_key.encode())[:20],
            identity[name]["image"],
            {"command": stage.command, "cwd": "/evaluator", "seed": seed},
            mounts,
            timeout,
            gpus=stage.resources.gpus,
            bridge=Path(__file__).with_name("workspace_worker.py"),
            resources=stage.resources,
        )
        receipts.append(receipt)
        if process.code or process.timed_out:
            raise ValueError(
                f"{name} evaluator failed or timed out; consult execution receipt {receipt}"
            )
        verify_tree(stage.source_dir, stage.source_manifest)
        if pipeline.private_dir:
            verify_tree(pipeline.private_dir, pipeline.private_manifest)
        if tree_manifest(public) != request["public"]:
            raise IntegrityError("Public evaluation inputs changed during execution")
        if candidate_source is not None and tree_manifest(candidate_source) != request["source"]:
            raise IntegrityError("Frozen candidate source changed during evaluation")
        if freeze_bundle(store, inputs, restored.specs) != prediction_bundle:
            raise IntegrityError("Read-only evaluation artifacts changed during execution")
        asset_mounts(
            runner.spec,
            name,
            protected=[pipeline.private_dir, *protected_source_paths(pipeline), candidate_source],
        )
        if name == "inference":
            prediction_bundle = freeze_bundle(store, output, stage.outputs)
            result = {"receipt": receipt, "output_bundle": prediction_bundle}
        else:
            path = output / "grade.json"
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
                raise ValueError("Scorer must produce a bounded regular grade.json")
            # Decode only after freezing: metrics and audit bind to the same immutable bytes.
            grade_ref = store.put_file(path, "trusted-score")
            scores = _grade(store, grade_ref)
            result = {"receipt": receipt, "metrics": scores, "grade_ref": grade_ref}
            if stage.outputs:
                if any(s.role != "diagnostic" for s in stage.outputs):
                    raise ValueError("Trusted scoring auxiliary outputs must be diagnostics")
                diagnostics_bundle = freeze_bundle(store, output, stage.outputs)
                result["diagnostics_bundle"] = diagnostics_bundle
        store.cache(stage_key, stage_request, result)
    result = {
        "metrics": scores,
        "input_bundle": bundle_ref,
        "prediction_bundle": prediction_bundle,
        "receipts": receipts,
        "pipeline": identity,
        "seed": seed,
        "grade_ref": grade_ref,
        "diagnostics_bundle": diagnostics_bundle,
        "public_manifest": request["public"],
        "candidate_manifest": request.get("source"),
        "request_ref": store.put_json(request, "evaluation-request"),
    }
    store.cache("pipeline/" + key, request, result)
    return result


def protected_source_paths(pipeline):
    return [stage.source_dir for stage in (pipeline.inference, pipeline.score) if stage]
