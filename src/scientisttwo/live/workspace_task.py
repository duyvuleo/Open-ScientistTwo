"""Freeze and evaluate local workspace tasks without benchmark-specific routing."""

import json
import math
from pathlib import Path
import time

from pydantic import BaseModel

from ..benchmarks.files import artifact_manifest, file_hash, tree_manifest, verify_tree, workspace
from ..benchmarks.manifest import preflight as native_preflight
from ..benchmarks.models import NativeSpec
from ..benchmarks.runner import NativeRunner
from ..process import run_process
from ..store import IntegrityError, Store
from .workspace_models import WorkspaceSpec
from ..runtime_contracts import asset_mounts
from ..artifacts import freeze_bundle
from .pipeline import evaluate_pipeline, PipelineInterrupted


def task_spec(value):
    cls = WorkspaceSpec if value.get("benchmark") == "workspace" else NativeSpec
    return cls.model_validate(value)


def freeze_workspace(value):
    spec = WorkspaceSpec.model_validate(value)
    for kind in ("public", "private", "candidate", "evaluator"):
        root = getattr(spec, kind + "_dir")
        if root is None:
            continue
        manifest = tree_manifest(root)
        if not root.is_dir() or (not manifest and kind in {"candidate", "evaluator"}):
            raise ValueError(f"Prepared {kind} input tree must be nonempty")
        setattr(spec, kind + "_manifest", manifest)
    spec.implementation = workspace_identity()
    if spec.evaluation_pipeline:
        for stage in [
            spec.evaluation_pipeline.score,
            *([spec.evaluation_pipeline.inference] if spec.evaluation_pipeline.inference else []),
        ]:
            verify_tree(stage.source_dir, stage.source_manifest)
        if spec.private_dir:
            verify_tree(spec.private_dir, spec.evaluation_pipeline.private_manifest)
    return spec


def workspace_identity():
    from ..benchmarks.files import file_hash

    return {
        name: file_hash(Path(__file__).with_name(name))
        for name in (
            "workspace_task.py",
            "workspace_models.py",
            "workspace_worker.py",
            "pipeline.py",
        )
    }


def preflight(spec):
    if not isinstance(spec, WorkspaceSpec):
        return native_preflight(spec)
    errors = []
    if spec.implementation != workspace_identity():
        errors.append("Workspace evaluator integration changed; freeze a fresh task")
    for kind in ("public", "private", "candidate", "evaluator"):
        root, expected = getattr(spec, kind + "_dir"), getattr(spec, kind + "_manifest")
        if root is None:
            if expected:
                errors.append(f"Missing {kind} tree for declared manifest")
            continue
        if not root.is_dir() or (not expected and kind in {"candidate", "evaluator"}):
            errors.append(f"Missing frozen {kind} input tree")
        else:
            verify_tree(root, expected)
    if spec.evaluation_pipeline:
        for stage in [
            spec.evaluation_pipeline.score,
            *([spec.evaluation_pipeline.inference] if spec.evaluation_pipeline.inference else []),
        ]:
            verify_tree(stage.source_dir, stage.source_manifest)
        if spec.private_dir:
            verify_tree(spec.private_dir, spec.evaluation_pipeline.private_manifest)
    return {
        "ready": not errors,
        "errors": errors,
        "task": {"task_id": spec.task_id, "benchmark": "workspace"},
        "checksums": {"status": "unavailable", "reason": "User-prepared workspace"},
        "live_calls": False,
        "data_provenance": spec.data_provenance,
    }


class WorkspaceResult(BaseModel):
    benchmark: str = "workspace"
    task_id: str
    status: str
    mode: str
    data_provenance: str
    score: float | None = None
    metric: str
    direction: str
    metrics: dict[str, float] = {}
    reason: str
    elapsed_seconds: float
    official_report_ref: str | None = None
    execution_ref: str | None = None
    artifacts_ref: str | None = None
    leaderboard_eligible: bool = False
    benchmark_data_verified: bool = False
    live_model_calls: int = 0


class WorkspaceRunner:
    """Final-only evaluation; protected labels/evaluator never enter candidate mounts."""

    _launch = NativeRunner._launch
    _image = NativeRunner._image

    def __init__(self, spec, output, *, executor=run_process):
        self.spec, self.executor = spec, executor
        output = Path(output).resolve()
        for root in (spec.public_dir, spec.private_dir, spec.candidate_dir, spec.evaluator_dir):
            if root is None:
                continue
            if output.is_relative_to(root.resolve()) or root.resolve().is_relative_to(output):
                raise ValueError("Workspace output must not overlap frozen inputs")
        self.store = Store(output)
        self.prefix = [spec.runtime] + (["--context", spec.context] if spec.context else [])
        self.started = time.monotonic()

    def close(self):
        self.store.close()

    def verify_executed_source(self, work):
        ref = self.store.get("workspace_initial_source")
        expected = self.store.json(ref) if ref else self.spec.candidate_manifest
        for name, record in expected.items():
            path = work / name
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != record["size"]
                or file_hash(path) != record["sha256"]
            ):
                raise IntegrityError("Candidate modified its frozen final source during execution")

    def finish(self, status, reason, **values):
        result = WorkspaceResult(
            task_id=self.spec.task_id,
            status=status,
            mode=self.spec.mode,
            data_provenance=self.spec.data_provenance,
            metric=self.spec.metric,
            direction=self.spec.direction,
            reason=reason,
            elapsed_seconds=time.monotonic() - self.started,
            **values,
        )
        self.store.set("workspace_result", self.store.put_json(result, "workspace-result"))
        if status in {"completed", "invalid_submission", "grading_failed"}:
            self.store.set("workspace_grading_pending", False)
        (self.store.root / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
        (self.store.root / "report.md").write_text(
            f"# {self.spec.task_id}\n\nStatus: {status}. {reason}\n\n"
            f"Metric: {result.metric}; score: {result.score}.\n\n"
            "User-prepared final evaluator; not an official benchmark result. "
            "No model is invoked by this evaluator.\n"
        )
        return result

    def run(self):
        spec = self.spec
        ref = self.store.put_json(spec, "workspace-specification")
        if self.store.get("workspace_spec") not in (None, ref):
            raise IntegrityError("Frozen workspace specification changed")
        self.store.set("workspace_spec", ref)
        errors = self.store.verify()
        if errors:
            raise IntegrityError(errors[0])
        checks = preflight(spec)
        if not checks["ready"]:
            return self.finish("blocked", "; ".join(checks["errors"]))
        saved = self.store.get("workspace_result")
        if saved:
            result = WorkspaceResult.model_validate(self.store.json(saved))
            if result.status in {"completed", "invalid_submission", "execution_failed"}:
                return result
        images = {"execution": self._image(spec.image), "grading": self._image(spec.grader_image)}
        if self.store.get("workspace_images") not in (None, images):
            raise IntegrityError("Workspace runtime images changed")
        self.store.set("workspace_images", images)
        work = self.store.root / "workspace"
        receipt = self.store.get("workspace_execution")
        bridge = Path(__file__).with_name("workspace_worker.py")
        if receipt is None:
            if self.store.get("workspace_started"):
                return self.finish("interrupted", "Unknown execution outcome; no automatic repeat")
            links = workspace(spec.public_dir, work, spec.candidate_dir)
            for directory in [work, *work.rglob("*")]:
                if directory.is_dir() and not directory.is_symlink():
                    directory.chmod(0o777)
            self.store.set("workspace_links", sorted(links))
            self.store.set(
                "workspace_initial_source",
                self.store.put_json(artifact_manifest(work, links), "workspace-initial-source"),
            )
            self.store.set("workspace_started", True)
            process, receipt = self._launch(
                "execute",
                images["execution"],
                {"command": spec.command, "cwd": "/workspace", "seed": spec.final_seed},
                [(work, "/workspace", "rw"), (spec.public_dir, "/data/public", "ro")]
                + asset_mounts(spec, "execute", protected=[spec.private_dir, spec.evaluator_dir]),
                spec.resources.seconds,
                gpus=spec.resources.gpus,
                bridge=bridge,
            )
            if process.code or process.timed_out:
                return self.finish(
                    "execution_failed", "Candidate failed or timed out", execution_ref=receipt
                )
            self.verify_executed_source(work)
            manifest = artifact_manifest(work, links)
            artifacts = {
                "files": manifest,
                "references": {name: self.store.put_file(work / name) for name in manifest},
            }
            self.store.set("workspace_artifacts", self.store.put_json(artifacts))
            self.store.set("workspace_execution", receipt)
        artifacts_ref = self.store.get("workspace_artifacts")
        self.verify_executed_source(work)
        if (
            artifact_manifest(work, set(self.store.get("workspace_links")))
            != self.store.json(artifacts_ref)["files"]
        ):
            raise IntegrityError("Frozen final outputs changed before grading")
        # Recheck protected inputs immediately before grading, including on resume.
        if not preflight(spec)["ready"]:
            raise IntegrityError("Workspace inputs changed before grading")
        if spec.evaluation_pipeline:
            bundle = freeze_bundle(self.store, work, spec.output_artifacts)
            try:
                verify_tree(spec.candidate_dir, spec.candidate_manifest)
                evaluated = evaluate_pipeline(
                    self,
                    "final",
                    spec.evaluation_pipeline,
                    bundle,
                    spec.public_dir,
                    candidate_source=spec.candidate_dir,
                    seed=spec.final_seed,
                )
            except PipelineInterrupted as error:
                return self.finish(
                    "interrupted", str(error), execution_ref=receipt, artifacts_ref=artifacts_ref
                )
            except (ValueError, OSError) as error:
                return self.finish(
                    "grading_failed", str(error), execution_ref=receipt, artifacts_ref=artifacts_ref
                )
            metrics = evaluated["metrics"]
            if spec.metric not in metrics or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in metrics.values()
            ):
                return self.finish(
                    "invalid_submission",
                    "Staged evaluator omitted valid finite metrics",
                    execution_ref=receipt,
                    artifacts_ref=artifacts_ref,
                )
            return self.finish(
                "completed",
                "Protected inference and trusted scoring completed in separate containers",
                score=metrics[spec.metric],
                metrics=metrics,
                official_report_ref=self.store.put_json(evaluated, "workspace-final-grade"),
                execution_ref=receipt,
                artifacts_ref=artifacts_ref,
            )
        if self.store.get("workspace_grading_pending"):
            return self.finish(
                "interrupted",
                "Unknown final grading outcome; no automatic repeat",
                execution_ref=receipt,
                artifacts_ref=artifacts_ref,
            )
        attempt = self.store.get("workspace_grading_attempts", 0)
        output = self.store.root / "grade-output" / str(attempt)
        output.mkdir(parents=True, exist_ok=False)
        output.chmod(0o777)
        job = {"command": spec.evaluator_command, "cwd": "/evaluator", "seed": spec.final_seed}
        self.store.set("workspace_grading_pending", True)
        process, grading_ref = self._launch(
            "grade",
            images["grading"],
            job,
            [
                (work, "/submission", "ro"),
                (spec.public_dir, "/data/public", "ro"),
                (spec.evaluator_dir, "/evaluator", "ro"),
                (output, "/grade-output", "rw"),
            ]
            + ([(spec.private_dir, "/data/private", "ro")] if spec.private_dir else []),
            spec.resources.grading_seconds,
            gpus=spec.resources.gpus,
            bridge=bridge,
        )
        self.store.set("workspace_grading", grading_ref)
        self.store.set("workspace_grading_attempts", attempt + 1)
        grade_path = output / "grade.json"
        if process.code or process.timed_out or not grade_path.is_file():
            return self.finish(
                "grading_failed",
                "Trusted evaluator failed or omitted grade.json",
                execution_ref=receipt,
                artifacts_ref=artifacts_ref,
            )
        if not preflight(spec)["ready"]:
            raise IntegrityError("Workspace inputs changed during grading")
        self.verify_executed_source(work)
        if (
            artifact_manifest(work, set(self.store.get("workspace_links")))
            != self.store.json(artifacts_ref)["files"]
        ):
            raise IntegrityError("Frozen final outputs changed during grading")
        if grade_path.is_symlink() or grade_path.stat().st_size > 1_000_000:
            raise IntegrityError("Invalid final grade artifact")
        grade = json.loads(grade_path.read_text())
        valid = grade.get("valid_submission")
        metrics = grade.get("metrics", {})
        if not isinstance(valid, bool) or not isinstance(metrics, dict):
            raise ValueError("Evaluator must return valid_submission and a metrics mapping")
        if any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
            for v in metrics.values()
        ):
            raise ValueError("Final evaluator returned nonfinite or nonnumeric metrics")
        if valid and spec.metric not in metrics:
            raise ValueError("Final evaluator omitted the declared primary metric")
        report = self.store.put_json(grade, "workspace-final-grade")
        return self.finish(
            "completed" if valid else "invalid_submission",
            "Protected workspace evaluation completed",
            score=metrics.get(spec.metric) if valid else None,
            metrics=metrics if valid else {},
            official_report_ref=report,
            execution_ref=receipt,
            artifacts_ref=artifacts_ref,
        )
