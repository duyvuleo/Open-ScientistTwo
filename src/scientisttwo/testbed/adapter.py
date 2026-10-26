"""Installed development adapter for frozen prepared CPU testbed tasks."""

from pathlib import Path
import ast
import hashlib
import json
from types import SimpleNamespace
import shutil

import pandas as pd

from ..benchmarks.files import file_hash, tree_manifest
from ..live.models import Candidate, SourceFile
from .catalog import get_recipe
from .metrics import score_frames
from .preparation import CPU_TASKS, manifest_hash, verify_prepared


_TRAIN_COMMAND = ("python", "train.py")
_EXECUTION_CONTEXT_MAX_CHARS = 12_000


def _source_excerpts(path, *, functions=(), training_task=None):
    """Slice trusted frozen source; never import/evaluate a worker or candidate."""
    raw = path.read_bytes()
    source = raw.decode("utf-8")
    tree = ast.parse(source)
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if any(name not in definitions for name in functions):
        raise ValueError("Frozen execution-context function is unavailable: " + path.name)
    selected = [definitions[name] for name in functions]
    if training_task is not None:
        train = definitions.get("train")
        if train is None:
            raise ValueError("Frozen training-context function is unavailable")
        default_names = {"batch_size", "optimizer_name", "optimizer_type", "optimizer_args"}
        defaults = [node for node in train.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id in default_names for target in node.targets)]
        if len(defaults) != len(default_names):
            raise ValueError("Frozen optimizer-default context is unavailable")
        selected += defaults
        selected += [node for node in train.body if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                     and node.test.left.id == "optimizer_name"]
        # Only literal task dispatch is interpreted, for selecting source text.
        # Numeric defaults, constraints and training code are never evaluated.
        input_branch = None
        for node in train.body:
            while isinstance(node, ast.If):
                test = node.test
                if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                        and test.left.id == "task" and len(test.ops) == len(test.comparators) == 1):
                    right = test.comparators[0]
                    matches = (isinstance(test.ops[0], ast.Eq) and isinstance(right, ast.Constant)
                               and right.value == training_task)
                    matches |= (isinstance(test.ops[0], ast.In) and isinstance(right, (ast.Set, ast.Tuple, ast.List))
                                and any(isinstance(item, ast.Constant) and item.value == training_task for item in right.elts))
                    if matches:
                        input_branch = node.body
                        break
                node = node.orelse[0] if len(node.orelse) == 1 else None
            if input_branch is not None:
                break
        if input_branch is None:
            raise ValueError("Frozen task input-context branch is unavailable")
        selected += input_branch
    lines = source.splitlines(keepends=True)
    fragments = [{"start_line": node.lineno, "end_line": node.end_lineno,
                  "text": "".join(lines[node.lineno - 1:node.end_lineno])}
                 for node in sorted(selected, key=lambda node: node.lineno)]
    return {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "fragments": fragments,
            "scope": "Exact read-only source excerpts; omitted code remains in the frozen reference"}


class PreparedDevelopment:
    """A task plugin, not controller routing. Final data never enter this context."""

    def __init__(self, native, config):
        self.native, self.config = native, config
        root = config.settings.get("prepared_root")
        if not root or not Path(root).is_absolute():
            raise ValueError("Testbed adapter requires absolute settings.prepared_root")
        self.root = Path(root)
        self.manifest = verify_prepared(self.root)
        self.recipe = get_recipe(self.manifest["task_id"])
        self.family = None
        self.vectorization = None
        self.snapshot_ref = None
        if self.recipe.task_id == "vectorization":
            from ..live.adapters import VectorizationDevelopment
            from ..live.models import DevelopmentSpec
            self.vectorization = VectorizationDevelopment(
                SimpleNamespace(public_dir=self.root / "reference"), DevelopmentSpec())
        elif self.recipe.task_id not in CPU_TASKS:
            from ..task_families import get_family
            self.family = get_family(self.recipe.task_id, settings=self.manifest["split_metadata"].get("resource_limits", {}))
        if config.metric != self.recipe.metric or config.direction != self.recipe.direction:
            raise ValueError("Development objective differs from frozen recipe")

    def identity(self):
        verify_prepared(self.root)
        return {"recipe": self.recipe.identity,
                "prepared_manifest": manifest_hash(self.manifest),
                **{name: file_hash(path) for name, path in self.source_files().items()}}

    def source_files(self):
        """Expose the complete trusted plugin closure for controller snapshots."""
        package = Path(__file__).parents[1]
        paths = [*Path(__file__).parent.glob("*.py"),
                 *(package / "task_families").rglob("*.py"),
                 package / "runtime_profiles.py", package / "runtime_locks.json"]
        return {"plugin/" + path.relative_to(package).as_posix(): path
                for path in sorted(paths) if path.is_file()}

    def live_readiness(self):
        """A live backend may only be enabled after independent offline validation."""
        from .readiness import verify_validation
        from .workflow_contracts import verified_workflow_contracts
        from .export import check_validated_protocol
        output = Path(self.config.settings.get("validation_output", self.root.parent / "validation")).resolve()
        errors = []
        try:
            receipt = verify_validation(self.root, output)
            check_validated_protocol(self.native, self.config, output, receipt)
            if receipt.get("actual_task_validated") is not True:
                errors.append("Full actual-task validation with independently verified dataset identity has not passed")
            contracts = verified_workflow_contracts(self.root.parent.parent)
            if contracts[self.recipe.adapter_family]["status"] != "pass":
                errors.append("The reusable execution family lacks verified measured workflow branch coverage")
        except (ValueError, OSError, KeyError) as error:
            errors.append(str(error))
        return {"ready": not errors, "errors": errors, "task_id": self.recipe.task_id,
                "adapter_family": self.recipe.adapter_family, "validation_output": str(output),
                "scope": "Task data/runtime/evaluator/recovery and reusable workflow coverage; live autonomous research remains unevaluated"}

    @property
    def timing_task(self):
        return self.recipe.task_id in {"vectorization", "llama-inference"}

    def context(self):
        if self.vectorization:
            return {"task_id": "vectorization", "metric": "latency_ratio", "direction": "minimize",
                    "contract": "Modify only Conv2DLayer.forward in train.py. Implement a stateless NumPy computation from the current input, kernel and bias: no memoized or hardcoded outputs, persistent caches, file/network reads, or timing-harness changes. Independent paired warmed timing compares the candidate to the immutable baseline; five warmups and ten interleaved pairs. Every output must match the independent oracle. Final workloads are held out. This controlled example measures correctness and latency; it does not establish algorithmic novelty or adversarial evaluator security.",
                    "starter_source": self.vectorization.source, "editable_files": ["train.py"],
                    "command": list(_TRAIN_COMMAND)}
        context = {"task_id": self.recipe.task_id, "metric": self.recipe.metric,
                "direction": self.recipe.direction, "targets": self.recipe.targets,
                "id_column": self.recipe.id_column, "editable_files": self.editable_paths(),
                "reference_baseline": self.recipe.reference_baseline,
                "input_files": ["train.csv", "evaluation.csv"],
                "submission": "submission.csv", "deviations": self.recipe.deviations,
                "command": list(_TRAIN_COMMAND)}
        if self.family:
            context.update(input_files=self.family.training_files + self.family.evaluation_files,
                           submission=self.family.output_paths,
                           base_snapshot_ref=self.snapshot_ref,
                           editable_source={name: (self.root / "reference" / name).read_text()
                                            for name in self.editable_paths()},
                           execution_contract=self._execution_context())
        return context

    def _execution_context(self):
        """Present frozen public harness rules without importing or restating them.

        Worker defaults and hyperparameter bounds are executable rules in the
        reference snapshot. Supplying those exact sources avoids maintaining a
        second validator or deriving contracts from prepared controls/results.
        """
        reference = self.root / "reference"
        excerpts = [_source_excerpts(reference / "array_data.py", functions=("archive_path", "open_arrays"))]
        if self.recipe.task_id == "CLRS":
            excerpts.append(_source_excerpts(reference / "clrs_worker.py", functions=("make_model", "feedback")))
        elif self.recipe.task_id == "llama-inference":
            excerpts.append(_source_excerpts(reference / "llama_model.py", functions=("load_optimized_model",)))
        else:
            functions = ["predict_batch"]
            if "transforms.py" in self.editable_paths():
                functions.append("transform_batch")
            excerpts.append(_source_excerpts(reference / "family_worker.py", functions=functions,
                                            training_task=self.recipe.task_id))
        context = {
            "base_configuration": json.loads((reference / "training.json").read_text()),
            "configuration_rules": (
                "Base configuration is shown before permitted candidate overrides and the "
                "controller-selected seed. training_contract.py defines allowed keys and "
                "bounds; the task-specific worker excerpts show defaults for omitted options and "
                "training/input handling. Array-loader excerpts define packed and directory array "
                "formats. These frozen harness sources are read-only context; edit only "
                "editable_files and use the exact command without added arguments."
            ),
            "read_only_sources": {name: (reference / name).read_text() for name in ("train.py", "training_contract.py")},
            "source_excerpts": excerpts,
        }
        normalization = self.manifest.get("split_metadata", {}).get("normalization")
        if isinstance(normalization, str):
            context["input_normalization"] = normalization
        if len(json.dumps(context, sort_keys=True)) > _EXECUTION_CONTEXT_MAX_CHARS:
            raise ValueError("Frozen execution context exceeds its 12000-character bound; revise the explicit presentation contract")
        return context

    def editable_paths(self):
        if self.family:
            descriptor = json.loads((self.root / "family-pipeline.json").read_text())
            return descriptor["editable_files"]
        return self.recipe.editable_files

    def source_tree(self):
        if self.family:
            return {"path": str(self.root / "reference"), "editable_paths": self.editable_paths()}
        return None

    def bind_source(self, snapshot_ref):
        self.snapshot_ref = snapshot_ref

    def _candidate(self, name):
        root = self.root / name
        if self.family:
            if self.snapshot_ref is None:
                raise ValueError("Source tree must be bound before requesting a checkpoint candidate")
            files = [SourceFile(path=path, content=(root / path).read_text())
                     for path in self.editable_paths()
                     if (root / path).read_bytes() != (self.root / "reference" / path).read_bytes()]
            return Candidate(files=files, base_snapshot_ref=self.snapshot_ref,
                             command=self.family.command("train"),
                             description=f"Prepared {name}; original source tree with bounded editable overlays")
        return Candidate(files=[SourceFile(path=path, content=(root / path).read_text())
                                for path in tree_manifest(root)],
                         command=["python", "train.py"],
                         description=f"Prepared {name}; no model generated this source")

    def baseline(self):
        return self._candidate("reference")

    def control(self):
        return self._candidate("control")

    def control_component(self):
        reference, control = tree_manifest(self.root / "reference"), tree_manifest(self.root / "control")
        if set(reference) != set(control):
            raise ValueError("Prepared matched controls must preserve the reference file inventory")
        changed = [name for name in reference if reference[name] != control[name]]
        if len(changed) > 1:
            raise ValueError("Prepared component-removal control changes more than one declared file")
        return changed[0] if changed else None

    def prepare(self, phase, destination):
        verify_prepared(self.root)
        if phase not in ({"subset", "full", "rebuttal", "final"} if self.timing_task else {"subset", "full", "rebuttal"}):
            raise ValueError("Final evidence cannot enter development preparation")
        prepared_phase = "screen" if phase == "subset" else "final" if phase == "final" else "full"
        source = self.root / "phases" / prepared_phase
        public = Path(destination) / "public"
        if public.exists():
            if tree_manifest(public) != tree_manifest(source / "public"):
                raise ValueError("Existing development public tree differs")
        else:
            shutil.copytree(source / "public", public)
        if self.vectorization:
            return {"public": str(public.resolve()), "files": tree_manifest(public),
                    "cases": self.manifest["split_metadata"]["cases"][prepared_phase]}
        if self.family:
            private = Path(destination) / "private"
            if private.exists():
                if tree_manifest(private) != tree_manifest(source / "private"):
                    raise ValueError("Existing protected development labels differ")
            else:
                shutil.copytree(source / "private", private)
            return {"public": str(public.resolve()), "private": str(private.resolve()),
                    "files": tree_manifest(public), "private_files": tree_manifest(private),
                    "recipe": self.recipe.identity}
        labels = pd.read_csv(source / "private/labels.csv", dtype={self.recipe.id_column: str})
        return {"public": str(public.resolve()), "files": tree_manifest(public),
                "labels": labels.to_dict("list"), "recipe": self.recipe.identity}

    def validate(self, candidate):
        if candidate.command != list(_TRAIN_COMMAND):
            raise ValueError(
                "Recipe requires command exactly " + json.dumps(list(_TRAIN_COMMAND))
                + "; do not add arguments or substitute an output path"
            )
        if self.vectorization:
            return self.vectorization.validate(candidate)
        if self.family:
            if candidate.base_snapshot_ref != self.snapshot_ref:
                raise ValueError("Checkpoint candidate must use the authorized source snapshot")
            allowed = set(self.editable_paths())
        else:
            allowed = set(self.recipe.editable_files)
        if not {f.path for f in candidate.files} <= allowed:
            raise ValueError("Candidate changes a file outside the recipe edit boundary")
        if not self.family and "train.py" not in {f.path for f in candidate.files}:
            raise ValueError("Candidate omits train.py")
        for file in candidate.files:
            if file.path.endswith(".py"):
                compile(file.content, file.path, "exec")

    def prediction_artifacts(self, prepared):
        if self.vectorization:
            return ["predictions.npz"]
        if self.family:
            return self.family.output_paths
        return ["submission.csv"]

    def artifacts(self, prepared):
        from ..artifacts import ArtifactSpec
        if self.vectorization:
            return [ArtifactSpec(path="predictions.npz", role="prediction", max_bytes=128_000_000)]
        if self.family:
            return self.family.artifacts("train")
        return [ArtifactSpec(path="submission.csv", role="prediction", max_bytes=64_000_000)]

    def _pipeline(self, private):
        from ..live.pipeline import EvaluationPipeline, EvaluationStage
        root = self.root / "evaluator"
        inference, scoring = root / "inference", root / "score"
        prediction_contracts = self.family.artifacts("infer")
        inference_resources = self.native.resources.model_copy(update={
            "output_max_bytes": sum(item.max_bytes for item in prediction_contracts),
            "output_max_files": sum(item.max_files for item in prediction_contracts),
        })
        return EvaluationPipeline(
            inference=EvaluationStage(kind="inference", image=self.native.image,
                                      source_dir=inference, source_manifest=tree_manifest(inference),
                                      command=self.family.command("infer"), resources=inference_resources,
                                      outputs=prediction_contracts),
            score=EvaluationStage(kind="score", image=self.native.grader_image,
                                  source_dir=scoring, source_manifest=tree_manifest(scoring),
                                  command=self.family.command("score"),
                                  resources=self.native.resources.model_copy(update={"gpus": None}),
                                  outputs=self.family.artifacts("score")),
            private_dir=private, private_manifest=tree_manifest(private),
        )

    def evaluation(self, prepared):
        return self._pipeline(Path(prepared["private"])) if self.family and not self.timing_task else None

    def diagnostic_capabilities(self):
        """Task declarations, not an assertion that every requested input is ready."""
        kinds = ["paired_differences", "paired_interval"]
        if self.timing_task:
            return kinds + ["timing_distribution"]
        if self.recipe.task_id in {"cifar10", "imdb", "ogbn-arxiv", "spaceship-titanic"}:
            return kinds + ["classification"]
        if self.recipe.task_id in {"house-price", "feedback"}:
            return kinds + ["regression_residuals"]
        # Missing clinical targets, masks, multilabel and language-model outputs
        # require different diagnostic contracts; do not advertise generic ones.
        return kinds

    def diagnostics_contract(self, kind, phase, *, prepared=None):
        """Frozen task-owned paths for isolated diagnostic scoring only."""
        from ..live.pipeline import EvaluationPipeline, EvaluationStage
        from ..research.diagnostic_runner import ScoringDiagnosticContract
        verify_prepared(self.root)
        if phase not in {"subset", "full", "rebuttal"}:
            raise ValueError("Final labels are unavailable to research diagnostics")
        if kind not in self.diagnostic_capabilities() or kind not in {"classification", "regression_residuals"}:
            raise ValueError("Task has no declared isolated scorer for this diagnostic")
        frozen_private = self.root / "phases" / ("screen" if phase == "subset" else "full") / "private"
        private = Path(prepared["private"]) if prepared and self.family else frozen_private
        if tree_manifest(private) != tree_manifest(frozen_private):
            raise ValueError("Diagnostic protected inputs differ from the frozen task partition")
        if self.family:
            contract = ScoringDiagnosticContract(kind=kind, format="class_scores_npz",
                prediction_path="predictions", label_path="labels.npz")
            pipeline = self._pipeline(private)
        else:
            contract = ScoringDiagnosticContract(kind=kind, format="cpu_frames", prediction_path="submission.csv",
                label_path="labels.csv", id_column=self.recipe.id_column, targets=self.recipe.targets)
            root = self.root / "evaluator"
            pipeline = EvaluationPipeline(score=EvaluationStage(kind="score", image=self.native.grader_image,
                source_dir=root, source_manifest=tree_manifest(root), command=["python", "evaluate.py"],
                resources=self.native.resources.model_copy(update={"gpus": None})),
                private_dir=private, private_manifest=tree_manifest(private))
        return {"contract": contract.model_dump(mode="json"), "pipeline": pipeline.model_dump(mode="json")}

    def final_evaluation(self):
        return self._pipeline(self.root / "phases/final/private") if self.family and not self.timing_task else None

    @property
    def evaluate_pair(self):
        return self._evaluate_pair if self.timing_task else None

    def _evaluate_pair(self, runner, key, candidate, phase="full", seeds=None):
        from ..live.paired_evaluation import execute_paired
        return execute_paired(runner, key, candidate, phase, seeds, run_pair=self._pair_runner())

    def _pair_runner(self):
        if self.vectorization:
            from ..live.timing_vectorization import run_vectorization_pair
            return run_vectorization_pair
        from ..live.timing_llama import run_llama_pair
        return run_llama_pair

    @property
    def evaluate_final(self):
        return self._evaluate_final if self.timing_task else None

    def _evaluate_final(self, runner, key, candidate):
        from ..live.paired_evaluation import execute_paired_final
        return execute_paired_final(runner, key, candidate, run_pair=self._pair_runner())

    @property
    def audit_timing(self):
        return self._audit_timing if self.timing_task else None

    def _audit_timing(self, store, prepared, report):
        if self.vectorization:
            from ..live.timing_vectorization import audit_vectorization_report
            return audit_vectorization_report(store, prepared, report)
        from ..live.timing_llama import audit_llama_report
        return audit_llama_report(store, prepared, report)

    def score(self, phase, prepared, workspace, seconds):
        if self.timing_task:
            raise ValueError("Timing tasks require the trusted paired harness, not candidate-written scores")
        if self.family:
            raise ValueError("Checkpoint tasks must use isolated inference and trusted scoring stages")
        if phase not in {"subset", "full", "rebuttal"}:
            raise ValueError("Final metrics are unavailable to development")
        prediction = pd.read_csv(Path(workspace) / "submission.csv",
                                 dtype={self.recipe.id_column: str})
        return score_frames(self.recipe.metric, pd.DataFrame(prepared["labels"]),
                            prediction, self.recipe.id_column, self.recipe.targets)
