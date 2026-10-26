"""Separately named CIFAR study adapter; never rewrites the canonical registry."""

import json
from pathlib import Path
import shutil
import tempfile

from ..benchmarks.files import file_hash, tree_manifest
from ..live.models import DevelopmentSpec
from ..live.workspace_models import WorkspaceSpec
from ..live.workspace_task import freeze_workspace
from ..store import IntegrityError, Store
from ..task_families import get_family
from ..testbed.adapter import PreparedDevelopment
from ..testbed.preparation import canonical, manifest_hash, verify_prepared
from .cifar_recipe import (
    PHASE_TRAINING_ROWS, VARIANT, recipe_candidate, study_recipe, validate_candidate, write_study_sources,
)


def _check_official_identity(manifest):
    identity, metadata = manifest.get("dataset_identity", {}), manifest.get("split_metadata", {})
    if (manifest.get("task_id") != "cifar10" or manifest.get("provenance") != "actual_data"
            or identity.get("status") != "verified"
            or identity.get("kind") != "official-published-archive-and-member-hashes"
            or identity.get("training_rows") != 50_000 or identity.get("final_rows") != 10_000
            or metadata.get("original_training_rows") != 50_000
            or metadata.get("original_final_rows") != 10_000):
        raise ValueError("Study requires independently verified complete official CIFAR-10 50000/10000 data")


def _check_counts(validation):
    expected = {phase: {"training": count, "evaluation": 10_000}
                for phase, count in PHASE_TRAINING_ROWS.items()}
    if validation.get("status") != "pass" or validation.get("counts") != expected:
        raise ValueError("Study requires fixed 20000/40000/50000 training and 10000 evaluation rows")


def prepare_study(source_prepared, destination):
    """Freeze a new study from already verified CIFAR data; no download or execution.

    Data are copied as regular files. The original prepared LeNet task is never
    changed, and the new model does not inherit its validation-ready claim.
    """
    source, destination = Path(source_prepared).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Study destination must be new and disjoint from prepared input")
    original = verify_prepared(source)
    _check_official_identity(original)
    validation = get_family("cifar10").validate_prepared(source, original["split_metadata"])
    _check_counts(validation)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cifar-study-data-", dir=destination.parent) as temporary:
        root = Path(temporary) / "prepared"
        source_identity = write_study_sources(root)
        shutil.copytree(source / "phases", root / "phases")
        if (source / "assets").exists():
            # Scratch classification requires no weights or other model assets.
            if tree_manifest(source / "assets"):
                raise ValueError("Scratch CIFAR study does not accept undeclared asset caches")
        (root / "provenance/parent-prepared-manifest.json").write_bytes(canonical(original))
        if verify_prepared(source) != original:
            raise ValueError("Prepared source changed during study freezing")
        value = {
            "schema_version": 1, "task_id": "cifar10", "variant": VARIANT,
            "recipe_sha256": study_recipe().identity, "provenance": "actual_data",
            "upstream_revision": study_recipe().upstream_revision,
            "dataset_identity": original["dataset_identity"],
            "split_metadata": original["split_metadata"], "data_contract": validation,
            "parent_manifest_sha256": manifest_hash(original),
            "source_identity": source_identity, "files": tree_manifest(root),
            "reference_baseline": study_recipe().reference_baseline,
            "deviations": study_recipe().deviations,
            "runtime_validation": "unverified", "execution_validation": "unverified",
            "live_model_calls": 0,
        }
        (root / "study-manifest.json").write_bytes(canonical(value))
        root.rename(destination)
    return value


def verify_study(root):
    root = Path(root)
    path = root / "study-manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Study manifest is absent or unsafe")
    value = json.loads(path.read_text())
    if value.get("variant") != VARIANT or value.get("recipe_sha256") != study_recipe().identity:
        raise ValueError("Study recipe differs from the frozen adapted protocol")
    _check_official_identity(value)
    _check_counts(value.get("data_contract", {}))
    files = tree_manifest(root)
    files.pop("study-manifest.json", None)
    if files != value.get("files"):
        raise ValueError("Study data, sources or evaluator changed after freezing")
    parent = json.loads((root / "provenance/parent-prepared-manifest.json").read_text())
    if manifest_hash(parent) != value.get("parent_manifest_sha256"):
        raise ValueError("Study parent data provenance changed")
    _check_official_identity(parent)
    if value["dataset_identity"] != parent["dataset_identity"] or value["split_metadata"] != parent["split_metadata"]:
        raise ValueError("Study changed the official dataset binding or split policy")
    # The parent hashes establish that copied image/label bytes stayed intact.
    copied = {name: item for name, item in files.items() if name.startswith("phases/")}
    expected = {name: item for name, item in parent["files"].items() if name.startswith("phases/")}
    if not copied or copied != expected:
        raise ValueError("Study partitions differ from their verified parent data")
    sources = value["source_identity"]
    if sources != json.loads((root / "study-sources.json").read_text()):
        raise ValueError("Study source identity record changed")
    if any(files.get(name) != item for name, item in sources["source_files"].items()):
        raise ValueError("Frozen study source closure changed")
    return value


class CifarRecipeDevelopment(PreparedDevelopment):
    """Fixed-model study using the existing isolated checkpoint execution family."""

    def __init__(self, native, config):
        self.native, self.config = native, config
        root = config.settings.get("prepared_root")
        if not root or not Path(root).is_absolute():
            raise ValueError("CIFAR study adapter requires absolute prepared_root")
        self.root = Path(root)
        self.manifest = verify_study(self.root)
        self.recipe, self.family = study_recipe(), get_family("cifar10")
        self.vectorization, self.snapshot_ref = None, None
        if config.metric != "accuracy" or config.direction != "maximize":
            raise ValueError("CIFAR study metric must be accuracy/maximize")
        if config.seeds != [0, 1, 2]:
            raise ValueError("CIFAR study requires three matched training seeds [0,1,2]")

    def identity(self):
        value = verify_study(self.root)
        return {"recipe": self.recipe.identity, "prepared_manifest": manifest_hash(value),
                **{name: file_hash(path) for name, path in self.source_files().items()}}

    def source_files(self):
        files = super().source_files()
        package = Path(__file__).parents[1]
        files.update({"plugin/" + path.relative_to(package).as_posix(): path
                      for path in sorted(Path(__file__).parent.rglob("*.py"))})
        return files

    def context(self):
        value = super().context()
        value["frozen_literature"] = self.sources()
        value["study_contract"] = {
            "variant": VARIANT, "architecture": "Frozen CIFAR-compatible ResNet18 from scratch",
            "question": "Can a training-recipe change improve three-seed development accuracy over the calibrated reference within the same ten-epoch budget, and which declared component explains the observed difference?",
            "claim_boundary": "Known optimizers and augmentations are allowed. This is a bounded empirical optimization study; do not describe a recipe search as a novel architecture or claim global novelty. The final utility target is a mean accuracy gain of at least 0.02 with a positive paired difference on every declared seed; only the frozen final evaluation can assess that target.",
            "model_sha256": self.manifest["source_identity"]["fixed_model_sha256"],
            "fixed_sources": {name: (self.root / "reference" / name).read_text()
                              for name in ("model.py", "transforms.py")},
            "rules": "Change only the declared JSON optimization and crop/flip fields. Ten full epochs, batch size128, no pretrained weights, ensembling, scheduler or architecture edits. Final labels and results are unavailable during refinement.",
            "phase_training_rows": dict(PHASE_TRAINING_ROWS),
        }
        return value

    def sources(self):
        from .literature import study_sources
        return study_sources()

    def validate(self, candidate):
        validate_candidate(candidate, snapshot_ref=self.snapshot_ref)

    def _calibration(self):
        configured = self.config.settings.get("calibration_root")
        if not configured or not Path(configured).is_absolute():
            raise ValueError("A verified absolute calibration_root is required before live research")
        root = Path(configured).resolve()
        if root.is_relative_to(self.root.resolve()) or not (root / "campaign.sqlite").is_file():
            raise ValueError("Calibration evidence must be an existing campaign outside the frozen task")
        receipt_ref = self.config.settings.get("calibration_receipt_ref")
        if not receipt_ref:
            raise ValueError("An immutable calibration_receipt_ref must be configured")
        from .calibration import verify_calibration

        store = Store(root)
        try:
            return verify_calibration(store, receipt_ref, native=self.native,
                                      development=self.config, adapter=self, require_actual=True)
        finally:
            store.close()

    def baseline(self):
        if not self.config.settings.get("calibration_root"):
            return super().baseline()
        result = self._calibration()
        return recipe_candidate(self.snapshot_ref, result["selected_overrides"],
                                description="Frozen development-calibrated study reference")

    def prepare(self, phase, destination):
        verify_study(self.root)
        if phase not in {"subset", "full", "rebuttal"}:
            raise ValueError("Final evidence cannot enter study development preparation")
        source = self.root / "phases" / ("screen" if phase == "subset" else "full")
        destination = Path(destination)
        for name in ("public", "private"):
            target = destination / name
            if target.exists():
                if tree_manifest(target) != tree_manifest(source / name):
                    raise ValueError("Existing study development data differ from the frozen partition")
            else:
                shutil.copytree(source / name, target)
        return {"public": str((destination / "public").resolve()),
                "private": str((destination / "private").resolve()),
                "files": tree_manifest(destination / "public"),
                "private_files": tree_manifest(destination / "private"), "recipe": self.recipe.identity}

    def diagnostics_contract(self, kind, phase, *, prepared=None):
        from ..research.diagnostic_runner import ScoringDiagnosticContract
        verify_study(self.root)
        if phase not in {"subset", "full", "rebuttal"} or kind != "classification":
            raise ValueError("Only non-final classification diagnostics are available")
        frozen = self.root / "phases" / ("screen" if phase == "subset" else "full") / "private"
        private = Path(prepared["private"]) if prepared else frozen
        if tree_manifest(private) != tree_manifest(frozen):
            raise ValueError("Study diagnostic labels differ from the frozen development partition")
        contract = ScoringDiagnosticContract(kind=kind, format="class_scores_npz",
                                             prediction_path="predictions", label_path="labels.npz")
        return {"contract": contract.model_dump(mode="json"),
                "pipeline": self._pipeline(private).model_dump(mode="json")}

    def live_readiness(self):
        # A new architecture cannot inherit the old LeNet validation receipt.
        errors = []
        try:
            verify_study(self.root)
            self._calibration()
        except (ValueError, OSError, KeyError, IntegrityError, ImportError) as error:
            errors.append(str(error))
        return {"ready": not errors, "task_id": "cifar10", "variant": VARIANT,
                "errors": errors,
                "scope": "Verified actual development calibration and source/runtime binding; no claim of end-to-end autonomous capability"}


def study_specs(prepared_root, *, image, grader_image, resources, runtime="docker", context=None):
    """Create frozen native/development specifications without constructing a backend."""
    root = Path(prepared_root).resolve()
    verify_study(root)
    if not resources.gpus:
        raise ValueError("Study requires an explicitly allocated CUDA device")
    native = WorkspaceSpec(
        name=VARIANT, task_id="cifar10", description="Adapted MLAgentBench fixed-ResNet18 training-recipe study",
        public_dir=root / "phases/final/public", private_dir=root / "phases/final/private",
        candidate_dir=root / "reference", evaluator_dir=root / "evaluator",
        command=["python", "train.py"], evaluator_command=["python", "evaluate.py"],
        metric="accuracy", direction="maximize", image=image, grader_image=grader_image,
        runtime=runtime, context=context, resources=resources,
    )
    development = DevelopmentSpec(adapter="cifar_recipe", metric="accuracy", direction="maximize",
                                  seeds=[0, 1, 2], prediction_columns=["label"],
                                  settings={"prepared_root": str(root), "runtime_profile": "torch-cuda"})
    adapter = CifarRecipeDevelopment(native, development)
    native.evaluation_pipeline = adapter.final_evaluation()
    native.output_artifacts = adapter.artifacts(None)
    return freeze_workspace(native), development
