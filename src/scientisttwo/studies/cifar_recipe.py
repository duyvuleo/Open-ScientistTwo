"""Local source preparation for the explicitly adapted CIFAR recipe study.

This module does not prepare data, register a task, run frameworks, or select an
LLM. An integrating adapter must bind these sources to independently verified
official CIFAR arrays and a frozen runtime. Existing LeNet recipes are unchanged.
"""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from ..benchmarks.files import file_hash, tree_manifest
from ..live.models import Candidate, SourceFile
from ..task_families import get_family
from ..testbed.catalog import get_recipe
from .cifar_baseline.training_contract import FIXED_CONFIGURATION, validate_overrides

VARIANT = "cifar10-resnet18-recipe-v1"
EDITABLE_FILES = ("training_config.json",)
PHASE_TRAINING_ROWS = {"screen": 20_000, "full": 40_000, "final": 50_000}
BASELINE_CONFIGURATION = {
    "optimizer": "sgd", "learning_rate": 0.1, "momentum": 0.9,
    "weight_decay": 0.0005, "gradient_clip_norm": 0.0,
    "crop_padding": 4, "horizontal_flip_probability": 0.5,
}


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def study_recipe():
    """Return a separate recipe identity, without altering the canonical catalog."""
    original = get_recipe("cifar10")
    return original.model_copy(update={
        "recipe_version": VARIANT,
        "reference_baseline": "scratch-cifar-resnet18-10epoch-sgd-crop-flip-v1",
        "editable_files": list(EDITABLE_FILES),
        "training_constraints": {**FIXED_CONFIGURATION, "initialization": "scratch",
                                 "architecture": "CIFAR-compatible ResNet18",
                                 "pretrained": False, "ensembles": False},
        "deviations": [
            "Starter test calls use development inputs; final labels removed from candidate mounts",
            "Named adapted study replaces the upstream LeNet reference with a fixed CIFAR-compatible ResNet18",
            "Only declared optimizer and crop/flip parameters are editable; model, transforms and worker code remain frozen",
            "Ten full epochs with fixed batch size 128; no scheduler, pretrained weights or ensembles",
            "Final training restores all 50000 original training rows after method freeze; official test labels remain protected",
        ],
    })


def calibration_plan():
    """Predeclare bounded development-only calibration, not a winning candidate."""
    return {
        "variant": VARIANT, "phase": "full", "seed": 0,
        "initial": dict(BASELINE_CONFIGURATION),
        "optional_sanity_check": {**BASELINE_CONFIGURATION, "learning_rate": 0.03},
        "max_training_runs": 2,
        "optional_trigger": "Only if the initial run is invalid or development accuracy is below 0.65",
        "selection_rule": (
            "Freeze the initial recipe when valid and development accuracy is at least 0.65. "
            "Otherwise permit exactly one predeclared 0.03 learning-rate sanity run and freeze it "
            "only if valid with development accuracy at least 0.65; otherwise stop as baseline-not-ready. "
            "Do not choose the better of two valid adequate runs or inspect final evidence."
        ),
        "accuracy_floor": 0.65,
        "floor_scope": "Predeclared sanity threshold, not a guarantee of a competitive or tuned reference",
        "required_artifacts": ["configuration", "source_identity", "protocol_identity", "training_receipt",
                               "predictions", "trusted_development_score", "resource_receipt"],
        "resource_measurements": ["training_seconds", "inference_seconds", "peak_gpu_memory_bytes",
                                  "peak_host_memory_bytes", "total_wall_seconds"],
        "post_calibration": "Freeze baseline source/configuration and matched per-experiment and campaign budgets before either live arm",
        "live_model_calls": 0, "final_access": False,
    }


def write_study_sources(destination):
    """Atomically emit a new source-only bundle; never overwrite a prepared task."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Study source destination must be new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    family = get_family("cifar10")
    templates = Path(__file__).with_name("cifar_baseline")
    with tempfile.TemporaryDirectory(prefix=".cifar-study-", dir=destination.parent) as temporary:
        root = Path(temporary) / "sources"
        root.mkdir()
        descriptor = family.write_sources(root)
        config = {"seed": 0, **FIXED_CONFIGURATION, **BASELINE_CONFIGURATION}
        for directory in (root / "reference", root / "control"):
            for name in ("model.py", "transforms.py", "training_contract.py"):
                shutil.copyfile(templates / name, directory / name)
            (directory / "training.json").write_text(_json(config))
            # The prepared control removes one augmentation component. It is for
            # adapter validation only, never an agent proposal or expected winner.
            overrides = {} if directory.name == "reference" else {"horizontal_flip_probability": 0.0}
            (directory / "training_config.json").write_text(_json(overrides))
        inference = root / "evaluator/inference"
        shutil.copyfile(templates / "training_contract.py", inference / "training_contract.py")
        (inference / "training.json").write_text(_json(config))
        descriptor.update(
            editable_files=list(EDITABLE_FILES), training_defaults=config,
            evaluator_identity={name: item["sha256"] for name, item in tree_manifest(root / "evaluator").items()},
            study_variant=VARIANT, study_recipe_sha256=study_recipe().identity,
        )
        (root / "family-pipeline.json").write_text(_json(descriptor))
        (root / "recipe.json").write_text(study_recipe().model_dump_json(indent=2) + "\n")
        (root / "calibration-plan.json").write_text(_json(calibration_plan()))
        provenance = root / "provenance"
        provenance.mkdir()
        shutil.copyfile(__file__, provenance / "cifar_recipe.py")
        identity = {
            "schema_version": 1, "variant": VARIANT, "recipe_sha256": study_recipe().identity,
            "editable_files": list(EDITABLE_FILES),
            "fixed_model_sha256": file_hash(root / "reference/model.py"),
            "study_preparation_sha256": file_hash(provenance / "cifar_recipe.py"),
            "reference_files": tree_manifest(root / "reference"),
            "evaluator_files": tree_manifest(root / "evaluator"),
            "source_files": tree_manifest(root),
            "phase_training_rows": PHASE_TRAINING_ROWS,
            "expected_steps": {phase: 10 * ((rows + 127) // 128) for phase, rows in PHASE_TRAINING_ROWS.items()},
            "training_seed_policy": {"screen": [0], "full": [0, 1, 2], "final": [0, 1, 2]},
            "normalization": "float32 NCHW: uint8 / 127.5 - 1",
            "data_validation": "unverified", "runtime_validation": "unverified",
            "calibration": "unfulfilled", "live_model_calls": 0,
        }
        (root / "study-sources.json").write_text(_json(identity))
        root.rename(destination)
    return identity


def verify_study_sources(root):
    root = Path(root)
    identity = json.loads((root / "study-sources.json").read_text())
    observed = tree_manifest(root)
    observed.pop("study-sources.json", None)
    if identity.get("variant") != VARIANT or identity.get("recipe_sha256") != study_recipe().identity:
        raise ValueError("Study recipe identity changed")
    if observed != identity.get("source_files"):
        raise ValueError("Frozen study source bundle changed")
    return identity


def baseline_candidate(snapshot_ref):
    return Candidate(files=[], base_snapshot_ref=snapshot_ref, command=["python", "train.py"],
                     description="Frozen scratch CIFAR-compatible ResNet18 reference; no model-generated source")


def recipe_candidate(snapshot_ref, overrides, *, description):
    validate_overrides(overrides)
    return Candidate(files=[SourceFile(path="training_config.json", content=_json(overrides))],
                     base_snapshot_ref=snapshot_ref, command=["python", "train.py"], description=description)


def validate_candidate(candidate, *, snapshot_ref):
    if candidate.base_snapshot_ref != snapshot_ref or candidate.command != ["python", "train.py"]:
        raise ValueError("Study candidate changed the authorized source snapshot or command")
    if any(source.path not in EDITABLE_FILES for source in candidate.files):
        raise ValueError("Only training_config.json is editable in the fixed-architecture study")
    for source in candidate.files:
        validate_overrides(json.loads(source.content))


def validate_materialized_workspace(workspace, source_identity):
    """Verify frozen code and parse overlays without executing submitted code."""
    workspace = Path(workspace)
    actual = tree_manifest(workspace)
    expected = source_identity["reference_files"]
    if set(actual) != set(expected):
        raise ValueError("Materialized study source inventory changed")
    if any(actual[name] != value for name, value in expected.items() if name not in EDITABLE_FILES):
        raise ValueError("Fixed architecture or execution code changed")
    if actual["model.py"]["sha256"] != source_identity["fixed_model_sha256"]:
        raise ValueError("Fixed model hash changed")
    overrides = validate_overrides(json.loads((workspace / "training_config.json").read_text()))
    return {"variant": VARIANT, "model_sha256": source_identity["fixed_model_sha256"],
            "source_sha256": hashlib.sha256(_json(actual).encode()).hexdigest(), "overrides": overrides}


def validate_training_receipt(receipt, configuration, *, phase):
    """Reject shortened or additional training even when execution itself exits zero."""
    if phase not in PHASE_TRAINING_ROWS:
        raise ValueError("Unknown study phase")
    if any(type(configuration.get(key)) is not int or configuration[key] != value
           for key, value in FIXED_CONFIGURATION.items()):
        raise ValueError("Frozen study training budget changed")
    expected = 10 * ((PHASE_TRAINING_ROWS[phase] + 127) // 128)
    if receipt.get("task") != "cifar10" or receipt.get("seed") != configuration.get("seed"):
        raise ValueError("Training receipt task or seed mismatch")
    if (type(receipt.get("steps")) is not int or receipt["steps"] != expected
            or receipt.get("epochs_started") != 10 or receipt.get("max_steps") != 3910):
        raise ValueError("Training did not complete exactly the frozen ten-epoch budget")
    return {"status": "pass", "phase": phase, "steps": expected, "epochs": 10, "batch_size": 128}
