"""Explicit local data preparation; no authentication, downloads, or model calls."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..benchmarks.files import file_hash, tree_manifest
from .catalog import get_recipe, UPSTREAM_REVISION
from .metrics import aligned_frames
from .templates import reference_source, control_source, evaluator_source

CPU_TASKS = {"house-price", "spaceship-titanic", "feedback",
             "amp-parkinsons-disease-progression-prediction"}
PHASES = ("screen", "full", "final")


def _vectorization_prepare(source, root, split_seed):
    from ..benchmarks.registry import verify_repository
    from ..live.adapters import VectorizationDevelopment
    from ..live.models import Candidate, DevelopmentSpec, SourceFile
    from ..live import timing_worker

    verify_repository(source, UPSTREAM_REVISION)
    env = source / "MLAgentBench/benchmarks/vectorization/env"
    adapter = VectorizationDevelopment(SimpleNamespace(public_dir=env), DevelopmentSpec(
        settings={"proposal_mode": "forward_body"}, split_seed=split_seed))
    for name in ("reference", "control", "evaluator"):
        (root / name).mkdir()
    (root / "reference/train.py").write_text(adapter.source)
    body = '''padding = self.padding if isinstance(self.padding, int) else 0
x = np.pad(features_batch, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
k = self.kernel_matrices
windows = np.lib.stride_tricks.sliding_window_view(x, k.shape[:2], axis=(1, 2))[:, ::self.stride, ::self.stride]
result = np.einsum("nhwcij,ijco->nhwo", windows, k, optimize=True) + self.biases
return np.maximum(result, 0) if self.activation == "relu" else result
'''
    control = adapter.materialize(Candidate(files=[SourceFile(path="train.py", content=body)],
                                             command=["python", "train.py"], description="Prepared NumPy vectorization control"))
    (root / "control/train.py").write_text(control.files[0].content)
    # Seal the actual transport code; execution uses the trusted paired harness,
    # never the upstream self-reported runtime CSV as an audited score.
    shutil.copyfile(timing_worker.__file__, root / "evaluator/timing_worker.py")
    phase_cases = {}
    for phase in PHASES:
        phase_root = root / "phases" / phase
        phase_adapter = VectorizationDevelopment(SimpleNamespace(public_dir=env), DevelopmentSpec(
            split_seed=split_seed + (1000 if phase == "final" else 0)))
        prepared = phase_adapter.prepare("subset" if phase == "screen" else "full", phase_root)
        (phase_root / "private").mkdir()
        phase_cases[phase] = prepared["cases"]
    return {"policy": "generated-benchmark-workloads; held-out-final-seed", "split_seed": split_seed,
            "final_seed": split_seed + 1000, "cases": phase_cases,
            "dataset_origin": "Pinned MLAgentBench convolution task; independently generated fixed arrays"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def manifest_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def preparation_sources():
    package = Path(__file__).parents[1]
    paths = [*Path(__file__).parent.glob("*.py"), *(package / "task_families").rglob("*.py")]
    return {path.relative_to(package).as_posix(): path for path in sorted(paths) if path.is_file()}


def prepare_plan(task_id, *, source=None):
    recipe = get_recipe(task_id)
    from .providers import KAGGLE_COMPETITIONS, PUBLIC_ARCHIVES
    return {
        "task_id": task_id, "canonical_task": True,
        "recipe": recipe.model_dump(), "recipe_sha256": recipe.identity,
        "source": str(Path(source).resolve()) if source else None,
        "minimum_disk_gb": recipe.minimum_disk_gb,
        "download_policy": "No automatic downloads or terms acceptance. Explicit benchmark acquire --allow-downloads supports recorded provider acquisition; benchmark prepare imports local bytes.",
        "acquisition_command": (["scientisttwo", "benchmark", "acquire", task_id, "--output", "SOURCE",
                                 "--allow-downloads", *(["--terms-accepted"] if recipe.access_requirements else []),
                                 *(["--include-images"] if task_id == "fathomnet" else [])]
                                if task_id in PUBLIC_ARCHIVES or task_id in KAGGLE_COMPETITIONS else None),
        "source_formats": (["bundle"] if task_id in {"CLRS", "llama-inference"}
                           else ["raw"] if task_id in {"vectorization", *CPU_TASKS} else ["raw", "bundle"]),
        "generation_command": (["scientisttwo", "benchmark", "generate-bundle", task_id,
                                "--image", "PINNED_IMAGE", "--output", "GENERATED_SOURCE"]
                               if task_id in {"CLRS", "llama-inference"} else None),
        "source_urls": [
            f"https://github.com/snap-stanford/MLAgentBench/tree/{UPSTREAM_REVISION}/MLAgentBench/benchmarks/{task_id}",
            *([f"https://www.kaggle.com/competitions/{KAGGLE_COMPETITIONS[task_id]}/data"]
              if task_id in KAGGLE_COMPETITIONS else []),
            *([PUBLIC_ARCHIVES[task_id]] if task_id in PUBLIC_ARCHIVES else []),
        ],
        "steps": ["Obtain required assets after accepting their terms outside this program",
                  recipe.preparation, "Run benchmark prepare with explicit provenance and source",
                  "Build and probe the recipe runtime", "Run benchmark validate on actual task data"],
        "live_calls": False,
    }


def _competition(task_id):
    return {"house-price": "home-data-for-ml-course", "feedback": "feedback-prize-english-language-learning",
            "identify-contrails": "google-research-identify-contrails-reduce-global-warming",
            "fathomnet": "fathomnet-out-of-sample-detection"}.get(task_id, task_id)


def _read(path, id_column):
    frame = pd.read_csv(path, dtype={id_column: str})
    if id_column not in frame or frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise ValueError(f"Input requires non-null unique {id_column}")
    return frame


def _partition(frame, fraction, seed, recipe):
    if recipe.task_id == "amp-parkinsons-disease-progression-prediction":
        groups = np.asarray(sorted(frame.patient_id.astype(str).unique()))
        train, test = train_test_split(groups, test_size=fraction, random_state=seed)
        mask = frame.patient_id.astype(str).isin(train)
        return frame.loc[mask].copy(), frame.loc[~mask].copy()
    stratify = frame[recipe.targets[0]] if recipe.metric == "accuracy" else None
    train, test = train_test_split(np.arange(len(frame)), test_size=fraction,
                                  random_state=seed, stratify=stratify)
    return frame.iloc[train].copy(), frame.iloc[test].copy()


def _parkinsons(source, *, include_supplemental=True):
    clinical = pd.read_csv(source / "train_clinical_data.csv", dtype={"patient_id": str})
    if "visit_id" not in clinical:
        raise ValueError("Clinical table requires visit_id")
    supplemental = source / "supplemental_clinical_data.csv"
    if supplemental.exists() and include_supplemental:
        extra = pd.read_csv(supplemental, dtype={"patient_id": str})
        extra = extra.loc[extra.patient_id.isin(set(clinical.patient_id))]
        clinical = pd.concat([clinical, extra], ignore_index=True)
    key = ["patient_id", "visit_month"]
    if clinical.duplicated(key).any():
        raise ValueError("Clinical/supplemental tables contain duplicate patient-month observations")
    required = {*key, *(f"updrs_{i}" for i in range(1, 5))}
    if not required <= set(clinical):
        raise ValueError("Clinical table lacks required patient, month, or UPDRS columns")
    indexed = clinical.set_index(key)
    rows = []
    for row in clinical.itertuples(index=False):
        for horizon in (0, 6, 12, 24):
            if row.visit_month + horizon > 108:
                continue
            future_key = (str(row.patient_id), row.visit_month + horizon)
            if future_key not in indexed.index:
                continue
            future = indexed.loc[future_key]
            for number in range(1, 5):
                rating = future[f"updrs_{number}"]
                if pd.notna(rating):
                    rows.append({"prediction_id": f"{row.patient_id}_{int(row.visit_month)}_updrs_{number}_plus_{horizon}_months",
                                 "patient_id": str(row.patient_id), "visit_month": row.visit_month,
                                 "horizon": horizon, "updrs_number": number, "rating": rating})
    frame = pd.DataFrame(rows)
    if frame.empty or frame.prediction_id.duplicated().any():
        raise ValueError("Clinical input produces empty/ambiguous forecast targets")
    return frame


def _cpu_prepare(recipe, source, root, labels_path, split_seed, derive_final=False):
    raw_parkinsons = recipe.task_id == "amp-parkinsons-disease-progression-prediction" and not (source / "train.csv").exists()
    frame = _parkinsons(source) if raw_parkinsons else _read(source / "train.csv", recipe.id_column)
    if not set(recipe.targets) <= set(frame) or frame[recipe.targets].isna().any().any():
        raise ValueError("Training targets must exist without missing values")
    if recipe.task_id == "feedback" and "full_text" not in frame:
        raise ValueError("Feedback requires full_text and all six target columns")
    if recipe.task_id == "amp-parkinsons-disease-progression-prediction":
        required = {"patient_id", "visit_month", "horizon", "updrs_number"}
        if not required <= set(frame):
            raise ValueError("Parkinson forecasts require patient, visit, horizon, and target number")
    if recipe.task_id == "feedback" and frame.full_text.duplicated().any():
        raise ValueError("Duplicate essays require explicit grouping before preparation")
    if labels_path is not None:
        final_inputs = _read(source / "test.csv", recipe.id_column)
        if set(recipe.targets) & set(final_inputs):
            raise ValueError("Public final inputs contain protected target columns")
        final_labels = _read(labels_path, recipe.id_column)
        aligned_frames(final_labels, final_inputs.assign(**{t: 0 for t in recipe.targets}), recipe.id_column, recipe.targets)
        if set(frame[recipe.id_column]) & set(final_inputs[recipe.id_column]):
            raise ValueError("Training and final identifiers overlap")
        if "patient_id" in frame and "patient_id" in final_inputs and set(frame.patient_id) & set(final_inputs.patient_id):
            raise ValueError("Patient leakage between training and final data")
        policy = "preserved-supplied-final; deterministic-80/20-training-development"
    elif derive_final:
        frame, final = _partition(frame, 0.2, split_seed, recipe)
        final_inputs = final.drop(columns=recipe.targets)
        final_labels = final[[recipe.id_column, *recipe.targets]]
        policy = "adapted-deterministic-60/20/20-from-training; no-published-final-equivalence"
    else:
        # Reproduce the pinned upstream preparation's training/final membership;
        # the competition's unlabeled test.csv is not MLAgentBench's final split.
        if recipe.task_id in {"house-price", "spaceship-titanic"}:
            boundary = int(len(frame) * 0.8)
            frame, final = frame.iloc[:boundary].copy(), frame.iloc[boundary:].copy()
            policy = "upstream-row-order-80/20-final; deterministic-training-development"
        elif recipe.task_id == "feedback":
            shuffled = frame.sample(frac=1, random_state=42).reset_index(drop=True)
            boundary = int(len(shuffled) * 0.98)
            frame, final = shuffled.iloc[:boundary].copy(), shuffled.iloc[boundary:].copy()
            policy = "upstream-seed42-shuffle-98/2-final; deterministic-training-development"
        elif raw_parkinsons:
            clinical = pd.read_csv(source / "train_clinical_data.csv", dtype={"patient_id": str})
            patients = clinical.patient_id.unique().tolist()
            if len(patients) < 5:
                raise ValueError("Parkinson canonical preparation needs at least five patient groups")
            held_out = set(random.Random(42).sample(patients, 2))
            original = _parkinsons(source, include_supplemental=False)
            final = original.loc[original.patient_id.astype(str).isin(held_out)].copy()
            frame = frame.loc[frame.patient_id.astype(str).isin(set(patients) - held_out)].copy()
            policy = "upstream-seed42-two-final-patients; group-consistent-development-and-all-tables"
        else:
            raise ValueError("Normalized Parkinson tables require supplied final labels or explicit derive_final=True")
        if frame.empty or final.empty:
            raise ValueError("Canonical upstream split produces an empty training or final partition")
        final_inputs = final.drop(columns=recipe.targets)
        final_labels = final[[recipe.id_column, *recipe.targets]]
    training, development = _partition(frame, 0.25 if derive_final else 0.2, split_seed + 1, recipe)
    screening, _ = _partition(training, 0.5, split_seed + 2, recipe)
    split_ids = {
        "train": training[recipe.id_column].astype(str).tolist(),
        "development": development[recipe.id_column].astype(str).tolist(),
        "final": final_inputs[recipe.id_column].astype(str).tolist(),
        "screening_train": screening[recipe.id_column].astype(str).tolist(),
    }
    partitions = {
        "screen": (screening, development.drop(columns=recipe.targets), development[[recipe.id_column, *recipe.targets]]),
        "full": (training, development.drop(columns=recipe.targets), development[[recipe.id_column, *recipe.targets]]),
        "final": (training, final_inputs, final_labels[[recipe.id_column, *recipe.targets]]),
    }
    for phase, (train, inputs, labels) in partitions.items():
        public, private = root / "phases" / phase / "public", root / "phases" / phase / "private"
        public.mkdir(parents=True)
        private.mkdir(parents=True)
        train.to_csv(public / "train.csv", index=False)
        inputs.to_csv(public / "evaluation.csv", index=False)
        labels.to_csv(private / "labels.csv", index=False)
        if raw_parkinsons:
            # Only training patients' measurements are exposed. Evaluation patient
            # features are represented in evaluation.csv without clinical targets.
            groups = set(train.patient_id.astype(str))
            for table in ("train_proteins.csv", "train_peptides.csv"):
                data = pd.read_csv(source / table, dtype={"patient_id": str})
                if "patient_id" not in data:
                    raise ValueError(f"{table} is missing patient_id")
                data.loc[data.patient_id.isin(groups)].to_csv(public / table, index=False)
                data.loc[data.patient_id.isin(set(inputs.patient_id.astype(str)))].to_csv(
                    public / table.replace("train_", "evaluation_"), index=False)
    for name in ("reference", "control", "evaluator"):
        (root / name).mkdir()
    (root / "reference/train.py").write_text(reference_source(recipe))
    (root / "control/train.py").write_text(control_source(recipe))
    (root / "evaluator/evaluate.py").write_text(evaluator_source(recipe))
    shutil.copyfile(Path(__file__).with_name("metrics.py"), root / "evaluator/metrics.py")
    return {"policy": policy, "split_seed": split_seed, "ids": split_ids,
            "row_counts": {k: len(v) for k, v in split_ids.items()},
            "patient_groups": {phase: sorted(set(values[0].patient_id.astype(str)))
                               for phase, values in partitions.items()} if "patient_id" in frame else {}}


def _copy_bundle(recipe, source, root):
    """Prepared GPU bundles have explicit schemas validated without loading frameworks."""
    from ..task_families import get_family

    family = get_family(recipe.task_id)
    for phase in PHASES:
        for kind in ("public", "private"):
            directory = source / "phases" / phase / kind
            if not directory.is_dir() or not tree_manifest(directory):
                raise ValueError(f"Prepared bundle missing nonempty phases/{phase}/{kind}")
    # Strictly copy named data trees; unrelated raw inputs, source metadata and
    # credentials never enter candidate-visible mounts.
    shutil.copytree(source / "phases", root / "phases")
    assets = source / "assets"
    if assets.exists():
        tree_manifest(assets)
        shutil.copytree(assets, root / "assets")
    if recipe.task_id == "CLRS":
        from .upstream_helpers import CLRS_HELPERS
        helpers = source / "source_helpers"
        if not helpers.is_dir() or not tree_manifest(helpers):
            raise ValueError("CLRS needs source_helpers from the pinned upstream environment")
        actual_helpers = {name: record["sha256"] for name, record in tree_manifest(helpers).items()}
        if actual_helpers != CLRS_HELPERS:
            raise ValueError("CLRS source_helpers differ from the complete pinned helper closure")
        shutil.copytree(helpers, root / "source_helpers")
    metadata_path = source / "split_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Prepared bundle needs split_metadata.json with dataset origin and official split provenance")
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict) or not metadata.get("dataset_origin"):
        raise ValueError("Prepared split metadata must identify the dataset origin")
    # Family adapters define and enforce data schemas; missing validation is an
    # integration gap, never a passing data gate.
    validator = getattr(family, "validate_prepared", None)
    if not callable(validator):
        raise ValueError("Task family has no prepared-data validator yet")
    validation = validator(root, metadata)
    metadata["input_validation"] = validation
    predictions = validation.get("predicted_output_bytes", {})
    if predictions:
        maximum = max(predictions.values())
        metadata["resource_forecast"] = {
            "predicted_output_bytes": predictions,
            "required_free_disk_mb": (maximum * 3 + (1 << 20) - 1) // (1 << 20) + 8192,
            "scope": "Peak single-evaluation staging and immutable copies; retained multi-experiment evidence consumes additional space",
        }
    writer = getattr(family, "write_sources", None)
    if not callable(writer):
        raise ValueError("Task family has no isolated source templates yet")
    writer(root)
    return metadata


def prepare(task_id, source, destination, *, provenance, labels=None,
            split_seed=1729, variant="audited-v1", terms_accepted=False, derive_final=False,
            source_format="raw", source_receipt=None):
    """Atomically prepare a new directory. Existing trees are never overwritten."""
    recipe = get_recipe(task_id)
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if provenance not in {"actual_data", "synthetic_fixture"}:
        raise ValueError("Explicit provenance must be actual_data or synthetic_fixture")
    if source_format not in {"raw", "bundle"}:
        raise ValueError("source_format must be raw or bundle")
    if recipe.access_requirements and provenance == "actual_data" and not terms_accepted:
        raise ValueError("Actual data requires explicit confirmation that applicable access terms were accepted")
    if destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Destination must be new and must not overlap input data")
    if not source.is_dir():
        raise ValueError("Source must be an existing local directory")
    manifest_source = (source / "MLAgentBench/benchmarks/vectorization/env"
                       if task_id == "vectorization" else source)
    source_files = tree_manifest(manifest_source)
    if not source_files:
        raise ValueError("Source data directory is empty")
    labels_path = Path(labels).resolve() if labels else None
    if derive_final and labels_path is not None:
        raise ValueError("Cannot both preserve supplied final labels and derive a new final holdout")
    if labels_path is not None and (not labels_path.is_file() or labels_path.is_symlink()):
        raise ValueError("Final labels must be a regular local file")
    labels_identity = file_hash(labels_path) if labels_path else None
    if task_id == "llama-inference" and variant == "audited-v1":
        metadata_file = source / "split_metadata.json"
        metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
        if metadata.get("model_identity") != "original-llama-7b":
            raise ValueError("Compatible alternative weights require an explicitly named adapted variant")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".testbed-prepare-", dir=destination.parent) as temporary:
        root = Path(temporary) / "prepared"
        root.mkdir()
        if task_id == "vectorization":
            split_metadata = _vectorization_prepare(source, root, split_seed)
        elif task_id in CPU_TASKS:
            split_metadata = _cpu_prepare(recipe, source, root, labels_path, split_seed, derive_final)
        else:
            prepared_source = source
            if source_format == "raw":
                from .acquisition import convert_raw
                prepared_source = Path(temporary) / "converted"
                convert_raw(task_id, source, prepared_source, split_seed=split_seed)
            split_metadata = _copy_bundle(recipe, prepared_source, root)
        from .source_identity import dataset_identity
        data_identity = dataset_identity(task_id, source, split_metadata, provenance=provenance,
                                         source_receipt=source_receipt)
        preparation_identity = {}
        for name, path in preparation_sources().items():
            target = root / "provenance/sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            preparation_identity[name] = file_hash(target)
        # Rehash inputs after preparation to reject concurrent source changes.
        if source_files != tree_manifest(manifest_source):
            raise ValueError("Source changed during preparation")
        if labels_path and file_hash(labels_path) != labels_identity:
            raise ValueError("Final labels changed during preparation")
        value = {
            "schema_version": 1, "task_id": task_id, "recipe_version": recipe.recipe_version,
            "recipe_sha256": recipe.identity, "upstream_revision": recipe.upstream_revision,
            "variant": variant, "provenance": provenance, "terms_accepted": terms_accepted,
            "source_format": source_format,
            "dataset_identity": data_identity,
            "preparation_identity": preparation_identity,
            "source_manifest": source_files,
            "final_labels_sha256": labels_identity,
            "split_metadata": split_metadata, "files": tree_manifest(root),
            "original_starter_status": recipe.original_starter_status,
            "reference_baseline": recipe.reference_baseline,
            "deviations": recipe.deviations, "live_model_calls": 0,
        }
        (root / "manifest.json").write_bytes(canonical(value))
        root.rename(destination)
    return {"prepared": str(destination), "manifest_sha256": manifest_hash(value), **value}


def verify_prepared(root):
    root = Path(root).resolve()
    path = root / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Prepared manifest is missing or unsafe")
    value = json.loads(path.read_text())
    recipe = get_recipe(value["task_id"])
    if value.get("recipe_sha256") != recipe.identity or value.get("upstream_revision") != UPSTREAM_REVISION:
        raise ValueError("Recipe or upstream identity changed after preparation")
    actual = tree_manifest(root)
    actual.pop("manifest.json")
    if actual != value.get("files"):
        raise ValueError("Prepared files changed, disappeared or appeared after freezing")
    if value.get("provenance") not in {"actual_data", "synthetic_fixture"} or value.get("live_model_calls") != 0:
        raise ValueError("Invalid prepared-data provenance")
    fingerprints = value.get("preparation_identity")
    if not fingerprints or any(file_hash(root / "provenance/sources" / name) != expected
                               for name, expected in fingerprints.items()):
        raise ValueError("Frozen preparation implementation provenance is missing or inconsistent")
    return value
