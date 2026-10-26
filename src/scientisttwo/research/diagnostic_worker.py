"""Standalone isolated scorer: no candidate import and no row-level label export."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _ids(values):
    result = np.asarray(values).astype(str)
    if result.ndim != 1 or not len(result) or len(set(result)) != len(result):
        raise ValueError("Diagnostic IDs must be nonempty, one-dimensional and unique")
    return result


def _npz(root):
    if root.is_symlink():
        raise ValueError("Diagnostic array input cannot be a symlink")
    if root.is_dir():
        paths = sorted(root.glob("*.npy"))
        if not paths or any(path.is_symlink() for path in paths):
            raise ValueError("Invalid array directory")
        return {path.stem: np.load(path, mmap_mode="r", allow_pickle=False) for path in paths}
    if not root.exists() and root.suffix == ".npz":
        return _npz(root.with_suffix(""))
    import zipfile
    with zipfile.ZipFile(root) as archive:
        if len(archive.infolist()) > 1000 or sum(item.file_size for item in archive.infolist()) > 512 * 1024 ** 2:
            raise ValueError("Diagnostic NPZ exceeds memory quota")
    with np.load(root, allow_pickle=False) as archive:
        return dict(archive)


def arrays(prediction_root, label_root, contract):
    if contract["format"] == "class_scores_npz":
        labels = _npz(label_root)
        ids = _ids(labels["ids"])
        truth = np.asarray(labels["y"])
        lookup = {value: index for index, value in enumerate(ids)}
        files = sorted(prediction_root.glob("predictions-*.npz")) if prediction_root.is_dir() else []
        files = files or [prediction_root / "predictions.npz" if prediction_root.is_dir() else prediction_root]
        values = np.empty(len(ids), dtype=np.int64)
        seen = set()
        class_count = None
        for path in files:
            batch = _npz(path)
            bids = _ids(batch["ids"])
            scores = np.asarray(batch["predictions"])
            if scores.ndim != 2 or len(scores) != len(bids) or not np.isfinite(scores).all():
                raise ValueError("Class-score diagnostic requires a finite row-aligned matrix")
            if class_count is not None and class_count != scores.shape[1]:
                raise ValueError("Shards have different class vocabularies")
            class_count = scores.shape[1]
            if any(value in seen or value not in lookup for value in bids):
                raise ValueError("Unexpected or duplicate diagnostic prediction ID")
            seen.update(bids)
            values[[lookup[value] for value in bids]] = scores.argmax(axis=1)
        if seen != set(ids) or truth.shape != values.shape:
            raise ValueError("Diagnostic predictions omit protected IDs or have mismatched targets")
        return truth, values, list(range(class_count))
    if contract["format"] == "cpu_frames":
        id_column, targets = contract["id_column"], contract["targets"]
        def read(path):
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if not {id_column, *targets} <= set(reader.fieldnames or []):
                    raise ValueError("Diagnostic frame omits ID or target columns")
                rows = list(reader)
            ids = _ids([row[id_column] for row in rows])
            return ids, [[row[name] for name in targets] for row in rows]
        ids, truth = read(label_root)
        pids, predictions = read(prediction_root)
        if set(ids) != set(pids):
            raise ValueError("Diagnostic prediction IDs do not exactly match labels")
        order = {value: index for index, value in enumerate(pids)}
        predictions = [predictions[order[value]] for value in ids]
        if contract["kind"] == "classification":
            if len(targets) != 1:
                raise ValueError("Frame classification requires one target")
            classes = sorted({row[0] for row in truth + predictions})
            if any(not value for value in classes):
                raise ValueError("Frame classification labels must be present")
            lookup = {value: index for index, value in enumerate(classes)}
            return np.array([lookup[row[0]] for row in truth]), np.array([lookup[row[0]] for row in predictions]), list(range(len(classes)))
        return np.asarray(truth, dtype=float), np.asarray(predictions, dtype=float), None
    raise ValueError("Undeclared diagnostic input format")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    if contract["split"] != "development":
        raise ValueError("Final evidence cannot enter research diagnostics")
    # This module is generated from frozen trusted pure functions, never a model response.
    from diagnostic_math import classification_diagnostics, regression_residuals
    actual, predicted, classes = arrays(args.predictions, args.labels, contract)
    if contract["kind"] == "classification":
        result = classification_diagnostics(actual, predicted, classes=classes)
    elif contract["kind"] == "regression_residuals":
        result = regression_residuals(actual, predicted, target_names=contract["targets"])
    else:
        raise ValueError("Diagnostic is not implemented in this trusted scorer")
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
