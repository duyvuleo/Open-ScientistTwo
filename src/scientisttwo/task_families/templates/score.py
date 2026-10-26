"""Trusted CPU scoring. Mount predictions and labels only, never candidate code."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def aligned(predicted, labels):
    ids = np.asarray(labels["ids"]).astype(str)
    pids = np.asarray(predicted["ids"]).astype(str)
    if ids.ndim != 1 or pids.ndim != 1 or len(set(ids)) != len(ids) or len(set(pids)) != len(pids):
        raise ValueError("Prediction IDs must be one-dimensional and unique")
    if set(ids) != set(pids):
        raise ValueError("Prediction IDs must exactly match protected label IDs")
    lookup = {value: index for index, value in enumerate(pids)}
    return np.asarray([lookup[value] for value in ids])


def compute(task, predicted, labels):
    order = aligned(predicted, labels)
    values = np.asarray(predicted["predictions"])[order]
    targets = np.asarray(labels["y"])
    if not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("Non-finite predictions or labels")
    if task in {"cifar10", "imdb", "ogbn-arxiv"}:
        if values.ndim != 2 or len(values) != len(targets) or targets.ndim != 1:
            raise ValueError("Expected class score matrix and class IDs")
        if np.any(targets < 0) or np.any(targets >= values.shape[1]) or np.any(targets != targets.astype(int)):
            raise ValueError("Invalid class IDs")
        return {"accuracy": float(np.mean(values.argmax(axis=1) == targets))}
    if task == "identify-contrails":
        if values.ndim == 4 and values.shape[1] == 1:
            values = values[:, 0]
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if values.shape != targets.shape or not np.isin(targets, [0, 1]).all():
            raise ValueError("Expected shape-matched binary masks")
        binary = values >= 0.5
        axes = tuple(range(1, binary.ndim))
        numerator = 2 * np.sum(binary * targets, axis=axes) + 1e-6
        denominator = binary.sum(axis=axes) + targets.sum(axis=axes) + 1e-6
        return {"dice": float(np.mean(numerator / denominator))}
    if task == "fathomnet":
        if values.shape != targets.shape or values.ndim != 2 or not np.isin(targets, [0, 1]).all():
            raise ValueError("Expected class scores and real multi-hot categories")
        if np.any(targets.sum(axis=1) == 0):
            raise ValueError("Every image must have real category ground truth")
        ranks = np.argsort(-values, axis=1, kind="stable")[:, :20]
        hits = np.take_along_axis(targets, ranks, axis=1)
        precision = np.cumsum(hits, axis=1) / np.arange(1, ranks.shape[1] + 1)
        ap = np.sum(precision * hits, axis=1) / np.minimum(targets.sum(axis=1), 20)
        return {"map_at_20": float(ap.mean())}
    if task == "CLRS":
        if values.shape != targets.shape or targets.ndim < 2:
            raise ValueError("Expected shape-matched Floyd-Warshall pointer matrices")
        return {"pointer_accuracy": float(np.mean(values == targets))}
    if task == "llama-inference":
        if values.shape != targets.shape or not np.array_equal(values, targets):
            raise ValueError("Generated tokens differ from frozen reference")
        # Timings are deliberately not read from candidate-written files.
        return {"output_equivalence": 1.0}
    raise ValueError("Unsupported scoring task")


def perplexity(predictions, labels):
    expected_ids = np.asarray(labels["ids"]).astype(str)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Duplicate labels")
    lookup = {value: index for index, value in enumerate(expected_ids)}
    seen = set()
    total_nll = 0.0
    token_count = 0
    document_counts, example_counts = {}, {}
    for path in sorted(predictions.glob("logits-*.npz")):
        with np.load(path, allow_pickle=False) as batch:
            ids = batch["ids"].astype(str)
            logits = np.asarray(batch["logits"], dtype=np.float64)
            if logits.ndim != 3 or not np.isfinite(logits).all() or len(ids) != len(logits):
                raise ValueError("Invalid language-model logits")
            for sample_id, sample in zip(ids, logits, strict=True):
                if sample_id not in lookup or sample_id in seen:
                    raise ValueError("Unexpected or repeated prediction ID")
                seen.add(sample_id)
                index = lookup[sample_id]
                target = np.asarray(labels["target_ids"][index])
                raw_mask = np.asarray(labels["attention_mask"][index])
                if target.dtype.kind not in "iu" or not np.isin(raw_mask, [0, 1]).all():
                    raise ValueError("Perplexity requires integer tokens and a binary protected mask")
                mask = raw_mask.astype(bool)
                if len(sample) != len(target) or len(mask) != len(target):
                    raise ValueError("Token count mismatch")
                if np.any(target[mask] < 0) or np.any(target[mask] >= sample.shape[-1]):
                    raise ValueError("Token outside frozen vocabulary")
                sample = sample[mask]
                target = target[mask]
                maximum = sample.max(axis=1)
                logsum = maximum + np.log(np.exp(sample - maximum[:, None]).sum(axis=1))
                total_nll += float(np.sum(logsum - sample[np.arange(len(target)), target]))
                token_count += len(target)
                example_counts[str(sample_id)] = len(target)
                doc_id = str(labels["doc_ids"][index]) if "doc_ids" in labels else str(sample_id)
                document_counts[doc_id] = document_counts.get(doc_id, 0) + len(target)
    if seen != set(expected_ids) or not token_count:
        raise ValueError("Missing prediction IDs or no evaluated tokens")
    value = float(np.exp(total_nll / token_count))
    if not np.isfinite(value):
        raise ValueError("Non-finite perplexity")
    expected_count = int(np.asarray(labels["attention_mask"]).sum())
    if token_count != expected_count:
        raise ValueError("Scored token count differs from the complete protected token mask")
    return {"perplexity": value, "mean_token_nll": total_nll / token_count,
            "token_count": token_count, "expected_token_count": expected_count,
            "document_token_counts": document_counts, "example_token_counts": example_counts}


def score_shards(task, directory, labels):
    ids = np.asarray(labels["ids"]).astype(str)
    if ids.ndim != 1 or not len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Protected IDs must be nonempty and unique")
    lookup, seen, totals = {value: index for index, value in enumerate(ids)}, set(), {}
    paths = sorted(directory.glob("predictions-*.npz"))
    if not paths:
        with np.load(directory / "predictions.npz", allow_pickle=False) as predicted:
            return compute(task, predicted, labels)
    for path in paths:
        with np.load(path, allow_pickle=False) as predicted:
            batch_ids = predicted["ids"].astype(str)
            if len(set(batch_ids)) != len(batch_ids) or any(value in seen or value not in lookup for value in batch_ids):
                raise ValueError("Unexpected or duplicate sharded prediction ID")
            seen.update(batch_ids)
            indices = [lookup[value] for value in batch_ids]
            batch_labels = {"ids": batch_ids, "y": labels["y"][indices]}
            metrics = compute(task, predicted, batch_labels)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value * len(batch_ids)
    if seen != set(ids):
        raise ValueError("Missing sharded prediction IDs")
    return {name: value / len(ids) for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnostics = {}
    from array_data import open_arrays
    with open_arrays(args.labels) as labels:
        if args.task == "babylm":
            result = perplexity(args.predictions, labels)
            metrics = {"perplexity": result.pop("perplexity")}
            diagnostics = result
        else:
            metrics = score_shards(args.task, args.predictions, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.task == "babylm":
        label_path = args.labels if args.labels.exists() else args.labels.with_suffix("")
        paths = sorted(label_path.iterdir()) if label_path.is_dir() else [label_path]
        label_files = {}
        for path in paths:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            label_files[path.name] = {"sha256": digest.hexdigest(), "size": path.stat().st_size}
        counts = {"schema_version": 1, "labels": label_files,
                  "by_document": diagnostics.pop("document_token_counts"),
                  "by_example": diagnostics.pop("example_token_counts")}
        encoded = json.dumps(counts, sort_keys=True).encode()
        (args.output.parent / "token-counts.json").write_bytes(encoded)
        diagnostics.update(token_counts_sha256=hashlib.sha256(encoded).hexdigest(),
                           document_count=len(counts["by_document"]), example_count=len(counts["by_example"]))
    args.output.write_text(json.dumps({"status": "completed", "valid_submission": True,
                                       "metrics": metrics, "diagnostics": diagnostics}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
