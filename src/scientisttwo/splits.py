"""Task-level partitions, independent of training, search, and repetition seeds."""

import hashlib
import numpy as np
from sklearn.model_selection import train_test_split
from .store import canonical, digest


def array_hash(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        h.update(canonical({"shape": value.shape, "dtype": value.dtype.str}))
        h.update(value.tobytes())
    return h.hexdigest()


def partition(y, split_seed, *, stratified=False):
    indices = np.arange(len(y))
    train, rest = train_test_split(
        indices, test_size=0.4, random_state=split_seed, stratify=y if stratified else None
    )
    dev, test = train_test_split(
        rest, test_size=0.5, random_state=split_seed, stratify=y[rest] if stratified else None
    )
    return train, dev, test


def validate_partitions(record):
    n = record["count"]
    groups = [record[name] for name in ("train", "development", "test")]
    flat = [index for group in groups for index in group]
    if any(not group for group in groups) or len(flat) != n or set(flat) != set(range(n)):
        raise ValueError("Partitions must be nonempty, disjoint, and cover every row exactly once")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in flat):
        raise ValueError("Partition indices must be integers")


def make_manifest(task, loader, *, stratified=False):
    if task.schema_version != 2:
        raise ValueError(
            "Task schema v1 used rotating partitions. Preserve its artifacts and create a v2 campaign."
        )
    result = {
        "version": 2,
        "adapter": task.adapter,
        "data_seed": task.data_seed,
        "split_seed": task.split_seed,
        "stratified": stratified,
        "datasets": {},
    }
    for name in task.datasets:
        x, y = loader(name, task.data_seed)
        train, dev, test = partition(y, task.split_seed, stratified=stratified)
        record = {
            "count": len(y),
            "data_hash": array_hash(x, y),
            "train": train.tolist(),
            "development": dev.tolist(),
            "test": test.tolist(),
        }
        validate_partitions(record)
        result["datasets"][name] = record
    return result


def bind_spec(spec, manifest):
    expected = digest(canonical(manifest))
    if spec.split_manifest_hash and spec.split_manifest_hash != expected:
        raise ValueError("Evaluation specification refers to different partitions")
    return spec.model_copy(update={"split_manifest_hash": expected})


def prepare_arrays(spec, loader, *, stratified=False):
    manifest = make_manifest(spec.task, loader, stratified=stratified)
    bind_spec(spec, manifest)
    for name in spec.datasets:
        if name not in manifest["datasets"]:
            raise ValueError("Evaluation dataset is outside the frozen task")
        x, y = loader(name, spec.task.data_seed)
        record = manifest["datasets"][name]
        train = np.array(record["train"])
        if spec.train_fraction < 1:
            train, _ = train_test_split(
                train,
                train_size=spec.train_fraction,
                random_state=spec.task.split_seed,
                stratify=y[train] if stratified else None,
            )
        evaluation = np.array(record[spec.split])
        for seed in spec.seeds:
            yield (
                name,
                seed,
                {
                    "x_train": x[train],
                    "y_train": y[train],
                    "x_eval": x[evaluation],
                    "row_ids": evaluation,
                },
                y[evaluation],
            )
