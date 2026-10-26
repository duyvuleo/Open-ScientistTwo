import numpy as np
import pytest

from scientisttwo.catalog import (
    classification_data,
    regression_data,
    optimization_data,
    OPTIMIZATION,
)
from scientisttwo.data import prepare
from scientisttwo.models import TaskSpec, EvalSpec
from scientisttwo.splits import make_manifest, prepare_arrays, validate_partitions
from scientisttwo.store import canonical, digest
from scientisttwo.tasks import load_task_adapter


@pytest.mark.parametrize(
    "family,name,metric",
    [
        ("classification", "wine", "macro_f1"),
        ("regression", "quadratic", "mse"),
        ("optimization", "simplex-mild", "objective_gap"),
    ],
)
def test_every_seed_and_rebuttal_keeps_test_rows_reserved(family, name, metric):
    task = TaskSpec(
        adapter=family,
        datasets=[name],
        subset_datasets=[name],
        seeds=[0, 1, 2],
        primary_metric=metric,
        metric_directions={metric: "maximize" if family == "classification" else "minimize"},
        guardrails={},
    )
    adapter = load_task_adapter(family, task)
    manifest = adapter.split_manifest(task)
    record = manifest["datasets"][name]
    test_ids = set(record["test"])
    for seeds in ([0, 1, 2], [10, 11, 12], [20, 21, 22], [3, 4, 23, 24]):
        repeat = task.model_copy(update={"seeds": seeds})
        assert adapter.split_manifest(repeat) == manifest
        for phase in ("subset", "full", "rebuttal"):
            spec = adapter.evaluation(repeat, phase)
            loader = {
                "classification": classification_data,
                "regression": regression_data,
                "optimization": optimization_data,
            }[family]
            for _, _, arrays, _ in prepare_arrays(
                spec, loader, stratified=family == "classification"
            ):
                assert set(arrays["row_ids"]) == set(record["development"])
                assert not test_ids.intersection(arrays["row_ids"])
                assert "y_eval" not in arrays and "y_test" not in arrays
                assert spec.split_manifest_hash == digest(canonical(manifest))
    assert not set(record["train"]) & test_ids


def test_partition_version_and_tampering_rejected():
    with pytest.raises(ValueError, match="schema v1"):
        list(prepare(EvalSpec.for_phase(TaskSpec(schema_version=1), "full")))
    spec = EvalSpec.for_phase(TaskSpec(), "full").model_copy(
        update={"split_manifest_hash": "wrong"}
    )
    with pytest.raises(ValueError, match="different partitions"):
        list(prepare(spec))
    with pytest.raises(ValueError, match="disjoint"):
        validate_partitions({"count": 4, "train": [0, 1], "development": [1], "test": [3]})


def test_data_change_changes_partition_manifest_identity():
    task = TaskSpec(datasets=["moons"], subset_datasets=["moons"])
    original = make_manifest(task, classification_data, stratified=True)
    changed = make_manifest(
        task.model_copy(update={"data_seed": 4}), classification_data, stratified=True
    )
    assert digest(canonical(original)) != digest(canonical(changed))


@pytest.mark.parametrize("name", list(OPTIMIZATION))
def test_optimization_oracle_satisfies_constraints_and_kkt(name):
    x, optimal = optimization_data(name, 5)
    mode, d, _ = OPTIMIZATION[name]
    q, center = x[:, 1 : 1 + d], x[:, 1 + d :]
    assert optimal.min() >= -1e-10
    if mode == "box":
        assert optimal.max() <= 1 + 1e-10
        np.testing.assert_allclose(optimal, np.clip(center, 0, 1), atol=1e-10)
    else:
        np.testing.assert_allclose(optimal.sum(axis=1), 1, atol=1e-10)
        for diagonal, c, solution in zip(q, center, optimal):
            active = solution > 1e-8
            multiplier = np.mean(diagonal[active] * (c[active] - solution[active]))
            np.testing.assert_allclose(
                diagonal[active] * (solution[active] - c[active]) + multiplier, 0, atol=1e-8
            )
            assert np.all(-diagonal[~active] * c[~active] + multiplier >= -1e-8)
