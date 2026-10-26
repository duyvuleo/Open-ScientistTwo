"""Trusted protocol checks; candidate source is never executed on the host."""

import ast
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scientisttwo.live.adapters import load_development
from scientisttwo.live.models import Candidate, DevelopmentSpec, SourceFile
from scientisttwo.live.workspace import (
    RegisteredDevelopment,
    artifact_paths,
    execution_descriptor,
    validate_adapter,
)


def setup(tmp_path, *, benchmark="mlagentbench", regression=False, settings=None, **kwargs):
    public = tmp_path / "public"
    public.mkdir()
    frame = pd.DataFrame(
        {
            "id": np.arange(80),
            "target": np.arange(80) * 0.5 if regression else [0, 1] * 40,
            "text": ["moon blue" if i % 2 else "sun red" for i in range(80)],
            "value": np.arange(80, dtype=float),
            "category": ["cat", "dog"] * 40,
            "group": np.repeat(np.arange(20), 4),
            "time": pd.date_range("2020-01-01", periods=40).repeat(2).astype(str),
        }
    )
    frame.to_csv(public / "train.csv", index=False)
    pd.DataFrame({"id": [1000], "text": ["PROTECTED FINAL CANARY"]}).to_csv(
        public / "test.csv", index=False
    )
    native = SimpleNamespace(
        public_dir=public,
        benchmark=benchmark,
        task_id="new-task",
        submission="submission.csv",
        candidate_dir=None,
        command=[],
    )
    config = DevelopmentSpec(
        adapter="csv",
        target="target",
        text_column="text",
        prediction_columns=["target"],
        metric="rmse" if regression else "accuracy",
        settings={"submission_columns": ["id", "target"], "id_columns": ["id"], **(settings or {})},
        **kwargs,
    )
    return native, config, load_development(native, config)


@pytest.mark.parametrize("benchmark", ["mlagentbench", "mle-bench", "workspace", "brand-new"])
def test_new_task_needs_only_configuration(tmp_path, benchmark):
    native, config, adapter = setup(tmp_path, benchmark=benchmark)
    prepared = adapter.prepare("full", tmp_path / "development")
    assert len(prepared["labels"]) == 16
    assert "PROTECTED FINAL CANARY" not in str(adapter.context())
    assert set(prepared["train_indices"]).isdisjoint(prepared["validation_indices"])
    assert adapter.prediction_artifacts(prepared) == ["submission.csv"]
    assert execution_descriptor(adapter, prepared, 3)["environment"]["SCIENTISTTWO_SEED"] == "3"


def test_mixed_tabular_baseline_fits_only_training_frame(tmp_path):
    _, _, adapter = setup(
        tmp_path,
        regression=True,
        settings={
            "features": {"text": ["text"], "numeric": ["value"], "categorical": ["category"]}
        },
    )
    source = adapter.baseline().files[0].content
    tree = ast.parse(source)
    fits = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "fit"
    ]
    assert len(fits) == 1
    assert ast.unparse(fits[0]) == "model.fit(train, train['target'])"
    assert 'SimpleImputer(strategy="median"' in source
    assert 'OneHotEncoder(handle_unknown="ignore")' in source
    assert "TfidfVectorizer()" in source
    assert "model.predict(test)" in source


@pytest.mark.parametrize("kind,column", [("group", "group"), ("time", "time")])
def test_declared_split_preserves_group_and_temporal_boundaries(tmp_path, kind, column):
    _, _, adapter = setup(tmp_path, settings={"split": {"kind": kind, "column": column}})
    full, subset, valid = (
        adapter.frame.iloc[rows] for rows in (adapter.train, adapter.subset, adapter.valid)
    )
    assert set(subset.id) <= set(full.id)
    if kind == "group":
        assert set(full.group).isdisjoint(valid.group)
        assert set(subset.group).isdisjoint(valid.group)
    else:
        assert pd.to_datetime(full.time).max() < pd.to_datetime(valid.time).min()
        assert pd.to_datetime(subset.time).max() < pd.to_datetime(valid.time).min()


def test_scoring_recalculates_multiple_metrics_and_checks_identifiers(tmp_path):
    from scientisttwo.live.models import MetricSpec

    _, _, adapter = setup(
        tmp_path,
        regression=True,
        metric_specs={"mae": MetricSpec(direction="minimize", max_degradation=0.1)},
        settings={"features": {"numeric": ["value"]}},
    )
    prepared = adapter.prepare("full", tmp_path / "development")
    work = tmp_path / "work"
    work.mkdir()
    pd.DataFrame(
        {"id": prepared["identifiers"]["id"], "target": np.asarray(prepared["labels"]) + 2}
    ).to_csv(work / "submission.csv", index=False)
    assert adapter.score("full", prepared, work, 500) == {"rmse": 2.0, "mae": 2.0}
    fake = pd.read_csv(work / "submission.csv").iloc[::-1]
    fake.to_csv(work / "submission.csv", index=False)
    with pytest.raises(ValueError, match="identifiers"):
        adapter.score("full", prepared, work, 500)


@pytest.mark.parametrize("mutation", ["missing", "extra", "nan"])
def test_prediction_contract_rejects_invalid_outputs(tmp_path, mutation):
    _, _, adapter = setup(tmp_path)
    prepared = adapter.prepare("full", tmp_path / "development")
    frame = pd.DataFrame({"id": prepared["identifiers"]["id"], "target": prepared["labels"]})
    if mutation == "missing":
        frame = frame.drop(columns=["id"])
    elif mutation == "extra":
        frame["score"] = 1
    else:
        frame.loc[0, "target"] = np.nan
    frame.to_csv(tmp_path / "submission.csv", index=False)
    with pytest.raises(ValueError):
        adapter.score("full", prepared, tmp_path, 1)


@pytest.mark.parametrize(
    "settings",
    [
        {"features": {"numeric": ["target"]}},
        {"features": {"numeric": ["id"]}},
        {"features": {"numeric": ["value"], "text": ["value"]}},
        {"split": {"kind": "time", "column": "target"}},
        {"prediction_file": "../submission.csv"},
    ],
)
def test_bad_declarations_fail_before_execution(tmp_path, settings):
    with pytest.raises(ValueError):
        setup(tmp_path, settings=settings)


class Plugin:
    def identity(self):
        return {"fixture": "trusted synthetic protocol"}

    def context(self):
        return {"contract": "Synthetic plugin, no executions"}

    def baseline(self):
        return Candidate(
            files=[SourceFile(path="main.py", content="print('baseline')")],
            command=["python", "main.py"],
            description="Synthetic",
        )

    def prepare(self, phase, destination):
        return {}

    def validate(self, candidate):
        pass

    def score(self, phase, prepared, workspace, seconds):
        return 0.0


def test_plugin_provenance_and_legacy_generic_transport(tmp_path):
    entry = SimpleNamespace(
        name="custom",
        value="installed_plugin:Factory",
        dist=SimpleNamespace(metadata={"Name": "installed-plugin"}, version="1.2.3"),
    )
    plugin = RegisteredDevelopment(Plugin(), entry)
    assert plugin.identity()["plugin_version"] == "1.2.3"
    assert len(plugin.identity()["plugin_source"]) == 64
    assert execution_descriptor(plugin, {}, 7) == {
        "kind": "command",
        "parameters": {},
        "environment": {"SCIENTISTTWO_SEED": "7"},
    }
    assert artifact_paths(plugin, {}) == ["submission.csv"]
    with pytest.raises(ValueError, match="missing methods"):
        validate_adapter(SimpleNamespace())


@pytest.mark.parametrize("paths", [["../outside"], [".env"], ["a", "a"], [], [12]])
def test_plugin_prediction_paths_are_confined(paths):
    plugin = SimpleNamespace(prediction_artifacts=lambda prepared: paths)
    with pytest.raises(ValueError):
        artifact_paths(plugin, {})


def test_trusted_driver_descriptor_is_data_not_host_execution(tmp_path):
    marker = tmp_path / "must-not-exist"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    plugin = SimpleNamespace(
        execution=lambda prepared, seed: {
            "kind": "python_driver",
            "source": source,
            "parameters": {"shape": [3, 4]},
        }
    )
    assert execution_descriptor(plugin, {}, 1)["source"] == source
    assert not marker.exists()
    plugin.execution = lambda prepared, seed: {"kind": "command", "environment": {"API_KEY": "x"}}
    with pytest.raises(ValueError, match="environment"):
        execution_descriptor(plugin, {}, 1)


def test_workspace_explicit_baseline_is_read_not_imported(tmp_path):
    native, config, _ = setup(tmp_path, benchmark="workspace")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    marker = tmp_path / "must-not-exist"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    (baseline / "main.py").write_text(source)
    native.candidate_dir, native.command = baseline, ["python", "main.py"]
    adapter = load_development(native, config)
    assert adapter.baseline().files[0].content == source
    assert "baseline_source/main.py" in adapter.identity()
    assert not marker.exists()
