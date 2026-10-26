"""CPU-only study contract tests; these do not establish CUDA or CIFAR readiness."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scientisttwo.benchmarks.files import tree_manifest
from scientisttwo.live.models import Candidate, SourceFile
from scientisttwo.studies.cifar_baseline.training_contract import load_configuration, validate_overrides
from scientisttwo.studies.cifar_recipe import (
    BASELINE_CONFIGURATION, FIXED_CONFIGURATION, VARIANT, baseline_candidate, calibration_plan,
    recipe_candidate, study_recipe, validate_candidate, validate_materialized_workspace,
    validate_training_receipt, verify_study_sources, write_study_sources,
)
from scientisttwo.testbed.catalog import get_recipe

SNAPSHOT = "artifacts/" + "a" * 64 + ".json"


def test_study_is_distinct_and_canonical_lenet_is_unchanged(tmp_path):
    old = get_recipe("cifar10").model_dump()
    identity = write_study_sources(tmp_path / "study")
    assert get_recipe("cifar10").model_dump() == old
    assert study_recipe().identity != get_recipe("cifar10").identity
    assert study_recipe().recipe_version == VARIANT
    assert identity["editable_files"] == ["training_config.json"]
    assert identity["expected_steps"] == {"screen": 1570, "full": 3130, "final": 3910}
    assert identity["training_seed_policy"]["final"] == [0, 1, 2]
    assert identity["data_validation"] == identity["runtime_validation"] == "unverified"
    assert identity["calibration"] == "unfulfilled"
    assert identity == verify_study_sources(tmp_path / "study")
    assert not (tmp_path / "study/phases").exists()
    with pytest.raises(ValueError, match="must be new"):
        write_study_sources(tmp_path / "study")


def test_source_bundle_pins_the_complete_worker_and_evaluator_closure(tmp_path):
    identity = write_study_sources(tmp_path / "study")
    root = tmp_path / "study"
    descriptor = json.loads((root / "family-pipeline.json").read_text())
    assert descriptor["editable_files"] == ["training_config.json"]
    assert "family_worker.py" in identity["reference_files"]
    assert "array_data.py" in identity["reference_files"]
    assert descriptor["evaluator_identity"] == {
        name: value["sha256"] for name, value in tree_manifest(root / "evaluator").items()
    }
    for source in root.rglob("*.py"):
        compile(source.read_text(), str(source), "exec")
    (root / "evaluator/score/array_data.py").write_text("# modified imported helper\n")
    with pytest.raises(ValueError, match="source bundle changed"):
        verify_study_sources(root)


def test_materialized_workspace_preserves_architecture_and_all_noneditable_files(tmp_path):
    identity = write_study_sources(tmp_path / "study")
    workspace = tmp_path / "workspace"
    shutil.copytree(tmp_path / "study/reference", workspace)
    result = validate_materialized_workspace(workspace, identity)
    assert result["model_sha256"] == identity["fixed_model_sha256"]
    (workspace / "training_config.json").write_text('{"learning_rate": 0.03}')
    assert validate_materialized_workspace(workspace, identity)["overrides"] == {"learning_rate": 0.03}
    (workspace / "model.py").write_text("raise RuntimeError('Must not execute')\n")
    with pytest.raises(ValueError, match="architecture or execution code changed"):
        validate_materialized_workspace(workspace, identity)


@pytest.mark.parametrize("bad", [
    {"epochs": 11}, {"epochs": 9}, {"epochs": 10}, {"batch_size": 64}, {"hidden_size": 64},
    {"scheduler": "cosine"}, {"pretrained": True}, {"learning_rate": float("nan")},
    {"learning_rate": True}, {"weight_decay": 0.2}, {"optimizer": []}, {"optimizer": "other"},
    {"crop_padding": 1.5}, {"crop_padding": 5}, {"crop_padding": True},
    {"horizontal_flip_probability": -0.1}, {"horizontal_flip_probability": 1.01}, [],
])
def test_invalid_or_protected_candidate_parameters_fail_before_execution(bad):
    with pytest.raises(ValueError):
        validate_overrides(bad)


def test_seed_is_injected_by_controller_and_budget_is_fixed(tmp_path, monkeypatch):
    write_study_sources(tmp_path / "study")
    root = tmp_path / "study/reference"
    monkeypatch.setenv("SCIENTISTTWO_SEED", "2")
    (root / "training_config.json").write_text('{"learning_rate":0.03,"crop_padding":0}')
    config = load_configuration(root / "training.json", root, "cifar10")
    assert config["seed"] == 2
    assert config["epochs"] == 10 and config["batch_size"] == 128
    assert config["learning_rate"] == 0.03 and config["crop_padding"] == 0
    assert config["momentum"] == BASELINE_CONFIGURATION["momentum"]
    config["epochs"] = 9
    (root / "training.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="budget changed"):
        load_configuration(root / "training.json", root, "cifar10")


def test_candidates_only_allow_configuration_on_the_bound_snapshot():
    validate_candidate(baseline_candidate(SNAPSHOT), snapshot_ref=SNAPSHOT)
    candidate = recipe_candidate(SNAPSHOT, {"weight_decay": 0.001}, description="One declared change")
    validate_candidate(candidate, snapshot_ref=SNAPSHOT)
    candidate.command += ["--config", "other.json"]
    with pytest.raises(ValueError, match="command"):
        validate_candidate(candidate, snapshot_ref=SNAPSHOT)
    candidate = Candidate(files=[SourceFile(path="transforms.py", content="pass\n")],
                          base_snapshot_ref=SNAPSHOT, command=["python", "train.py"], description="Bypass attempt")
    with pytest.raises(ValueError, match="Only training_config"):
        validate_candidate(candidate, snapshot_ref=SNAPSHOT)


def test_reference_architecture_factory_explicitly_disables_pretrained_and_uses_cifar_stem(monkeypatch):
    # Exercise the actual model factory against a minimal API double, without
    # importing or installing Torch on the controller host.
    calls = {}
    def resnet18(**kwargs):
        calls["resnet18"] = kwargs
        return SimpleNamespace()
    def conv(*args, **kwargs):
        calls["conv1"] = (args, kwargs)
        return "cifar-stem"
    torch = ModuleType("torch")
    torch.nn = SimpleNamespace(Conv2d=conv, Identity=lambda: "identity")
    vision = ModuleType("torchvision")
    models = ModuleType("torchvision.models")
    models.resnet18 = resnet18
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchvision", vision)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)
    from scientisttwo.studies.cifar_baseline.model import build_model
    model = build_model("cifar10", {}, Path("/unused-assets"))
    assert calls["resnet18"] == {"weights": None, "num_classes": 10}
    assert calls["conv1"] == ((3, 64), {"kernel_size": 3, "stride": 1, "padding": 1, "bias": False})
    assert model.maxpool == "identity" and model.conv1 == "cifar-stem"


def test_inference_transform_does_not_import_framework_or_change_inputs(tmp_path, monkeypatch):
    write_study_sources(tmp_path / "study")
    path = tmp_path / "study/reference/transforms.py"
    spec = importlib.util.spec_from_file_location("study_transform_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import numpy as np
    inputs = np.zeros((2, 3, 32, 32), dtype=np.float32)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert module.transform_batch("cifar10", inputs, training=False, config={}) is inputs


def test_training_transform_crops_flips_and_preserves_each_image(monkeypatch):
    # The array double checks indexing and row preservation without claiming a
    # Torch runtime smoke test. Real CUDA execution remains a separate gate.
    import numpy as np
    from scientisttwo.studies.cifar_baseline.transforms import transform_batch

    class Tensor(np.ndarray):
        @property
        def device(self):
            return "cpu"

        def flip(self, axis):
            return np.flip(self, axis).view(Tensor)

    def tensor(value):
        return np.asarray(value).view(Tensor)

    rng = np.random.default_rng(7)
    torch = ModuleType("torch")
    torch.randint = lambda low, high, size, device: tensor(rng.integers(low, high, size))
    torch.arange = lambda count, device: tensor(np.arange(count))
    torch.rand = lambda count, device: tensor(rng.random(count))
    torch.where = lambda condition, yes, no: tensor(np.where(condition, yes, no))
    functional = ModuleType("torch.nn.functional")
    functional.pad = lambda values, padding, value: tensor(np.pad(
        values, ((0, 0), (0, 0), (padding[2], padding[3]), (padding[0], padding[1])), constant_values=value))
    neural = ModuleType("torch.nn")
    neural.functional = functional
    torch.nn = neural
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", neural)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functional)
    inputs = tensor(np.stack([np.full((3, 32, 32), 0.25), np.full((3, 32, 32), 0.75)]))
    result = transform_batch("cifar10", inputs, training=True,
                             config={"crop_padding": 4, "horizontal_flip_probability": 1.0})
    assert result.shape == inputs.shape
    for index, value in enumerate((0.25, 0.75)):
        assert set(np.unique(result[index])) == {-1.0, value}
    ramp = tensor(np.broadcast_to(np.arange(32), (2, 3, 32, 32)))
    flipped = transform_batch("cifar10", ramp, training=True,
                              config={"crop_padding": 0, "horizontal_flip_probability": 1.0})
    np.testing.assert_array_equal(flipped, ramp[..., ::-1])


@pytest.mark.parametrize("phase,steps", [("screen", 1570), ("full", 3130), ("final", 3910)])
def test_receipt_requires_all_ten_epochs_with_matching_seed(phase, steps):
    config = {**FIXED_CONFIGURATION, "seed": 2}
    receipt = {"task": "cifar10", "seed": 2, "steps": steps, "epochs_started": 10, "max_steps": 3910}
    assert validate_training_receipt(receipt, config, phase=phase)["status"] == "pass"
    for change in ({"steps": steps - 1}, {"steps": steps + 1}, {"epochs_started": 9}, {"seed": 1}):
        with pytest.raises(ValueError):
            validate_training_receipt({**receipt, **change}, config, phase=phase)


def test_calibration_is_bounded_development_only_without_claiming_a_winner():
    plan = calibration_plan()
    assert plan["max_training_runs"] == 2 and plan["phase"] == "full"
    assert plan["final_access"] is False and plan["live_model_calls"] == 0
    assert plan["initial"]["learning_rate"] == 0.1
    assert plan["optional_sanity_check"]["learning_rate"] == 0.03
    assert {key for key in plan["initial"] if plan["initial"][key] != plan["optional_sanity_check"][key]} == {"learning_rate"}
    assert "0.65" in plan["optional_trigger"]
