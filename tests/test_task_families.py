import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from scientisttwo.task_families import family_task_ids, get_family, template_sources


def scorer():
    root = Path(__file__).parents[1] / "src/scientisttwo/task_families/templates/score.py"
    spec = importlib.util.spec_from_file_location("family_scorer_test", root)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_framework_task_has_training_inference_scoring_sources():
    assert set(family_task_ids()) == {"cifar10", "imdb", "ogbn-arxiv", "identify-contrails", "fathomnet", "CLRS", "babylm", "llama-inference"}
    for task in family_task_ids():
        family = get_family(task)
        sources = template_sources(task)
        for name, text in sources.items():
            if name.endswith(".py"):
                compile(text, name, "exec")
        assert "/data/private/labels.npz" not in family.command("infer")
        assert "/candidate/model.py" not in family.command("score")
        assert family.artifacts()


def test_id_alignment_and_invalid_predictions():
    compute = scorer().compute
    labels = {"ids": np.array(["a", "b"]), "y": np.array([0, 1])}
    prediction = {"ids": np.array(["b", "a"]), "predictions": np.array([[0, 1], [1, 0]])}
    assert compute("cifar10", prediction, labels) == {"accuracy": 1.0}
    prediction["ids"] = np.array(["a", "a"])
    with pytest.raises(ValueError, match="unique"):
        compute("cifar10", prediction, labels)
    prediction["ids"] = np.array(["a", "b"])
    prediction["predictions"] = np.array([[np.nan, 1], [1, 0]])
    with pytest.raises(ValueError, match="Non-finite"):
        compute("cifar10", prediction, labels)


def test_true_category_map20_and_binary_dice():
    compute = scorer().compute
    labels = {"ids": np.array(["a"]), "y": np.array([[1, 0, 1]])}
    prediction = {"ids": labels["ids"], "predictions": np.array([[0.9, 0.8, 0.7]])}
    assert compute("fathomnet", prediction, labels)["map_at_20"] == pytest.approx((1 + 2 / 3) / 2)
    masks = {"ids": np.array(["a"]), "y": np.array([[[1, 0], [1, 0]]])}
    prediction = {"ids": masks["ids"], "predictions": np.array([[[0.9, 0.2], [0.1, 0.2]]])}
    assert compute("identify-contrails", prediction, masks)["dice"] == pytest.approx(2 / 3)
    labels = {"ids": np.array(["large", "small"]), "y": np.array([[[1, 1], [1, 1]], [[1, 0], [0, 0]]])}
    prediction = {"ids": labels["ids"], "predictions": np.array([[[[1, 1], [1, 1]]], [[[0, 0], [0, 0]]]])}
    expected = (1 + 1e-6 / (1 + 1e-6)) / 2
    assert compute("identify-contrails", prediction, labels)["dice"] == pytest.approx(expected)
    assert abs(expected - 8 / 9) > 0.3  # A global pixel Dice would produce a different metric.


def test_perplexity_uses_protected_targets_and_rejects_missing_batches(tmp_path):
    module = scorer()
    labels = {"ids": np.array(["doc-1"]), "target_ids": np.array([[0, 1]]), "attention_mask": np.array([[1, 1]])}
    np.savez(tmp_path / "logits-000000.npz", ids=labels["ids"], logits=np.zeros((1, 2, 3)))
    assert module.perplexity(tmp_path, labels)["perplexity"] == pytest.approx(3)
    labels["ids"] = np.array(["unknown"])
    with pytest.raises(ValueError, match="Unexpected"):
        module.perplexity(tmp_path, labels)


def _image_bundle(root):
    for phase in ["screen", "full", "final"]:
        public = root / "phases" / phase / "public"
        private = root / "phases" / phase / "private"
        public.mkdir(parents=True)
        private.mkdir(parents=True)
        ids = ["train-a"] if phase == "screen" else ["train-a", "train-b"]
        if phase == "final":
            ids += ["dev"]
        np.savez(public / "train.npz", ids=np.array(ids), x=np.zeros((len(ids), 3, 32, 32)), y=np.zeros(len(ids)))
        eval_ids = np.array(["final" if phase == "final" else "dev"])
        np.savez(public / "evaluation.npz", ids=eval_ids, x=np.zeros((1, 3, 32, 32)))
        np.savez(private / "labels.npz", ids=eval_ids, y=np.zeros(1))


def test_prepared_labels_never_enter_public_evaluation(tmp_path):
    _image_bundle(tmp_path)
    family = get_family("cifar10")
    checked = family.validate_prepared(tmp_path, {"dataset_origin": "explicit synthetic contract fixture"})
    assert checked["scope"] == "prepared_input_contract_only"
    assert checked["actual_task_execution"] == "unverified"
    public = tmp_path / "phases/final/public/evaluation.npz"
    np.savez(public, ids=np.array(["final"]), x=np.zeros((1, 3, 32, 32)), y=np.zeros(1))
    with pytest.raises(ValueError, match="protected label"):
        family.validate_prepared(tmp_path, {"dataset_origin": "fixture"})


def test_source_emission_and_frozen_stage_boundary(tmp_path):
    family = get_family("cifar10")
    receipt = family.write_sources(tmp_path)
    assert receipt["editable_files"] == ["model.py", "training_config.json", "transforms.py"]
    assert (tmp_path / "reference/model.py").read_bytes() == (tmp_path / "control/model.py").read_bytes()
    assert (tmp_path / "evaluator/score/evaluate.py").is_file()
    assert "candidate_model" not in (tmp_path / "evaluator/score/evaluate.py").read_text()
    assert json.loads((tmp_path / "reference/training.json").read_text())["max_epochs"] == 10
    assert receipt["runtime_validation"] == "unverified"


def test_babylm_inference_does_not_load_candidate_forward():
    source = template_sources("babylm")["family_worker.py"]
    assert "type(model) is not GPT2LMHeadModel" in source
    assert "GPT2LMHeadModel(GPT2Config.from_dict(architecture))" in source
    assert "weights_only=True" in source


def test_llama_timing_not_taken_from_predictions():
    labels = {"ids": np.array(["a"]), "y": np.array([[123]])}
    predictions = {"ids": labels["ids"], "predictions": labels["y"], "time": 0.00001}
    result = scorer().compute("llama-inference", predictions, labels)
    assert result == {"output_equivalence": 1.0}


def test_editable_hyperparameters_cannot_override_caps_or_seed(tmp_path, monkeypatch):
    get_family("cifar10").write_sources(tmp_path)
    directory = tmp_path / "reference"
    spec = importlib.util.spec_from_file_location("training_contract_test", directory / "training_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("SCIENTISTTWO_SEED", "2")
    (directory / "training_config.json").write_text(json.dumps({"optimizer": "adamw", "learning_rate": 0.003, "epochs": 10}))
    config = module.load_configuration(directory / "training.json", directory, "cifar10")
    assert config["seed"] == 2 and config["epochs"] == 10 and config["optimizer"] == "adamw"
    for invalid in [{"epochs": 11}, {"max_epochs": 1000}, {"seed": 999}, {"learning_rate": float("nan")}]:
        (directory / "training_config.json").write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            module.load_configuration(directory / "training.json", directory, "cifar10")


def test_memory_mapped_arrays_and_bounded_validation(tmp_path):
    from scientisttwo.task_families.array_data import all_finite, open_arrays
    directory = tmp_path / "train"
    directory.mkdir()
    np.save(directory / "ids.npy", np.array(["a", "b"]))
    np.save(directory / "x.npy", np.zeros((2, 3, 32, 32), dtype=np.float32))
    with open_arrays(directory) as archive:
        assert isinstance(archive["x"], np.memmap)
        assert all_finite(archive["x"])

    class GuardedArray:
        def __len__(self):
            return 1000

        def __getitem__(self, key):
            assert isinstance(key, slice) and key.stop - key.start <= 32
            return np.zeros((min(key.stop, 1000) - key.start, 2))

    assert all_finite(GuardedArray())


def test_large_packed_arrays_rejected_before_numpy_load(tmp_path, monkeypatch):
    from scientisttwo.task_families import array_data
    from types import SimpleNamespace

    class FakeZip:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def infolist(self):
            return [SimpleNamespace(file_size=1024**3)]

    monkeypatch.setattr(array_data.zipfile, "ZipFile", lambda path: FakeZip())
    monkeypatch.setattr(array_data.np, "load", lambda *a, **kw: pytest.fail("Oversized packed data reached numpy"))
    with pytest.raises(ValueError, match="memory-mapped"):
        with array_data.open_arrays(tmp_path / "large.npz"):
            pass


def test_prediction_shards_score_all_rows_without_loading_full_targets(tmp_path):
    ids = np.array([str(index) for index in range(10)])

    class GuardedTargets:
        def __getitem__(self, indices):
            assert isinstance(indices, list) and len(indices) <= 2
            return np.array([index % 2 for index in indices])

    for offset in range(0, 10, 2):
        np.savez(tmp_path / f"predictions-{offset}.npz", ids=ids[offset:offset + 2], predictions=np.array([[1, 0], [0, 1]]))
    assert scorer().score_shards("cifar10", tmp_path, {"ids": ids, "y": GuardedTargets()}) == {"accuracy": 1.0}


def test_clrs_initial_state_and_ood_arrays_follow_locked_floyd_protocol():
    from scientisttwo.task_families.array_data import ArrayDirectory
    from scientisttwo.task_families.clrs_inputs import inference_hints, validate_trajectory
    weights = np.array([[[0, 2, 0], [0, 0, 3], [1, 0, 0]]], dtype=float)
    adjacency = (weights != 0).astype(float)
    adjacency[:, np.arange(3), np.arange(3)] = 1
    hints = inference_hints(weights, adjacency)
    assert np.array_equal(hints["Pi_h"][0], [[[0, 0, 0], [1, 1, 1], [2, 2, 2]]])
    assert np.array_equal(hints["D"][0], weights)
    assert np.array_equal(hints["msk"][0], adjacency)
    assert np.array_equal(hints["k"][0], [[1, 0, 0]])
    assert all(not value[1:].any() for value in hints.values())
    arrays = ArrayDirectory(ids=np.array(["graph"]), lengths=np.array([3]),
                            **{"input:A": weights, "input:adj": adjacency, "input:pos": np.zeros((1, 3))})
    validate_trajectory(arrays, nodes=3, supervised=False)
    with pytest.raises(ValueError, match="graph sizes"):
        validate_trajectory(arrays, nodes=64, supervised=False)


def test_controls_are_actual_bounded_changes(tmp_path):
    family = get_family("cifar10")
    family.write_sources(tmp_path)
    assert (tmp_path / "reference/training_config.json").read_bytes() != (tmp_path / "control/training_config.json").read_bytes()
    family = get_family("llama-inference")
    family.write_sources(tmp_path / "llama")
    assert (tmp_path / "llama/reference/optimization_config.json").read_bytes() != (tmp_path / "llama/control/optimization_config.json").read_bytes()


def test_prediction_quota_can_be_configured_without_truncating_data():
    family = get_family("babylm", settings={"prediction_max_bytes": 2 * 1024**4})
    assert family.artifacts("infer")[0].max_bytes == 2 * 1024**4
    with pytest.raises(ValueError):
        get_family("babylm", settings={"truncate_test": True})


def test_packaged_generators_serialize_features_without_protected_labels(tmp_path):
    from types import SimpleNamespace
    source = Path(__file__).parents[1] / "src/scientisttwo/task_families/templates/prepare_clrs.py"
    specification = importlib.util.spec_from_file_location("clrs_generator_test", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    feedback = SimpleNamespace(features=SimpleNamespace(lengths=np.array([4, 4]),
        inputs=[SimpleNamespace(name="A", data=np.zeros((2, 4, 4)))],
        hints=[SimpleNamespace(name="Pi_h", data=np.ones((4, 2, 4, 4)))]),
        outputs=[SimpleNamespace(name="Pi", data=np.ones((2, 4, 4)))])
    public = module._archive(feedback, 0, 2, ["a", "b"], labels=False)
    assert set(public) == {"ids", "lengths", "input:A"}
    train = module._archive(feedback, 0, 1, ["a", "b"])
    assert train["hint:Pi_h"].shape == (4, 1, 4, 4)
    assert train["output:Pi"].shape == (1, 4, 4)


def test_llama_generator_requires_complete_fixed_requests(tmp_path):
    source = Path(__file__).parents[1] / "src/scientisttwo/task_families/templates/prepare_llama.py"
    specification = importlib.util.spec_from_file_location("llama_generator_test", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    text = tmp_path / "wiki.test.tokens"
    text.write_text("short\nlong\nlong\n")

    def tokenizer(value, **kwargs):
        count = 1 if value.strip() == "short" else 128
        return {"input_ids": [1] * count, "attention_mask": [1] * count}

    ids, inputs, _ = module.select_requests(text, tokenizer, "final", count=2)
    assert ids == ["wikitext-final-1", "wikitext-final-2"] and len(inputs[0]) == 128
    with pytest.raises(ValueError, match="silent reduced"):
        module.select_requests(text, tokenizer, "final")
