import json
from pathlib import Path

import numpy as np
import pytest

from scientisttwo.live.source_tree import freeze_source_tree
from scientisttwo.store import Store
from scientisttwo.runtime_profiles import runtime_profile
from scientisttwo.testbed.adapter import PreparedDevelopment
from scientisttwo.testbed.preparation import prepare, verify_prepared
from scientisttwo.testbed.validation import create_validation_spec


def family_bundle(path):
    for phase in ("screen", "full", "final"):
        public, private = path / "phases" / phase / "public", path / "phases" / phase / "private"
        public.mkdir(parents=True)
        private.mkdir(parents=True)
        count = 10 if phase == "screen" else 20
        train_ids = np.asarray([f"train-{i}" for i in range(count)] + ([f"dev-{i}" for i in range(8)] if phase == "final" else []))
        count = len(train_ids)
        evaluation_ids = np.asarray([f"{'final' if phase == 'final' else 'dev'}-{i}" for i in range(8)])
        np.savez(public / "train.npz", ids=train_ids, x=np.zeros((count, 3, 32, 32), dtype="float32"), y=np.arange(count) % 10)
        np.savez(public / "evaluation.npz", ids=evaluation_ids, x=np.zeros((8, 3, 32, 32), dtype="float32"))
        np.savez(private / "labels.npz", ids=evaluation_ids, y=np.arange(8))
    (path / "split_metadata.json").write_text(json.dumps({"dataset_origin": "EXPLICIT SYNTHETIC CONTRACT FIXTURE"}))
    return path


def test_family_preparation_source_tree_pipeline_and_assets(tmp_path):
    source = family_bundle(tmp_path / "source")
    output = tmp_path / "prepared"
    prepare("cifar10", source, output, provenance="synthetic_fixture", source_format="bundle")
    verify_prepared(output)
    config = create_validation_spec(output, image="test:torch", grader_image="test:numpy", level="full", gpus="device=0")
    assert config.native.evaluation_pipeline.inference is not None
    assert config.native.evaluation_pipeline.score.image == "test:numpy"
    assert config.native.evaluation_pipeline.private_dir == output / "phases/final/private"
    minimum_tmp = runtime_profile("torch-cuda").requirements.min_tmp_mb
    assert minimum_tmp == 4096  # Required free capacity remains unchanged.
    assert config.native.resources.tmp_mb >= minimum_tmp + 64
    # Both protected inference and scoring inherit the same capacity headroom.
    for stage in (config.native.evaluation_pipeline.inference, config.native.evaluation_pipeline.score):
        assert stage.resources.tmp_mb == config.native.resources.tmp_mb
        assert stage.resources.tmp_mb - 16 >= minimum_tmp
    adapter = PreparedDevelopment(config.native, config.development)
    descriptor = adapter.source_tree()
    assert set(descriptor["editable_paths"]) == {"model.py", "training_config.json", "transforms.py"}
    store = Store(tmp_path / "snapshot")
    try:
        ref = freeze_source_tree(store, Path(descriptor["path"]), descriptor["editable_paths"])
        adapter.bind_source(ref)
        baseline = adapter.baseline()
        assert baseline.files == [] and baseline.base_snapshot_ref == ref
        adapter.validate(baseline)
        phase = adapter.prepare("subset", tmp_path / "work")
        pipeline = adapter.evaluation(phase)
        assert pipeline.private_dir.is_relative_to(tmp_path / "work")
        assert not pipeline.private_dir.is_relative_to(Path(phase["public"]))
        assert adapter.artifacts(phase)[0].role == "checkpoint"
        assert pipeline.inference.outputs[0].role == "prediction"
        assert "labels.npz" not in json.dumps(adapter.context())
    finally:
        store.close()


def test_family_evaluation_labels_cannot_be_in_public_inputs(tmp_path):
    source = family_bundle(tmp_path / "source")
    path = source / "phases/full/public/evaluation.npz"
    np.savez(path, ids=np.asarray([f"dev-{i}" for i in range(8)]),
             x=np.zeros((8, 3, 32, 32)), y=np.arange(8))
    with pytest.raises(ValueError, match="protected label"):
        prepare("cifar10", source, tmp_path / "prepared", provenance="synthetic_fixture", source_format="bundle")


def test_framework_variant_cannot_hide_missing_assets(tmp_path):
    source = family_bundle(tmp_path / "source")
    with pytest.raises(ValueError, match="tokenizer"):
        prepare("babylm", source, tmp_path / "prepared", provenance="synthetic_fixture", source_format="bundle")
