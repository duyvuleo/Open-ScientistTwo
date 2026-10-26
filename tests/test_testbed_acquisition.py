import json

import numpy as np
import pandas as pd
import pytest

from scientisttwo.testbed.acquisition import convert_raw
from scientisttwo.testbed.preparation import prepare, verify_prepared


def test_cifar_raw_binary_conversion_protects_original_final_labels(tmp_path):
    source = tmp_path / "raw"
    batches = source / "cifar-10-batches-bin"
    batches.mkdir(parents=True)
    rows = np.zeros((20, 3073), dtype=np.uint8)
    rows[:, 0] = np.arange(20) % 10
    rows[:, 1:] = 127
    for number in range(1, 6):
        rows.tofile(batches / f"data_batch_{number}.bin")
    rows.tofile(batches / "test_batch.bin")
    output = tmp_path / "prepared"
    prepare("cifar10", source, output, provenance="synthetic_fixture", source_format="raw")
    manifest = verify_prepared(output)
    assert manifest["split_metadata"]["raw_format"] == "official-cifar10-binary"
    assert manifest["split_metadata"]["original_final_rows"] == 20
    public = output / "phases/final/public/evaluation"
    assert {path.stem for path in public.glob("*.npy")} == {"ids", "x"}
    assert all(name.startswith("cifar-test-") for name in np.load(public / "ids.npy"))
    assert np.load(public / "x.npy", mmap_mode="r").shape == (20, 3, 32, 32)
    assert len(np.load(output / "phases/full/public/train/ids.npy")) == 80
    assert len(np.load(output / "phases/final/public/train/ids.npy")) == 100


def test_pickle_cifar_never_loaded_on_host(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "data_batch_1").write_bytes(b"untrusted-pickle")
    with pytest.raises(ValueError, match="pickle"):
        convert_raw("cifar10", raw, tmp_path / "converted")


def test_imdb_official_text_layout_to_private_label_bundle(tmp_path):
    source = tmp_path / "raw"
    for split, count in (("train", 50), ("test", 10)):
        for label in ("neg", "pos"):
            root = source / "aclImdb" / split / label
            root.mkdir(parents=True)
            for index in range(count):
                (root / f"{index}_5.txt").write_text(f"Fixture {split} {label} review {index}")
    assets = source / "assets/distilbert"
    assets.mkdir(parents=True)
    (assets / "config.json").write_text('{"fixture": true}')
    output = tmp_path / "prepared"
    prepare("imdb", source, output, provenance="synthetic_fixture", source_format="raw")
    verify_prepared(output)
    rows = [json.loads(line) for line in (output / "phases/final/public/evaluation.jsonl").read_text().splitlines()]
    assert len(rows) == 20 and all(set(row) == {"id", "text"} for row in rows)
    assert (output / "assets/distilbert/config.json").is_file()


def test_ogb_official_masks_expose_features_but_only_training_labels(tmp_path):
    source = tmp_path / "raw"
    root = source / "ogbn_arxiv"
    (root / "raw").mkdir(parents=True)
    (root / "split/time").mkdir(parents=True)
    values = {
        "raw/node-feat.csv.gz": np.arange(400).reshape(100, 4),
        "raw/node-label.csv.gz": np.arange(100) % 4,
        "raw/edge.csv.gz": np.stack([np.arange(100), (np.arange(100) + 1) % 100], axis=1),
        "split/time/train.csv.gz": np.arange(60), "split/time/valid.csv.gz": np.arange(60, 80),
        "split/time/test.csv.gz": np.arange(80, 100),
    }
    for name, value in values.items():
        pd.DataFrame(value).to_csv(root / name, index=False, header=False)
    output = tmp_path / "prepared"
    prepare("ogbn-arxiv", source, output, provenance="synthetic_fixture", source_format="raw")
    verify_prepared(output)
    with np.load(output / "phases/full/public/train.npz") as public:
        assert len(public["x"]) == 100
        assert len(public["y"]) == len(public["train_indices"]) == 60
        assert not set(public["train_indices"]) & set(range(80, 100))
    with np.load(output / "phases/final/private/labels.npz") as private:
        assert private["ids"].tolist() == list(map(str, range(80, 100)))


def test_babylm_raw_document_split_and_fixed_tokenization(tmp_path):
    tokenizers = pytest.importorskip("tokenizers", reason="Optional preparation dependency")
    source = tmp_path / "raw"
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    (source / "assets/tokenizer").mkdir(parents=True)
    (source / "assets/model_config").mkdir()
    tokenizer.save(str(source / "assets/tokenizer/tokenizer.json"))
    (source / "assets/model_config/config.json").write_text(json.dumps({"model_type": "gpt2", "eos_token_id": 1, "vocab_size": 4}))
    for directory, split in (("babylm_10M", "train"), ("babylm_dev", "dev"), ("babylm_test", "test")):
        root = source / "babylm_data" / directory
        root.mkdir(parents=True)
        (root / f"fixture.{split}").write_text("hello world hello\nworld hello world\nhello hello world\n")
    output = tmp_path / "prepared"
    prepare("babylm", source, output, provenance="synthetic_fixture", source_format="raw")
    manifest = verify_prepared(output)
    assert manifest["split_metadata"]["tokenization"]["truncation"] is False
    assert manifest["split_metadata"]["document_split_manifest"]["test"]["count"] == 3
    with np.load(output / "phases/final/private/labels.npz") as labels:
        assert labels["attention_mask"].sum() == 6
        np.testing.assert_array_equal(labels["target_ids"][0, :2], [3, 2])
    with np.load(output / "phases/final/public/evaluation.npz") as inputs:
        assert "target_ids" not in inputs.files


def test_contrails_raw_converter_streams_record_arrays_and_protects_masks(tmp_path):
    source = tmp_path / "raw"
    for split, count in (("train", 20), ("validation", 5)):
        for index in range(count):
            record = source / split / f"{split}-{index}"
            record.mkdir(parents=True)
            for band in (11, 14, 15):
                np.save(record / f"band_{band}.npy", np.full((8, 8, 5), 250 + band, dtype=np.float32))
            np.save(record / "human_pixel_masks.npy", np.ones((8, 8, 1), dtype=np.uint8))
    output = tmp_path / "converted"
    metadata = convert_raw("identify-contrails", source, output)
    assert metadata["frame"] == 4
    inputs = output / "phases/final/public/evaluation"
    assert not (inputs / "y.npy").exists()
    assert np.load(inputs / "x.npy", mmap_mode="r").shape == (5, 3, 8, 8)
    labels = np.load(output / "phases/final/private/labels/y.npy", mmap_mode="r")
    assert labels.shape == (5, 1, 8, 8)
    prepared = tmp_path / "prepared"
    prepare("identify-contrails", output, prepared, provenance="synthetic_fixture", source_format="bundle")
    verify_prepared(prepared)


def test_fathomnet_raw_uses_actual_categories_and_never_synthesizes_osd(tmp_path):
    image_module = pytest.importorskip("PIL.Image", reason="Optional preparation dependency")
    source = tmp_path / "raw"
    (source / "multilabel_classification").mkdir(parents=True)
    (source / "images").mkdir()
    pd.DataFrame({"name": ["one", "two", "three"]}).to_csv(source / "category_key.csv", index=False)
    pd.DataFrame({"id": [f"image-{i}" for i in range(30)], "categories": ["[1, 3]"] * 30}).to_csv(
        source / "multilabel_classification/train.csv", index=False)
    for index in range(30):
        image_module.new("RGB", (4, 4), color=(index, 0, 0)).save(source / "images" / f"image-{index}.png")
    output = tmp_path / "converted"
    metadata = convert_raw("fathomnet", source, output)
    assert metadata["osd"].startswith("unavailable")
    labels = np.load(output / "phases/final/private/labels/y.npy", mmap_mode="r")
    assert labels.shape == (1, 3)
    np.testing.assert_array_equal(labels, [[1, 0, 1]])
    prepared = tmp_path / "prepared"
    prepare("fathomnet", output, prepared, provenance="synthetic_fixture", source_format="bundle")
    verify_prepared(prepared)
