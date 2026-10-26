import json

import pytest

from scientisttwo.testbed.source_identity import dataset_identity


def test_actual_data_assertion_cannot_certify_dataset_identity(tmp_path):
    result = dataset_identity("cifar10", tmp_path, {}, provenance="actual_data")
    assert result["status"] == "unverified"
    assert result["kind"] == "user_asserted_actual_data"


def test_fake_provider_metadata_cannot_replace_real_official_archive(tmp_path):
    archive = tmp_path / "cifar.tar.gz"
    archive.write_bytes(b"fabricated dataset archive")
    receipt = tmp_path / "source_receipt.json"
    receipt.write_text(json.dumps({"task_id": "cifar10", "archive": {"path": str(archive),
                    "md5": "c32a1d4ab5d03f1284b67883e8d87530", "size": 170052171}}))
    with pytest.raises(ValueError, match="published official"):
        dataset_identity("cifar10", tmp_path, {"original_training_rows": 50000,
                         "original_final_rows": 10000}, provenance="actual_data")


def test_kaggle_verified_boolean_without_acquisition_cannot_certify_data(tmp_path):
    (tmp_path / "source_receipt.json").write_text(json.dumps({"task_id": "house-price", "verified": True}))
    with pytest.raises(ValueError, match="completed supported acquisition"):
        dataset_identity("house-price", tmp_path, {}, provenance="actual_data")


def test_fixture_identity_never_becomes_actual_provider_proof(tmp_path):
    result = dataset_identity("vectorization", tmp_path, {}, provenance="synthetic_fixture")
    assert result["status"] == "synthetic_fixture"
