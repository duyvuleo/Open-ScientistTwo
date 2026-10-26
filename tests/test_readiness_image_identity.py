import copy

import pytest

from scientisttwo.testbed.readiness import _require_immutable_images


@pytest.mark.parametrize("prefix", ["", "sha256:"])
def test_full_image_config_ids_pass_without_changing_saved_representation(prefix):
    images = {"execution": prefix + "1a" * 32, "grading": prefix + "2b" * 32}
    before = copy.deepcopy(images)
    _require_immutable_images(images)
    assert images == before


def test_mixed_runtime_inspection_spellings_are_both_immutable():
    _require_immutable_images({"execution": "a" * 64, "grading": "sha256:" + "b" * 64})


@pytest.mark.parametrize("value", [
    "latest", "python:3.12", "repo@sha256:" + "a" * 64,
    "sha256:", "sha256:abc", "a" * 63, "a" * 65,
    "sha256:" + "a" * 63, "sha256:" + "a" * 65,
    "sha256:" + "G" * 64, "a" * 64 + "\n", " " + "a" * 64,
    "a" * 64 + "\n" + "b" * 64, None, 123, ["a" * 64],
])
def test_tags_truncated_hashes_and_ambiguous_values_remain_rejected(value):
    with pytest.raises(ValueError, match="complete immutable"):
        _require_immutable_images({"execution": "sha256:" + "a" * 64, "grading": value})


@pytest.mark.parametrize("images", [{}, None, [], ["a" * 64]])
def test_image_mapping_is_required(images):
    with pytest.raises(ValueError, match="complete immutable"):
        _require_immutable_images(images)
