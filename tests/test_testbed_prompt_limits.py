"""Synthetic preparation verifies frozen prompt contracts without model or runtime calls."""
import json

import pytest

from scientisttwo.testbed import validation
from scientisttwo.testbed.preparation import prepare
from test_testbed_preparation import raw_data


@pytest.fixture
def prepared(tmp_path):
    source = raw_data(tmp_path)
    root = tmp_path / "prepared"
    prepare("house-price", source, root, provenance="synthetic_fixture")
    return root


def test_validation_prompt_default_and_explicit_limit_are_frozen(prepared):
    default = validation.create_validation_spec(prepared, image="fixture:test")
    expanded = validation.create_validation_spec(prepared, image="fixture:test", max_prompt_chars=600_000)
    assert default.max_prompt_chars == 200_000
    assert expanded.max_prompt_chars == 600_000
    expected = default.model_dump(mode="json")
    expected["max_prompt_chars"] = 600_000
    assert expanded.model_dump(mode="json") == expected
    with pytest.raises(ValueError, match="max_prompt_chars"):
        validation.create_validation_spec(prepared, image="fixture:test", max_prompt_chars=999)


def test_validate_forwards_limit_and_rejects_changed_resume_without_overwriting(prepared, tmp_path, monkeypatch):
    class StopBeforeRuntime(Exception):
        pass

    def no_runtime(*args):
        raise StopBeforeRuntime

    monkeypatch.setattr(validation, "load_development", no_runtime)
    output = tmp_path / "validation"
    with pytest.raises(StopBeforeRuntime):
        validation.validate(prepared, output, image="fixture:test", max_prompt_chars=600_000)
    path = output / "validation-spec.json"
    frozen = path.read_bytes()
    assert json.loads(frozen)["max_prompt_chars"] == 600_000
    with pytest.raises(ValueError, match="configuration changed on resume"):
        validation.validate(prepared, output, image="fixture:test", max_prompt_chars=200_000)
    assert path.read_bytes() == frozen
    with pytest.raises(StopBeforeRuntime):
        validation.validate(prepared, output, image="fixture:test", max_prompt_chars=600_000)
    assert path.read_bytes() == frozen
