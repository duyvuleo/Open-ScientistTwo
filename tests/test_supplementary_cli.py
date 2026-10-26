"""Synthetic metadata tests; manifest creation makes no model or runtime calls."""

import json
import re

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from scientisttwo.cli import app
from scientisttwo.live import cli as live_cli
from scientisttwo.live.models import LiveSpec
from scientisttwo.testbed import export
from scientisttwo.testbed.adapter import PreparedDevelopment
from test_testbed_export_binding import configured, forbid_live_backend


@pytest.mark.parametrize("value,expected", [(None, 2), (10, 10)])
def test_create_cli_freezes_exact_supplementary_seed_allowance_without_live_calls(
    tmp_path, monkeypatch, value, expected
):
    _, _, original, _ = configured(tmp_path, monkeypatch)
    native = tmp_path / "native.json"
    native.write_text(original.native.model_dump_json())
    development = tmp_path / "development.json"
    development.write_text(original.development.model_dump_json())
    backend_calls = forbid_live_backend(monkeypatch)
    preflight_calls = []

    def preflight(spec):
        preflight_calls.append(spec)
        return {"ready": True, "errors": [], "synthetic_test": True}

    monkeypatch.setattr(live_cli, "preflight", preflight)
    monkeypatch.setattr(live_cli, "load_development", PreparedDevelopment)
    output = tmp_path / "live.json"
    args = [
        "live",
        "create",
        str(native),
        str(output),
        "--development",
        str(development),
        "--model",
        "gpt-6-astra",
        "--max-calls",
        "12",
        "--seconds",
        "1200",
    ]
    if value is not None:
        args += ["--supplementary-seed-count", str(value)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    saved = LiveSpec.model_validate_json(output.read_text())
    assert saved.supplementary_seed_count == expected
    assert saved.development.seeds == original.development.seeds
    assert json.loads(result.stdout)["live_model_calls"] == 0
    assert len(preflight_calls) == 1
    assert not backend_calls


@pytest.mark.parametrize("value,expected", [(None, 2), (10, 10)])
def test_export_cli_freezes_exact_supplementary_seed_allowance_without_live_calls(
    tmp_path, monkeypatch, value, expected
):
    root, _, original, _ = configured(tmp_path, monkeypatch)
    backend_calls = forbid_live_backend(monkeypatch)
    calls, original_export = [], export.export_live

    def observed_export(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_export(*args, **kwargs)

    monkeypatch.setattr(export, "export_live", observed_export)
    output = tmp_path / "exported.json"
    args = [
        "benchmark",
        "export-live",
        "house-price",
        str(output),
        "--root",
        str(root),
        "--model",
        "gpt-6-astra",
        "--max-calls",
        "12",
        "--campaign-seconds",
        "1200",
    ]
    if value is not None:
        args += ["--supplementary-seed-count", str(value)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    saved = LiveSpec.model_validate_json(output.read_text())
    assert saved.supplementary_seed_count == expected
    assert calls[0]["supplementary_seed_count"] == expected
    assert saved.implementation == original.implementation
    assert saved.development.seeds == original.development.seeds
    assert json.loads(result.stdout)["live_model_calls"] == 0
    assert not backend_calls


@pytest.mark.parametrize("command", ["create", "export"])
@pytest.mark.parametrize("value", [0, 33])
def test_cli_rejects_out_of_range_allowances_before_callbacks_or_writes(
    tmp_path, monkeypatch, command, value
):
    calls = []
    backend_calls = forbid_live_backend(monkeypatch)

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Invalid CLI allowance must be rejected before callbacks")

    monkeypatch.setattr(live_cli, "preflight", forbidden)
    monkeypatch.setattr(export, "export_live", forbidden)
    output = tmp_path / "absent" / "must-not-exist.json"
    if command == "create":
        args = [
            "live",
            "create",
            str(tmp_path / "missing-native.json"),
            str(output),
            "--model",
            "gpt-6-astra",
            "--max-calls",
            "12",
            "--seconds",
            "1200",
        ]
    else:
        args = [
            "benchmark",
            "export-live",
            "house-price",
            str(output),
            "--root",
            str(tmp_path / "missing-registry"),
            "--model",
            "gpt-6-astra",
            "--max-calls",
            "12",
            "--campaign-seconds",
            "1200",
        ]
    result = CliRunner().invoke(app, args + ["--supplementary-seed-count", str(value)])
    assert result.exit_code == 2
    assert "supplementary-seed-count" in re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.output)
    assert not output.exists() and not output.parent.exists()
    assert not calls and not backend_calls


def test_typed_manifest_defaults_and_bounds_are_independent_of_cli(tmp_path, monkeypatch):
    _, _, original, _ = configured(tmp_path, monkeypatch)
    payload = original.model_dump(mode="json")
    payload.pop("supplementary_seed_count", None)
    assert LiveSpec.model_validate(payload).supplementary_seed_count == 2
    assert (
        LiveSpec.model_validate(
            {**payload, "supplementary_seed_count": 32}
        ).supplementary_seed_count
        == 32
    )
    for count in (0, 33):
        with pytest.raises(ValidationError):
            LiveSpec.model_validate({**payload, "supplementary_seed_count": count})
