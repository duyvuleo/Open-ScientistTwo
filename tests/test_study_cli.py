"""Installed adapter discovery and model-free preparation command boundaries."""
from importlib.metadata import entry_points
import json

from typer.testing import CliRunner

from scientisttwo.cli import app
from scientisttwo.live.adapters import implementation_identity
from scientisttwo.studies import cli


def test_installed_cifar_entry_point_loads_real_adapter():
    found = list(entry_points(group="scientisttwo.native_development", name="cifar_recipe"))
    assert len(found) == 1
    assert found[0].load() is cli.CifarRecipeDevelopment


def test_study_preparation_does_not_construct_agent(monkeypatch, tmp_path):
    from scientisttwo import backends
    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation cannot construct a live backend")
    monkeypatch.setattr(backends.CodexBackend, "__init__", forbidden)
    monkeypatch.setattr(cli, "prepare_study", lambda source, output: {"variant": "synthetic-cli"})
    result = CliRunner().invoke(app, ["study", "prepare", str(tmp_path / "source"), str(tmp_path / "output")])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["live_model_calls"] == 0
    assert report["runtime_validation"] == "unverified"


def test_source_identity_includes_study_execution_and_protocol():
    identity = implementation_identity()
    assert {"studies/runner.py", "studies/direct_control.py", "studies/calibration.py",
            "studies/cifar_recipe.py", "studies/media_relay.py"} <= set(identity)
