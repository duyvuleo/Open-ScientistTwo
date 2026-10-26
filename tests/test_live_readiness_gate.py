"""Readiness must block a live backend before it is constructed."""

from types import SimpleNamespace

import pytest

from scientisttwo.live import cli, suite
from scientisttwo.live.workspace import live_readiness
from test_live_suite import specification


def blocked_adapter():
    return SimpleNamespace(live_readiness=lambda: {
        "ready": False, "blockers": ["Actual-data validation is incomplete"]
    })


def test_optional_adapter_gate_preserves_other_plugins():
    assert live_readiness(object(), require=True)["ready"]
    assert not live_readiness(blocked_adapter())["ready"]
    with pytest.raises(ValueError, match="Actual-data validation"):
        live_readiness(blocked_adapter(), require=True)
    with pytest.raises(ValueError, match="boolean"):
        live_readiness(SimpleNamespace(live_readiness=lambda: {"ready": "yes"}))


def test_live_command_checks_gate_before_backend(tmp_path, monkeypatch):
    config = specification(tmp_path).assignments[0].campaign
    monkeypatch.setattr(cli, "load_development", lambda *_: blocked_adapter())
    monkeypatch.setattr(cli, "CodexBackend", lambda *a, **k: pytest.fail("Live backend constructed"))
    with pytest.raises(ValueError, match="Actual-data validation"):
        cli.execute(config, tmp_path / "output", True)


def test_suite_assignment_checks_gate_before_backend(tmp_path, monkeypatch):
    from scientisttwo import backends
    from scientisttwo.live import adapters

    config = specification(tmp_path).assignments[0].campaign
    monkeypatch.setattr(adapters, "load_development", lambda *_: blocked_adapter())
    monkeypatch.setattr(backends, "CodexBackend", lambda *a, **k: pytest.fail("Live backend constructed"))
    shared = SimpleNamespace(assignment="trial-0", suite=SimpleNamespace(
        store=SimpleNamespace(root=tmp_path / "suite"), config=specification(tmp_path)))
    with pytest.raises(ValueError, match="Actual-data validation"):
        suite.execute_assignment(config, tmp_path / "output", shared)
