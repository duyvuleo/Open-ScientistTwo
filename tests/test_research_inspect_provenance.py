"""Inspect uses persisted backend provenance; these records are test fixtures."""

import json

import pytest
from typer.testing import CliRunner

from scientisttwo.cli import app
from scientisttwo.store import Store


@pytest.mark.parametrize("synthetic,expected_calls", [(False, 2), (True, 0)])
def test_root_inspect_does_not_label_live_backend_as_offline(tmp_path, synthetic, expected_calls):
    root = tmp_path / "fixture-campaign"
    store = Store(root)
    state = {"stage": "search", "status": "running", "reason": "Synthetic inspection fixture"}
    provenance = {"backend": "codex-fixture" if not synthetic else "fixture", "synthetic": synthetic,
                  "model": "test-model-only"}
    store.set("engine_version", "research-v2")
    store.set("live_state", state)
    store.set("live_backend", provenance)
    store.db.execute("CREATE TABLE live_operations (key TEXT,kind TEXT,status TEXT)")
    store.db.executemany("INSERT INTO live_operations VALUES (?,?,?)", [
        ("call-one", "model", "success"), ("call-two", "model", "failed"),
        ("compile", "probe", "success"), ("experiment", "experiment", "success"),
    ])
    store.db.commit()
    store.close()
    response = CliRunner().invoke(app, ["inspect", "fixture-campaign", "--runs-dir", str(tmp_path)])
    assert response.exit_code == 0, response.output
    inspected = json.loads(response.stdout)
    assert inspected["backend"] == provenance
    assert inspected["state"] == state
    assert inspected["live_calls"] is (not synthetic)
    assert inspected["live_model_calls"] == expected_calls
