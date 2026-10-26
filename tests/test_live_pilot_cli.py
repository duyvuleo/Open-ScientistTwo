"""CLI authorization/configuration contracts; no real model or container calls."""

import json

import pytest
from typer.testing import CliRunner

from scientisttwo import cli
from scientisttwo.models import Profile, TaskSpec
from scientisttwo.research import live_pilot, live_pilot_cli
from scientisttwo.research.policy import ResearchPolicy
from scientisttwo.store import IntegrityError, Store, canonical, digest


IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def inputs(tmp_path):
    task = TaskSpec()
    options = live_pilot.LivePilotOptions(
        model="gpt-6-astra", max_calls=40, token_stop_threshold=400_000,
        campaign_seconds=600, call_seconds=30, max_experiments=8,
        workflow=ResearchPolicy(seed_ideas=2, profile="research-pilot"),
        search_rounds=1, engineering_retries=0, ablation_refinements=0,
        review_rounds=0, meta_refinements=0)
    task_path, options_path = tmp_path / "task.json", tmp_path / "options.json"
    task_path.write_text(task.model_dump_json())
    options_path.write_text(options.model_dump_json())
    return task_path, options_path, task, options


def manifest(tmp_path, inputs):
    task_path, options_path, _, _ = inputs
    output = tmp_path / "manifest.json"
    result = CliRunner().invoke(cli.app, ["live-pilot", "create", str(task_path),
        str(options_path), str(output), "--image", IMAGE, "--context", "synthetic-context",
        "--experiment-seconds", "25"])
    assert result.exit_code == 0, result.exception or result.output
    return output, result


def forbidden(*args, **kwargs):
    raise AssertionError("This CLI contract cannot construct a backend or run a campaign")


def test_create_freezes_bounded_manifest_without_backend_or_campaign(tmp_path, inputs, monkeypatch):
    from scientisttwo.research import pilot_bridge

    monkeypatch.setattr(live_pilot, "CodexBackend", forbidden)
    monkeypatch.setattr(live_pilot, "DirectPilotExecutor", forbidden)
    monkeypatch.setattr(pilot_bridge, "create_pilot_campaign", forbidden)
    monkeypatch.setattr(live_pilot_cli, "run_live_pilot", forbidden)
    path, result = manifest(tmp_path, inputs)
    frozen = live_pilot_cli.LivePilotManifest.model_validate_json(path.read_text())
    payload = json.loads(result.stdout)
    assert payload["status"] == "prepared_no_calls"
    assert payload["live_model_calls"] == 0
    assert payload["manifest_sha256"] == digest(canonical(frozen))
    assert frozen.task == inputs[2] and frozen.options == inputs[3]
    assert frozen.profile.live_calls is False
    assert frozen.profile.experiment_seconds == 25
    assert frozen.profile.campaign_seconds == frozen.options.campaign_seconds == 600
    assert frozen.runtime == "docker" and frozen.context == "synthetic-context"
    assert frozen.image == IMAGE
    assert not list(tmp_path.rglob("campaign.sqlite"))
    assert not list(tmp_path.rglob("model-ledger"))


@pytest.mark.parametrize("command", ["run", "resume"])
def test_opt_in_required_before_manifest_or_store_access(tmp_path, monkeypatch, command):
    monkeypatch.setattr(live_pilot_cli, "run_live_pilot", forbidden)
    monkeypatch.setattr(live_pilot_cli, "resume_live_pilot", forbidden)
    target = tmp_path / "absent-campaign"
    args = ["live-pilot", command]
    args += [str(tmp_path / "absent-manifest.json"), str(target)] if command == "run" else [str(target)]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "requires --allow-live" in str(result.exception)
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_implementation_drift_rejected_before_store_or_backend(tmp_path, inputs, monkeypatch):
    path, _ = manifest(tmp_path, inputs)
    monkeypatch.setattr(live_pilot_cli, "implementation_identity", lambda: {"changed": "source"})
    monkeypatch.setattr(live_pilot_cli, "run_live_pilot", forbidden)
    monkeypatch.setattr(live_pilot, "CodexBackend", forbidden)
    target = tmp_path / "campaign"
    result = CliRunner().invoke(cli.app, ["live-pilot", "run", str(path), str(target), "--allow-live"])
    assert isinstance(result.exception, IntegrityError)
    assert "Implementation changed" in str(result.exception)
    assert not target.exists()


@pytest.mark.parametrize("status,exit_code", [("complete", 0), ("blocked", 2)])
def test_authorized_run_delegates_exact_frozen_limits(tmp_path, inputs, monkeypatch, status, exit_code):
    path, _ = manifest(tmp_path, inputs)
    calls = []

    def run(store, task, profile, **kwargs):
        calls.append((task, profile, kwargs))
        frozen = live_pilot_cli.LivePilotManifest.model_validate_json(path.read_text())
        assert store.json(store.get("live_pilot_launch_ref")) == frozen.model_dump(mode="json")
        assert task == frozen.task and profile == frozen.profile
        assert kwargs == {"options": frozen.options, "runtime": "docker", "image": IMAGE,
                          "context": "synthetic-context", "allow_live": True,
                          "executable": "synthetic-codex-not-invoked"}
        store.set("live_pilot_model_accounting", store.put_json({"claimed_calls": 0, "synthetic_test": True}))
        return {"status": status, "reason": "Synthetic delegated CLI result"}

    monkeypatch.setattr(live_pilot_cli, "run_live_pilot", run)
    monkeypatch.setattr(live_pilot, "CodexBackend", forbidden)
    target = tmp_path / "campaign"
    result = CliRunner().invoke(cli.app, ["live-pilot", "run", str(path), str(target),
        "--allow-live", "--executable", "synthetic-codex-not-invoked"])
    assert result.exit_code == exit_code, result.exception or result.output
    assert len(calls) == 1
    payload = json.loads(result.stdout)
    assert payload["state"]["status"] == status
    assert payload["directory"] == str(target.resolve())
    assert payload["model_accounting_ref"]


def test_explicit_resume_delegates_to_saved_campaign_without_new_limits(tmp_path, monkeypatch):
    target = tmp_path / "campaign"
    with_store = Store(target)
    with_store.set("live_pilot_options_ref", "synthetic-options-marker")
    with_store.close()
    calls = []

    def resume(store, **kwargs):
        calls.append(kwargs)
        assert store.get("live_pilot_options_ref") == "synthetic-options-marker"
        return {"status": "complete", "reason": "Synthetic resumed result"}

    monkeypatch.setattr(live_pilot_cli, "resume_live_pilot", resume)
    result = CliRunner().invoke(cli.app, ["live-pilot", "resume", str(target), "--allow-live",
                                        "--executable", "synthetic-codex-not-invoked"])
    assert result.exit_code == 0, result.exception or result.output
    assert calls == [{"allow_live": True, "executable": "synthetic-codex-not-invoked"}]


@pytest.mark.parametrize("engine", ["research-v2", None])
def test_ordinary_offline_resume_refuses_live_store_before_any_engine(tmp_path, inputs, monkeypatch, engine):
    target = tmp_path / "campaign"
    store = Store(target)
    try:
        store.set("task_ref", store.put_json(inputs[2]))
        store.set("profile_ref", store.put_json(Profile()))
        store.set("runtime", {"runtime": "docker", "image": IMAGE, "context": "synthetic-context"})
        store.set("live_pilot_options_ref", store.put_json(inputs[3]))
        if engine:
            store.set("engine_version", engine)
        before = [dict(row) for row in store.db.execute("SELECT * FROM metadata ORDER BY key")]
    finally:
        store.close()
    monkeypatch.setattr(cli, "execute_legacy", forbidden)
    monkeypatch.setattr(cli, "FixtureBackend", forbidden)
    monkeypatch.setattr(live_pilot, "CodexBackend", forbidden)
    result = CliRunner().invoke(cli.app, ["resume", str(target)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError), result.exception
    assert "live-pilot resume --allow-live" in str(result.exception)
    store = Store(target)
    try:
        assert [dict(row) for row in store.db.execute("SELECT * FROM metadata ORDER BY key")] == before
        assert store.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
        assert store.db.execute("SELECT count(*) FROM calls").fetchone()[0] == 0
    finally:
        store.close()
