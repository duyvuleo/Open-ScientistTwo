"""Native live dispatch contracts with a local fake CLI, never real model calls.

The native adapter and controller's role/cache/accounting paths are real. The
research-stage driver is deliberately synthetic; these are not benchmark scores.
"""
from contextlib import closing
import json

import pytest
from typer.testing import CliRunner

from scientisttwo.backends import BackendError
from scientisttwo.benchmarks.models import NativeSpec
from scientisttwo.live import cli
from scientisttwo.live.adapters import implementation_identity
from scientisttwo.live.models import DevelopmentSpec, LiveSpec
from scientisttwo.research.controller import ResearchController
from scientisttwo.research import live_pilot
from scientisttwo.research.policy import ResearchPolicy
from scientisttwo.store import IntegrityError, Store

import test_study_media_relay as relay_tests
from test_study_media_relay import Answer


fake_cli = relay_tests.fake_cli


def native_config(tmp_path, **updates):
    public = tmp_path / "native-inputs"
    public.mkdir()
    (public / "train.py").write_text(
        "# Explicitly synthetic source for a native-dispatch contract test.\n"
        "class Conv2DLayer:\n"
        "    def forward(self, features_batch):\n"
        "        return features_batch\n"
    )
    values = dict(
        engine_version="research-v2", mode="research",
        native=NativeSpec(name="native-dispatch-contract", benchmark="mlagentbench",
            task_id="vectorization", repository=tmp_path / "repository", revision="a" * 40,
            mode="baseline", data_provenance="contract_fixture", public_dir=public),
        development=DevelopmentSpec(metric="process_seconds", direction="minimize"),
        model="gpt-6-astra", model_retries=0, max_calls=4, max_experiments=1,
        token_stop_threshold=100, call_seconds=30, campaign_seconds=120,
        workflow=ResearchPolicy(writer_backend="canonical"),
        implementation=implementation_identity(),
    )
    values.update(updates)
    return LiveSpec(**values)


def one_role(self):
    """Synthetic end state after one genuine controller/fake-CLI invocation."""
    self.store.set("live_spec_ref", self.store.put_json(self.config))
    result = self.ask("native-contract", "critic", {"fixture": "synthetic dispatch"}, Answer)
    assert result.answer == "actual fake CLI"
    state = {"stage": "complete", "status": "complete", "outcome": "no_improvement",
             "reason": "Synthetic dispatch contract, no experiments executed"}
    self.store.set("live_state", state)
    return state


def forbidden(*args, **kwargs):
    raise AssertionError("No backend, model executable or experiment may run")


def test_native_run_and_completed_resume_use_durable_receipts_without_new_calls(
    tmp_path, monkeypatch, fake_cli
):
    executable, counter = fake_cli
    config = native_config(tmp_path)
    manifest, output = tmp_path / "live.json", tmp_path / "campaign"
    manifest.write_text(config.model_dump_json())
    monkeypatch.setattr(ResearchController, "run", one_role)
    monkeypatch.setattr(cli, "CodexBackend", forbidden)  # Plain legacy path must not be used.
    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(manifest), str(output), "--allow-live",
                                    "--executable", str(executable)])
    assert result.exit_code == 0, result.exception or result.output
    assert json.loads(result.stdout)["state"]["status"] == "complete"
    with closing(Store(output)) as store:
        summary_ref = store.get("live_pilot_model_accounting")
        summary = store.json(summary_ref)
        assert summary["claimed_calls"] == summary["completed_receipts"] == 1
        assert summary["known_observed_tokens"] == 12 and summary["usage_complete"]
        assert summary["transport"] == "direct_local_no_queue"
        operations = [dict(row) for row in store.db.execute("SELECT * FROM live_operations")]
        assert len(operations) == 1 and operations[0]["tokens"] == 12
        assert operations[0]["status"] == "success"
        assert store.get("task_ref") is None  # This is a native campaign, not a bundled pilot.
        deadline = store.get("live_deadline")
    monkeypatch.setattr(live_pilot, "CodexBackend", forbidden)
    monkeypatch.setattr(live_pilot, "DirectPilotExecutor", forbidden)
    monkeypatch.setattr(ResearchController, "run", forbidden)
    resumed = runner.invoke(cli.app, ["resume", str(output), "--allow-live",
                                     "--executable", str(executable)])
    assert resumed.exit_code == 0, resumed.exception or resumed.output
    with closing(Store(output)) as store:
        assert store.get("live_deadline") == deadline
        assert store.get("live_pilot_model_accounting") == summary_ref
        assert [dict(row) for row in store.db.execute("SELECT * FROM live_operations")] == operations
    assert len(counter.read_text().splitlines()) == 1
    assert not list(output.rglob("*.request.json"))


@pytest.mark.parametrize("update,match", [
    ({"token_stop_threshold": None}, "token_stop_threshold"),
    ({"model_retries": 1}, "model_retries"),
    ({"model": "scripted-fixture-v1"}, "non-fixture"),
    ({"implementation": {"changed": "source"}}, "source"),
])
def test_native_live_preconditions_fail_before_backend(tmp_path, monkeypatch, update, match):
    config = native_config(tmp_path, **update)
    monkeypatch.setattr(live_pilot, "DirectPilotExecutor", forbidden)
    monkeypatch.setattr(live_pilot, "CodexBackend", forbidden)
    with pytest.raises((ValueError, IntegrityError), match=match):
        cli.execute(config, tmp_path / "campaign", True)


def test_unknown_native_call_usage_is_retained_and_cannot_dispatch_on_resume(
    tmp_path, monkeypatch, fake_cli
):
    executable, counter = fake_cli
    executable.write_text(executable.read_text().replace(
        "'turn.completed','usage':{'input_tokens':10,'cached_input_tokens':3,'output_tokens':2}",
        "'turn.failed'"))
    config = native_config(tmp_path)
    output = tmp_path / "campaign"
    monkeypatch.setattr(ResearchController, "run", one_role)
    for _ in range(2):
        with pytest.raises(BackendError):
            cli.execute(config, output, True, executable=str(executable))
    with closing(Store(output)) as store:
        summary = store.json(store.get("live_pilot_model_accounting"))
        assert summary["claimed_calls"] == 1 and summary["unknown_calls"]
        assert not summary["usage_complete"]
        assert store.db.execute("SELECT count(*) FROM live_operations").fetchone()[0] == 1
    assert len(counter.read_text().splitlines()) == 1


def test_final_freeze_still_blocks_native_agent_invocations(tmp_path, monkeypatch, fake_cli):
    executable, counter = fake_cli
    config = native_config(tmp_path)
    output = tmp_path / "campaign"
    with closing(Store(output)) as store:
        store.set("live_frozen", {"fixture": "synthetic final barrier"})
    monkeypatch.setattr(ResearchController, "run", one_role)
    with pytest.raises(IntegrityError, match="final candidate freeze"):
        cli.execute(config, output, True, executable=str(executable))
    assert not counter.exists()
    with closing(Store(output)) as store:
        assert store.json(store.get("live_pilot_model_accounting"))["claimed_calls"] == 0


def test_legacy_native_execution_preserves_its_existing_backend(tmp_path, monkeypatch):
    config = native_config(tmp_path, engine_version="legacy-live-v1")
    called = []
    monkeypatch.setattr(live_pilot, "run_prepared_live", forbidden)
    monkeypatch.setattr(cli, "CodexBackend", lambda *a, **kw: called.append(kw) or object())
    monkeypatch.setattr(cli, "LiveController", lambda *a, **kw: type("Controller", (), {
        "run": lambda self: {"status": "complete", "outcome": "no_improvement"}})())
    cli.execute(config, tmp_path / "campaign", True)
    assert len(called) == 1 and called[0]["allow_live"] is True
