"""Live-pilot contracts with an executable fake CLI; no actual model or container."""
from copy import deepcopy
import time

import pytest

from scientisttwo.backends import CodexBackend, BackendError
from scientisttwo.models import Profile, TaskSpec
from scientisttwo.process import ProcessResult
from scientisttwo.research import live_pilot as live
from scientisttwo.research.pilot_bridge import create_pilot_campaign
from scientisttwo.research.policy import ResearchPolicy
from scientisttwo.store import Store, IntegrityError

from test_research_pilot_bridge import SyntheticContainer
import test_study_media_relay as relay_tests
from test_study_media_relay import Answer

fake_cli = relay_tests.fake_cli


IMAGE = "sha256:" + "a" * 64


def options(**updates):
    values = dict(model="gpt-6-astra", max_calls=100, token_stop_threshold=1_500_000,
                  campaign_seconds=7200, call_seconds=600, max_experiments=24,
                  workflow=ResearchPolicy(seed_ideas=2, novelty_attempts=1, max_ablation_plans=1,
                                          writing_refinements=0, profile="research-pilot"))
    values.update(updates)
    return live.LivePilotOptions.model_validate(values)


def setup(tmp_path, chosen=None):
    store = Store(tmp_path / "campaign")
    worker = SyntheticContainer(store)
    config, adapter, runner = create_pilot_campaign(store, TaskSpec(), Profile(), image=IMAGE,
        live_options=chosen or options(), container_runner=worker)
    return store, config, adapter, runner, worker


def test_live_factory_freezes_options_before_any_backend_and_reconstructs_for_audit(tmp_path):
    declared = options()
    store, config, _, runner, worker = setup(tmp_path, declared)
    assert config.model == "gpt-6-astra" and config.max_calls == 100
    assert config.call_seconds == 600 and config.campaign_seconds == 7200
    assert config.engine_version == "research-v2" and config.workflow.seed_ideas == 2
    assert config.token_stop_threshold == 1_500_000 and config.max_experiments == 24
    assert store.json(store.get("live_pilot_options_ref")) == declared.model_dump(mode="json")
    assert store.json(store.get("profile_ref"))["live_calls"] is False
    assert store.get("live_pilot_execution_policy")["synthetic_agents"] is False
    assert not worker.executions and not runner.budget.summary()
    assert not (store.root / "model-ledger").exists()
    restored, _, _ = create_pilot_campaign(store, TaskSpec(), Profile(), image=IMAGE,
                                           container_runner=worker)
    assert restored == config  # audit_research's existing reconstruction path.


def test_offline_factory_remains_fixture_only_and_cannot_be_upgraded(tmp_path):
    store = Store(tmp_path)
    config, _, _ = create_pilot_campaign(store, TaskSpec(), Profile())
    assert config.model == "scripted-fixture-v1"
    assert store.get("live_pilot_options_ref") is None
    with pytest.raises(IntegrityError, match="offline frozen"):
        create_pilot_campaign(store, TaskSpec(), Profile(), image=IMAGE, live_options=options())


@pytest.mark.parametrize("changed", ["options", "image", "task", "profile", "source"])
def test_live_resume_rejects_every_configuration_drift(tmp_path, monkeypatch, changed):
    store, _, _, _, worker = setup(tmp_path)
    arguments = {"task": TaskSpec(), "profile": Profile(), "image": IMAGE, "container_runner": worker}
    if changed == "options":
        arguments["live_options"] = options(max_calls=101)
    elif changed == "image":
        arguments["image"] = "sha256:" + "b" * 64
    elif changed == "task":
        arguments["task"] = TaskSpec(split_seed=1)
    elif changed == "profile":
        arguments["profile"] = Profile(experiment_seconds=121)
    else:
        monkeypatch.setattr("scientisttwo.research.pilot_bridge.implementation_identity", lambda: {"changed": "source"})
    with pytest.raises(IntegrityError):
        create_pilot_campaign(store, **arguments)


def test_all_execution_and_resume_paths_require_explicit_opt_in_before_side_effects(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(ValueError, match="allow_live"):
        live.run_live_pilot(store, TaskSpec(), Profile(), options=options())
    with pytest.raises(ValueError, match="allow_live"):
        live.resume_live_pilot(store)
    assert store.get("task_ref") is None and store.get("live_pilot_options_ref") is None


def test_completed_resume_never_constructs_backend_or_executor_and_adds_no_operations(tmp_path, monkeypatch):
    store, config, _, _, worker = setup(tmp_path)
    state = {"stage": "complete", "status": "complete", "outcome": "development_improvement",
             "reason": "Synthetic completed campaign"}
    store.set("live_spec_ref", store.put_json(config))
    store.set("live_state", state)

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed resume cannot construct a backend/executor")

    monkeypatch.setattr(live, "CodexBackend", forbidden)
    monkeypatch.setattr(live, "DirectPilotExecutor", forbidden)
    before = [dict(row) for row in store.db.execute("SELECT * FROM live_operations")]
    assert live.resume_live_pilot(store, allow_live=True, container_runner=worker) == state
    assert [dict(row) for row in store.db.execute("SELECT * FROM live_operations")] == before


def test_real_fake_cli_roundtrip_has_durable_direct_accounting_without_queue(tmp_path, fake_cli):
    cli, counter = fake_cli
    store, config, _, _, _ = setup(tmp_path)
    executor = live.DirectPilotExecutor(store.root / "model-ledger", config, time.time() + 60,
                                        executable=str(cli), allow_live=True)
    backend = CodexBackend(store.root / "agent", config.model, allow_live=True, timeout=30, executor=executor)
    result = backend.invoke("critic", {"observation": "synthetic input"}, Answer)
    assert result.payload == {"answer": "actual fake CLI"}
    assert result.synthetic is False and result.usage["input_tokens"] == 10
    accounting = executor.accounting()
    assert accounting["claimed_calls"] == accounting["completed_receipts"] == 1
    assert accounting["known_observed_tokens"] == 12 and accounting["usage_complete"]
    assert accounting["transport"] == "direct_local_no_queue"
    assert len(counter.read_text().splitlines()) == 1
    assert not list(store.root.rglob("*.request.json"))


@pytest.mark.parametrize("failure", ["unknown_usage", "lost_completion"])
def test_unknown_local_model_outcome_blocks_every_later_dispatch(tmp_path, fake_cli, failure):
    cli, counter = fake_cli
    store, config, _, _, _ = setup(tmp_path)
    calls = []

    def executor(*args, **kwargs):
        calls.append(True)
        if failure == "lost_completion":
            raise RuntimeError("Synthetic crash after possible provider dispatch")
        return ProcessResult(1, '{"type":"turn.failed"}\n', "private fixture diagnostic", .1)

    direct = live.DirectPilotExecutor(store.root / "model-ledger", config, time.time() + 60,
                                      executable=str(cli), executor=executor, allow_live=True)
    backend = CodexBackend(store.root / "agent", config.model, allow_live=True, timeout=30, executor=direct)
    with pytest.raises((BackendError, RuntimeError)):
        backend.invoke("critic", {}, Answer)
    with pytest.raises(live.relay.UnknownDispatch):
        backend.invoke("critic", {"new": "request"}, Answer)
    assert len(calls) == 1 and not counter.exists()
    assert direct.accounting()["unknown_calls"] and not direct.accounting()["usage_complete"]


def test_known_failed_usage_still_exhausts_direct_token_allowance(tmp_path, fake_cli):
    cli, _ = fake_cli
    store, config, _, _, _ = setup(tmp_path, options(token_stop_threshold=12))
    direct = live.DirectPilotExecutor(store.root / "model-ledger", config, time.time() + 60,
        executable=str(cli), allow_live=True,
        executor=lambda *args, **kwargs: ProcessResult(1, '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":2}}', '', .1))
    backend = CodexBackend(store.root / "agent", config.model, allow_live=True, timeout=30, executor=direct)
    with pytest.raises(BackendError):
        backend.invoke("critic", {}, Answer)
    with pytest.raises(live.relay.RelayError, match="threshold"):
        backend.invoke("critic", {}, Answer)
    assert direct.accounting()["known_observed_tokens"] == 12


def test_frozen_literature_is_part_of_live_adapter_identity(tmp_path):
    from scientisttwo.research.literature import SourceRecord, SourcePassage
    from scientisttwo.store import digest
    text = "Synthetic supplied passage, not a real scientific source."
    source = SourceRecord(id="synthetic-paper", title="Synthetic contract reference", provider="local",
        retrieved_at="2026-10-24T00:00:00Z", metadata_status="supplied", passages=[SourcePassage(id="passage", text=text,
            sha256=digest(text.encode()), scope="supplied_excerpt")])
    declared = options(frozen_literature=[source])
    before = deepcopy(declared)
    store, _, adapter, _, _ = setup(tmp_path, declared)
    assert adapter.sources()[-1]["id"] == "synthetic-paper"
    assert "frozen_literature" in adapter.identity()
    assert declared == before and store.json(store.get("live_pilot_options_ref"))["frozen_literature"]


def test_unpinned_worker_image_and_network_literature_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="pinned worker"):
        create_pilot_campaign(Store(tmp_path), TaskSpec(), Profile(), live_options=options())
    with pytest.raises(ValueError, match="frozen local"):
        options(workflow=ResearchPolicy(network_literature=True))
