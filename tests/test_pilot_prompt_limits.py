"""Offline context ceilings are explicit and frozen; no model/container calls."""

import pytest
from typer.testing import CliRunner

from scientisttwo import cli
from scientisttwo.models import Profile, TaskSpec
from scientisttwo.research.live_pilot import LivePilotOptions
from scientisttwo.research.pilot_bridge import create_pilot_campaign
from scientisttwo.research.policy import ResearchPolicy
from scientisttwo.store import IntegrityError, Store
from test_research_pilot_bridge import SyntheticContainer


@pytest.mark.parametrize("limit", [None, 1_000_000])
def test_offline_factory_preserves_default_or_freezes_explicit_limit(tmp_path, limit):
    store = Store(tmp_path / "campaign")
    worker = SyntheticContainer(store)
    try:
        if limit is not None:
            store.set("pilot_controller_options", {"max_prompt_chars": limit})
        config, _, _ = create_pilot_campaign(store, TaskSpec(), Profile(), container_runner=worker)
        expected = 200_000 if limit is None else limit
        assert config.max_prompt_chars == expected
        frozen_ref = store.get("pilot_live_spec_ref")
        assert store.json(frozen_ref)["max_prompt_chars"] == expected
        again, _, _ = create_pilot_campaign(store, TaskSpec(), Profile(), container_runner=worker)
        assert again == config
        store.set("pilot_controller_options", {"max_prompt_chars": expected + 1})
        with pytest.raises(IntegrityError, match="specification changed"):
            create_pilot_campaign(store, TaskSpec(), Profile(), container_runner=worker)
        assert store.get("pilot_live_spec_ref") == frozen_ref
        assert not worker.executions
    finally:
        store.close()


def test_offline_cli_records_character_limit_separately_from_writer_options(tmp_path, monkeypatch):
    captured = []

    def capture(store, *args, **kwargs):
        captured.append((store.get("pilot_controller_options"), store.get("pilot_workflow_options")))

    monkeypatch.setattr(cli, "execute", capture)
    result = CliRunner().invoke(cli.app, ["run", "pilot", "--runs-dir", str(tmp_path),
                                          "--max-prompt-chars", "1000000"])
    assert result.exit_code == 0, result.exception or result.output
    assert captured[0][0] == {"max_prompt_chars": 1_000_000}
    assert "max_prompt_chars" not in captured[0][1]
    result = CliRunner().invoke(cli.app, ["run", "pilot", "--runs-dir", str(tmp_path),
                                          "--max-prompt-chars", "999"])
    assert result.exit_code != 0
    assert len(captured) == 1


def test_explicit_live_options_keep_their_own_prompt_contract(tmp_path):
    store = Store(tmp_path / "campaign")
    worker = SyntheticContainer(store)
    try:
        options = LivePilotOptions(model="configured-model-not-invoked", max_calls=40,
            token_stop_threshold=400_000, campaign_seconds=600, call_seconds=30, max_experiments=20,
            max_prompt_chars=345_678, workflow=ResearchPolicy())
        config, _, _ = create_pilot_campaign(store, TaskSpec(), Profile(),
            image="sha256:" + "a" * 64, live_options=options, container_runner=worker)
        assert config.max_prompt_chars == 345_678
        assert store.get("pilot_controller_options") is None
        assert not worker.executions
        store.set("pilot_controller_options", {"max_prompt_chars": 1_000_000})
        with pytest.raises(IntegrityError, match="Offline controller options"):
            create_pilot_campaign(store, TaskSpec(), Profile(),
                image="sha256:" + "a" * 64, live_options=options, container_runner=worker)
    finally:
        store.close()
