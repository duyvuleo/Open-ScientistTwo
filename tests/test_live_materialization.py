"""Trusted proposal assembly and its evidence binding; no model or host code execution."""

from types import SimpleNamespace

import pytest

from scientisttwo.live.adapters import implementation_identity
from scientisttwo.live.audit import audit_live
from scientisttwo.live.models import LiveSpec
from scientisttwo.live.runtime import Budget, DevelopmentRunner
from scientisttwo.store import Store, IntegrityError
from test_live_development import VECTOR_TEMPLATE, body_proposal, vector_body_setup, runtime_setup


def test_final_body_materialization_replays_crlf_exactly(tmp_path, monkeypatch):
    adapter = vector_body_setup(tmp_path, source=VECTOR_TEMPLATE.replace("\n", "\r\n"))
    config = LiveSpec(
        native=adapter.native,
        development=adapter.config,
        model="not-invoked",
        max_calls=1,
        campaign_seconds=300,
        implementation=implementation_identity(),
    )
    store = Store(tmp_path / "campaign")
    budget = Budget(store, config)
    runner = DevelopmentRunner(store, config, adapter, budget)
    store.set("live_images", {"execution": "sha256:fixture", "grading": "sha256:fixture"})
    proposal = body_proposal("return features_batch.copy()")
    expected = adapter.materialize(proposal)
    calls = []

    class FinalDouble:
        def __init__(self, spec, *args, **kwargs):
            assert (spec.candidate_dir / "train.py").read_bytes() == expected.files[
                0
            ].content.encode()
            calls.append(spec)

        def run(self):
            if len(calls) == 1:
                raise KeyboardInterrupt
            return SimpleNamespace(
                model_dump=lambda **kwargs: {"status": "completed", "elapsed_seconds": 0}
            )

        def close(self):
            pass

    monkeypatch.setattr("scientisttwo.live.runtime.NativeRunner", FinalDouble)
    with pytest.raises(KeyboardInterrupt):
        runner.final("selected", proposal)
    assert runner.final("selected", proposal)["status"] == "completed"
    assert calls[0].candidate_manifest == calls[1].candidate_manifest
    evidence = store.json(
        store.db.execute(
            "SELECT output_ref FROM calls WHERE key='materialization/final/selected'"
        ).fetchone()[0]
    )
    assert store.json(evidence["proposal_ref"]) == proposal.model_dump()
    assert store.json(evidence["code_ref"]) == expected.model_dump()
    store.close()


def test_materialization_changed_on_resume_is_rejected(tmp_path):
    adapter = vector_body_setup(tmp_path)
    config = LiveSpec(
        native=adapter.native,
        development=adapter.config,
        model="not-invoked",
        max_calls=1,
        campaign_seconds=300,
        implementation=implementation_identity(),
    )
    store = Store(tmp_path / "campaign")
    runner = DevelopmentRunner(store, config, adapter, Budget(store, config))
    proposal = body_proposal("return features_batch.copy()")
    runner.materialize("same", proposal)
    changed = adapter.materialize(body_proposal("return features_batch + 1"))
    adapter.materialize = lambda _: changed
    with pytest.raises(IntegrityError, match="materialization changed"):
        runner.materialize("same", proposal)
    store.close()


def test_audit_binds_observation_to_materialized_source(tmp_path, monkeypatch):
    store, adapter, runner, _ = runtime_setup(tmp_path, monkeypatch)
    observation = runner.execute("first", adapter.baseline())
    assert observation.status == "success"
    other = adapter.baseline().model_copy(update={"description": "Another source receipt"})
    forged = observation.model_copy(update={"code_ref": store.put_json(other)})
    with store.db:
        store.db.execute(
            "UPDATE calls SET output_ref=? WHERE key='development/first'", (store.put_json(forged),)
        )
    result = audit_live(store.root)
    assert result["status"] == "fail"
    assert any("materialization receipt" in error for error in result["errors"])
    store.close()
