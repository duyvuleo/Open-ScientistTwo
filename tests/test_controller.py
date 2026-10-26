import json
import time

import pytest

from scientisttwo.backends import FixtureBackend
from scientisttwo.controller import Controller, initialize
from scientisttwo.models import Profile, Stage, TaskSpec
from scientisttwo.store import Store
from helpers import ScalarTask, SyntheticRunner


def campaign(tmp_path, **options):
    task = TaskSpec(
        adapter="scalar-test",
        datasets=["quadratic"],
        subset_datasets=["quadratic"],
        seeds=[0],
        primary_metric="loss",
        metric_directions={"loss": "minimize"},
        guardrails={},
    )
    adapter = ScalarTask(**options)
    store = Store(tmp_path / "run")
    profile = Profile()
    initialize(store, task, profile, adapter)
    runner = SyntheticRunner(store)
    controller = Controller(
        store, task, profile, adapter, FixtureBackend(adapter.fixture_response), runner
    )
    return controller, runner, store, adapter


def test_general_minimization_task_full_workflow(tmp_path):
    controller, runner, store, adapter = campaign(tmp_path)
    state = controller.run()
    assert state.outcome == "candidate_supported"
    assert store.get("best")["result"]["rows"][0]["metrics"] == {"loss": 7}
    assert any("rebuttal" in key for key in runner.executed)
    assert runner.executed[-2:] == ["final/baseline", "final/candidate"]
    assert all('"split": "test"' not in json.dumps(inputs) for _, inputs in adapter.received)
    assert "synthetic_test" in (store.root / "report.md").read_text()
    before = list(runner.executed)
    assert controller.run().stage == Stage.COMPLETE
    assert runner.executed == before


def test_no_winner_is_valid_negative_outcome(tmp_path):
    controller, runner, store, _ = campaign(tmp_path, reject=True)
    assert controller.run().outcome == "no_improvement"
    assert store.get("best") is None
    assert not any(key.startswith("final/") for key in runner.executed)


@pytest.mark.parametrize("success", [True, False])
def test_meta_refinement_preserves_or_replaces_incumbent(tmp_path, success):
    controller, runner, store, _ = campaign(tmp_path, refine_meta=True, improved_revision=success)
    state = controller.run()
    assert state.outcome == "candidate_supported"
    best = store.get("best")["idea"]["id"]
    assert ("meta-refine" in best) == success
    assert any("cycle/1/ablation" in key for key in runner.executed) == success


def test_ablation_refinement_replans_after_update(tmp_path):
    controller, runner, store, _ = campaign(tmp_path, refine_ablation=True)
    controller.run()
    assert any("ablation/1/" in key for key in runner.executed)
    assert "ablation-0-refine" in store.get("best")["idea"]["id"]


def test_engineering_budget_and_lineage(tmp_path):
    controller, runner, store, _ = campaign(tmp_path, engineering=True)
    assert controller.run().outcome == "candidate_supported"
    assert any("/subset/1" in key for key in runner.executed)
    assert all("/subset/2" not in key for key in runner.executed)
    assert store.get("best")["idea"]["parent_ids"]


def test_failed_baseline_is_not_hypothesis_rejection(tmp_path):
    controller, runner, store, _ = campaign(tmp_path)
    runner.fail_baseline = True
    assert controller.run().outcome == "baseline_failed"
    assert len(runner.executed) == 2
    assert not controller.traces


def test_interrupt_resume_reuses_completed_experiments(tmp_path):
    controller, runner, store, adapter = campaign(tmp_path)
    runner.interrupt_at = "search/0/1/subset/0"
    with pytest.raises(KeyboardInterrupt):
        controller.run()
    assert store.state().status == "interrupted"
    before = list(runner.executed)
    resumed = Controller(
        store,
        controller.task,
        controller.profile,
        adapter,
        FixtureBackend(adapter.fixture_response),
        runner,
    )
    assert resumed.run().outcome == "candidate_supported"
    assert all(runner.executed.count(key) == 1 for key in before)
    count = store.db.execute("SELECT count(*) FROM events").fetchone()[0]
    resumed.run()
    assert store.db.execute("SELECT count(*) FROM events").fetchone()[0] == count


def test_budget_stops_before_any_new_calls(tmp_path):
    controller, runner, store, adapter = campaign(tmp_path)
    store.set("deadline", time.time() - 1)
    assert controller.run().outcome == "budget_exhausted"
    assert not runner.executed and not adapter.received


def test_offline_rejects_live_backend(tmp_path):
    controller, runner, store, adapter = campaign(tmp_path)
    backend = FixtureBackend(adapter.fixture_response)
    backend.synthetic = False
    with pytest.raises(ValueError, match="forbids live"):
        Controller(store, controller.task, controller.profile, adapter, backend, runner)


def test_paper_configuration():
    p = Profile.named("paper")
    assert (
        p.limitation_rounds,
        p.seed_ideas,
        p.search_rounds,
        p.engineering_rounds,
        p.ablation_rounds,
        p.peer_rounds,
        p.meta_rounds,
    ) == (16, 5, 4, 2, 1, 2, 1)
    assert p.live_calls is False


def test_critic_cannot_override_guardrails(tmp_path):
    from scientisttwo.models import Comparison

    controller, _, _, adapter = campaign(tmp_path)
    adapter.compare = lambda *args: Comparison(
        valid=True, improved=True, guardrails_pass=False, delta=1, reason="Guardrail failed"
    )
    assert controller.run().outcome == "no_improvement"
    assert all(t.decision.decision == "Bad" for t in controller.traces)


def test_same_controller_can_resume_after_interrupt(tmp_path):
    controller, runner, store, _ = campaign(tmp_path)
    runner.interrupt_at = "search/0/1/subset/0"
    with pytest.raises(KeyboardInterrupt):
        controller.run()
    assert controller.run().outcome == "candidate_supported"
    assert len(runner.executed) == len(set(runner.executed))
