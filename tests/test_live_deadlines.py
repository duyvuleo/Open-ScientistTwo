"""Synthetic-clock deadline tests: no research agents or benchmark performance claims."""
import json
import time

import pytest

from scientisttwo.environment_probe import probe_runtime
from scientisttwo.live.paired_evaluation import execute_paired
from scientisttwo.live.runtime import Budget, LiveBudgetExhausted
from scientisttwo.live.suite import SuiteBudget
from scientisttwo.process import ProcessResult
from scientisttwo.store import Store
from test_live_development import runtime_setup
from test_live_suite import specification
from test_paired_evaluation import setup as paired_setup, synthetic_pair
from test_staged_evaluation import StagedTransportDouble, staged_development


def deadline(store, monkeypatch, seconds=1):
    clock = [time.time()]
    store.set("live_deadline", clock[0] + seconds)
    monkeypatch.setattr("scientisttwo.live.runtime.time.time", lambda: clock[0])
    return clock


def test_executor_clips_campaign_and_shared_time_but_allows_cleanup(tmp_path, monkeypatch):
    store, _, runner, _ = runtime_setup(tmp_path, monkeypatch)
    suite_store = Store(tmp_path / "suite")
    try:
        clock = deadline(store, monkeypatch, seconds=10)
        suite = SuiteBudget(suite_store, specification(tmp_path, count=1), clock=lambda: clock[0])
        suite_store.set("suite_deadline", clock[0] + 0.25)
        runner.budget = Budget(store, runner.config, shared=suite.for_assignment("trial-0"))
        timeouts = []
        runner.raw_executor = lambda argv, timeout, **kwargs: (timeouts.append(timeout) or ProcessResult(0, "", "", 0))
        runner.executor([*runner.prefix, "image", "inspect", "rm"], 30)
        assert timeouts[-1] == pytest.approx(0.25)
        clock[0] += 1
        with pytest.raises(LiveBudgetExhausted):
            runner.executor([*runner.prefix, "run", "image"], 30)
        runner.executor([*runner.prefix, "rm", "--force", "owned-container"], 15)
        assert timeouts == [0.25, 15]
    finally:
        store.close()
        suite_store.close()


def test_deadline_between_seeds_commits_attempt_and_never_launches_next_seed(tmp_path, monkeypatch):
    store, adapter, runner, fake = runtime_setup(tmp_path, monkeypatch)
    try:
        clock = deadline(store, monkeypatch)
        def execute(argv, timeout, **kwargs):
            result = fake(argv, timeout, **kwargs)
            if "run" in argv:
                assert timeout <= 1
                clock[0] += 2
            return result
        runner.raw_executor = execute
        with pytest.raises(LiveBudgetExhausted):
            runner.execute("deadline", adapter.baseline())
        assert len([argv for argv, _ in fake.calls if "run" in argv]) == 1
        row = store.db.execute("SELECT * FROM live_operations WHERE key='experiment/deadline'").fetchone()
        assert row["status"] == "timeout" and row["result_ref"] is not None
        assert store.json(row["result_ref"])["metrics"] == {}
        before = len(fake.calls)
        assert runner.execute("deadline", adapter.baseline()).status == "timeout"
        assert len(fake.calls) == before
    finally:
        store.close()


def test_deadline_after_inference_preserves_completed_stage_without_starting_score(tmp_path, monkeypatch):
    fake = StagedTransportDouble()
    store, runner, adapter = staged_development(tmp_path, monkeypatch, executor=fake)
    try:
        clock = deadline(store, monkeypatch)
        def execute(argv, timeout, **kwargs):
            result = fake(argv, timeout, **kwargs)
            if "run" in argv and fake.stages[-1][0] == "inference":
                clock[0] += 2
            return result
        runner.raw_executor = execute
        with pytest.raises(LiveBudgetExhausted):
            runner.execute("deadline", adapter.baseline())
        assert [stage for stage, _ in fake.stages] == ["train", "inference"]
        assert store.get("pipeline-stage/deadline/0/score/started") is None
        assert store.db.execute("SELECT output_ref FROM calls WHERE key='pipeline-stage/deadline/0/inference'").fetchone()
        assert store.db.execute("SELECT status FROM live_operations WHERE key='experiment/deadline'").fetchone()[0] == "timeout"
    finally:
        store.close()


def test_final_budget_stop_finishes_reserved_operation(tmp_path, monkeypatch):
    fake = StagedTransportDouble()
    store, runner, adapter = staged_development(tmp_path, monkeypatch, executor=fake)
    try:
        clock = deadline(store, monkeypatch)
        def execute(argv, timeout, **kwargs):
            result = fake(argv, timeout, **kwargs)
            if "run" in argv:
                clock[0] += 2
            return result
        runner.raw_executor = execute
        with pytest.raises(LiveBudgetExhausted):
            runner.final("selected", adapter.baseline())
        row = store.db.execute("SELECT * FROM live_operations WHERE key='final/selected'").fetchone()
        assert row["status"] == "timeout" and store.json(row["result_ref"])["status"] == "timeout"
        assert [stage for stage, _ in fake.stages] == ["train"]
    finally:
        store.close()


def test_paired_seed_deadline_commits_partial_evidence_and_operation(tmp_path, monkeypatch):
    store, adapter, runner = paired_setup(tmp_path, monkeypatch)
    try:
        clock = deadline(store, monkeypatch)
        calls = []
        def pair(*args, **kwargs):
            calls.append(kwargs["seed"])
            assert kwargs["timeout"] <= 1
            result = synthetic_pair(*args, **kwargs)
            clock[0] += 2
            return result
        with pytest.raises(LiveBudgetExhausted):
            execute_paired(runner, "deadline", adapter.baseline(), run_pair=pair)
        assert calls == [0]
        row = store.db.execute("SELECT * FROM live_operations WHERE key='experiment/deadline'").fetchone()
        result = store.json(row["result_ref"])
        assert row["status"] == "timeout" and len(result["timing_refs"]) == 1
        assert result["metrics"] == {}
    finally:
        store.close()


def test_runtime_probe_is_bounded_and_accounted_in_campaign_and_suite(tmp_path, monkeypatch):
    store, _, runner, _ = runtime_setup(tmp_path, monkeypatch)
    suite_store = Store(tmp_path / "suite")
    try:
        clock = deadline(store, monkeypatch)
        suite = SuiteBudget(suite_store, specification(tmp_path, count=1, max_experiments=1), clock=lambda: clock[0])
        suite_store.set("suite_deadline", clock[0] + 0.5)
        runner.budget = Budget(store, runner.config, shared=suite.for_assignment("trial-0"))
        timeouts = []
        def execute(argv, timeout, **kwargs):
            timeouts.append(timeout)
            report = {"status": "pass", "platform": "synthetic probe fixture", "python": "3.12",
                      "checks": [{"name": "cpu_identity", "status": "pass", "value": {"fixture": "synthetic"}}]}
            return ProcessResult(0, json.dumps(report), "", 0.01)
        runner.raw_executor = execute
        assert probe_runtime(runner, "cpu-tabular")["status"] == "verified"
        assert timeouts == [pytest.approx(0.5)]
        local, shared = runner.budget.summary(), suite.summary()
        assert len(local) == len(shared) == 1
        assert local[0]["kind"] == shared[0]["kind"] == "probe"
        assert local[0]["status"] == shared[0]["status"] == "success"
        suite.ensure_available("experiment")  # Infrastructure work does not consume a scientific experiment slot.
    finally:
        store.close()
        suite_store.close()


def test_probe_expiry_is_recorded_without_leaving_a_running_reservation(tmp_path, monkeypatch):
    store, _, runner, _ = runtime_setup(tmp_path, monkeypatch)
    try:
        def fail(*args):
            raise LiveBudgetExhausted("Synthetic probe deadline")
        monkeypatch.setattr("scientisttwo.environment_probe._run_probe", fail)
        with pytest.raises(LiveBudgetExhausted):
            probe_runtime(runner, "cpu-tabular")
        row = runner.budget.summary()[-1]
        assert row["kind"] == "probe" and row["status"] == "failed" and row["result_ref"]
    finally:
        store.close()
