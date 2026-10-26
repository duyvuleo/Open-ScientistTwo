"""Explicitly synthetic suite traces; never execute models or candidate code."""

import json

import pytest
from typer.testing import CliRunner

from scientisttwo.benchmarks.models import NativeSpec
from scientisttwo.live.models import DevelopmentSpec, LiveSpec
from scientisttwo.live.runtime import AmbiguousCall, Budget, LiveBudgetExhausted
from scientisttwo.live.suite import (
    SuiteAssignment,
    SuiteBudget,
    SuiteSpec,
    check_suite,
    inspect_suite,
    run_suite,
    suite_identity,
)
from scientisttwo.live.suite_cli import app
from scientisttwo.store import IntegrityError, Store


def specification(tmp_path, count=3, **limits):
    native = NativeSpec(
        name="synthetic",
        benchmark="mlagentbench",
        task_id="vectorization",
        repository=tmp_path / "repo",
        revision="a" * 40,
        mode="baseline",
        data_provenance="upstream_no_data",
        public_dir=tmp_path / "public",
    )
    campaign = LiveSpec(
        native=native,
        development=DevelopmentSpec(),
        model="synthetic-fixture-only",
        max_calls=10,
        max_experiments=10,
        campaign_seconds=100,
    )
    return SuiteSpec(
        name="synthetic-suite",
        wall_seconds=limits.get("wall_seconds", 500),
        max_calls=limits.get("max_calls", 20),
        max_experiments=limits.get("max_experiments", 20),
        token_stop_threshold=limits.get("token_stop_threshold"),
        implementation=suite_identity(),
        assignments=[
            SuiteAssignment(id=f"trial-{r}", repetition=r, campaign=campaign) for r in range(count)
        ],
    )


class SyntheticExecutor:
    def __init__(self, outcomes=None, calls=1, tokens=7):
        self.executed = []
        self.outcomes = outcomes or {}
        self.calls, self.tokens = calls, tokens

    def __call__(self, config, output, shared):
        self.executed.append(output.name)
        for i in range(self.calls):
            key = f"model/{i}"
            prior = shared.reserve(key, "model", {"synthetic": True, "i": i})
            if prior:
                raise AmbiguousCall("Synthetic unknown attempt")
            shared.finish(key, "success", {"synthetic": True}, 0.1, self.tokens)
        shared.reserve("experiment/0", "experiment", {"synthetic": True})
        shared.finish("experiment/0", "success", {"synthetic": True}, 0.2)
        outcome = self.outcomes.get(output.name, "development_improvement")
        return {
            "status": "blocked" if outcome == "execution_error" else "complete",
            "outcome": outcome,
            "reason": "Synthetic trace only",
        }


def test_frozen_denominator_negative_failure_and_resume(tmp_path):
    config = specification(tmp_path)
    executor = SyntheticExecutor({"trial-1": "no_improvement", "trial-2": "execution_error"})
    output = tmp_path / "suite"
    result = run_suite(config, output, allow_live=True, executor=executor)
    assert result["status"] == "blocked"
    assert result["improvement_denominator"] == 3 and result["development_improvements"] == 1
    assert result["improvement_rate_over_all_declared_assignments"] == 1 / 3
    assert result["model_attempts"] == result["experiment_attempts"] == 3
    assert result["observed_tokens"] == 21
    second = run_suite(config, output, allow_live=True, executor=executor)
    assert second == result and executor.executed == ["trial-0", "trial-1", "trial-2"]
    saved, readback = inspect_suite(output)
    assert saved == config and readback == result


@pytest.mark.parametrize(
    "limit,expected", [("max_calls", 1), ("max_experiments", 1), ("token_stop_threshold", 7)]
)
def test_aggregate_limits_keep_unstarted_assignments_in_denominator(tmp_path, limit, expected):
    config = specification(tmp_path, **{limit: expected})
    executor = SyntheticExecutor()
    result = run_suite(config, tmp_path / "suite", allow_live=True, executor=executor)
    assert executor.executed == ["trial-0"]
    assert result["status"] == "budget_exhausted"
    assert result["improvement_denominator"] == 3
    assert [r["status"] for r in result["assignments"]] == [
        "complete",
        "not_started",
        "not_started",
    ]


def test_global_call_limit_stops_an_active_assignment(tmp_path):
    config = specification(tmp_path, max_calls=2)
    executor = SyntheticExecutor(calls=4)
    result = run_suite(config, tmp_path / "suite", allow_live=True, executor=executor)
    assert result["model_attempts"] == 2
    assert result["assignments"][0]["outcome"] == "budget_exhausted"
    assert executor.executed == ["trial-0"]


def test_shared_deadline_and_completion_reconciliation(tmp_path):
    config = specification(tmp_path, count=1, wall_seconds=5)
    store = Store(tmp_path / "budget")
    now = [100.0]
    suite = SuiteBudget(store, config, clock=lambda: now[0])
    shared = suite.for_assignment("trial-0")
    assert shared.remaining() == 5
    shared.reserve("first", "model", {})
    now[0] += 6
    with pytest.raises(LiveBudgetExhausted, match="wall-clock"):
        shared.remaining()
    # A committed child receipt can be mirrored after a deadline without a new reservation.
    shared.finish("first", "success", {"synthetic": True}, 1, 10)
    shared.finish("first", "success", {"synthetic": True}, 1, 10)
    assert len(suite.summary()) == 1 and suite.summary()[0]["tokens"] == 10
    with pytest.raises(IntegrityError, match="Conflicting"):
        shared.finish("first", "success", {"synthetic": True}, 1, 11)
    with pytest.raises(LiveBudgetExhausted):
        shared.reserve("second", "model", {})
    store.close()


def test_attempted_failure_consumes_budget_and_cannot_change_request(tmp_path):
    config = specification(tmp_path, count=1, max_calls=1)
    store = Store(tmp_path / "budget")
    shared = SuiteBudget(store, config).for_assignment("trial-0")
    shared.reserve("call", "model", {"version": 1})
    shared.finish("call", "failed", {"synthetic": True}, 2)
    with pytest.raises(IntegrityError, match="inputs changed"):
        shared.reserve("call", "model", {"version": 2})
    with pytest.raises(LiveBudgetExhausted):
        shared.reserve("retry", "model", {"version": 1})
    store.close()


def test_crash_after_shared_reservation_does_not_repeat_unknown_call(tmp_path):
    config = specification(tmp_path, count=1)
    output = tmp_path / "suite"

    def crash(config, output, shared):
        shared.reserve("unknown", "model", {"synthetic": True})
        raise SystemExit("Synthetic abrupt process death")

    with pytest.raises(SystemExit):
        run_suite(config, output, allow_live=True, executor=crash)
    executor = SyntheticExecutor()
    result = run_suite(config, output, allow_live=True, executor=executor)
    assert not executor.executed
    assert result["assignments"][0]["outcome"] == "ambiguous_operation"
    assert result["model_attempts"] == 1 and result["unresolved_operations"] == 1


def test_manifest_evidence_and_implementation_mutation_rejected(tmp_path):
    config = specification(tmp_path, count=1)
    output = tmp_path / "suite"
    run_suite(config, output, allow_live=True, executor=SyntheticExecutor())
    changed = config.model_copy(update={"max_calls": 1000})
    with pytest.raises(IntegrityError, match="Frozen suite changed"):
        run_suite(changed, output, allow_live=True, executor=SyntheticExecutor())
    changed = config.model_copy(update={"implementation": {"fake": "hash"}})
    with pytest.raises(IntegrityError, match="implementation changed"):
        run_suite(changed, output, allow_live=True, executor=SyntheticExecutor())
    artifact = next((output / "artifacts").glob("*.json"))
    artifact.write_text("{}")
    with pytest.raises(IntegrityError):
        inspect_suite(output)
    with pytest.raises(IntegrityError):
        run_suite(config, output, allow_live=True, executor=SyntheticExecutor())


def test_authorization_output_isolation_and_no_overwrite(tmp_path):
    config = specification(tmp_path, count=1)
    executor = SyntheticExecutor()
    with pytest.raises(ValueError, match="allow-live"):
        run_suite(config, tmp_path / "suite", executor=executor)
    assert not (tmp_path / "suite").exists() and not executor.executed
    for output in (tmp_path, tmp_path / "repo" / "output", tmp_path / "public"):
        with pytest.raises(ValueError, match="overlap"):
            run_suite(config, output, allow_live=True, executor=executor)
    output = tmp_path / "linked-suite"
    output.mkdir()
    (output / "campaigns").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        run_suite(config, output, allow_live=True, executor=executor)


def test_create_and_check_are_model_free_and_explicit(tmp_path):
    config = specification(tmp_path, count=1)
    source = tmp_path / "campaign.json"
    source.write_text(config.assignments[0].campaign.model_dump_json())
    target = tmp_path / "suite.json"
    args = [
        "create",
        str(target),
        "--campaign",
        str(source),
        "--repetitions",
        "2",
        "--max-calls",
        "50",
        "--max-experiments",
        "60",
        "--seconds",
        "900",
    ]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["live_model_calls"] == 0
    frozen = SuiteSpec.model_validate_json(target.read_text())
    assert len(frozen.assignments) == 2 and frozen.max_calls == 50
    source.write_text("{}")  # The suite embeds its input, rather than following a mutable path.
    checks = check_suite(frozen, checker=lambda c: {"ready": True})
    assert checks["ready"] and checks["live_model_calls"] == 0
    assert runner.invoke(app, args).exit_code != 0
    assert runner.invoke(app, ["resume", str(tmp_path / "run")]).exit_code != 0


def test_recovery_after_completed_assignment_does_not_repeat_it(tmp_path):
    config = specification(tmp_path, count=2)
    output = tmp_path / "suite"
    executed = []
    worker = SyntheticExecutor()

    def interrupted(config, output, shared):
        executed.append(output.name)
        if output.name == "trial-1":
            raise KeyboardInterrupt
        return worker(config, output, shared)

    with pytest.raises(KeyboardInterrupt):
        run_suite(config, output, allow_live=True, executor=interrupted)
    result = run_suite(config, output, allow_live=True, executor=worker)
    assert result["status"] == "complete"
    assert worker.executed == ["trial-0", "trial-1"]
    assert result["model_attempts"] == 2


def test_child_budget_mirrors_shared_caps_and_minimum_deadline(tmp_path):
    config = specification(tmp_path, count=2, max_calls=1, wall_seconds=5)
    store = Store(tmp_path / "suite-budget")
    suite = SuiteBudget(store, config, clock=lambda: 100)
    first = Store(tmp_path / "first")
    second = Store(tmp_path / "second")
    a = Budget(first, config.assignments[0].campaign, shared=suite.for_assignment("trial-0"))
    b = Budget(second, config.assignments[1].campaign, shared=suite.for_assignment("trial-1"))
    assert a.remaining() <= 5
    a.reserve("call", "model", {"synthetic": True})
    a.finish("call", "success", {"synthetic": True}, 0.1, 7)
    with pytest.raises(LiveBudgetExhausted):
        b.reserve("call", "model", {"synthetic": True})
    assert not b.summary() and len(suite.summary()) == 1
    for s in (first, second, store):
        s.close()


def test_child_reconciles_committed_local_result_after_shared_finish_crash(tmp_path):
    config = specification(tmp_path, count=1, wall_seconds=5)
    store = Store(tmp_path / "suite-budget")
    now = [100]
    suite = SuiteBudget(store, config, clock=lambda: now[0])
    shared = suite.for_assignment("trial-0")
    local = Store(tmp_path / "child")
    budget = Budget(local, config.assignments[0].campaign, shared=shared)
    request = {"synthetic": True}
    budget.reserve("call", "model", request)
    # Simulate the process dying after local completion committed but before its mirror.
    unshared = Budget(local, config.assignments[0].campaign)
    unshared.finish("call", "success", {"synthetic": True}, 0.1, 7)
    assert shared.summary()[0]["status"] == "running"
    now[0] += 6
    assert budget.reserve("call", "model", request)["status"] == "success"
    assert shared.summary()[0]["status"] == "success"
    assert shared.summary()[0]["tokens"] == 7
    budget.reserve("call", "model", request)
    assert len(shared.summary()) == 1
    local.close()
    store.close()


def test_shared_only_reservation_cannot_be_reissued_by_child(tmp_path):
    config = specification(tmp_path, count=1)
    store = Store(tmp_path / "suite-budget")
    shared = SuiteBudget(store, config).for_assignment("trial-0")
    local = Store(tmp_path / "child")
    budget = Budget(local, config.assignments[0].campaign, shared=shared)
    shared.reserve("call", "model", {"synthetic": True})
    with pytest.raises(AmbiguousCall):
        budget.reserve("call", "model", {"synthetic": True})
    assert not budget.summary()
    local.close()
    store.close()
