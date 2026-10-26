"""Synthetic timers and operation records, never measured compute claims."""

import time
from types import SimpleNamespace

import pytest

from scientisttwo.live.runtime import AmbiguousCall, Budget, LiveBudgetExhausted
from scientisttwo.store import IntegrityError, Store
from scientisttwo.studies.budget import StudyBudget


def setup(tmp_path, *, seconds=10.0, final_seconds=20.0, **limits):
    config = SimpleNamespace(campaign_seconds=60, max_calls=5, max_experiments=5,
                             token_stop_threshold=100, **limits)
    store = Store(tmp_path)
    budget = StudyBudget(store, config, max_experiment_seconds=seconds,
                         max_final_seconds=final_seconds, max_final_wall_seconds=90)
    now = [0.0]
    budget._clock = lambda: now[0]
    return store, budget, now


def complete(budget, key, seconds, *, status="success", request=None):
    budget.reserve(key, "experiment", request or {})
    budget.finish(key, status, {"synthetic": True}, seconds)


def release(store, budget):
    ref = store.put_json({"status": "BOTH_ARMS_FROZEN", "synthetic": True})
    store.set("study_final_release_ref", ref)
    return budget.activate_final_grant(ref, time.time() + 80)


def test_active_elapsed_caps_timeout_without_double_charging_committed_seconds(tmp_path):
    _, budget, clock = setup(tmp_path)
    with budget.experiment_scope():
        budget.reserve("experiment/a", "experiment", {})
        clock[0] = 3
        assert budget.remaining() == pytest.approx(7)
        budget.finish("experiment/a", "success", {}, 3.0)
        assert budget.remaining() == pytest.approx(7)
    assert budget.compute_summary()["buckets"]["development"]["charged_seconds"] == 3
    assert budget.remaining() > 10  # Model/reporting time uses campaign deadline.


def test_scoring_diagnostics_are_capped_without_an_explicit_runner_scope(tmp_path):
    _, budget, clock = setup(tmp_path)
    budget.reserve("diagnostic/classification", "experiment", {})
    clock[0] = 9.5
    assert budget.remaining() == pytest.approx(0.5)
    clock[0] = 10.5
    with pytest.raises(LiveBudgetExhausted, match="wall-time"):
        budget.remaining()
    budget.finish("diagnostic/classification", "failed", {}, 10.5)
    assert budget.remaining() > 10


def test_failed_attempts_consume_compute_but_allow_reporting_model_calls(tmp_path):
    _, budget, _ = setup(tmp_path)
    complete(budget, "experiment/a", 10, status="timeout")
    with pytest.raises(LiveBudgetExhausted, match="allowance"):
        budget.reserve("experiment/retry", "experiment", {})
    budget.reserve("model/report", "model", {})
    budget.finish("model/report", "success", {}, 1, 15)
    budget.reserve("probe/pdf", "probe", {})
    budget.finish("probe/pdf", "success", {}, 1)
    assert budget.compute_summary()["buckets"]["development"]["charged_seconds"] == 10


def test_completed_experiment_replay_is_allowed_after_compute_exhaustion(tmp_path):
    _, budget, _ = setup(tmp_path)
    complete(budget, "experiment/a", 10)
    with budget.experiment_scope():
        prior = budget.reserve("experiment/a", "experiment", {})
        assert prior["status"] == "success"
    assert len(budget.summary()) == 1


def test_limits_are_immutable_on_resume_and_cannot_be_added_retroactively(tmp_path):
    store, budget, _ = setup(tmp_path)
    with pytest.raises(IntegrityError, match="cannot change"):
        StudyBudget(store, budget.config, max_experiment_seconds=20, max_final_seconds=20,
                    max_final_wall_seconds=90)
    other = Store(tmp_path / "other")
    basic = Budget(other, budget.config)
    basic.reserve("experiment/a", "experiment", {})
    with pytest.raises(IntegrityError, match="untracked"):
        StudyBudget(other, budget.config, max_experiment_seconds=10, max_final_seconds=20,
                    max_final_wall_seconds=90)


def test_unknown_completion_blocks_new_experiment_and_is_reported(tmp_path):
    store, budget, _ = setup(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with budget.experiment_scope():
            budget.reserve("experiment/unknown", "experiment", {})
            raise KeyboardInterrupt()
    resumed = StudyBudget(store, budget.config, max_experiment_seconds=10, max_final_seconds=20,
                          max_final_wall_seconds=90)
    with pytest.raises(AmbiguousCall, match="unresolved"):
        resumed.reserve("experiment/new", "experiment", {})
    assert resumed.compute_summary()["unknown_operations"] == ["experiment/unknown"]
    assert resumed.compute_summary()["accounting_status"] == "incomplete"


def test_final_compute_separate_per_label_and_from_development(tmp_path):
    store, budget, _ = setup(tmp_path)
    complete(budget, "experiment/search", 10)
    release(store, budget)
    with budget.experiment_scope(final_label="baseline"):
        complete(budget, "final/baseline-seed-0", 20)
    with budget.experiment_scope(final_label="selected"):
        budget.reserve("final/selected-seed-0", "experiment", {})
        assert budget.remaining() == pytest.approx(20)
        budget.finish("final/selected-seed-0", "success", {}, 4)
    summary = budget.compute_summary()["buckets"]
    assert summary["development"]["charged_seconds"] == 10
    assert summary["final/baseline"]["charged_seconds"] == 20
    assert summary["final/selected"]["remaining_seconds"] == 16


def test_final_aggregate_is_not_charged_or_blocked_by_exhausted_compute(tmp_path):
    store, budget, _ = setup(tmp_path)
    release(store, budget)
    complete(budget, "final/baseline-seed-0", 20)
    budget.reserve("final/baseline", "experiment", {"execution_type": "aggregation"})
    with pytest.raises(IntegrityError, match="zero charged"):
        budget.finish("final/baseline", "success", {}, 20)
    budget.finish("final/baseline", "success", {}, 0)
    assert budget.compute_summary()["buckets"]["final/baseline"]["charged_seconds"] == 20


def test_final_window_is_separate_without_resetting_expired_campaign_deadline(tmp_path):
    store, budget, _ = setup(tmp_path)
    expired = time.time() - 100
    store.set("live_deadline", expired)
    with pytest.raises(LiveBudgetExhausted, match="Campaign"):
        budget.remaining()
    activation = release(store, budget)
    assert budget.remaining() > 60
    assert store.get("live_deadline") == expired
    assert budget.activate_final_grant(activation["grant_ref"], activation["deadline"]) == activation
    with pytest.raises(IntegrityError, match="extended"):
        budget.activate_final_grant(activation["grant_ref"], activation["deadline"] + 1)


def test_final_release_blocks_new_models_development_and_diagnostics(tmp_path):
    store, budget, _ = setup(tmp_path)
    release(store, budget)
    for key, kind in (("model/new", "model"), ("experiment/new", "experiment"),
                      ("diagnostic/new", "experiment")):
        with pytest.raises(IntegrityError, match="after final"):
            budget.reserve(key, kind, {})
    budget.reserve("probe/final-pdf", "probe", {})
    budget.finish("probe/final-pdf", "success", {}, 1)


def test_final_seed_space_and_activation_are_mandatory(tmp_path):
    store, budget, _ = setup(tmp_path)
    with pytest.raises(IntegrityError, match="activation"):
        budget.reserve("final/selected-seed-0", "experiment", {})
    with pytest.raises(ValueError, match="Only fixed final"):
        budget.reserve("experiment/free", "experiment", {"execution_type": "aggregation"})
    release(store, budget)
    with pytest.raises(ValueError, match="fixed seeds"):
        budget.reserve("final/selected-seed-3", "experiment", {})


def test_final_seed_count_does_not_spend_search_operation_ceiling(tmp_path):
    store, budget, _ = setup(tmp_path)
    budget.config.max_experiments = 1
    complete(budget, "experiment/a", 1)
    release(store, budget)
    for seed in range(3):
        complete(budget, f"final/selected-seed-{seed}", 1)
    assert len(budget.summary()) == 4


def test_nested_scope_does_not_double_active_elapsed(tmp_path):
    _, budget, clock = setup(tmp_path)
    with budget.experiment_scope():
        clock[0] = 2
        with budget.experiment_scope():
            clock[0] = 3
            assert budget.remaining() == pytest.approx(7)


@pytest.mark.parametrize("limit", [0, -1, True, float("inf"), float("nan")])
def test_invalid_time_limits_rejected(tmp_path, limit):
    with pytest.raises(ValueError, match="finite and positive"):
        setup(tmp_path, seconds=limit)
