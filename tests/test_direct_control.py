"""Synthetic control-loop traces only; these scores are never experimental claims."""

import pytest

from scientisttwo.backends import BackendError
from scientisttwo.benchmarks.models import NativeSpec
from scientisttwo.live.adapters import implementation_identity
from scientisttwo.live.models import Candidate, DevelopmentSpec, LiveSpec, Observation, SourceFile
from scientisttwo.live.runtime import Budget
from scientisttwo.models import AgentResult
from scientisttwo.store import IntegrityError, Store, canonical, digest
from scientisttwo.studies.direct_control import DirectControl, DirectProposal


def candidate(value):
    return Candidate(files=[SourceFile(path="train.py", content=f"# synthetic trace {value}")],
                     command=["python", "train.py"], description="Synthetic, never executed")


class Adapter:
    def baseline(self):
        return candidate(0.5)

    def context(self):
        return {"editable_files": ["train.py"], "synthetic": True}

    def identity(self):
        return {"adapter": "synthetic-direct-control-test"}

    def live_readiness(self):
        return {"ready": True}


class Backend:
    name = "fixture"
    synthetic = True

    def __init__(self, actions):
        self.actions, self.calls = iter(actions), []

    def invoke(self, role, inputs, schema):
        assert role == "direct_coding"
        assert schema is DirectProposal
        self.calls.append(inputs)
        action = next(self.actions)
        if isinstance(action, BaseException):
            raise action
        return AgentResult(payload=action, backend=self.name, model="scripted-fixture-v1",
                           synthetic=self.synthetic, usage={"input_tokens": 10, "output_tokens": 5})


class Runner:
    def __init__(self, store, budget):
        self.store, self.budget = store, budget
        self.executions, self.preflights = [], 0
        self.fail_values = set()
        self.unknown_value = None
        self.short_seeds = False
        self.change_protocol = False

    def preflight(self):
        self.preflights += 1

    def execute(self, key, code, phase, seeds):
        request = {"code": code.model_dump(), "phase": phase, "seeds": seeds}
        prior = self.budget.reserve("experiment/" + key, "experiment", request)
        if prior is not None:
            if prior["result_ref"]:
                return Observation.model_validate(self.store.json(prior["result_ref"]))
            return Observation(id=key, phase=phase, status="interrupted", protocol_hash="unknown",
                               code_ref=self.store.put_json(code), metric="accuracy")
        self.executions.append((key, phase, seeds))
        value = float(code.files[0].content.split()[-1])
        if value == self.unknown_value:
            raise KeyboardInterrupt("Synthetic abrupt stop after reservation")
        failed = value in self.fail_values
        actual_seeds = seeds[:1] if self.short_seeds and phase == "full" else seeds
        result = Observation(
            id=key, phase=phase, status="execution_failed" if failed else "success",
            protocol_hash=("wrong" if self.change_protocol and value != 0.5
                           else digest(canonical([phase, seeds]))),
            code_ref=self.store.put_json(code), metric="accuracy",
            value=None if failed else value, values=[] if failed else [value] * len(actual_seeds),
            prediction_refs=[] if failed else [self.store.put_json({"synthetic_seed": s})
                                               for s in actual_seeds],
            reason="Synthetic execution failed" if failed else "Synthetic success",
        )
        self.budget.finish("experiment/" + key, result.status, result, 0)
        return result

    def final(self, *args, **kwargs):
        raise AssertionError("Control must never open final evaluation")


def evaluate(value, phase="full"):
    return {"action": "evaluate", "reason": "Synthetic proposal", "phase": phase,
            "candidate": candidate(value).model_dump()}


def setup(tmp_path, actions, **settings):
    config = LiveSpec(
        native=NativeSpec(name="direct-test", benchmark="mlagentbench", task_id="vectorization",
                          repository=tmp_path / "repo", revision="a" * 40, mode="baseline",
                          data_provenance="upstream_no_data", public_dir=tmp_path / "public"),
        development=DevelopmentSpec(metric="accuracy", direction="maximize", seeds=[7, 11, 13]),
        model="scripted-fixture-v1", max_calls=10, max_experiments=10, campaign_seconds=60,
        implementation=implementation_identity(),
    ).model_copy(update=settings)
    store = Store(tmp_path / "run")
    budget, backend = Budget(store, config), Backend(actions)
    runner = Runner(store, budget)
    control = DirectControl(store, config, Adapter(), backend, runner)
    return control, store, backend, runner


def test_single_role_full_selection_and_no_final(tmp_path):
    control, store, backend, runner = setup(tmp_path, [evaluate(0.7), evaluate(0.6),
                                                     {"action": "stop", "reason": "Done"}])
    result = control.run(max_iterations=4)
    assert result.status == "frozen" and result.improved
    assert store.json(result.selected_observation_ref)["value"] == 0.7
    assert [item[2] for item in runner.executions] == [[7], [7, 11, 13], [7, 11, 13], [7, 11, 13]]
    assert len(backend.calls) == 3
    assert store.json(result.freeze_ref)["final_evaluation"] == "not_started"
    assert sum(item["tokens"] for item in result.usage) == 45
    assert store.verify() == []
    assert store.get("live_frozen") is None


def test_subset_success_cannot_replace_full_incumbent(tmp_path):
    control, store, _, _ = setup(tmp_path, [evaluate(0.99, "subset")])
    result = control.run(max_iterations=1)
    assert not result.improved
    assert store.json(result.selected_observation_ref)["value"] == 0.5


def test_execution_failure_is_feedback_and_counted(tmp_path):
    control, store, backend, runner = setup(tmp_path, [evaluate(0.9), evaluate(0.7)])
    runner.fail_values.add(0.9)
    result = control.run(max_iterations=2)
    history = store.json(result.history_ref)
    assert history[0]["status"] == "execution_failed" and not history[0]["selected"]
    assert backend.calls[1]["history"][0]["observation"]["status"] == "execution_failed"
    assert backend.calls[1]["last_attempt_candidate"] == candidate(0.9).model_dump()
    assert backend.calls[1]["incumbent"] == candidate(0.5).model_dump()
    assert len([op for op in result.usage if op["kind"] == "experiment"]) == 4
    assert result.improved


def test_completed_resume_has_no_new_calls_experiments_or_preflight(tmp_path):
    control, _, backend, runner = setup(tmp_path, [evaluate(0.7)])
    first = control.run(max_iterations=1)
    assert control.run(max_iterations=1) == first
    assert len(backend.calls) == 1 and len(runner.executions) == 3 and runner.preflights == 1


def test_changed_budget_or_iteration_bound_cannot_resume(tmp_path):
    control, _, _, _ = setup(tmp_path, [evaluate(0.7)])
    control.run(max_iterations=1)
    with pytest.raises(IntegrityError, match="inputs"):
        control.run(max_iterations=2)


def test_invalid_output_consumes_usage_and_can_receive_feedback(tmp_path):
    control, store, backend, _ = setup(tmp_path, [{"action": "stop", "reason": "Invalid",
                                                 "candidate": candidate(0.8).model_dump()},
                                                evaluate(0.7)])
    result = control.run(max_iterations=2)
    assert result.improved
    assert store.json(result.history_ref)[0]["status"] == "invalid_proposal"
    assert backend.calls[1]["history"][0]["error"]
    assert sum(op["tokens"] for op in result.usage) == 30
    assert any(op["status"] == "invalid_output" for op in result.usage)


def test_transport_retry_consumes_call_budget(tmp_path):
    control, _, backend, _ = setup(tmp_path, [BackendError("Synthetic transport error"), evaluate(0.8)],
                                  max_calls=1, model_retries=1)
    result = control.run(max_iterations=2)
    assert result.status == "frozen" and not result.improved
    assert "Budget exhausted" in result.reason and len(backend.calls) == 1
    assert len([op for op in result.usage if op["kind"] == "model"]) == 1


def test_token_stop_threshold_prevents_next_call(tmp_path):
    control, _, backend, _ = setup(tmp_path, [evaluate(0.8), evaluate(0.9)], token_stop_threshold=10)
    result = control.run(max_iterations=2)
    assert result.status == "frozen" and result.improved
    assert len(backend.calls) == 1 and "threshold" in result.reason


def test_experiment_limit_preserves_incumbent(tmp_path):
    control, store, _, runner = setup(tmp_path, [evaluate(0.8)], max_experiments=2)
    result = control.run(max_iterations=1)
    assert not result.improved and result.status == "frozen"
    assert store.json(result.selected_observation_ref)["value"] == 0.5
    assert len(runner.executions) == 2


def test_unknown_execution_is_never_repeated_or_frozen(tmp_path):
    control, _, backend, runner = setup(tmp_path, [evaluate(0.8)])
    runner.unknown_value = 0.8
    with pytest.raises(KeyboardInterrupt):
        control.run(max_iterations=1)
    result = control.run(max_iterations=1)
    assert result.status == "blocked" and result.freeze_ref is None
    assert len(runner.executions) == 3 and len(backend.calls) == 1
    assert "Unknown experiment" in result.reason


def test_unknown_model_completion_never_repeats(tmp_path):
    control, _, backend, runner = setup(tmp_path, [KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        control.run(max_iterations=1)
    result = control.run(max_iterations=1)
    assert result.status == "blocked" and result.freeze_ref is None
    assert len(backend.calls) == 1 and len(runner.executions) == 2
    assert "Unknown model" in result.reason


def test_incomparable_improvement_cannot_win(tmp_path):
    control, _, _, runner = setup(tmp_path, [evaluate(0.9)])
    runner.change_protocol = True
    assert not control.run(max_iterations=1).improved


def test_incomplete_seed_result_is_integrity_error(tmp_path):
    control, _, _, runner = setup(tmp_path, [])
    runner.short_seeds = True
    with pytest.raises(IntegrityError, match="matched seed"):
        control.run(max_iterations=1)


def test_live_enablement_is_explicit_and_readiness_required(tmp_path):
    control, store, backend, runner = setup(tmp_path, [])
    backend.synthetic = False
    with pytest.raises(ValueError, match="allow_live"):
        DirectControl(store, control.config, Adapter(), backend, runner)
    adapter = Adapter()
    adapter.live_readiness = lambda: {"ready": False, "errors": ["Synthetic blocked gate"]}
    result = DirectControl(store, control.config, adapter, backend, runner, allow_live=True).run(
        max_iterations=1)
    assert result.status == "blocked" and not backend.calls and not runner.executions


def test_modified_evidence_fails_resume(tmp_path):
    control, store, _, _ = setup(tmp_path, [evaluate(0.8)])
    result = control.run(max_iterations=1)
    (store.root / result.freeze_ref).write_text("tampered")
    with pytest.raises(IntegrityError, match="modified"):
        control.run(max_iterations=1)


def test_control_cannot_share_research_store_or_unequal_budget(tmp_path):
    control, store, backend, runner = setup(tmp_path, [])
    other_store = Store(tmp_path / "other")
    with pytest.raises(ValueError, match="own store"):
        DirectControl(other_store, control.config, Adapter(), backend, runner)
    store.set("live_spec_ref", "research-existing")
    with pytest.raises(IntegrityError, match="separate run store"):
        control.run(max_iterations=1)


def test_oversized_context_blocks_without_repeated_model_attempts(tmp_path):
    control, _, backend, runner = setup(tmp_path, [], max_prompt_chars=1000)
    result = control.run(max_iterations=3)
    assert result.status == "blocked" and result.freeze_ref is None
    assert "max_prompt_chars" in result.reason
    assert not backend.calls and len(runner.executions) == 2


def test_baseline_failure_is_not_readiness_or_scientific_success(tmp_path):
    control, _, backend, runner = setup(tmp_path, [])
    runner.fail_values.add(0.5)
    result = control.run(max_iterations=1)
    assert result.status == "blocked" and not result.improved and result.freeze_ref is None
    assert "Baseline failed" in result.reason
    assert not backend.calls and len(runner.executions) == 1


def test_refuses_non_three_seed_control(tmp_path):
    control, store, backend, runner = setup(tmp_path, [])
    config = control.config.model_copy(update={"development": DevelopmentSpec(seeds=[1])})
    with pytest.raises(ValueError, match="three matched"):
        DirectControl(store, config, Adapter(), backend, runner)
