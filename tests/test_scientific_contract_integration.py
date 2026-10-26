"""Synthetic controller traces for declared actions and faithful parent editing.

No test in this module invokes a live model, a container, or real ML training.
The fixture runner's values are explicitly synthetic control-flow evidence.
"""

from copy import deepcopy
import json

import pytest

from scientisttwo.live.controller import LiveController
from scientisttwo.live.models import Candidate, SourceFile
from scientisttwo.live.output_contracts import AgentOutputError
from scientisttwo.live.parent_edits import ParentChangePlan
from scientisttwo.live.runtime import LiveBudgetExhausted
from scientisttwo.models import CriticDecision
from scientisttwo.store import digest
from test_live import Adapter, Agents, candidate, concern_responses, idea, setup


class Scripted(Agents):
    def __init__(self, **builders):
        super().__init__()
        self.builders = builders

    def respond(self, role, inputs):
        if role in self.builders:
            self.seen.append((role, inputs))
            return self.builders[role](inputs)
        return super().respond(role, inputs)


@pytest.fixture
def scenario(tmp_path, request):
    number = 0

    def make(agents=None, **settings):
        nonlocal number
        number += 1
        controller, store, backend, runner = setup(tmp_path / str(number), agents, **settings)
        request.addfinalizer(store.close)
        store.set("live_state", {"stage": "search", "status": "running"})
        return controller, store, backend, runner

    return make


def incumbent(controller, code=None):
    code = code or candidate(8)
    observation = controller.runner.execute("fixture-parent", code, "full")
    trial = controller.record("fixture-parent", idea("parent"), code, observation,
                              CriticDecision(decision="Good", reason="Synthetic fixture"))
    controller.best = trial
    controller.store.set("live_best", trial)
    return trial, observation


def plan(inputs, identifier="control", *, concern=0, seed_count=2, kind="paired_control", **updates):
    result = {
        "id": identifier,
        "concern_ids": [inputs["concerns"][concern]["id"]],
        "purpose": "Synthetic exact action contract",
        "seed_count": seed_count,
        "execution_kind": kind,
        "evaluation_split": "development",
        "diagnostics": [{"kind": "metric", "metric": "process_seconds"}],
        "changed_file": None if kind == "repeat" else "train.py",
        "candidate": deepcopy(inputs["incumbent"]["candidate"]) if kind == "repeat" else candidate(9).model_dump(),
    }
    result.update(updates)
    return result


def envelope(*plans):
    return {"plans": list(plans), "unaddressed_concerns": []}


def record(store, key):
    return store.json(store.get("supplementary/" + key))


def experiment_rows(store):
    return [dict(row) for row in store.db.execute("SELECT * FROM live_operations WHERE kind='experiment' ORDER BY rowid")]


def test_ten_requested_seeds_are_unfulfilled_without_a_two_seed_substitute(scenario):
    agents = Scripted(rebuttal=lambda inputs: envelope(plan(inputs, seed_count=10)))
    controller, store, _, runner = scenario(agents)
    best, _ = incumbent(controller)
    before = list(runner.executions)
    result = controller.supplementary_controls("review-0", best, {"concerns": ["Run ten independent seeds"]})
    assert result == [] and runner.executions == before
    saved = record(store, "review-0")
    decision = saved["negotiation"]["decisions"][0]
    assert decision["requested_seed_count"] == 10 and decision["supported_seed_count"] == 2
    assert decision["status"] == "unfulfilled" and decision["accepted_seeds"] == []
    assert saved["executions"] == []
    assert saved["coverage"][0]["status"] == "unfulfilled"
    assert "No truncated substitute" in " ".join(decision["reasons"])


def test_unsupported_orientation_and_final_requests_do_not_block_valid_independent_work(scenario):
    def requests(inputs):
        return envelope(
            plan(inputs, "orientation", concern=0, diagnostics=[{"kind": "orientation_consistency"}]),
            plan(inputs, "protected-final", concern=1, evaluation_split="final"),
            plan(inputs, "supported", concern=2, seed_count=1),
        )
    controller, store, _, runner = scenario(Scripted(rebuttal=requests))
    best, _ = incumbent(controller)
    result = controller.supplementary_controls("mixed", best, {"concerns": ["Orientation", "Final", "Paired metric"]})
    saved = record(store, "mixed")
    assert [row["status"] for row in saved["negotiation"]["decisions"]] == ["unfulfilled", "unfulfilled", "accepted"]
    assert [row["status"] for row in saved["coverage"]] == ["unfulfilled", "unfulfilled", "executed"]
    assert len(result) == 2 and all(observation.phase == "rebuttal" for observation in result)
    assert [len(observation.values) for observation in result] == [1, 1]
    assert runner.executions == ["fixture-parent", "mixed/matched/1", "mixed/2/control"]
    assert runner.finals == []


def test_shared_repeat_and_two_controls_execute_three_aggregate_operations(scenario):
    def requests(inputs):
        return envelope(plan(inputs, "repeat", kind="repeat"),
                        plan(inputs, "first-control", candidate=candidate(9).model_dump()),
                        plan(inputs, "second-control", candidate=candidate(9.5).model_dump()))
    controller, store, _, runner = scenario(Scripted(rebuttal=requests))
    best, _ = incumbent(controller)
    result = controller.supplementary_controls("shared", best, {"concerns": ["Check replication"]})
    assert runner.executions == ["fixture-parent", "shared/matched/2", "shared/1/control", "shared/2/control"]
    assert len(result) == 3 and len(experiment_rows(store)) == 4
    saved = record(store, "shared")
    executions = saved["executions"]
    assert [row["evaluation_count"] for row in executions] == [2, 4, 4]
    assert {row["observation_ids"][0] for row in executions} == {"shared/matched/2"}
    assert saved["coverage"][0]["status"] == "executed"


def test_new_rounds_get_disjoint_seeds_and_same_key_resume_has_no_operations(scenario):
    controller, store, backend, runner = scenario(
        Scripted(rebuttal=lambda inputs: envelope(plan(inputs, kind="repeat"))))
    best, _ = incumbent(controller)
    first = controller.supplementary_controls("round-0", best, {"concerns": ["Repeat"]})
    second = controller.supplementary_controls("round-1", best, {"concerns": ["Repeat"]})
    seeds0 = record(store, "round-0")["executions"][0]["seeds"]
    seeds1 = record(store, "round-1")["executions"][0]["seeds"]
    assert len(seeds0) == len(seeds1) == 2 and not set(seeds0) & set(seeds1)
    assert not (set(seeds0) | set(seeds1)) & set(controller.config.development.seeds)
    operations, seen, executed = controller.budget.summary(), len(backend.seen), list(runner.executions)
    resumed = LiveController(store, controller.config, Adapter(), backend, runner, budget=controller.budget)
    # run() reconstructs committed trials while replaying earlier research stages.
    # This focused direct-stage replay supplies that same committed history.
    resumed.trials = deepcopy(controller.trials)
    assert resumed.supplementary_controls("round-0", best, {"concerns": ["Repeat"]}) == first
    assert resumed.supplementary_controls("round-1", best, {"concerns": ["Repeat"]}) == second
    assert resumed.budget.summary() == operations and len(backend.seen) == seen
    assert runner.executions == executed


@pytest.mark.parametrize("status,outcome", [("complete", "no_improvement"), ("blocked", "invalid_agent_output")])
def test_terminal_resume_preserves_durable_coverage_without_new_work(scenario, status, outcome):
    controller, store, backend, runner = scenario(Scripted(rebuttal=lambda inputs: envelope(plan(inputs, seed_count=10))))
    best, _ = incumbent(controller)
    controller.supplementary_controls("unfulfilled", best, {"concerns": ["Ten seeds"]})
    coverage = controller.supplementary_coverage()
    controller.finish(outcome, "Synthetic terminal condition", status=status)
    operations, seen = controller.budget.summary(), len(backend.seen)
    resumed = LiveController(store, controller.config, Adapter(), backend, runner, budget=controller.budget)
    assert resumed.run()["status"] == status
    assert resumed.supplementary_coverage() == coverage
    assert resumed.budget.summary() == operations and len(backend.seen) == seen
    report = json.loads((store.root / "summary.json").read_text())
    assert report["supplementary_requests"][0]["coverage"] == coverage


def test_invalid_last_source_plan_blocks_all_earlier_accepted_execution(scenario):
    invalid = candidate(9).model_dump()
    invalid["files"].append({"path": "unexpected.py", "content": "# undeclared extra change"})
    controller, store, _, runner = scenario(Scripted(rebuttal=lambda inputs: envelope(
        plan(inputs, "first-valid", kind="repeat"), plan(inputs, "last-invalid", candidate=invalid))))
    best, _ = incumbent(controller)
    operations = experiment_rows(store)
    with pytest.raises(AgentOutputError, match="exactly its declared"):
        controller.supplementary_controls("source-invalid", best, {"concerns": ["Two checks"]})
    assert experiment_rows(store) == operations and runner.executions == ["fixture-parent"]
    assert store.json(store.get("agent-output/source-invalid/plans"))["status"] == "rejected"


def test_insufficient_whole_block_budget_starts_no_partial_substitute(scenario):
    controller, store, _, runner = scenario(Scripted(rebuttal=lambda inputs: envelope(plan(inputs))), max_experiments=2)
    best, _ = incumbent(controller)
    with pytest.raises(LiveBudgetExhausted, match="exact supplementary block"):
        controller.supplementary_controls("too-large", best, {"concerns": ["Paired control"]})
    assert runner.executions == ["fixture-parent"]
    assert record(store, "too-large")["executions"] == []
    assert record(store, "too-large")["coverage"][0]["status"] == "unfulfilled"


def draft_response(inputs, *, dishonest=False):
    observation = inputs["evidence"][0]
    responses = concern_responses(inputs)
    if dishonest:
        responses[0]["status"] = "executed"
    return {"title": "Synthetic draft", "claims": [{"experiment_id": observation["id"],
            "metric": observation["metric"], "value": observation["value"]}],
            "limitations": ["Synthetic values; no ML training"], "source_ids": ["task"],
            "concern_responses": responses}


@pytest.mark.parametrize("retries", [0, 1])
def test_unfulfilled_request_cannot_be_reported_executed_and_correction_is_durable(scenario, retries):
    agents = Scripted(
        rebuttal=lambda inputs: envelope(plan(inputs, seed_count=10)),
        draft=lambda inputs: draft_response(inputs, dishonest="output_correction" not in inputs),
    )
    controller, store, backend, runner = scenario(agents, agent_output_retries=retries)
    best, observation = incumbent(controller)
    controller.supplementary_controls("oversized", best, {"concerns": ["Ten seeds"]})
    if retries == 0:
        with pytest.raises(AgentOutputError, match="cannot be marked executed"):
            controller.draft("honest-draft", best, [observation])
        assert store.get("live_draft") is None
    else:
        draft = controller.draft("honest-draft", best, [observation])
        assert draft.concern_responses[0].status == "unaddressed"
        accepted = store.json(store.get("accepted-agent-output/honest-draft"))
        assert accepted["response_key"] == "honest-draft/output-correction/1"
        operations, seen = controller.budget.summary(), len(backend.seen)
        assert controller.draft("honest-draft", best, [observation]) == draft
        assert controller.budget.summary() == operations and len(backend.seen) == seen
    assert store.json(store.get("agent-output/honest-draft"))["status"] == "rejected"
    assert runner.executions == ["fixture-parent"]


def parent_with_initializer():
    source = candidate(8)
    return Candidate(command=source.command, description="Synthetic parent with recorded initializer expressions",
        files=source.files + [SourceFile(path="model.py", content=(
            'nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")\n'
            'nn.init.ones_(layer.weight)\n'
            'nn.init.zeros_(layer.bias)\n')),
            SourceFile(path="transforms.py", content="def transform(x):\n    return x\n")])


@pytest.mark.parametrize("retries", [0, 1])
def test_parent_edit_scope_preserves_initializer_and_binds_corrected_response_proof(scenario, retries):
    def implement(inputs):
        corrected = "output_correction" in inputs
        model = next(item for item in inputs["parent"]["files"] if item["path"] == "model.py")
        assert 'nn.init.kaiming_normal_' in model["content"]
        assert model["sha256"] == digest(model["content"].encode())
        return {
            "parent_source_ref": inputs["parent"]["parent_source_ref"],
            "edits": [{"path": "transforms.py" if corrected else "model.py",
                       "before": "return x" if corrected else "nn.init.ones_(layer.weight)",
                       "after": "return x.flip(-1)" if corrected else "nn.init.zeros_(layer.weight)",
                       "reason": "Synthetic source-scope regression"}],
            "description": "Synthetic patch",
        }
    controller, store, backend, runner = scenario(Scripted(implement=implement), agent_output_retries=retries)
    parent = parent_with_initializer()
    best, _ = incumbent(controller, parent)
    args = ("evolved-implementation", "implement", idea("Flips only"), best,
            ParentChangePlan(changed_files=["transforms.py"], reason="Change only input flips"))
    if retries == 0:
        with pytest.raises(AgentOutputError, match="independently declared"):
            controller.implement_change(*args)
        assert not store.db.execute("SELECT 1 FROM calls WHERE key='parent-edit/evolved-implementation'").fetchone()
    else:
        edited = controller.implement_change(*args)
        assert edited.files[1] == parent.files[1]
        assert edited.files[2].content.endswith("return x.flip(-1)\n")
        proof = store.json(store.db.execute("SELECT output_ref FROM calls WHERE key='parent-edit/evolved-implementation'").fetchone()[0])
        accepted = store.json(store.get("accepted-agent-output/evolved-implementation"))
        assert proof["response_key"] == accepted["response_key"] == "evolved-implementation/output-correction/1"
        assert proof["request_ref"] == accepted["request_ref"]
        assert proof["response_ref"] == accepted["response_ref"]
        assert proof["source_hashes"]["model.py"]["before"] == proof["source_hashes"]["model.py"]["after"]
        operations, seen = controller.budget.summary(), len(backend.seen)
        assert controller.implement_change(*args) == edited
        assert controller.budget.summary() == operations and len(backend.seen) == seen
    assert runner.executions == ["fixture-parent"]
