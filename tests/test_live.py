"""Synthetic branch traces; none of these scores are experimental measurements."""

import json

import pytest
from typer.testing import CliRunner

from scientisttwo.backends import BackendError, FixtureBackend, structured_schema
from scientisttwo.cli import app
from scientisttwo.live.adapters import implementation_identity
from scientisttwo.live.controller import LiveController
from scientisttwo.live.models import (
    Candidate,
    DevelopmentSpec,
    Hypothesis,
    LiveSpec,
    Observation,
    SourceFile,
)
from scientisttwo.live.runtime import Budget
from scientisttwo.benchmarks.models import NativeSpec
from scientisttwo.store import Store, canonical, digest


def candidate(value):
    return Candidate(
        files=[
            SourceFile(
                path="train.py",
                content=f"# SYNTHETIC TRACE ONLY\nTRACE_VALUE = {value}\n#value={value}",
            )
        ],
        command=["python", "train.py"],
        description="Synthetic trace, never executed",
    )


def idea(name):
    return Hypothesis(
        title=name,
        hypothesis="Synthetic test",
        novelty_score=5,
        novelty_scope="Synthetic",
        source_ids=["task"],
    )


def synthetic_patch(inputs, value):
    """Replace the one explicit synthetic value in the exact supplied parent."""
    parent = inputs["parent"]
    source = next(item["content"] for item in parent["files"] if item["path"] == "train.py")
    after = candidate(value).files[0].content
    if source == after:
        # An additional synthetic refinement measures a distinct, worse method;
        # declaring a no-op as a scientific method revision is forbidden.
        after = candidate(value + 0.25).files[0].content
    return {
        "parent_source_ref": parent["parent_source_ref"],
        "edits": [
            {
                "path": "train.py",
                "before": source,
                "after": after,
                "reason": "Change the declared synthetic trace value only",
            }
        ],
        "description": "Synthetic anchored parent edit; never executed",
    }


def concern_responses(inputs):
    return [
        {
            "concern_id": item["concern_id"],
            "status": {"executed": "executed", "partial": "limited", "unfulfilled": "unaddressed"}[
                item["status"]
            ],
            "experiment_ids": item["observation_ids"],
            "explanation": "Synthetic trace of execution coverage only; semantic resolution unverified",
        }
        for item in inputs.get("supplementary_coverage", [])
    ]


class Adapter:
    def identity(self):
        return {"fixture": "synthetic"}

    def context(self):
        return {"contract": "Synthetic test only"}

    def baseline(self):
        return candidate(10)


class Agents(FixtureBackend):
    def __init__(
        self,
        *,
        reject=False,
        engineer=False,
        refine=False,
        worse=False,
        meta=False,
        bad_claim=False,
    ):
        super().__init__(self.respond)
        self.seen = []
        self.reject, self.engineer, self.refine = reject, engineer, refine
        self.worse, self.meta, self.bad_claim = worse, meta, bad_claim

    def respond(self, role, inputs):
        self.seen.append((role, inputs))
        if role in {"limitations", "verify_limitations"}:
            return {"limitations": ["Synthetic"], "relevant": True, "reason": "Synthetic"}
        if role == "seeds":
            return {"ideas": [idea(f"seed-{i}").model_dump() for i in range(inputs["count"])]}
        if role == "evolve":
            assert inputs["attempts"]
            return {
                "idea": idea("evolved").model_dump(),
                "change_plan": {
                    "changed_files": ["train.py"],
                    "reason": "Change only the synthetic trace value",
                },
            }
        if role == "implement":
            if "parent" in inputs:
                value = 7 if inputs["idea"]["title"] == "evolved" else 11 if self.worse else 6
                return synthetic_patch(inputs, value)
            return candidate(8).model_dump()
        if role == "engineer":
            assert inputs["feedback"]["observation"]
            if "parent" in inputs:
                return synthetic_patch(inputs, 6.5)
            return candidate(6.5).model_dump()
        if role == "critic":
            choice = (
                "Bad"
                if self.reject
                else "Engineer"
                if self.engineer and inputs.get("can_engineer")
                else "Good"
            )
            return {"decision": choice, "reason": "Synthetic critique"}
        if role == "ablations":
            return {
                "plans": [
                    {
                        "purpose": "Synthetic control",
                        "changed_file": "train.py",
                        "candidate": candidate(9).model_dump(),
                    }
                ]
            }
        if role == "rebuttal":
            return {
                "plans": [
                    {
                        "id": "synthetic-control",
                        "concern_ids": [item["id"] for item in inputs["concerns"]],
                        "purpose": "Measure the declared synthetic control",
                        "seed_count": inputs["action_contract"]["max_seed_count"],
                        "execution_kind": "paired_control",
                        "evaluation_split": "development",
                        "diagnostics": [
                            {"kind": "metric", "metric": inputs["action_contract"]["metrics"][0]}
                        ],
                        "changed_file": "train.py",
                        "candidate": candidate(9).model_dump(),
                    }
                ],
                "unaddressed_concerns": [],
            }
        if role == "ablation_critic":
            return {"decision": "Refine" if self.refine else "Good", "reason": "Synthetic"}
        if role == "refine":
            return {
                "idea": idea("revision").model_dump(),
                "change_plan": {
                    "changed_files": ["train.py"],
                    "reason": "Measure a distinct synthetic revision",
                },
            }
        if role == "draft":
            e = next(e for e in inputs["evidence"] if e["status"] == "success")
            return {
                "title": "Synthetic draft",
                "claims": [
                    {
                        "experiment_id": e["id"],
                        "metric": e["metric"],
                        "value": 999 if self.bad_claim else e["value"],
                    }
                ],
                "limitations": ["Synthetic"],
                "source_ids": ["task"],
                "concern_responses": concern_responses(inputs),
            }
        if role == "review":
            return {"score": 7, "concerns": ["Synthetic repeat"]}
        if role == "meta":
            return {"decision": "ReviseMethod" if self.meta else "Accept", "reason": "Synthetic"}
        raise AssertionError(role)


class Runner:
    def __init__(self, store, budget, *, fail_first=False):
        self.store, self.budget, self.fail_first = store, budget, fail_first
        self.executions, self.finals = [], []

    def preflight(self):
        pass

    def execute(self, key, code, phase="full", seeds=None):
        request = {"code": code.model_dump(), "phase": phase, "seeds": seeds}
        cached = self.store.cached("development/" + key, request)
        if cached:
            return Observation.model_validate(cached)
        self.budget.reserve("experiment/" + key, "experiment", request)
        self.executions.append(key)
        value = float(code.files[0].content.split("#value=")[1])
        failed = self.fail_first and key == "search/0/0/subset/0"
        log = self.store.put_json({"log_ref": self.store.put(b"SYNTHETIC LOG", "log")})
        actual_seeds = (
            seeds
            if seeds is not None
            else self.budget.config.development.seeds[:1]
            if phase == "subset"
            else self.budget.config.development.seeds
        )
        result = Observation(
            id=key,
            phase=phase,
            status="execution_failed" if failed else "success",
            protocol_hash=digest(canonical([phase, seeds])),
            code_ref=self.store.put_json(code),
            metric="process_seconds",
            value=None if failed else value,
            values=[] if failed else [value] * len(actual_seeds),
            prediction_refs=[]
            if failed
            else [self.store.put_json({"synthetic": True, "seed": seed}) for seed in actual_seeds],
            log_refs=[log],
            seconds=0.0,
        )
        self.budget.finish("experiment/" + key, result.status, result, 0)
        self.store.cache("development/" + key, request, result)
        return result

    def final(self, key, code):
        assert self.store.get("live_frozen")
        self.finals.append(key)
        return {"status": "completed", "score": 123, "live_model_calls": 0}


def setup(tmp_path, agents=None, **settings):
    native = NativeSpec(
        name="test",
        benchmark="mlagentbench",
        task_id="vectorization",
        repository=tmp_path / "repo",
        revision="a" * 40,
        mode="baseline",
        data_provenance="upstream_no_data",
        public_dir=tmp_path / "public",
    )
    config = LiveSpec(
        native=native,
        development=DevelopmentSpec(),
        model="scripted-fixture-v1",
        mode="research",
        max_calls=100,
        campaign_seconds=600,
        max_experiments=100,
        implementation=implementation_identity(),
    )
    config = config.model_copy(update=settings)
    store = Store(tmp_path / "run")
    budget = Budget(store, config)
    backend = agents or Agents()
    runner = Runner(store, budget)
    controller = LiveController(store, config, Adapter(), backend, runner, budget=budget)
    return controller, store, backend, runner


def test_research_all_stages_resume_and_final_score_exclusion(tmp_path):
    controller, store, backend, runner = setup(tmp_path)
    result = controller.run()
    assert result["status"] == "complete" and result["outcome"] == "development_improvement"
    roles = {r for r, _ in backend.seen}
    assert {"evolve", "ablations", "draft", "review", "rebuttal", "meta"} <= roles
    assert "123" not in json.dumps(backend.seen)
    seen, executed = len(backend.seen), len(runner.executions)
    assert controller.run() == result
    assert len(backend.seen) == seen and len(runner.executions) == executed
    assert runner.finals == ["baseline", "selected"]
    assert (store.root / "manuscript.tex").is_file()
    assert json.loads((store.root / "summary.json").read_text())["synthetic_agents"]
    store.close()


def test_rejection_keeps_negative_outcome_and_baseline(tmp_path):
    controller, store, _, runner = setup(tmp_path, Agents(reject=True))
    assert controller.run()["outcome"] == "no_improvement"
    assert store.get("live_best") is None and runner.finals == ["baseline"]
    store.close()


def test_engineering_separate_from_scientific_rejection(tmp_path):
    controller, store, backend, runner = setup(tmp_path, Agents(engineer=True))
    runner.fail_first = True
    assert controller.run()["status"] == "complete"
    assert "engineer" in {role for role, _ in backend.seen}
    trials = [
        store.json(r[0])
        for r in store.db.execute("SELECT output_ref FROM calls WHERE key LIKE 'live-trial/%'")
    ]
    assert any(t["result"]["status"] == "execution_failed" for t in trials)
    assert any(t["parents"] for t in trials)
    store.close()


@pytest.mark.parametrize("worse", [False, True])
def test_refinement_and_incumbent_preservation(tmp_path, worse):
    controller, store, backend, _ = setup(tmp_path, Agents(refine=True, meta=True, worse=worse))
    assert controller.run()["status"] == "complete"
    best = store.get("live_best")
    assert best["result"]["value"] == (7 if worse else 6)
    drafts = [inputs for role, inputs in backend.seen if role == "draft"]
    assert all(d["incumbent"]["result"]["value"] == (7 if worse else 6) for d in drafts)
    store.close()


def test_budgets_and_no_implicit_retry_of_unknown_call(tmp_path):
    controller, store, backend, runner = setup(tmp_path, max_calls=1)
    assert controller.run()["outcome"] == "budget_exhausted"
    assert len(backend.seen) == 1 and not runner.executions
    store.close()
    controller, store, backend, _ = setup(tmp_path / "ambiguous")
    original = backend.invoke

    def interrupt(*args):
        raise KeyboardInterrupt

    backend.invoke = interrupt
    with pytest.raises(KeyboardInterrupt):
        controller.run()
    backend.invoke = original
    result = controller.run()
    assert result["status"] == "blocked" and "possibly billed" in result["reason"]
    assert not backend.seen
    store.close()


def test_failed_calls_charge_budget_and_resume_cached_results(tmp_path):
    controller, store, backend, _ = setup(tmp_path, mode="smoke", max_calls=20)
    original = backend.invoke
    count = 0

    def once(role, inputs, schema):
        nonlocal count
        count += 1
        if count == 1:
            raise BackendError("simulated service failure")
        return original(role, inputs, schema)

    backend.invoke = once
    assert controller.run()["status"] == "complete"
    rows = controller.budget.summary()
    assert sum(r["status"] == "failed" for r in rows) == 1
    assert sum(r["kind"] == "model" for r in rows) == count
    store.close()


def test_draft_claim_integrity(tmp_path):
    from scientisttwo.store import IntegrityError

    controller, store, _, _ = setup(tmp_path, Agents(bad_claim=True))
    with pytest.raises(IntegrityError, match="Unsupported"):
        controller.run()
    assert not store.get("live_frozen")
    store.close()


@pytest.mark.parametrize("path", ["../x.py", "/x.py", ".codex/config.json", "a/.env", "auth.json"])
def test_candidate_path_restrictions(path):
    with pytest.raises(ValueError):
        SourceFile(path=path, content="fixture")


def test_structured_schema_closed_and_required():
    schema = structured_schema(Candidate.model_json_schema())
    assert schema["required"] == list(schema["properties"])
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["SourceFile"]["additionalProperties"] is False


def test_live_cli_no_implicit_authorization(tmp_path):
    controller, store, _, _ = setup(tmp_path)
    manifest = tmp_path / "live.json"
    manifest.write_text(controller.config.model_dump_json())
    result = CliRunner().invoke(app, ["live", "run", str(manifest), str(tmp_path / "new")])
    assert result.exit_code != 0
    assert "--allow-live" in str(result.exception)
    assert not (tmp_path / "new").exists()
    store.close()


def test_completed_reservations_recover_after_deadline_without_new_work(tmp_path):
    import time
    from scientisttwo.live.runtime import LiveBudgetExhausted

    controller, store, _, _ = setup(tmp_path)
    budget = controller.budget
    budget.reserve("receipt", "model", {"input": 1})
    budget.finish("receipt", "success", {"done": True}, 0)
    store.set("live_deadline", time.time() - 1)
    assert budget.reserve("receipt", "model", {"input": 1})["status"] == "success"
    with pytest.raises(LiveBudgetExhausted):
        budget.reserve("new-work", "model", {"input": 2})
    store.close()


def test_successful_meta_revision_restarts_downstream(tmp_path):
    controller, store, backend, _ = setup(tmp_path, Agents(meta=True))
    assert controller.run()["status"] == "complete"
    draft_best = [
        inputs["incumbent"]["result"]["value"] for role, inputs in backend.seen if role == "draft"
    ]
    assert draft_best == [7, 7, 6, 6]
    store.close()


def test_freeze_resume_never_invokes_agents_again(tmp_path):
    controller, store, backend, runner = setup(tmp_path, mode="smoke")
    original = runner.final
    interrupted = False
    saved = {}

    def final(key, code):
        nonlocal interrupted
        if key in saved:
            return saved[key]
        if key == "selected" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        saved[key] = original(key, code)
        return saved[key]

    runner.final = final
    with pytest.raises(KeyboardInterrupt):
        controller.run()
    count = len(backend.seen)
    assert store.get("live_frozen")
    assert controller.run()["status"] == "complete"
    assert len(backend.seen) == count
    assert runner.finals == ["baseline", "selected"]
    store.close()


def test_review_can_request_unchanged_repeat_with_fresh_seeds(tmp_path):
    agents = Agents()
    previous = agents.responder

    def respond(role, inputs):
        if role == "rebuttal":
            return {
                "plans": [
                    {
                        "id": "synthetic-repeat",
                        "concern_ids": [item["id"] for item in inputs["concerns"]],
                        "purpose": "repeat on fresh development seeds",
                        "seed_count": inputs["action_contract"]["max_seed_count"],
                        "execution_kind": "repeat",
                        "evaluation_split": "development",
                        "diagnostics": [
                            {"kind": "metric", "metric": inputs["action_contract"]["metrics"][0]}
                        ],
                        "changed_file": None,
                        "candidate": inputs["incumbent"]["candidate"],
                    }
                ],
                "unaddressed_concerns": [],
            }
        return previous(role, inputs)

    agents.responder = respond
    controller, store, _, _ = setup(tmp_path, agents)
    assert controller.run()["status"] == "complete"
    assert any("rebuttal" in r[0] for r in store.db.execute("SELECT key FROM calls"))
    store.close()


def test_scripted_research_restores_frozen_reference_missing_from_overlay(tmp_path):
    from scientisttwo.live.parent_edits import MethodChange, ParentPatch
    from scientisttwo.testbed.validation import ScriptedResearch

    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "model.py").write_text("# synthetic prepared reference\nWIDTH = 4\n")

    class PreparedOverlay:
        root = tmp_path

        def baseline(self):
            return Candidate(
                files=[],
                base_snapshot_ref="artifacts/" + "a" * 64 + ".json",
                command=["python", "train.py"],
                description="Synthetic frozen overlay",
            )

        def control(self):
            result = self.baseline()
            result.files = [
                SourceFile(path="model.py", content="# synthetic prepared reference\nWIDTH = 8\n")
            ]
            return result

    adapter = PreparedOverlay()
    backend = ScriptedResearch(adapter)
    parent = {
        "parent_source_ref": "artifacts/" + "b" * 64 + ".json",
        "editable_paths": ["model.py"],
        "files": [{"path": "model.py", "content": adapter.control().files[0].content}],
    }
    change = MethodChange.model_validate(backend.respond("evolve", {"parent": parent}))
    assert change.change_plan.changed_files == ["model.py"]
    patch = ParentPatch.model_validate(
        backend.respond(
            "implement",
            {
                "idea": change.idea.model_dump(),
                "change_plan": change.change_plan.model_dump(),
                "parent": parent,
            },
        )
    )
    assert patch.edits[0].before == adapter.control().files[0].content
    assert patch.edits[0].after == (reference / "model.py").read_text()
    assert patch.parent_source_ref == parent["parent_source_ref"]
    # A negative-search route can select a reference parent. The scripted agent
    # then explicitly selects the other prepared source instead of declaring a
    # no-op reference restoration as a method change.
    parent["files"][0]["content"] = patch.edits[0].after
    change = MethodChange.model_validate(backend.respond("evolve", {"parent": parent}))
    assert change.idea.title.startswith("Restore prepared control")
    patch = ParentPatch.model_validate(
        backend.respond(
            "implement",
            {
                "idea": change.idea.model_dump(),
                "change_plan": change.change_plan.model_dump(),
                "parent": parent,
            },
        )
    )
    assert patch.edits[0].before == (reference / "model.py").read_text()
    assert patch.edits[0].after == adapter.control().files[0].content


def test_scripted_research_patches_supplied_body_representation_without_wrapper(tmp_path):
    from scientisttwo.live.parent_edits import MethodChange, ParentPatch
    from scientisttwo.testbed.validation import ScriptedResearch

    class PreparedBodies:
        def baseline(self):
            return Candidate(
                files=[
                    SourceFile(
                        path="train.py", content="return slow_forward(self, features_batch)\n"
                    )
                ],
                command=["python", "train.py"],
                description="Synthetic body fixture",
            )

        def control(self):
            return Candidate(
                files=[
                    SourceFile(
                        path="train.py", content="return vectorized_forward(self, features_batch)\n"
                    )
                ],
                command=["python", "train.py"],
                description="Synthetic body fixture",
            )

    adapter = PreparedBodies()
    backend = ScriptedResearch(adapter)
    source = adapter.control().files[0].content
    parent = {
        "parent_source_ref": "artifacts/" + "b" * 64 + ".json",
        "editable_paths": ["train.py"],
        "files": [
            {
                "path": "train.py",
                "content": source,
                "representation": "proposal",
                "materialized_content": "class FixedTemplate:\n    def forward(self, features_batch):\n        "
                + source,
            }
        ],
    }
    change = MethodChange.model_validate(backend.respond("refine", {"parent": parent}))
    patch = ParentPatch.model_validate(
        backend.respond(
            "implement",
            {
                "parent": parent,
                "idea": change.idea.model_dump(),
                "change_plan": change.change_plan.model_dump(),
            },
        )
    )
    assert patch.edits[0].before == source
    assert patch.edits[0].after == adapter.baseline().files[0].content
    assert "FixedTemplate" not in patch.model_dump_json()
