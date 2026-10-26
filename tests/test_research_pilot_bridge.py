"""Synthetic worker outcomes validate the bridge; no candidate executes on the host."""

from copy import deepcopy
import time

import pytest

from scientisttwo.live.models import Candidate, Hypothesis, LimitationAssessment, SourceFile
from scientisttwo.live.parent_edits import MethodChange, ParentPatch, parent_context
from scientisttwo.live.source_tree import candidate_snapshot
from scientisttwo.models import ExperimentResult, MetricRow, Profile, TaskSpec
from scientisttwo.research.pilot_bridge import COMMAND, audit_pilot_campaign, create_pilot_campaign
from scientisttwo.store import IntegrityError, Store, canonical, digest


class SyntheticContainer:
    """Durable test double for the worker's legacy job/result protocol."""

    image_id = None

    def __init__(self, store):
        self.store = store
        self.executions = []
        self.status = "success"
        self.crash = None
        self.omit_row = False
        self.dataset_values = {}

    def preflight(self):
        self.image_id = "sha256:" + "1" * 64

    def execute(self, key, idea, source, spec):
        old = self.store.db.execute("SELECT * FROM jobs WHERE key=?", (key,)).fetchone()
        if old and old["result_ref"]:
            return ExperimentResult.model_validate(self.store.json(old["result_ref"]))
        if self.crash == "before-job":
            self.crash = None
            raise KeyboardInterrupt()
        self.executions.append((key, source, spec))
        with self.store.db:
            self.store.db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?,NULL)",
                (
                    key,
                    digest(canonical({"source": source, "spec": spec.model_dump(mode="json")})),
                    "running",
                    time.time(),
                    120,
                    0,
                ),
            )
        rows = (
            [
                MetricRow(
                    dataset=dataset,
                    seed=seed,
                    metrics={
                        "macro_f1": self.dataset_values.get(dataset, 0.5) + seed / 1000,
                        "accuracy": self.dataset_values.get(dataset, 0.5) + seed / 1000,
                    },
                    training_seconds=0.1,
                    inference_seconds=0.01,
                    count=10,
                )
                for dataset in spec.datasets
                for seed in spec.seeds
            ]
            if self.status == "success"
            else []
        )
        if self.omit_row:
            rows.pop()
        result = ExperimentResult(
            id=key,
            idea_id=idea.id,
            status=self.status,
            protocol_hash=digest(canonical(spec)),
            code_ref=self.store.put(source.encode(), "py"),
            spec_ref=self.store.put_json(spec),
            split_manifest_ref=self.store.get("split_manifest_ref"),
            rows=rows,
            observations_ref=self.store.put_json({"synthetic_test_only": True}) if rows else None,
            logs_ref=self.store.put(b"Synthetic worker receipt; no code executed", "log"),
            measurement_origin="synthetic_test",
            duration_seconds=1.5,
        )
        ref = self.store.put_json(result)
        with self.store.db:
            self.store.db.execute(
                "UPDATE jobs SET status=?,result_ref=?,charged=? WHERE key=?",
                (result.status, ref, 1.5, key),
            )
        if self.crash == "after-result":
            self.crash = None
            raise KeyboardInterrupt()
        return result


@pytest.fixture
def bridge(tmp_path):
    store = Store(tmp_path)
    worker = SyntheticContainer(store)
    config, adapter, runner = create_pilot_campaign(
        store, TaskSpec(), Profile(), container_runner=worker
    )
    yield store, config, adapter, runner, worker
    store.close()


def test_factory_uses_shared_engine_and_original_split_identity(bridge):
    store, config, adapter, runner, worker = bridge
    assert config.engine_version == "research-v2"
    assert config.workflow.profile == "offline-pilot"
    assert not config.workflow.network_literature
    assert not worker.executions
    assert store.json(store.get("task_ref")) == TaskSpec().model_dump(mode="json")
    assert set(store.json(store.get("split_manifest_ref"))["datasets"]) == {
        "iris",
        "wine",
        "digits",
    }
    repeated, _, _ = create_pilot_campaign(store, TaskSpec(), Profile(), container_runner=worker)
    assert repeated == config


@pytest.mark.parametrize("changed", ["task", "profile", "runtime"])
def test_factory_refuses_changed_resume_inputs(bridge, changed):
    store, _, _, _, worker = bridge
    task, profile, runtime = TaskSpec(), Profile(), "docker"
    if changed == "task":
        task = task.model_copy(update={"split_seed": 40})
    if changed == "profile":
        profile = profile.model_copy(update={"search_rounds": 3})
    if changed == "runtime":
        runtime = "podman"
    with pytest.raises(IntegrityError):
        create_pilot_campaign(store, task, profile, runtime, container_runner=worker)


def test_candidate_is_only_parsed_never_imported_on_host(bridge, tmp_path):
    _, _, adapter, _, _ = bridge
    marker = tmp_path / "must-not-exist"
    source = f"open({str(marker)!r}, 'w').write('forbidden')\ndef build(seed):\n    return None\n"
    candidate = Candidate(
        files=[SourceFile(path="candidate.py", content=source)],
        command=COMMAND,
        description="Untrusted test source",
    )
    adapter.validate(candidate)
    assert not marker.exists()
    for changed in [
        candidate.model_copy(update={"command": ["sh", "-c", "bad"]}),
        candidate.model_copy(update={"files": [SourceFile(path="other.py", content=source)]}),
    ]:
        with pytest.raises(ValueError):
            adapter.validate(changed)


def test_complete_grid_and_per_seed_means_survive_bridge(bridge):
    store, _, adapter, runner, worker = bridge
    baseline = adapter.baseline()
    result = runner.execute("baseline/full", baseline, "full")
    assert len(result.metric_rows) == 9
    assert result.values == [0.5, 0.501, 0.502]
    assert result.metrics == {"accuracy": 0.501, "macro_f1": 0.501}
    assert result.legacy_result_ref
    original = store.json(result.legacy_result_ref)
    assert result.metric_rows == original["rows"]
    assert original["measurement_origin"] == "synthetic_test"
    assert len(result.prediction_refs) == 3
    assert [store.json(ref)["seed"] for ref in result.prediction_refs] == [0, 1, 2]
    assert len(result.log_refs) == 1
    receipt = store.json(result.log_refs[0])
    assert receipt["kind"] == "pilot_batch" and receipt["seeds"] == [0, 1, 2]
    assert receipt["legacy_result_ref"] == result.legacy_result_ref
    assert result.source_tree_ref and result.protocol_ref
    operations = runner.budget.summary()
    assert sum(row["seconds"] for row in operations if row["kind"] == "experiment") == 1.5
    assert store.db.execute("SELECT SUM(charged) FROM jobs").fetchone()[0] == 1.5
    assert runner.execute("baseline/full", baseline, "full") == result
    assert len(worker.executions) == 1


def test_screening_and_rebuttal_keep_declared_grid_and_partition(bridge):
    store, _, adapter, runner, worker = bridge
    subset = runner.execute("screen", adapter.baseline(), "subset")
    assert {row["dataset"] for row in subset.metric_rows} == {"iris", "wine"}
    assert {row["seed"] for row in subset.metric_rows} == {0}
    extra = runner.execute("supplement", adapter.baseline(), "rebuttal", [7, 9])
    assert {row["seed"] for row in extra.metric_rows} == {7, 9}
    assert len(extra.metric_rows) == 6
    spec = store.json(store.json(extra.legacy_result_ref)["spec_ref"])
    assert spec["split"] == "development"
    assert spec["split_manifest_hash"]
    assert len(worker.executions) == 2
    with pytest.raises(ValueError):
        runner.execute("invalid-seeds", adapter.baseline(), "full", [])


def test_dataset_guardrail_cannot_be_hidden_by_aggregate_gain(bridge):
    _, _, adapter, runner, worker = bridge
    baseline = runner.execute("reference", adapter.baseline())
    worker.dataset_values = {"iris": 0.9, "wine": 0.9, "digits": 0.4}
    candidate = runner.execute("candidate", adapter.candidate_for(adapter.seed_ideas(1)[0]))
    assert candidate.value > baseline.value
    compared = adapter.compare(candidate, baseline)
    assert not compared["valid"] and not compared["improved"]
    assert not compared["guardrails_pass"] and compared["constraint_failures"]


def test_tampered_aggregate_or_grid_fails_comparison(bridge):
    _, _, adapter, runner, _ = bridge
    result = runner.execute("reference", adapter.baseline())
    changed = result.model_copy(deep=True)
    changed.metric_rows[0]["metrics"]["accuracy"] += 0.1
    assert not adapter.compare(changed, result)["valid"]
    changed = result.model_copy(deep=True)
    changed.metric_values["accuracy"][0] += 0.1
    assert not adapter.compare(changed, result)["valid"]


def test_missing_grid_cell_becomes_invalid_not_a_silent_partial_mean(bridge):
    _, _, adapter, runner, worker = bridge
    worker.omit_row = True
    result = runner.execute("missing", adapter.baseline())
    assert result.status == "invalid" and result.value is None and not result.values
    assert "grid" in result.reason and len(result.metric_rows) == 8
    assert runner.execute("missing", adapter.baseline()) == result
    assert len(worker.executions) == 1


@pytest.mark.parametrize("status", ["timeout", "execution_failed", "interrupted"])
def test_execution_failures_stay_separate_and_consume_budget(bridge, status):
    _, _, adapter, runner, worker = bridge
    worker.status = status
    result = runner.execute("failure", adapter.baseline())
    assert result.status == status
    assert not result.metrics and not result.prediction_refs
    assert not adapter.compare(result, result)["valid"]
    operation = next(row for row in runner.budget.summary() if row["kind"] == "experiment")
    assert operation["status"] == status and operation["seconds"] == 1.5


def test_unknown_bridge_admission_is_not_reexecuted(bridge):
    _, _, adapter, runner, worker = bridge
    worker.crash = "before-job"
    with pytest.raises(KeyboardInterrupt):
        runner.execute("uncertain", adapter.baseline())
    result = runner.execute("uncertain", adapter.baseline())
    assert result.status == "interrupted"
    assert not worker.executions
    assert "refusing" in result.reason


def test_worker_completion_recovers_without_duplicate_execution(bridge):
    _, _, adapter, runner, worker = bridge
    worker.crash = "after-result"
    with pytest.raises(KeyboardInterrupt):
        runner.execute("recover", adapter.baseline())
    result = runner.execute("recover", adapter.baseline())
    assert result.status == "success"
    assert len(worker.executions) == 1
    assert (
        next(row for row in runner.budget.summary() if row["kind"] == "experiment")["seconds"]
        == 1.5
    )


def test_budget_completion_before_cache_recovers_committed_observation(bridge, monkeypatch):
    store, _, adapter, runner, worker = bridge
    original_cache = store.cache

    def interrupted(key, *args, **kwargs):
        if key == "development/recover":
            raise KeyboardInterrupt()
        return original_cache(key, *args, **kwargs)

    monkeypatch.setattr(store, "cache", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.execute("recover", adapter.baseline())
    monkeypatch.setattr(store, "cache", original_cache)
    result = runner.execute("recover", adapter.baseline())
    assert result.status == "success" and len(worker.executions) == 1
    assert store.db.execute("SELECT 1 FROM calls WHERE key='development/recover'").fetchone()


def test_final_requires_frozen_exact_source_and_runs_once(bridge):
    store, _, adapter, runner, worker = bridge
    baseline = adapter.baseline()
    with pytest.raises(IntegrityError):
        runner.final("baseline", baseline)
    with pytest.raises(ValueError):
        runner.execute("unsafe", baseline, "final")
    assert not worker.executions
    store.set(
        "live_frozen",
        {"baseline": baseline.model_dump(), "best": {"candidate": baseline.model_dump()}},
    )
    final = runner.final("baseline", baseline)
    assert final["status"] == "completed" and final["live_model_calls"] == 0
    assert worker.executions[0][2].split == "test"
    assert runner.final("baseline", baseline) == final
    assert len(worker.executions) == 1
    with pytest.raises(IntegrityError):
        runner.final("selected", adapter.candidate_for(adapter.seed_ideas(1)[0]))


def test_legacy_scripted_roles_translate_to_typed_research_contracts(bridge):
    store, _, adapter, _, _ = bridge
    limited = adapter.fixture_response("verify_limitations", {})
    assert LimitationAssessment.model_validate(limited).relevant
    seed = adapter.fixture_response("seeds", {"count": 3})["ideas"][0]
    assert Hypothesis.model_validate(seed).adapter_payload["method"] == "svc"
    candidate = Candidate.model_validate(adapter.fixture_response("implement", {"idea": seed}))
    tree = candidate_snapshot(store, candidate)
    parent = parent_context(store, candidate, store.put_json(tree), editable_paths=["candidate.py"])
    evolved = MethodChange.model_validate(
        adapter.fixture_response("evolve", {"parent": parent, "attempts": []})
    )
    assert evolved.idea.adapter_payload["parameters"]["strength"] == 6
    patch = ParentPatch.model_validate(
        adapter.fixture_response(
            "implement",
            {
                "idea": evolved.idea.model_dump(),
                "parent": parent,
                "change_plan": evolved.change_plan.model_dump(),
            },
        )
    )
    assert patch.edits[0].before == candidate.files[0].content
    assert patch.edits[0].after != patch.edits[0].before


def test_no_mutation_of_original_task_or_profile(bridge):
    store, _, _, _, worker = bridge
    task, profile = TaskSpec(), Profile()
    before = deepcopy((task.model_dump(), profile.model_dump()))
    create_pilot_campaign(store, task, profile, container_runner=worker)
    assert before == (task.model_dump(), profile.model_dump())


def test_audit_does_not_promote_synthetic_outcomes_to_real_scoring(bridge):
    store, _, adapter, runner, _ = bridge
    runner.execute("reference", adapter.baseline())
    audit = audit_pilot_campaign(store, adapter)
    assert audit["artifacts"] == "pass" and audit["ledger"] == "pass"
    assert audit["scoring"] == "unverified"
    assert audit["observations"][0]["measurement_origin"] == "synthetic_test"
    assert "Synthetic" in audit["observations"][0]["reason"]


def test_audit_checks_underlying_and_shared_ledgers(bridge):
    store, _, adapter, runner, _ = bridge
    runner.execute("reference", adapter.baseline())
    with store.db:
        store.db.execute("UPDATE jobs SET charged=99")
    audit = audit_pilot_campaign(store, adapter)
    assert audit["artifacts"] == "fail"
    assert any("ledger" in error for error in audit["errors"])


@pytest.mark.parametrize("changed", [False, True])
def test_negative_campaign_reproduction_uses_frozen_reference_and_reports_drift(bridge, changed):
    store, _, adapter, runner, worker = bridge
    baseline = adapter.baseline()
    result = runner.execute("baseline/full", baseline, "full")
    store.set("live_frozen", {"best": {"candidate": baseline.model_dump(),
                                      "result": result.model_dump()},
                              "baseline": baseline.model_dump(), "improved": False})
    assert store.get("live_best") is None
    if changed:
        worker.dataset_values = {"iris": .8}
    report = audit_pilot_campaign(store, adapter, runner, reexecute=True)
    assert len(worker.executions) == 2
    assert worker.executions[-1][2].split == "development"
    assert report["reexecution"] == ("fail" if changed else "pass")
    assert report["reexecution_measurement_origin"] == "synthetic_test"
    assert bool(report["errors"]) == changed
    # Replay of the same audit operation does not run the selected method twice.
    audit_pilot_campaign(store, adapter, runner, reexecute=True)
    assert len(worker.executions) == 2


def test_audit_detects_manufactured_container_score_files(bridge):
    store, _, adapter, runner, _ = bridge
    result = runner.execute("reference", adapter.baseline())
    original = store.json(result.legacy_result_ref)
    original["measurement_origin"] = "container"
    new_ref = store.put_json(original)
    altered = result.model_copy(update={"legacy_result_ref": new_ref})
    # Adversarial fixture rebinds all public records to a fabricated container claim.
    # The trusted scorer must still reject its non-prediction output payload.
    altered_ref = store.put_json(altered)
    with store.db:
        store.db.execute("UPDATE jobs SET result_ref=?", (new_ref,))
        store.db.execute(
            "UPDATE live_operations SET result_ref=? WHERE kind='experiment'", (altered_ref,)
        )
        store.db.execute(
            "UPDATE calls SET output_ref=? WHERE key='development/reference'", (altered_ref,)
        )
    audit = audit_pilot_campaign(store, adapter)
    assert audit["artifacts"] == "fail"
    assert audit["scoring"] == "unverified"


def test_shared_controller_completes_synthetic_cycle_and_resume_has_no_operations(bridge):
    from scientisttwo.research.controller import ResearchController
    from scientisttwo.research.fixtures import ResearchFixtures

    store, config, adapter, runner, worker = bridge
    original_execute = worker.execute

    def scripted(key, idea, source, spec):
        worker.dataset_values = {
            d: (
                0.8
                if "estimator = SVC(" in source
                else 0.6
                if "estimator = KNeighbors" in source
                else 0.5
            )
            for d in spec.datasets
        }
        return original_execute(key, idea, source, spec)

    worker.execute = scripted
    backend = ResearchFixtures(adapter.fixture_response)
    state = ResearchController(store, config, adapter, backend, runner, budget=runner.budget).run()
    assert state["status"] == "complete" and state["outcome"] == "development_improvement"
    roles = {role for role, _ in backend.seen}
    assert {
        "novelty_checker",
        "baseline_plan",
        "full_set_coder",
        "selection_review",
        "ablation_planner",
        "ablation_coder",
        "control_reviewer",
        "initial_drafter",
        "draft_enhancer",
        "writing_critic",
        "scientific_review",
        "rebuttal_planner",
        "meta",
    } <= roles
    before_operations, before_calls, before_jobs = (
        runner.budget.summary(),
        len(backend.seen),
        len(worker.executions),
    )
    resumed = ResearchController(
        store, config, adapter, backend, runner, budget=runner.budget
    ).run()
    assert resumed == state
    assert runner.budget.summary() == before_operations
    assert len(backend.seen) == before_calls and len(worker.executions) == before_jobs
    audit = audit_pilot_campaign(store, adapter, include_final=True)
    assert audit["artifacts"] == "pass" and audit["ledger"] == "pass"
    assert audit["scoring"] == "unverified"  # Every metric in this test is explicitly synthetic.
