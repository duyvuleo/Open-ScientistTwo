"""Synthetic paired-study evidence only; no CUDA, provider calls or real scores."""

from types import SimpleNamespace

import pytest

from scientisttwo.artifacts import ArtifactSpec, freeze_bundle
from scientisttwo.live.models import DevelopmentSpec, LiveSpec, Observation
from scientisttwo.live.source_tree import candidate_snapshot
from scientisttwo.live.workspace_models import WorkspaceSpec
from scientisttwo.store import IntegrityError, Store, canonical, digest
from scientisttwo.studies.budget import StudyBudget
from scientisttwo.studies.cifar_recipe import (
    BASELINE_CONFIGURATION, FIXED_CONFIGURATION, baseline_candidate, validate_candidate, write_study_sources,
)
from scientisttwo.studies.coordinator import (
    StudyResearchController, audit_study, create_study_plan, final_grant_verifier, grant_final_access,
    release_finals, study_verdicts,
)
from scientisttwo.studies.runner import StudyDevelopmentRunner, candidate_source_hash
from scientisttwo.studies import runner as runner_module
from scientisttwo.studies import coordinator as coordinator_module


class Adapter:
    def __init__(self, root):
        self.root, self.snapshot_ref = root, None

    def source_tree(self):
        return {"path": self.root / "reference", "editable_paths": ["training_config.json"]}

    def bind_source(self, ref):
        self.snapshot_ref = ref

    def baseline(self):
        return baseline_candidate(self.snapshot_ref)

    def context(self):
        return {"synthetic_fixture": True, "editable_files": ["training_config.json"]}

    def identity(self):
        return {"adapter": "synthetic-coordinator-fixture"}

    def validate(self, candidate):
        validate_candidate(candidate, snapshot_ref=self.snapshot_ref)


def setup(tmp_path):
    source = tmp_path / "shared-source"
    write_study_sources(source)
    for name in ("public", "private"):
        (source / name).mkdir()
        (source / name / "synthetic.txt").write_text("Synthetic, never executed")
    native = WorkspaceSpec(name="synthetic-paired-study", task_id="cifar10",
        public_dir=source / "public", private_dir=source / "private", candidate_dir=source / "reference",
        evaluator_dir=source / "evaluator", command=["python", "train.py"], evaluator_command=["python", "evaluate.py"],
        metric="accuracy", direction="maximize", image="sha256:" + "a" * 64, grader_image="sha256:" + "b" * 64)
    config = LiveSpec(engine_version="research-v2", mode="research", native=native,
        development=DevelopmentSpec(metric="accuracy", direction="maximize", seeds=[0, 1, 2]),
        model="synthetic-no-calls", max_calls=10, max_experiments=10, campaign_seconds=1000,
        token_stop_threshold=10_000)
    owners = []
    for name in ("scientist", "direct"):
        store = Store(tmp_path / name)
        spec = config.model_copy(deep=True)
        budget = StudyBudget(store, spec, max_experiment_seconds=1000, max_final_seconds=1000,
                             max_final_wall_seconds=1000)
        adapter = Adapter(source)
        runner = StudyDevelopmentRunner(store, spec, adapter, budget)
        store.set("live_images", {"execution": native.image, "grading": native.grader_image})
        owner = StudyResearchController.__new__(StudyResearchController) if name == "scientist" else SimpleNamespace()
        owner.store, owner.config, owner.adapter, owner.budget, owner.runner = store, spec, adapter, budget, runner
        owner.backend = SimpleNamespace(name="fixture", synthetic=True)
        owners.append(owner)
    research, direct = owners
    research.research_enabled = True
    research.research_final_appendix = lambda results: research.store.set("synthetic_final_appendix", True)
    def finish(outcome, reason, status="complete"):
        state = {"stage": "complete", "status": status, "outcome": outcome, "reason": reason}
        research.store.set("live_state", state)
        return state
    research.finish = finish
    return Store(tmp_path / "study-evidence"), research, direct


def checkpoint(store, directory, seed, phase):
    root = directory / "checkpoint"
    root.mkdir(parents=True)
    configuration = {"seed": seed, **FIXED_CONFIGURATION, **BASELINE_CONFIGURATION}
    (root / "config.json").write_bytes(canonical(configuration))
    (root / "training.json").write_bytes(canonical({"task": "cifar10", "seed": seed,
        "steps": 3910 if phase == "final" else 3130, "max_steps": 3910, "epochs_started": 10}))
    (root / "weights.pt").write_text("SYNTHETIC checkpoint; never deserialize")
    return freeze_bundle(store, directory, [ArtifactSpec(path="checkpoint", kind="directory")])


def measured(owner, key, values):
    """Construct explicitly synthetic but internally consistent immutable evidence."""
    store, candidate = owner.store, owner.adapter.baseline()
    protocol = {"phase": "full", "seeds": [0, 1, 2], "synthetic_fixture": True,
                "runtime": {"images": store.get("live_images")},
                "pipeline": {"private_dir": str(store.root / "owned-development-labels")}}
    bundles, predictions, evaluations, logs = [], [], [], []
    def receipt(image):
        return store.put_json({"returncode": 0, "timed_out": False, "limit_exceeded": False,
                               "image_id": image, "synthetic_fixture": True})
    for seed, value in enumerate(values):
        directory = store.root / "synthetic" / key / str(seed)
        bundle = checkpoint(store, directory, seed, "full")
        prediction_dir = directory / "predictions"
        prediction_dir.mkdir()
        (prediction_dir / "values.txt").write_text("SYNTHETIC predictions")
        prediction = freeze_bundle(store, prediction_dir, [ArtifactSpec(path="values.txt")])
        grade = store.put_json({"valid_submission": True, "metrics": {"accuracy": value}, "synthetic_fixture": True})
        evaluations.append(store.put_json({"seed": seed, "metrics": {"accuracy": value},
            "input_bundle": bundle, "prediction_bundle": prediction, "grade_ref": grade,
            "receipts": [receipt(owner.config.native.image), receipt(owner.config.native.grader_image)]}))
        bundles.append(bundle)
        predictions.append(prediction)
        logs.append(receipt(owner.config.native.image))
    observed = Observation(id=key, phase="full", status="success", metric="accuracy", value=sum(values) / 3,
        values=values, protocol_hash=digest(canonical(protocol)), protocol_ref=store.put_json(protocol),
        code_ref=store.put_json(candidate), source_tree_ref=store.put_json(candidate_snapshot(store, candidate)),
        bundle_refs=bundles, evaluation_refs=evaluations, prediction_refs=predictions, log_refs=logs)
    owner.budget.reserve("experiment/" + key, "experiment", {"synthetic": key})
    owner.budget.finish("experiment/" + key, "success", observed, 0.01)
    return observed


def freeze_arms(research, direct):
    baseline = measured(research, "baseline/full", [0.65, 0.66, 0.67])
    chosen = measured(research, "selected/full", [0.69, 0.70, 0.71])
    candidate = research.adapter.baseline()
    frozen = {"baseline": candidate.model_dump(), "best": {"id": "selected", "candidate": candidate.model_dump(),
                                                         "result": chosen.model_dump()}, "improved": True}
    research.store.set("live_frozen", frozen)
    research.store.set("live_state", {"stage": "freeze", "status": "running"})
    research.store.set("research_manuscript_bundle", research.store.put_json({"synthetic": "manuscript fixture"}))
    state = research.finalize(frozen)
    assert state["status"] == "frozen"
    direct_baseline = measured(direct, "direct/baseline/full", baseline.values)
    direct_selected = measured(direct, "direct/selected/full", [0.68, 0.69, 0.70])
    direct_frozen = {"identity_ref": direct.store.put_json({"max_iterations": 4}),
        "selection_basis": "full_development_only", "seeds": [0, 1, 2],
        "selected_candidate_ref": direct.store.put_json(direct.adapter.baseline()),
        "selected_observation_ref": direct.store.put_json(direct_selected),
        "baseline_observation_ref": direct.store.put_json(direct_baseline)}
    direct.store.set("direct_frozen_ref", direct.store.put_json(direct_frozen))


def test_plan_precedes_model_and_training_and_does_not_start_arm_deadlines(tmp_path):
    study, research, direct = setup(tmp_path)
    result = create_study_plan(study, research, direct, direct_max_iterations=4)
    assert result["status"] == "planned" and result["final_access"] is False
    assert research.store.get("live_deadline") is None and direct.store.get("live_deadline") is None
    assert create_study_plan(study, research, direct, direct_max_iterations=4) == result
    assert study.json(result["plan_ref"])["arm_order"] == ["scientist", "direct"]


@pytest.mark.parametrize("field,value", [("model", "different-model"), ("max_calls", 11), ("token_stop_threshold", 99_999)])
def test_unequal_arm_model_or_budgets_are_rejected(tmp_path, field, value):
    study, research, direct = setup(tmp_path)
    setattr(direct.config, field, value)
    with pytest.raises(IntegrityError, match="arms differ"):
        create_study_plan(study, research, direct, direct_max_iterations=4)


def test_a_plan_cannot_be_retrofitted_after_an_arm_experiment(tmp_path):
    study, research, direct = setup(tmp_path)
    research.budget.reserve("experiment/too-early", "experiment", {})
    with pytest.raises(IntegrityError, match="before any arm"):
        create_study_plan(study, research, direct, direct_max_iterations=4)


def test_research_freezes_without_final_calls_and_requires_peer_freeze(tmp_path):
    study, research, direct = setup(tmp_path)
    create_study_plan(study, research, direct, direct_max_iterations=4)
    with pytest.raises(IntegrityError):
        grant_final_access(study, research, direct)
    freeze_arms(research, direct)
    assert not research.store.db.execute("SELECT 1 FROM live_operations WHERE key LIKE 'final/%'").fetchone()
    ref = grant_final_access(study, research, direct)
    assert grant_final_access(study, research, direct) == ref
    assert study.json(ref)["status"] == "BOTH_ARMS_FROZEN"


def test_grant_reverifies_frozen_ledgers_and_only_authorizes_the_shared_reference_once(tmp_path):
    study, research, direct = setup(tmp_path)
    create_study_plan(study, research, direct, direct_max_iterations=4)
    freeze_arms(research, direct)
    ref = grant_final_access(study, research, direct)
    verifier = final_grant_verifier(study, research, direct)
    code_hash = candidate_source_hash(direct.store, direct.adapter.baseline())
    with pytest.raises(IntegrityError, match="does not authorize"):
        verifier(direct.runner, ref, "baseline", code_hash)
    assert verifier(direct.runner, ref, "selected", code_hash)["status"] == "BOTH_ARMS_FROZEN"
    with pytest.raises(IntegrityError, match="after final release"):
        direct.budget.reserve("model/after-freeze", "model", {})
    # The idle scientific arm has not yet been activated in this callback-only
    # test; adding development evidence still invalidates the existing grant.
    research.budget.reserve("experiment/extra", "experiment", {"synthetic": True})
    research.budget.finish("experiment/extra", "invalid", {"synthetic": "extra"}, 0.01)
    with pytest.raises(IntegrityError, match="changed after freeze"):
        verifier(research.runner, ref, "selected", code_hash)


class SyntheticFinal:
    calls = []

    def __init__(self, spec, output, *, executor):
        self.spec, self.store = spec, Store(output)

    def run(self):
        self.calls.append((self.store.root.parent.parent.name, self.store.root.name, self.spec.final_seed))
        self.store.set("workspace_spec", self.store.put_json(self.spec))
        bundle = checkpoint(self.store, self.store.root / "synthetic-output", self.spec.final_seed, "final")
        score = 0.70 + 0.01 * self.spec.final_seed
        report = self.store.put_json({"seed": self.spec.final_seed, "input_bundle": bundle,
                                     "metrics": {"accuracy": score}, "synthetic_fixture": True,
                                     "pipeline": {"inference": {"synthetic": True},
                                                  "score": {"resources": {"gpus": None, "seconds": 1000}}}})
        raw = {"status": "completed", "metric": "accuracy", "direction": "maximize", "score": score,
               "metrics": {"accuracy": score}, "elapsed_seconds": 0.01,
               "official_report_ref": report, "live_model_calls": 0}
        self.store.set("workspace_result", self.store.put_json(raw))
        return SimpleNamespace(model_dump=lambda **kwargs: raw)

    def close(self):
        self.store.close()


def test_release_runs_nine_seed_jobs_zero_models_and_completed_resume_is_idle(tmp_path, monkeypatch):
    study, research, direct = setup(tmp_path)
    create_study_plan(study, research, direct, direct_max_iterations=4)
    freeze_arms(research, direct)
    SyntheticFinal.calls = []
    monkeypatch.setattr(runner_module, "WorkspaceRunner", SyntheticFinal)
    first = release_finals(study, research, direct)
    assert len(SyntheticFinal.calls) == 9
    assert first["reference"]["status"] == first["scientist"]["status"] == first["direct"]["status"] == "completed"
    assert research.store.get("synthetic_final_appendix") is True
    assert first["live_model_calls_during_final"] == 0
    before = [owner.budget.summary() for owner in (research, direct)]
    assert release_finals(study, research, direct) == first
    assert [owner.budget.summary() for owner in (research, direct)] == before
    assert len(SyntheticFinal.calls) == 9
    assert research.store.get("study_final_budget_activation")["deadline"] == direct.store.get("study_final_budget_activation")["deadline"]


def test_separate_verdicts_do_not_turn_completion_or_a_reference_gain_into_added_value():
    def final(values):
        return {"status": "completed", "per_seed_values": values}
    evidence = {"reference": final([0.70, 0.71, 0.72]), "scientist": final([0.73, 0.74, 0.75]),
                "direct": final([0.74, 0.75, 0.76]), "research_state": {"status": "complete"},
                "synthetic_agents": False}
    audits = {name: {"status": "pass"} for name in ("scientist", "direct")}
    report = study_verdicts(evidence, audits=audits)
    assert report["workflow"]["status"] == "pass"
    assert report["useful_improvement"]["status"] == "pass"
    assert report["added_value"]["status"] == "not_demonstrated"
    assert report["mechanism"]["status"] == "unverified"
    evidence["scientist"] = final([0.72, 0.76, 0.70])
    assert study_verdicts(evidence, audits=audits)["useful_improvement"]["status"] == "not_demonstrated"
    evidence["direct"] = {"status": "failed"}
    assert study_verdicts(evidence, audits=audits)["workflow"]["status"] == "fail"


def test_research_run_requires_plan_before_entering_the_controller(tmp_path):
    _, research, _ = setup(tmp_path)
    with pytest.raises(IntegrityError, match="paired study plan"):
        research.run()


def test_audit_requires_replays_and_bounds_them_to_cpu_scoring(tmp_path, monkeypatch):
    from scientisttwo.live import audit as live_audit
    study, research, direct = setup(tmp_path)
    create_study_plan(study, research, direct, direct_max_iterations=4)
    freeze_arms(research, direct)
    monkeypatch.setattr(runner_module, "WorkspaceRunner", SyntheticFinal)
    release_finals(study, research, direct)
    # The static auditors are tested elsewhere against real controller layouts.
    # Here their test doubles isolate acceptance gating and replay admission.
    monkeypatch.setattr(live_audit, "audit_live", lambda *args, **kwargs: {"status": "pass", "synthetic": True})
    monkeypatch.setattr(coordinator_module, "_direct_integrity", lambda owner: {"status": "pass", "synthetic": True})
    monkeypatch.setattr(live_audit, "_audit_workspace_final", lambda *args, **kwargs: {"status": "pass"})
    calls = []
    def replay(store, config, report, prepared, **kwargs):
        assert report["pipeline"]["inference"] is None
        assert report["pipeline"]["score"]["resources"]["gpus"] is None
        assert 0 < report["pipeline"]["score"]["resources"]["seconds"] <= 60
        calls.append(kwargs["proof_id"])
        return report["metrics"], {"id": kwargs["proof_id"], "synthetic": True}
    monkeypatch.setattr(live_audit, "replay_evaluator", replay)
    unverified = audit_study(study, research, direct)
    assert unverified["status"] == "unverified" and calls == []
    denied = audit_study(study, research, direct, reexecute_evaluators=True, max_score_replays=1)
    assert denied["status"] == "fail" and "count bound" in denied["error"] and calls == []
    checked = audit_study(study, research, direct, reexecute_evaluators=True,
                          max_score_replays=9, max_audit_seconds=60)
    assert checked["status"] == "pass" and len(calls) == 9
    assert checked["training_or_inference_reruns"] == checked["live_model_calls"] == 0
    assert not checked["verdicts"]["workflow"]["autonomous_live_validation"]


def test_actual_final_appendix_dispatch_uses_the_activated_final_budget(tmp_path, monkeypatch):
    from scientisttwo.research.stages import ResearchStages
    from scientisttwo.research.manuscript import build_evidence_ledger, fixture_manuscript, render_manuscript
    from test_research_compiler import FakeContainers, IMAGE

    study, research, direct = setup(tmp_path)
    # Both arms declare the compiler before the shared plan is frozen.
    for owner in (research, direct):
        owner.config.workflow.compiler_image = IMAGE
    create_study_plan(study, research, direct, direct_max_iterations=4)
    freeze_arms(research, direct)
    ledger = build_evidence_ledger(research.store, [])
    rendered = render_manuscript(research.store, fixture_manuscript(ledger), ledger)
    research.store.set("research_manuscript_bundle", research.store.put_json(rendered))
    transport = FakeContainers()
    research.runner.raw_executor = transport
    research.research_final_appendix = lambda results: ResearchStages.research_final_appendix(research, results)
    monkeypatch.setattr(runner_module, "WorkspaceRunner", SyntheticFinal)
    result = release_finals(study, research, direct)
    assert result["research_state"]["status"] == "complete"
    final_bundle = research.store.json(research.store.get("research_final_bundle"))
    assert final_bundle["compilation"]["status"] == "success"
    probes = [row for row in research.budget.summary() if row["key"].startswith("artifact/compile/")]
    assert len(probes) == 1 and probes[0]["kind"] == "probe"
    assert research.store.get("study_final_budget_activation") is not None
    assert all("--gpus" not in argv for argv, _, _ in transport.calls)
