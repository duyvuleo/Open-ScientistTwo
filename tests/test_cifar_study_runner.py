"""Explicitly synthetic checkpoint traces; no actual training or final scoring."""

import json
from types import SimpleNamespace

import pytest

from scientisttwo.artifacts import ArtifactSpec, freeze_bundle
from scientisttwo.live.models import DevelopmentSpec, LiveSpec, Observation
from scientisttwo.live.runtime import Budget, DevelopmentRunner
from scientisttwo.live.source_tree import candidate_snapshot, freeze_source_tree
from scientisttwo.live.workspace_models import WorkspaceSpec
from scientisttwo.store import IntegrityError, Store, canonical, digest
from scientisttwo.studies import runner as module
from scientisttwo.studies.cifar_recipe import (
    BASELINE_CONFIGURATION, FIXED_CONFIGURATION, baseline_candidate, recipe_candidate,
    validate_candidate, write_study_sources,
)
from scientisttwo.studies.runner import (
    StudyDevelopmentRunner, candidate_source_hash, validate_observation_receipts, verify_final_aggregation,
)


def checkpoint(store, directory, seed, *, phase="full", change=None, overrides=None):
    root = directory / "checkpoint"
    root.mkdir(parents=True)
    config = {"seed": seed, **FIXED_CONFIGURATION, **BASELINE_CONFIGURATION, **(overrides or {})}
    receipt = {"task": "cifar10", "seed": seed, "steps": {"screen": 1570, "full": 3130, "final": 3910}[phase],
               "max_steps": 3910, "epochs_started": 10}
    receipt.update(change or {})
    (root / "config.json").write_bytes(canonical(config))
    (root / "training.json").write_bytes(canonical(receipt))
    (root / "weights.pt").write_bytes(b"SYNTHETIC checkpoint; do not deserialize")
    return freeze_bundle(store, directory, [ArtifactSpec(path="checkpoint", kind="directory")])


def setup(tmp_path):
    source = tmp_path / "study"
    write_study_sources(source)
    for name in ("public", "private"):
        (source / name).mkdir()
        (source / name / "synthetic.txt").write_text("SYNTHETIC data; never execute")
    native = WorkspaceSpec(name="synthetic-study", task_id="cifar10", public_dir=source / "public",
                           private_dir=source / "private", candidate_dir=source / "reference",
                           evaluator_dir=source / "evaluator", command=["python", "train.py"],
                           evaluator_command=["python", "evaluate.py"], metric="accuracy", direction="maximize",
                           image="sha256:" + "a" * 64, grader_image="sha256:" + "b" * 64)
    config = LiveSpec(native=native, development=DevelopmentSpec(metric="accuracy", direction="maximize", seeds=[0, 1, 2]),
                      model="SYNTHETIC-no-calls", max_calls=1, max_experiments=20, campaign_seconds=60)
    store = Store(tmp_path / "campaign")
    snapshot = freeze_source_tree(store, source / "reference", ["training_config.json"])
    store.set("authorized_source_snapshot", snapshot)
    store.set("live_images", {"execution": native.image, "grading": native.grader_image})
    candidate = baseline_candidate(snapshot)
    adapter = SimpleNamespace(validate=lambda item: validate_candidate(item, snapshot_ref=snapshot),
                              identity=lambda: {"synthetic": "study-contract-test"})
    budget = Budget(store, config)
    runner = StudyDevelopmentRunner(store, config, adapter, budget,
        final_grant_ref="synthetic-grant-reference", final_grant_verifier=lambda owner, ref, label, source_hash: {
            "status": "BOTH_ARMS_FROZEN", "grant_digest": "d" * 64,
            "candidate_source_hash": source_hash, "seeds": [0, 1, 2]})
    return runner, candidate


def observation(runner, candidate, *, change=None, phase="full", seeds=(0, 1, 2)):
    store = runner.store
    protocol = {"phase": phase, "seeds": list(seeds)}
    overrides = json.loads(candidate.files[0].content) if candidate.files else {}
    bundles = [checkpoint(store, store.root / f"outputs-{index}-{seed}", seed,
                          phase="screen" if phase == "subset" else "full", overrides=overrides,
                          change=change if seed == 1 else None)
               for index, seed in enumerate(seeds)]
    return Observation(id="synthetic", phase=phase, status="success", metric="accuracy", value=0.7,
                       values=[0.7] * len(seeds), protocol_ref=store.put_json(protocol),
                       protocol_hash=digest(canonical(protocol)), code_ref=store.put_json(candidate),
                       source_tree_ref=store.put_json(candidate_snapshot(store, candidate)),
                       prediction_refs=[store.put(b"SYNTHETIC prediction")], bundle_refs=bundles)


def test_every_development_seed_receipt_is_validated(tmp_path):
    runner, candidate = setup(tmp_path)
    result = observation(runner, candidate)
    proof = validate_observation_receipts(runner.store, result, expected_seeds=[0, 1, 2])
    assert [row["seed"] for row in proof["receipts"]] == [0, 1, 2]
    with pytest.raises(IntegrityError, match="requested evaluation"):
        validate_observation_receipts(runner.store, result, expected_seeds=[0])


@pytest.mark.parametrize("kind", ["repeat", "paired_control"])
def test_rebuttal_fresh_seeds_reach_execution_and_bind_each_receipt(tmp_path, monkeypatch, kind):
    runner, candidate = setup(tmp_path)
    if kind == "paired_control":
        candidate = recipe_candidate(candidate.base_snapshot_ref, {"horizontal_flip_probability": 0.0},
                                     description="Synthetic matched control removing horizontal flips")
    raw = observation(runner, candidate, phase="rebuttal", seeds=[3, 4])
    calls = []

    def execute(owner, key, submitted, phase, seeds):
        calls.append((key, submitted, phase, seeds))
        return raw

    monkeypatch.setattr(DevelopmentRunner, "execute", execute)
    assert runner.execute(kind, candidate, phase="rebuttal", seeds=[3, 4]) == raw
    assert calls == [(kind, candidate, "rebuttal", [3, 4])]
    proof = validate_observation_receipts(runner.store, raw, expected_seeds=[3, 4])
    assert proof["seeds"] == [3, 4]
    assert [row["seed"] for row in proof["receipts"]] == [3, 4]
    with pytest.raises(IntegrityError, match="requested evaluation"):
        validate_observation_receipts(runner.store, raw, expected_seeds=[4, 3])
    # Rebinding the protocol cannot relabel checkpoints from different executions.
    changed = {"phase": "rebuttal", "seeds": [4, 3]}
    swapped = raw.model_copy(update={"protocol_ref": runner.store.put_json(changed),
                                     "protocol_hash": digest(canonical(changed))})
    with pytest.raises(IntegrityError, match="declared seed"):
        validate_observation_receipts(runner.store, swapped, expected_seeds=[4, 3])


@pytest.mark.parametrize("phase,seeds", [
    ("rebuttal", [-1]), ("rebuttal", [2**32]), ("rebuttal", [True]),
    ("rebuttal", [3.0]), ("rebuttal", [[3]]), ("rebuttal", []),
    ("rebuttal", [3, 3]), ("rebuttal", [3, 4, 5]),
    ("subset", [3]), ("full", [3]), ("final", [0]), ("unknown", [0]),
])
def test_invalid_phase_seed_request_is_rejected_before_execution(tmp_path, monkeypatch, phase, seeds):
    runner, candidate = setup(tmp_path)
    monkeypatch.setattr(DevelopmentRunner, "execute",
                        lambda *args, **kwargs: pytest.fail("Invalid seeds reached execution"))
    with pytest.raises(ValueError, match="phase-bounded training seeds"):
        runner.execute("bad-seeds", candidate, phase=phase, seeds=seeds)
    assert not runner.budget.summary()


@pytest.mark.parametrize("seeds", [[-1], [2**32], [True], [3.0], [[3]], [], [3, 3], list(range(3, 36))])
def test_rebuttal_receipt_validation_rejects_invalid_protocol_seed_lists(tmp_path, seeds):
    runner, candidate = setup(tmp_path)
    raw = observation(runner, candidate, phase="rebuttal", seeds=[3, 4])
    protocol = {"phase": "rebuttal", "seeds": seeds}
    changed = raw.model_copy(update={"protocol_ref": runner.store.put_json(protocol),
                                     "protocol_hash": digest(canonical(protocol))})
    with pytest.raises(IntegrityError, match="training seeds"):
        validate_observation_receipts(runner.store, changed)


def test_rebuttal_supports_declared_maximum_count_and_uint32_boundary(tmp_path, monkeypatch):
    runner, candidate = setup(tmp_path)
    runner.config.supplementary_seed_count = 32
    seeds = [*range(3, 34), 2**32 - 1]
    raw = observation(runner, candidate, phase="rebuttal", seeds=seeds)
    monkeypatch.setattr(DevelopmentRunner, "execute", lambda *args, **kwargs: raw)
    assert runner.execute("boundary", candidate, phase="rebuttal", seeds=seeds) == raw
    proof = validate_observation_receipts(runner.store, raw, expected_seeds=seeds)
    assert len(proof["receipts"]) == 32 and proof["receipts"][-1]["seed"] == 2**32 - 1


@pytest.mark.parametrize("phase", ["subset", "full"])
def test_selection_phase_receipts_still_reject_fresh_supplementary_seeds(tmp_path, phase):
    runner, candidate = setup(tmp_path)
    raw = observation(runner, candidate, phase=phase, seeds=[3])
    with pytest.raises(IntegrityError, match="training seeds"):
        validate_observation_receipts(runner.store, raw, expected_seeds=[3])


def test_calibration_full_single_seed_and_protocol_phase_binding_are_preserved(tmp_path, monkeypatch):
    runner, candidate = setup(tmp_path)
    raw = observation(runner, candidate, seeds=[0])
    monkeypatch.setattr(DevelopmentRunner, "execute", lambda *args, **kwargs: raw)
    assert runner.execute("calibration", candidate, seeds=[0]) == raw
    protocol = {"phase": "rebuttal", "seeds": [0]}
    changed = raw.model_copy(update={"protocol_ref": runner.store.put_json(protocol),
                                     "protocol_hash": digest(canonical(protocol))})
    with pytest.raises(IntegrityError, match="protocol/phase binding"):
        validate_observation_receipts(runner.store, changed, expected_seeds=[0])


def test_invalid_receipt_blocks_promotion_without_rewriting_execution(tmp_path, monkeypatch):
    runner, candidate = setup(tmp_path)
    raw = observation(runner, candidate, change={"steps": 1})
    runner.budget.reserve("experiment/synthetic", "experiment", {"synthetic": True})
    runner.budget.finish("experiment/synthetic", "success", raw, 1.0)
    before = runner.budget.summary()
    monkeypatch.setattr(DevelopmentRunner, "execute", lambda *args, **kwargs: raw)
    with pytest.raises(IntegrityError, match="immutable execution retained"):
        runner.execute("synthetic", candidate)
    assert runner.budget.summary() == before
    proof = runner.store.cached("study-training-validation/synthetic", {
        "observation_sha256": digest(canonical(raw)), "seeds": [0, 1, 2]})
    assert proof["status"] == "fail" and proof["promotion"] == "blocked"
    assert runner.store.json(proof["raw_observation_ref"])["status"] == "success"


def test_successful_development_proof_is_idempotent(tmp_path, monkeypatch):
    runner, candidate = setup(tmp_path)
    raw = observation(runner, candidate)
    monkeypatch.setattr(DevelopmentRunner, "execute", lambda *args, **kwargs: raw)
    assert runner.execute("synthetic", candidate) == raw
    count = runner.store.db.execute("SELECT count(*) FROM artifacts").fetchone()[0]
    assert runner.execute("synthetic", candidate) == raw
    assert runner.store.db.execute("SELECT count(*) FROM artifacts").fetchone()[0] == count


class SyntheticFinal:
    calls = []
    fail_seed = None
    interrupt_seed = None
    bad_training_seed = None

    def __init__(self, spec, output, *, executor):
        self.spec, self.store = spec, Store(output)

    def run(self):
        seed = self.spec.final_seed
        self.calls.append(seed)
        self.store.set("workspace_spec", self.store.put_json(self.spec))
        if seed == self.interrupt_seed:
            raise RuntimeError("Synthetic process disappearance")
        raw = {"status": "execution_failed", "score": None, "elapsed_seconds": 0.01,
               "metric": "accuracy", "direction": "maximize", "live_model_calls": 0,
               "reason": "Explicitly synthetic final trace"}
        if seed != self.fail_seed:
            bundle = checkpoint(self.store, self.store.root / "outputs", seed, phase="final",
                                change={"steps": 1} if seed == self.bad_training_seed else None)
            score = [0.68, 0.70, 0.72][seed]
            report = {"seed": seed, "metrics": {"accuracy": score}, "input_bundle": bundle}
            raw.update(status="completed", score=score, metrics={"accuracy": score},
                       official_report_ref=self.store.put_json(report))
        self.store.set("workspace_result", self.store.put_json(raw))
        return SimpleNamespace(model_dump=lambda **kwargs: raw)

    def close(self):
        self.store.close()


@pytest.fixture
def fake_finals(monkeypatch):
    SyntheticFinal.calls = []
    SyntheticFinal.fail_seed = SyntheticFinal.interrupt_seed = SyntheticFinal.bad_training_seed = None
    monkeypatch.setattr(module, "WorkspaceRunner", SyntheticFinal)
    return SyntheticFinal


def test_three_final_seeds_charge_actual_runs_and_resume_without_new_operations(tmp_path, fake_finals):
    runner, candidate = setup(tmp_path)
    result = runner.final("baseline", candidate)
    assert result["status"] == "completed" and result["per_seed_values"] == [0.68, 0.70, 0.72]
    assert result["score"] == pytest.approx(0.70)
    assert fake_finals.calls == [0, 1, 2]
    rows = runner.budget.summary()
    assert len(rows) == 4 and all(row["kind"] == "experiment" for row in rows)
    aggregate = next(row for row in rows if row["key"] == "final/baseline")
    assert aggregate["seconds"] == 0
    assert sum(row["seconds"] for row in rows) == result["elapsed_seconds"]
    assert sorted(path.parent.name for path in (runner.store.root / "final").glob("*/campaign.sqlite")) == [
        "baseline-seed-0", "baseline-seed-1", "baseline-seed-2"]
    assert runner.final("baseline", candidate) == result
    assert runner.budget.summary() == rows and fake_finals.calls == [0, 1, 2]


@pytest.mark.parametrize("grant", [None, True, {"status": "BOTH_ARMS_FROZEN", "seeds": [0]}])
def test_final_denied_before_any_final_runner_or_ledger_operation(tmp_path, fake_finals, grant):
    runner, candidate = setup(tmp_path)
    runner.final_grant_verifier = lambda *args: grant
    with pytest.raises(IntegrityError, match="grant"):
        runner.final("selected", candidate)
    assert not fake_finals.calls and not runner.budget.summary()
    assert not (runner.store.root / "final").exists()


def test_failed_final_seed_never_reports_partial_mean(tmp_path, fake_finals):
    runner, candidate = setup(tmp_path)
    fake_finals.fail_seed = 1
    result = runner.final("selected", candidate)
    assert result["status"] == "failed" and result["score"] is None and result["metrics"] == {}
    assert result["per_seed_values"] == [] and result["partial_seed_values"] == [0.68]
    assert fake_finals.calls == [0, 1]


def test_final_training_violation_preserves_original_result_and_blocks_aggregate(tmp_path, fake_finals):
    runner, candidate = setup(tmp_path)
    fake_finals.bad_training_seed = 0
    result = runner.final("baseline", candidate)
    assert result["score"] is None and result["status"] == "failed"
    first = result["seed_results"][0]
    assert first["status"] == "invalid" and first["failure_proof_ref"]
    child = Store(runner.store.root / first["child_store"])
    try:
        assert child.json(first["raw_result_ref"])["status"] == "completed"
        assert child.json(first["failure_proof_ref"])["promotion"] == "blocked"
    finally:
        child.close()


def test_unknown_final_completion_is_not_reexecuted(tmp_path, fake_finals):
    runner, candidate = setup(tmp_path)
    fake_finals.interrupt_seed = 0
    with pytest.raises(RuntimeError, match="process disappearance"):
        runner.final("selected", candidate)
    assert fake_finals.calls == [0]
    result = runner.final("selected", candidate)
    assert result["status"] == "interrupted" and result["score"] is None
    assert fake_finals.calls == [0]


def test_aggregate_tampering_and_candidate_change_are_detected(tmp_path, fake_finals):
    runner, candidate = setup(tmp_path)
    result = runner.final("baseline", candidate)
    with pytest.raises(IntegrityError, match="three executed"):
        verify_final_aggregation(runner.store, {**result, "score": 0.99})
    changed = json.loads(json.dumps(result))
    changed["seed_results"][0]["score"] = 0.99
    with pytest.raises(IntegrityError, match="immutable seed"):
        verify_final_aggregation(runner.store, changed)
    before = candidate_source_hash(runner.store, candidate)
    assert len(before) == 64
