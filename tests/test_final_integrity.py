"""Final evidence stays separate, bound, and auditable without another agent call."""

import pytest

from scientisttwo.live.audit import audit_live, _replay_legacy_final
from scientisttwo.live.paired_evaluation import execute_paired_final
from scientisttwo.live.models import Observation
from scientisttwo.live.workspace_task import WorkspaceRunner
from scientisttwo.store import IntegrityError, Store
from test_paired_evaluation import setup as paired_setup, synthetic_pair
from test_staged_evaluation import StagedTransportDouble, make_staged_task, staged_development
from test_workspace_task import make_workspace, WorkspaceTransportDouble


def paired_audit_setup(tmp_path, monkeypatch):
    store, adapter, runner = paired_setup(tmp_path, monkeypatch)
    store.set("live_spec_ref", store.put_json(runner.config))
    monkeypatch.setattr("scientisttwo.live.audit.load_development", lambda *args: adapter)
    store.set("live_frozen", {"fixture": "synthetic frozen method"})
    final = execute_paired_final(runner, "selected", adapter.baseline(), run_pair=synthetic_pair)
    assert final["status"] == "completed"
    return store, adapter, runner, final


def replace_final(store, observation):
    ref = store.put_json(observation)
    with store.db:
        store.db.execute("UPDATE calls SET output_ref=? WHERE key='final-paired/selected'", (ref,))
        store.db.execute(
            "UPDATE live_operations SET result_ref=? WHERE key='final/selected'", (ref,)
        )


def test_final_paired_audit_checks_evidence_without_exposing_it_to_development(
    tmp_path, monkeypatch
):
    store, _, runner, final = paired_audit_setup(tmp_path, monkeypatch)
    try:
        before = runner.budget.summary()
        audit = audit_live(store.root)
        assert audit["status"] == "pass", audit
        assert any(
            check.get("final_experiment") == "selected"
            and check.get("scope") == "protected_final_only"
            for check in audit["checks"]
        )
        assert not any(check.get("experiment") == "selected" for check in audit["checks"])
        assert runner.budget.summary() == before
        assert audit["verification"]["complete"] is True
        assert audit["verification"]["container_replay_count"] == 0
        assert len(audit["verification"]["paired_correctness_recomputations"]) == 1
        store.set(
            "live_draft",
            store.put_json(
                {
                    "claims": [
                        {
                            "experiment_id": "selected",
                            "metric": "latency_ratio",
                            "value": final["score"],
                        }
                    ]
                }
            ),
        )
        failed = audit_live(store.root)
        assert failed["status"] == "fail" and "Unsupported draft claim" in failed["errors"]
    finally:
        store.close()


def test_final_paired_audit_recalculates_stored_timing_summaries(tmp_path, monkeypatch):
    store, _, _, final = paired_audit_setup(tmp_path, monkeypatch)
    try:
        observation = Observation.model_validate(store.json(final["observation_ref"]))
        evidence = store.json(observation.timing_refs[0])
        report = store.json(evidence["report_ref"])
        report["candidate"]["median_seconds"] *= 10
        evidence["report_ref"] = store.put_json(report)
        observation.timing_refs[0] = store.put_json(evidence)
        replace_final(store, observation)
        failed = audit_live(store.root)
        assert failed["status"] == "fail", failed
        assert any("summaries" in error for error in failed["errors"])
    finally:
        store.close()


@pytest.mark.parametrize(
    "change", ["missing_freeze", "wrong_phase", "missing_timing", "wrong_source"]
)
def test_final_paired_audit_rejects_missing_or_misbound_evidence(tmp_path, monkeypatch, change):
    store, _, _, final = paired_audit_setup(tmp_path, monkeypatch)
    try:
        observation = Observation.model_validate(store.json(final["observation_ref"]))
        if change == "missing_freeze":
            store.set("live_frozen", None)
        elif change == "wrong_phase":
            observation.phase = "full"
            replace_final(store, observation)
        elif change == "missing_timing":
            observation.timing_refs = []
            replace_final(store, observation)
        else:
            observation.code_ref = store.put_json({"fixture": "unbound source"})
            replace_final(store, observation)
        assert audit_live(store.root)["status"] == "fail"
    finally:
        store.close()


def test_completed_final_metrics_must_equal_trusted_pipeline_grade(tmp_path, monkeypatch):
    fake = StagedTransportDouble()
    store, runner, _ = staged_development(tmp_path, monkeypatch, executor=fake)
    child = WorkspaceRunner(runner.spec, store.root / "final" / "selected", executor=fake)
    try:
        outcome = child.run()
        assert outcome.status == "completed"
        assert audit_live(store.root)["status"] == "pass"
        forged = outcome.model_copy(update={"score": 123.0, "metrics": {"mse": 123.0}})
        child.store.set("workspace_result", child.store.put_json(forged))
        failed = audit_live(store.root)
        assert failed["status"] == "fail", failed
        assert any("Final workspace metrics differ" in error for error in failed["errors"])
    finally:
        child.close()
        store.close()


def test_final_execution_cannot_rewrite_its_frozen_source(tmp_path):
    spec = make_staged_task(tmp_path / "inputs")
    fake = StagedTransportDouble()

    def mutate(argv, timeout, **kwargs):
        result = fake(argv, timeout, **kwargs)
        if "run" in argv and fake.stages[-1][0] == "train":
            work = fake.stages[-1][1]["/workspace"]
            (work / "custom.py").write_text("raise RuntimeError('changed after training')\n")
        return result

    runner = WorkspaceRunner(spec, tmp_path / "final", executor=mutate)
    try:
        with pytest.raises(IntegrityError, match="frozen final source"):
            runner.run()
        assert [stage for stage, _ in fake.stages] == ["train"]
        assert runner.run().status == "interrupted"
        assert len(fake.stages) == 1
    finally:
        runner.close()


def test_legacy_final_replay_retains_real_execution_proof_after_temporary_cleanup(
    tmp_path, monkeypatch
):
    spec = make_workspace(tmp_path / "inputs")
    fake = WorkspaceTransportDouble()
    runner = WorkspaceRunner(spec, tmp_path / "final", executor=fake)
    proofs = Store(tmp_path / "audit-proofs")
    monkeypatch.setattr("scientisttwo.live.audit.run_process", fake)
    try:
        result = runner.run()
        assert result.status == "completed"
        metrics, record = _replay_legacy_final(
            runner.store, proofs, spec, result.model_dump(), "final/fixture"
        )
        assert metrics == result.metrics
        proof = proofs.json(record["evidence_ref"])
        assert proof["id"] == "final/fixture" and proof["status"] == "pass"
        assert proof["source_grade_ref"] == result.official_report_ref
        assert proofs.json(proof["replayed_grade_ref"])["metrics"] == metrics
        assert all(proofs.json(ref)["returncode"] == 0 for ref in proof["execution_refs"])
        assert len([argv for argv in fake.calls if "run" in argv]) == 3
        assert proofs.verify() == []
    finally:
        runner.close()
        proofs.close()
