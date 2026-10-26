"""Regression checks for observed generic-runner bugs, using synthetic transport only."""

import json
from pathlib import Path

import pytest

from scientisttwo.benchmarks.files import tree_manifest
from scientisttwo.live.audit import audit_live
from scientisttwo.live.models import LiveSpec
from scientisttwo.live.runtime import Budget, DevelopmentRunner, validate_prepared
from scientisttwo.live.workspace_task import WorkspaceRunner
from scientisttwo.process import ProcessResult
from scientisttwo.store import Store

from test_live_development import runtime_setup
from test_workspace_task import (
    CalibrationDevelopment,
    WorkspaceTransportDouble,
    development,
    make_workspace,
)


def mounts_for(argv):
    return {
        argv[index + 1].rsplit(":", 2)[1]: Path(argv[index + 1].rsplit(":", 2)[0])
        for index, arg in enumerate(argv[:-1])
        if arg == "-v"
    }


class StaleGradeTransport(WorkspaceTransportDouble):
    """First failed grader leaves output; second successful process omits its output."""

    def __init__(self):
        super().__init__()
        self.grading_directories = []

    def __call__(self, argv, timeout, **kwargs):
        if "run" in argv:
            mounts = mounts_for(argv)
            if "/grade-output" in mounts:
                self.calls.append(argv)
                output = mounts["/grade-output"]
                self.grading_directories.append(output)
                if len(self.grading_directories) == 1:
                    (output / "grade.json").write_text(
                        json.dumps(
                            {
                                "valid_submission": True,
                                "metrics": {"calibration_error": 123.0},
                            }
                        )
                    )
                    return ProcessResult(1, "SYNTHETIC failed grader left output", "", 0)
                return ProcessResult(0, "SYNTHETIC grader omitted output", "", 0)
        return super().__call__(argv, timeout, **kwargs)


def test_failed_grading_retry_cannot_reuse_previous_output(tmp_path):
    spec = make_workspace(tmp_path / "inputs")
    fake = StaleGradeTransport()
    runner = WorkspaceRunner(spec, tmp_path / "run", executor=fake)
    try:
        first, second = runner.run(), runner.run()
        assert first.status == second.status == "grading_failed"
        assert second.score is None
        assert len(set(fake.grading_directories)) == 2
        assert (fake.grading_directories[0] / "grade.json").exists()
        assert not (fake.grading_directories[1] / "grade.json").exists()
    finally:
        runner.close()


def test_uncommitted_final_grading_is_never_repeated(tmp_path):
    spec = make_workspace(tmp_path / "inputs")
    fake = WorkspaceTransportDouble()
    grades = []

    def interrupted(argv, timeout, **kwargs):
        if "run" in argv and "/grade-output" in mounts_for(argv):
            grades.append(argv)
            raise KeyboardInterrupt
        return fake(argv, timeout, **kwargs)

    runner = WorkspaceRunner(spec, tmp_path / "run", executor=interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.run()
        result = runner.run()
        assert result.status == "interrupted"
        assert "grading" in result.reason and len(grades) == 1
    finally:
        runner.close()


def test_final_seed_is_explicit_and_shared_by_execution_and_grading(tmp_path):
    spec = make_workspace(tmp_path / "inputs").model_copy(update={"final_seed": 37})
    fake = WorkspaceTransportDouble()
    jobs = []

    def capture(argv, timeout, **kwargs):
        if "run" in argv:
            jobs.append(json.loads((mounts_for(argv)["/control"] / "job.json").read_text()))
        return fake(argv, timeout, **kwargs)

    runner = WorkspaceRunner(spec, tmp_path / "run", executor=capture)
    try:
        assert runner.run().status == "completed"
        assert len(jobs) == 2 and all(job["seed"] == 37 for job in jobs)
        assert runner.store.json(runner.store.get("workspace_spec"))["final_seed"] == 37
    finally:
        runner.close()


def test_grader_completion_before_final_receipt_cannot_trigger_regrade(tmp_path, monkeypatch):
    spec = make_workspace(tmp_path / "inputs")
    fake = WorkspaceTransportDouble()
    runner = WorkspaceRunner(spec, tmp_path / "run", executor=fake)
    original = runner.store.put_json

    def interrupted(value, kind="evidence"):
        if kind == "workspace-final-grade":
            raise KeyboardInterrupt
        return original(value, kind)

    try:
        monkeypatch.setattr(runner.store, "put_json", interrupted)
        with pytest.raises(KeyboardInterrupt):
            runner.run()
        monkeypatch.setattr(runner.store, "put_json", original)
        before = sum("run" in argv for argv in fake.calls)
        result = runner.run()
        assert result.status == "interrupted"
        assert sum("run" in argv for argv in fake.calls) == before
    finally:
        runner.close()


@pytest.mark.parametrize("protected", ["private_dir", "evaluator_dir"])
def test_adapter_cannot_redirect_candidate_public_mount_to_protected_tree(tmp_path, protected):
    spec = make_workspace(tmp_path / "inputs")

    class Misdirected(CalibrationDevelopment):
        def prepare(self, phase, destination):
            root = getattr(self.native, protected)
            return {"public": str(root), "files": tree_manifest(root)}

    config = LiveSpec(
        native=spec,
        development=development(),
        model="synthetic-not-invoked",
        max_calls=1,
        campaign_seconds=60,
    )
    store = Store(tmp_path / "run")
    fake = WorkspaceTransportDouble()
    runner = DevelopmentRunner(
        store, config, Misdirected(spec, config.development), Budget(store, config), executor=fake
    )
    try:
        with pytest.raises(ValueError, match="owned development"):
            runner.preflight()
        assert all("run" not in argv for argv in fake.calls)
    finally:
        store.close()


def test_owned_preparation_symlink_cannot_alias_protected_tree(tmp_path):
    spec = make_workspace(tmp_path / "inputs")
    destination = tmp_path / "phase"
    destination.mkdir()
    link = destination / "public"
    link.symlink_to(spec.private_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="owned development"):
        validate_prepared(
            {"public": str(link), "files": tree_manifest(spec.private_dir)}, destination
        )


@pytest.mark.parametrize("mutation", ["escape", "missing", "wrong_primary"])
def test_audit_rejects_malformed_prediction_bundle_before_replay(tmp_path, monkeypatch, mutation):
    store, adapter, runner, _ = runtime_setup(tmp_path, monkeypatch)
    try:
        observation = runner.execute("probe", adapter.baseline())
        assert observation.status == "success"
        serialized = observation.model_dump(mode="json")
        if mutation == "escape":
            serialized["artifact_refs"][0] = {"../escape.csv": observation.prediction_refs[0]}
        elif mutation == "missing":
            serialized["artifact_refs"][0] = {}
        else:
            serialized["artifact_refs"][0]["submission.csv"] = observation.code_ref
        changed = store.put_json(serialized, "SYNTHETIC-malformed-observation")
        with store.db:
            store.db.execute(
                "UPDATE calls SET output_ref=? WHERE key='development/probe'", (changed,)
            )
            # Keep the execution-ledger binding coherent so this test reaches
            # the artifact-contract check rather than the earlier receipt check.
            store.db.execute(
                "UPDATE live_operations SET result_ref=? WHERE key='experiment/probe'", (changed,)
            )
        report = audit_live(store.root)
        assert report["status"] == "fail"
        assert any("output contract" in error for error in report["errors"])
        assert not (tmp_path / "escape.csv").exists()
    finally:
        store.close()
