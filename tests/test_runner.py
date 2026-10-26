import json
from pathlib import Path
import time

import pytest

from scientisttwo.data import prepare
from scientisttwo.models import EvalSpec, Profile, TaskSpec
from scientisttwo.process import ProcessResult
from scientisttwo.runner import (
    BudgetExhausted,
    ContainerRunner,
    PrerequisiteError,
    evaluator_identity,
)
from scientisttwo.store import IntegrityError, Store
from scientisttwo.tasks import ClassificationTask


class FakeContainer:
    """Simulates the wire protocol, never executes candidate code on the host."""

    def __init__(self, spec, *, interrupt=False, timeout=False):
        self.spec = spec
        self.interrupt = interrupt
        self.timeout = timeout
        self.calls = []

    def __call__(self, argv, timeout, **kwargs):
        self.calls.append(argv)
        if "info" in argv:
            return ProcessResult(0, "arm64", "", 0.01)
        if "inspect" in argv:
            return ProcessResult(0, "sha256:fixture-image", "", 0.01)
        if "run" not in argv:
            return ProcessResult(0, "", "", 0.01)
        if self.timeout:
            return ProcessResult(-9, "", "", 0.1, timed_out=True)
        mounts = {
            value.split(":")[-2]: Path(value.rsplit(":", 2)[0])
            for i, value in enumerate(argv)
            if i and argv[i - 1] == "-v"
        }
        output = mounts["/output"]
        manifest = json.loads((mounts["/input"] / "manifest.json").read_text())
        observations = {}
        for name, seed, arrays, labels in prepare(self.spec):
            observations[f"{name}:{seed}"] = {
                "row_ids": arrays["row_ids"].tolist(),
                "predictions": labels.tolist(),
                "training_seconds": 0.1,
                "inference_seconds": 0.01,
            }
        (output / "observations.json").write_text(json.dumps(observations))
        (output / "completed.json").write_text(
            json.dumps({"request_hash": manifest["request_hash"]})
        )
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt
        return ProcessResult(0, "test container protocol", "", 0.1)


def setup(tmp_path, **options):
    store = Store(tmp_path)
    store.set("evaluator", evaluator_identity())
    store.set("deadline", time.time() + 30)
    adapter = ClassificationTask()
    task = TaskSpec(datasets=["iris"], subset_datasets=["iris"], seeds=[0])
    spec = EvalSpec.for_phase(task, "full")
    fake = FakeContainer(spec, **options)
    runner = ContainerRunner(store, Profile(), adapter=adapter, executor=fake)
    return store, runner, spec, fake, adapter.baseline()


def test_container_flags_and_idempotent_execution(tmp_path):
    store, runner, spec, fake, idea = setup(tmp_path)
    first = runner.execute("job", idea, "test-only source", spec)
    assert first.status == "success"
    second = runner.execute("job", idea, "test-only source", spec)
    assert first == second
    calls = [c for c in fake.calls if "run" in c]
    assert len(calls) == 1
    assert {"--network=none", "--read-only", "--cap-drop=ALL", "--pull=never"} <= set(calls[0])
    assert all(".env" not in value and "docker.sock" not in value for value in calls[0])


def test_completed_output_survives_controller_crash_without_reexecution(tmp_path):
    store, runner, spec, fake, idea = setup(tmp_path, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        runner.execute("job", idea, "test-only source", spec)
    result = runner.execute("job", idea, "test-only source", spec)
    assert result.status == "success"
    assert len([c for c in fake.calls if "run" in c]) == 1
    assert store.db.execute("SELECT charged FROM jobs").fetchone()[0] > 0


def test_timeout_removes_container_and_charges_budget(tmp_path):
    store, runner, spec, fake, idea = setup(tmp_path, timeout=True)
    assert runner.execute("job", idea, "test-only source", spec).status == "timeout"
    assert any("rm" in call for call in fake.calls)
    assert store.db.execute("SELECT charged FROM jobs").fetchone()[0] > 0


def test_expired_budget_does_not_launch(tmp_path):
    store, runner, spec, fake, idea = setup(tmp_path)
    store.set("deadline", time.time() - 1)
    with pytest.raises(BudgetExhausted):
        runner.execute("job", idea, "test-only source", spec)
    assert not any("run" in call for call in fake.calls)


def test_modified_evaluator_blocks_execution(tmp_path):
    store, runner, spec, fake, idea = setup(tmp_path)
    store.set("evaluator", {"modified": True})
    with pytest.raises(IntegrityError, match="evaluator"):
        runner.execute("job", idea, "test-only source", spec)


def test_daemon_failure_has_no_host_fallback(tmp_path):
    store, _, spec, _, idea = setup(tmp_path)

    def unavailable(*args, **kwargs):
        return ProcessResult(1, "", "daemon unavailable", 0)

    runner = ContainerRunner(store, Profile(), adapter=ClassificationTask(), executor=unavailable)
    with pytest.raises(PrerequisiteError):
        runner.execute("job", idea, "never execute me", spec)
    assert not store.db.execute("SELECT * FROM jobs").fetchall()


def test_partial_output_is_interrupted_and_never_used_as_scientific_evidence(tmp_path):
    store, runner, spec, fake, idea = setup(tmp_path, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        runner.execute("job", idea, "test-only source", spec)
    for marker in tmp_path.glob("work/*/output/completed.json"):
        marker.unlink()
    result = runner.execute("job", idea, "test-only source", spec)
    assert result.status == "interrupted" and not result.rows
    assert len([c for c in fake.calls if "run" in c]) == 1


def test_modified_observations_are_detected_before_cached_result_reuse(tmp_path):
    store, runner, spec, fake, idea = setup(tmp_path)
    result = runner.execute("job", idea, "test-only source", spec)
    (store.root / result.observations_ref).write_text("{}")
    with pytest.raises(IntegrityError, match="modified"):
        runner.execute("job", idea, "test-only source", spec)
