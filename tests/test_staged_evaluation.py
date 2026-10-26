"""Staged checkpoint evaluation: synthetic fixtures, real isolation when opted in.

Fake transports exercise contracts only. The optional container test executes the
same train/infer/score sources and derives every numeric result from execution.
"""

import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest

from scientisttwo.artifacts import ArtifactSpec, freeze_bundle, load_bundle
from scientisttwo.benchmarks.files import tree_manifest
from scientisttwo.benchmarks.models import NativeResources
from scientisttwo.live.adapters import implementation_identity
from scientisttwo.live.audit import audit_live
from scientisttwo.live.models import Candidate, DevelopmentSpec, LiveSpec, SourceFile
from scientisttwo.live.pipeline import (
    EvaluationPipeline,
    EvaluationStage,
    evaluate_pipeline,
)
from scientisttwo.live.runtime import Budget, DevelopmentRunner
from scientisttwo.live.workspace_models import WorkspaceSpec
from scientisttwo.live.workspace_task import WorkspaceRunner, freeze_workspace
from scientisttwo.process import ProcessResult
from scientisttwo.runtime_contracts import RuntimeAsset
from scientisttwo.store import Store


TRAIN = """from pathlib import Path
import numpy as np
assert not Path("/data/private").exists()
assert not Path("/evaluator").exists()
assert not Path("/predictions").exists()
x = np.load("/data/public/train-x.npy", allow_pickle=False)
y = np.load("/data/public/train-y.npy", allow_pickle=False)
weights = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
Path("checkpoint/tokenizer").mkdir(parents=True)
np.save("checkpoint/weights.npy", weights)
np.save("checkpoint/padding.npy", np.zeros(400000, dtype=np.float64))
Path("checkpoint/tokenizer/config.json").write_text('{"fixture":"synthetic"}')
"""

CUSTOM = """import numpy as np
def predict(values, weights):
    return np.column_stack([values, np.ones(len(values))]) @ weights
"""

INFER = """from pathlib import Path
import sys
import numpy as np
assert not Path("/data/private").exists()
assert not Path("/predictions").exists()
assert not Path("/grade-output").exists()
assert Path("/candidate/custom.py").is_file()
assert Path("/artifacts/checkpoint/padding.npy").stat().st_size > 3000000
sys.path.insert(0, "/candidate")
from custom import predict
weights = np.load("/artifacts/checkpoint/weights.npy", allow_pickle=False)
values = np.load("/data/public/test-x.npy", allow_pickle=False)
np.save("/output/predictions.npy", predict(values, weights))
"""

SCORE = """from pathlib import Path
import json
import numpy as np
assert not Path("/candidate").exists()
assert not Path("/workspace").exists()
assert not Path("/artifacts").exists()
assert not Path("/data/public").exists()
prediction = np.load("/predictions/predictions.npy", allow_pickle=False)
labels = np.load("/data/private/labels.npy", allow_pickle=False)
valid = prediction.shape == labels.shape and np.isfinite(prediction).all()
metrics = {"mse": float(np.mean((prediction - labels) ** 2))} if valid else {}
Path("/grade-output/grade.json").write_text(json.dumps({"valid_submission": bool(valid), "metrics": metrics}))
"""


def checkpoint_contract():
    return ArtifactSpec(
        path="checkpoint", kind="directory", role="checkpoint", max_bytes=8_000_000, max_files=10
    )


def make_staged_task(root, *, context=None, image="synthetic-staged-image"):
    root = Path(root).resolve()
    public, private, candidate, evaluator = [
        root / name for name in ("public", "private", "candidate", "evaluator")
    ]
    inference, scoring = evaluator / "inference", evaluator / "score"
    for directory in (public, private, candidate, inference, scoring):
        directory.mkdir(parents=True)
    x = np.linspace(-4, 4, 30)
    np.save(public / "train-x.npy", x)
    np.save(public / "train-y.npy", 3 * x + 2)
    np.save(public / "test-x.npy", np.array([101.0, 103.0, 109.0]))
    np.save(private / "labels.npy", np.array([305.0, 311.0, 329.0]))
    (private / "FINAL_ONLY_CANARY.txt").write_text("FINAL_STAGED_LABELS_MUST_NOT_ENTER_REFINEMENT")
    (candidate / "train.py").write_text(TRAIN)
    (candidate / "custom.py").write_text(CUSTOM)
    (inference / "infer.py").write_text(INFER)
    (scoring / "score.py").write_text(SCORE)
    resources = NativeResources(seconds=60, grading_seconds=60)
    pipeline = EvaluationPipeline(
        inference=EvaluationStage(
            kind="inference",
            image=image,
            source_dir=inference,
            source_manifest=tree_manifest(inference),
            command=["python", "/evaluator/infer.py"],
            outputs=[ArtifactSpec(path="predictions.npy", role="prediction")],
            resources=resources,
        ),
        score=EvaluationStage(
            kind="score",
            image=image,
            source_dir=scoring,
            source_manifest=tree_manifest(scoring),
            command=["python", "/evaluator/score.py"],
            resources=resources,
        ),
        private_dir=private,
        private_manifest=tree_manifest(private),
    )
    return freeze_workspace(
        WorkspaceSpec(
            name="staged-fixture",
            task_id="staged-fixture",
            data_provenance="contract_fixture",
            public_dir=public,
            private_dir=private,
            candidate_dir=candidate,
            evaluator_dir=evaluator,
            command=["python", "train.py"],
            evaluator_command=["python", "score.py"],
            submission="checkpoint",
            metric="mse",
            direction="minimize",
            image=image,
            grader_image=image,
            context=context,
            resources=resources,
            output_artifacts=[checkpoint_contract()],
            evaluation_pipeline=pipeline,
        ).model_dump(mode="json")
    )


class StagedAdapter:
    def __init__(self, native):
        self.native = native

    def identity(self):
        return {"fixture": "synthetic-checkpoint-v1", "scope": "staged-contract-test"}

    def context(self):
        return {"contract": "Synthetic train/checkpoint/infer/score test; no final labels"}

    def baseline(self):
        return Candidate(
            files=[
                SourceFile(path="train.py", content=TRAIN),
                SourceFile(path="custom.py", content=CUSTOM),
            ],
            command=["python", "train.py"],
            description="Synthetic checkpoint baseline",
        )

    def validate(self, candidate):
        assert candidate.command == ["python", "train.py"]

    def prepare(self, phase, destination):
        public, private = Path(destination) / "public", Path(destination) / "private"
        public.mkdir(parents=True)
        private.mkdir()
        for name in ("train-x.npy", "train-y.npy"):
            shutil.copyfile(self.native.public_dir / name, public / name)
        values = np.array([1.0, 2.0]) if phase == "subset" else np.array([-2.0, 0.0, 3.0])
        np.save(public / "test-x.npy", values)
        np.save(private / "labels.npy", 3 * values + 2)
        return {
            "public": str(public.resolve()),
            "private": str(private.resolve()),
            "files": tree_manifest(public),
            "private_files": tree_manifest(private),
        }

    def artifacts(self, prepared):
        return [checkpoint_contract()]

    def evaluation(self, prepared):
        return self.native.evaluation_pipeline.model_copy(
            update={
                "private_dir": Path(prepared["private"]),
                "private_manifest": prepared["private_files"],
            }
        )

    def score(self, *args):
        raise AssertionError("Checkpoint scoring must never run in the host adapter")


class StagedTransportDouble:
    """Mirrors stage outputs for contract tests; all receipts explicitly synthetic."""

    def __init__(self):
        self.stages = []
        self.interrupt = None
        self.fail_gpu = False
        self.grade_metrics = None

    def __call__(self, argv, timeout, **kwargs):
        if "inspect" in argv:
            return ProcessResult(0, "sha256:staged-fixture", "", 0.01)
        if "run" not in argv:
            return ProcessResult(0, "", "", 0.01)
        mounts = {
            value.rsplit(":", 2)[1]: Path(value.rsplit(":", 2)[0])
            for index, value in enumerate(argv)
            if index and argv[index - 1] == "-v"
        }
        if "/workspace" in mounts:
            stage = "train"
        elif "/artifacts" in mounts:
            stage = "inference"
        else:
            stage = "score"
        self.stages.append((stage, mounts))
        assert "--network=none" in argv and "--read-only" in argv
        if self.interrupt == stage:
            raise KeyboardInterrupt
        if self.fail_gpu and stage == "inference":
            assert "--gpus" in argv
            return ProcessResult(125, "", "synthetic missing GPU runtime", 0.1)
        if stage == "train":
            assert "/data/private" not in mounts and "/evaluator" not in mounts
            root = mounts["/workspace"] / "checkpoint"
            (root / "tokenizer").mkdir(parents=True)
            np.save(root / "weights.npy", np.array([3.0, 2.0]))
            np.save(root / "padding.npy", np.zeros(400000, dtype=np.float64))
            (root / "tokenizer/config.json").write_text('{"fixture":"synthetic"}')
        elif stage == "inference":
            assert "/candidate" in mounts and "/data/private" not in mounts
            assert (mounts["/candidate"] / "custom.py").is_file()
            assert (mounts["/artifacts"] / "checkpoint/padding.npy").stat().st_size > 3_000_000
            x = np.load(mounts["/data/public"] / "test-x.npy", allow_pickle=False)
            np.save(mounts["/output"] / "predictions.npy", 3 * x + 2)
        else:
            assert not {"/candidate", "/artifacts", "/workspace", "/data/public"} & set(mounts)
            prediction = np.load(mounts["/predictions"] / "predictions.npy", allow_pickle=False)
            labels = np.load(mounts["/data/private"] / "labels.npy", allow_pickle=False)
            (mounts["/grade-output"] / "grade.json").write_text(
                json.dumps(
                    {
                        "valid_submission": True,
                        "metrics": self.grade_metrics
                        if self.grade_metrics is not None
                        else {"mse": float(np.mean((prediction - labels) ** 2))},
                        "fixture": "synthetic transport",
                    }
                )
            )
        return ProcessResult(0, "synthetic staged transport", "", 0.1)


def staged_development(
    tmp_path, monkeypatch, *, executor=None, context=None, image="synthetic-staged-image"
):
    native = make_staged_task(tmp_path / "inputs", context=context, image=image)
    config = LiveSpec(
        native=native,
        development=DevelopmentSpec(
            adapter="staged-fixture", metric="mse", direction="minimize", seeds=[0]
        ),
        model="not-invoked",
        max_calls=1,
        campaign_seconds=300,
        implementation=implementation_identity(),
    )
    adapter = StagedAdapter(native)
    store = Store(tmp_path / "campaign")
    runner = DevelopmentRunner(
        store,
        config,
        adapter,
        Budget(store, config),
        **({"executor": executor} if executor else {}),
    )
    runner.preflight()
    store.set("live_spec_ref", store.put_json(config))
    store.set("live_adapter_identity", adapter.identity())
    monkeypatch.setattr("scientisttwo.live.audit.load_development", lambda *args: adapter)
    return store, runner, adapter


def test_staged_checkpoint_streaming_mount_boundaries_audit_and_resume(tmp_path, monkeypatch):
    transport = StagedTransportDouble()
    store, runner, adapter = staged_development(tmp_path, monkeypatch, executor=transport)
    original_read = store.read

    def bounded_read(ref):
        assert not ref.endswith(".npy"), "Model arrays must not be loaded through Store.read"
        return original_read(ref)

    monkeypatch.setattr(store, "read", bounded_read)
    try:
        result = runner.execute("checkpoint", adapter.baseline())
        assert result.status == "success" and result.value == 0.0
        assert [stage for stage, _ in transport.stages] == ["train", "inference", "score"]
        bundle = load_bundle(store, result.bundle_refs[0])
        assert (
            len(bundle.files) == 3 and sum(file.size for file in bundle.files.values()) > 3_000_000
        )
        evaluation = store.json(result.evaluation_refs[0])
        assert evaluation["input_bundle"] == result.bundle_refs[0]
        assert evaluation["candidate_manifest"]["custom.py"]
        before = len(transport.stages)
        assert runner.execute("checkpoint", adapter.baseline()) == result
        assert len(transport.stages) == before
        audit = audit_live(store.root)
        assert audit["status"] == "pass", audit
        assert audit["container_evaluator_reexecution"] == "not_requested"
        assert audit["verification"]["required_container_replays"] == ["development/checkpoint/0"]
        assert audit["verification"]["container_replay_count"] == 0
        assert audit["verification"]["complete"] is False
        assert len(transport.stages) == before
        monkeypatch.setattr("scientisttwo.live.audit.run_process", transport)
        audit = audit_live(store.root, reexecute_evaluators=True)
        assert audit["status"] == "pass", audit
        assert audit["verification"]["container_replay_count"] == 1
        assert audit["verification"]["complete"] is True
        proof = store.json(audit["verification"]["container_replays"][0]["evidence_ref"])
        assert proof["verification_kind"] == "container_evaluator_replay"
        assert store.json(proof["replayed_grade_ref"])["metrics"] == result.metrics
        assert all(store.json(ref)["returncode"] == 0 for ref in proof["execution_refs"])
        assert [stage for stage, _ in transport.stages[before:]] == ["score"]
    finally:
        store.close()


def test_staged_final_workspace_runs_train_infer_score_once(tmp_path):
    spec = make_staged_task(tmp_path / "inputs")
    fake = StagedTransportDouble()
    runner = WorkspaceRunner(spec, tmp_path / "final", executor=fake)
    try:
        result = runner.run()
        assert result.status == "completed" and result.score == 0.0
        assert result.live_model_calls == 0
        assert [stage for stage, _ in fake.stages] == ["train", "inference", "score"]
        assert runner.run() == result and len(fake.stages) == 3
        grade = runner.store.json(result.official_report_ref)
        assert grade["seed"] == spec.final_seed
        assert grade["pipeline"]["private_dir"] == str(spec.private_dir)
    finally:
        runner.close()


@pytest.mark.parametrize("interrupted", ["inference", "score"])
def test_unknown_inference_completion_is_not_repeated(tmp_path, interrupted):
    spec = make_staged_task(tmp_path / "inputs")
    fake = StagedTransportDouble()
    fake.interrupt = interrupted
    runner = WorkspaceRunner(spec, tmp_path / "final", executor=fake)
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.run()
        fake.interrupt = None
        result = runner.run()
        assert result.status == "interrupted"
        assert [stage for stage, _ in fake.stages] == (
            ["train", "inference"]
            if interrupted == "inference"
            else ["train", "inference", "score"]
        )
    finally:
        runner.close()


def test_checkpoint_corruption_and_changed_grader_rejected(tmp_path, monkeypatch):
    fake = StagedTransportDouble()
    store, runner, adapter = staged_development(tmp_path, monkeypatch, executor=fake)
    try:
        result = runner.execute("first", adapter.baseline())
        payload = load_bundle(store, result.bundle_refs[0]).files["checkpoint/padding.npy"]
        (store.root / payload.ref).write_bytes(b"fabricated checkpoint")
        audit = audit_live(store.root)
        assert audit["status"] == "fail"
        assert any("modified" in message for message in audit["errors"])
        source = runner.spec.evaluation_pipeline.score.source_dir / "score.py"
        source.write_text(source.read_text() + "\n# modified grader\n")
        with pytest.raises(ValueError, match="Frozen input tree changed"):
            runner.execute("changed-grader", adapter.baseline())
        assert len(fake.stages) == 3
    finally:
        store.close()


def test_missing_gpu_does_not_execute_scorer(tmp_path):
    spec = make_staged_task(tmp_path / "inputs")
    spec.evaluation_pipeline.inference.resources.gpus = "all"
    fake = StagedTransportDouble()
    fake.fail_gpu = True
    runner = WorkspaceRunner(spec, tmp_path / "final", executor=fake)
    try:
        result = runner.run()
        assert result.status == "grading_failed" and result.score is None
        assert [stage for stage, _ in fake.stages] == ["train", "inference"]
    finally:
        runner.close()


@pytest.mark.parametrize("which", ["public", "candidate", "score-asset"])
def test_protected_labels_cannot_be_reintroduced_through_mounts(tmp_path, which):
    spec = make_staged_task(tmp_path / "inputs")
    fake = StagedTransportDouble()
    runner = WorkspaceRunner(spec, tmp_path / "final", executor=fake)
    checkpoint = tmp_path / "checkpoint-input"
    (checkpoint / "checkpoint").mkdir(parents=True)
    (checkpoint / "checkpoint/weights.bin").write_bytes(b"fixture")
    ref = freeze_bundle(runner.store, checkpoint, [checkpoint_contract()])
    public, candidate = spec.public_dir, spec.candidate_dir
    if which == "public":
        public = spec.private_dir.parent
    elif which == "candidate":
        candidate = spec.private_dir.parent
    else:
        spec.assets = [
            RuntimeAsset(
                name="leaked",
                role="runtime",
                path=spec.candidate_dir,
                manifest=tree_manifest(spec.candidate_dir),
                stages=["score"],
            )
        ]
    try:
        with pytest.raises(ValueError, match="overlaps"):
            evaluate_pipeline(
                runner,
                "bad-mount",
                spec.evaluation_pipeline,
                ref,
                public,
                candidate_source=candidate,
            )
        assert not fake.stages
    finally:
        runner.close()


def test_scorer_rejects_checkpoint_bundle_without_inference(tmp_path):
    spec = make_staged_task(tmp_path / "inputs")
    fake = StagedTransportDouble()
    runner = WorkspaceRunner(spec, tmp_path / "final", executor=fake)
    root = tmp_path / "checkpoint-input"
    (root / "checkpoint").mkdir(parents=True)
    (root / "checkpoint/weights.bin").write_bytes(b"fixture")
    ref = freeze_bundle(runner.store, root, [checkpoint_contract()])
    pipeline = spec.evaluation_pipeline.model_copy(update={"inference": None})
    try:
        with pytest.raises(ValueError, match="cannot receive checkpoint"):
            evaluate_pipeline(runner, "bad-role", pipeline, ref, spec.public_dir)
        assert not fake.stages
    finally:
        runner.close()


@pytest.mark.parametrize("metrics", [{"mse": float("nan")}, {"mse": True}, {}])
def test_invalid_stage_metrics_never_become_scientific_results(tmp_path, monkeypatch, metrics):
    fake = StagedTransportDouble()
    fake.grade_metrics = metrics
    store, runner, adapter = staged_development(tmp_path, monkeypatch, executor=fake)
    try:
        result = runner.execute("invalid-grade", adapter.baseline())
        assert result.status == "invalid" and result.value is None
        assert result.metrics == {} and result.metric_values == {}
        assert [stage for stage, _ in fake.stages] == ["train", "inference", "score"]
    finally:
        store.close()


def test_audit_rejects_grade_from_another_candidate(tmp_path, monkeypatch):
    fake = StagedTransportDouble()
    store, runner, adapter = staged_development(tmp_path, monkeypatch, executor=fake)
    try:
        first = runner.execute("first", adapter.baseline())
        candidate = adapter.baseline()
        candidate.files[1].content += "\n# different frozen candidate source\n"
        second = runner.execute("second", candidate)
        assert first.status == second.status == "success"
        assert first.evaluation_refs != second.evaluation_refs
        forged = first.model_copy(update={"evaluation_refs": second.evaluation_refs})
        with store.db:
            store.db.execute(
                "UPDATE calls SET output_ref=? WHERE key='development/first'",
                (store.put_json(forged),),
            )
        audit = audit_live(store.root)
        assert audit["status"] == "fail", audit
        assert any(
            "bind" in error.lower() or "pipeline" in error.lower() or "evaluation" in error.lower()
            for error in audit["errors"]
        )
    finally:
        store.close()


def test_evaluator_change_during_execution_is_rejected(tmp_path, monkeypatch):
    fake = StagedTransportDouble()

    def mutate(argv, timeout, **kwargs):
        result = fake(argv, timeout, **kwargs)
        if "run" in argv and fake.stages[-1][0] == "score":
            directory = fake.stages[-1][1]["/evaluator"]
            (directory / "score.py").write_text(SCORE + "\n# changed while grading\n")
        return result

    store, runner, adapter = staged_development(tmp_path, monkeypatch, executor=mutate)
    try:
        result = runner.execute("changed-during-grade", adapter.baseline())
        assert result.status == "invalid" and result.value is None
        assert "Frozen input tree changed" in result.reason
    finally:
        store.close()


@pytest.mark.native_container
def test_real_cpu_checkpoint_development_final_and_replay(staged_container_root, monkeypatch):
    root, context, image = staged_container_root
    store, runner, adapter = staged_development(root, monkeypatch, context=context, image=image)
    try:
        result = runner.execute("checkpoint", adapter.baseline())
        assert result.status == "success" and result.value < 1e-20, result
        assert result.evaluation_refs and result.bundle_refs
        audit = audit_live(store.root, reexecute_evaluators=True)
        assert audit["status"] == "pass", audit
        frozen = store.root / "frozen-source"
        frozen.mkdir()
        for file in adapter.baseline().files:
            (frozen / file.path).write_text(file.content)
        final_spec = freeze_workspace(
            {
                **runner.spec.model_dump(mode="json"),
                "candidate_dir": str(frozen),
                "mode": "prepared_candidate",
            }
        )
        final = WorkspaceRunner(final_spec, store.root / "final" / "selected")
        try:
            outcome = final.run()
            assert outcome.status == "completed" and outcome.score < 1e-20, outcome
            assert final.run() == outcome
        finally:
            final.close()
        audit = audit_live(store.root, reexecute_evaluators=True)
        assert audit["status"] == "pass", audit
        assert any(check.get("final_workspace") == "selected" for check in audit["checks"])
        before = runner.budget.summary()
        assert runner.execute("checkpoint", adapter.baseline()) == result
        assert runner.budget.summary() == before
        (root / "staged-verification.json").write_text(
            json.dumps(
                {
                    "scope": "Synthetic task data and fixed source; actual CPU container training, checkpoint inference and scoring",
                    "development": result.model_dump(),
                    "final": outcome.model_dump(),
                    "audit": audit,
                    "live_model_calls": 0,
                    "research_tokens": 0,
                    "resume_added_operations": 0,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        store.close()


@pytest.fixture
def staged_container_root():
    context = os.environ.get("SCIENTISTTWO_NATIVE_CONTEXT")
    if not context:
        pytest.skip("Set SCIENTISTTWO_NATIVE_CONTEXT for real isolated staged CPU validation")
    image = os.environ.get("SCIENTISTTWO_WORKSPACE_IMAGE", "scientisttwo-native:0.1")
    output = os.environ.get("SCIENTISTTWO_STAGED_OUTPUT")
    if output:
        root = Path(output).resolve()
        root.mkdir(parents=True, exist_ok=True)
        yield root, context, image
    else:
        # Colima shares the workspace, not pytest's /private/var temporary root.
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="staged-validation-", dir=cache) as temporary:
            yield Path(temporary), context, image
