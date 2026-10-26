"""Real diagnostic arithmetic behind a synthetic transport; no container claim."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scientisttwo.artifacts import ArtifactSpec, freeze_bundle, load_bundle
from scientisttwo.benchmarks.files import tree_manifest
from scientisttwo.live.models import Observation
from scientisttwo.live.pipeline import EvaluationPipeline, EvaluationStage, PipelineInterrupted
from scientisttwo.live.runtime import Budget
from scientisttwo.process import ProcessResult
from scientisttwo.research.diagnostic_runner import ScoringDiagnosticContract, run_scoring_diagnostic, _source_files
from scientisttwo.research.diagnostic_worker import arrays
from scientisttwo.research.diagnostics import DiagnosticInputs, DiagnosticRequest, classification_diagnostics, regression_residuals, run_diagnostic
from scientisttwo.store import IntegrityError, Store, canonical, digest


class SyntheticTransport:
    def __init__(self, store):
        self.store, self.launches = store, []
        self.interrupt = self.fail = False
        self.budget = Budget(store, SimpleNamespace(campaign_seconds=300, max_calls=1, max_experiments=10, token_stop_threshold=1))

    def _image(self, image):
        return image

    def _launch(self, stage, image, job, mounts, timeout, *, resources, gpus, bridge):
        self.launches.append({"stage": stage, "image": image, "mounts": mounts, "gpus": gpus})
        if self.interrupt:
            raise KeyboardInterrupt
        locations = {target: Path(path) for path, target, mode in mounts}
        assert not {"/workspace", "/candidate", "/data/public", "/artifacts"} & locations.keys()
        assert gpus is None and len(locations) == 4
        assert all(mode == "ro" for path, target, mode in mounts if target != "/grade-output")
        contract = json.loads((locations["/evaluator"] / "contract.json").read_text())
        if not self.fail:
            truth, predictions, classes = arrays(locations["/predictions"] / contract["prediction_path"],
                                                locations["/data/private"] / contract["label_path"], contract)
            result = (classification_diagnostics(truth, predictions, classes=classes) if contract["kind"] == "classification"
                      else regression_residuals(truth, predictions, target_names=contract["targets"]))
            (locations["/grade-output"] / "diagnostic.json").write_bytes(canonical(result))
        return ProcessResult(1 if self.fail else 0, "synthetic", "", .1), self.store.put_json({"synthetic_receipt": True, "seed": job["seed"]})


def setup(tmp_path, *, format="class_scores_npz", kind="classification"):
    store = Store(tmp_path / "campaign")
    runner = SyntheticTransport(store)
    private, source, predictions = (tmp_path / name for name in ("private", "source", "predictions"))
    for path in (private, source, predictions):
        path.mkdir()
    (source / "evaluate.py").write_text("# Frozen synthetic scorer; not executed\n")
    if format == "class_scores_npz":
        np.savez(private / "labels.npz", ids=np.array(["a", "b", "c"]), y=np.array([0, 1, 0]))
        output = predictions / "predictions"
        output.mkdir()
        np.savez(output / "predictions-0.npz", ids=np.array(["c", "a"]), predictions=np.array([[0., 1.], [1., 0.]]))
        np.savez(output / "predictions-1.npz", ids=np.array(["b"]), predictions=np.array([[0., 1.]]))
        specs = [ArtifactSpec(path="predictions", kind="directory", role="prediction")]
        contract = ScoringDiagnosticContract(kind=kind, format=format, prediction_path="predictions", label_path="labels.npz")
    else:
        (private / "labels.csv").write_text("id,a,b\none,1,2\ntwo,3,4\n")
        (predictions / "submission.csv").write_text("id,a,b\ntwo,4,6\none,2,4\n")
        specs = [ArtifactSpec(path="submission.csv", role="prediction")]
        contract = ScoringDiagnosticContract(kind=kind, format=format, prediction_path="submission.csv", label_path="labels.csv", targets=["a", "b"])
    bundle = freeze_bundle(store, predictions, specs)
    stage = EvaluationStage(kind="score", image="sha256:synthetic", source_dir=source, source_manifest=tree_manifest(source), command=["python", "evaluate.py"])
    pipeline = EvaluationPipeline(score=stage, private_dir=private, private_manifest=tree_manifest(private))
    protocol = {"phase": "full", "seeds": [0, 1, 2], "pipeline": pipeline.model_dump(mode="json")}
    eval_refs = []
    for seed in protocol["seeds"]:
        result = {"pipeline": pipeline.model_dump(mode="json"), "seed": seed, "prediction_bundle": bundle}
        store.cache("pipeline/obs/" + str(seed), {}, result)
        eval_refs.append(store.put_json(result))
    observation = Observation(id="obs", phase="full", status="success", metric="accuracy", value=2/3, values=[2/3]*3,
                              code_ref=store.put_json({"synthetic_code": True}), protocol_ref=store.put_json(protocol),
                              protocol_hash=digest(canonical(protocol)), prediction_refs=[bundle]*3, evaluation_refs=eval_refs)
    store.cache("development/obs", {}, observation)
    ref = store.put_json(observation)
    request = DiagnosticRequest(id="class-check", kind=kind, expected_seeds=[0, 1, 2])
    return store, runner, request, ref, pipeline, contract


@pytest.mark.parametrize("format,kind", [("class_scores_npz", "classification"), ("cpu_frames", "regression_residuals")])
def test_actual_arithmetic_sealed_inputs_no_candidate_mounts_resume_and_export_safety(tmp_path, format, kind):
    store, runner, request, ref, pipeline, contract = setup(tmp_path, format=format, kind=kind)
    try:
        output = run_scoring_diagnostic(runner, request, ref, pipeline, contract)
        result = run_diagnostic(store, request, DiagnosticInputs(observation_refs=[ref], trusted_result_refs=[output], declared_kinds=[kind]))
        assert result.status == "success" and len(runner.launches) == 3
        assert store.json(output)["label_commitment"] == digest(canonical(pipeline.private_manifest))
        text = canonical(store.json(output)).decode()
        assert str(pipeline.private_dir) not in text and "one" not in text
        if kind == "classification":
            assert result.tables["per_seed"][0]["accuracy"] == pytest.approx(2/3)
            assert result.arrays["seed_0/confusion_matrix"] == [[1, 1], [0, 1]]
        else:
            assert result.tables["per_seed"][0]["mean_target_rmse"] == 1.5
        assert run_scoring_diagnostic(runner, request, ref, pipeline, contract) == output
        assert len(runner.launches) == 3
        assert len(runner.budget.summary()) == 1
        assert runner.budget.summary()[0]["status"] == "success"
    finally:
        store.close()


@pytest.mark.parametrize("damage", ["labels", "predictions", "evaluator"])
def test_completed_resume_detects_modified_dependency_closure(tmp_path, damage):
    store, runner, request, ref, pipeline, contract = setup(tmp_path)
    try:
        run_scoring_diagnostic(runner, request, ref, pipeline, contract)
        if damage == "labels":
            (pipeline.private_dir / "labels.npz").write_bytes(b"tampered")
        elif damage == "evaluator":
            (pipeline.score.source_dir / "evaluate.py").write_text("modified")
        else:
            obs = store.json(ref)
            bundle = load_bundle(store, obs["prediction_refs"][0], verify=False)
            (store.root / next(iter(bundle.files.values())).ref).write_bytes(b"tampered")
        with pytest.raises((ValueError, IntegrityError)):
            run_scoring_diagnostic(runner, request, ref, pipeline, contract)
        assert len(runner.launches) == 3
    finally:
        store.close()


@pytest.mark.parametrize("interrupt", [False, True])
def test_failed_and_unknown_attempts_consume_operation_and_never_repeat(tmp_path, interrupt):
    store, runner, request, ref, pipeline, contract = setup(tmp_path)
    try:
        runner.interrupt, runner.fail = interrupt, not interrupt
        with pytest.raises(KeyboardInterrupt if interrupt else ValueError):
            run_scoring_diagnostic(runner, request, ref, pipeline, contract)
        with pytest.raises(PipelineInterrupted):
            run_scoring_diagnostic(runner, request, ref, pipeline, contract)
        assert len(runner.launches) == 1
        assert len(runner.budget.summary()) == 1
        assert runner.budget.summary()[0]["status"] == ("running" if interrupt else "failed")
    finally:
        store.close()


def test_exact_seed_request_not_reduced_before_launch(tmp_path):
    store, runner, request, ref, pipeline, contract = setup(tmp_path)
    try:
        request.expected_seeds = list(range(10))
        with pytest.raises(ValueError, match="seed coverage"):
            run_scoring_diagnostic(runner, request, ref, pipeline, contract)
        assert runner.launches == []
    finally:
        store.close()


def test_unimplemented_requested_metric_not_substituted_by_classification_accuracy(tmp_path):
    store, runner, request, ref, pipeline, contract = setup(tmp_path)
    try:
        request.metric = "roc_auc"
        with pytest.raises(ValueError, match="not produced"):
            run_scoring_diagnostic(runner, request, ref, pipeline, contract)
        assert runner.launches == []
    finally:
        store.close()


def test_generated_math_dependency_closure_compiles_without_package_imports():
    files = _source_files()
    for name, source in files.items():
        compile(source, name, "exec")
    assert "pydantic" not in files["diagnostic_math.py"]
    assert "scientisttwo" not in files["diagnostic_math.py"]


@pytest.mark.parametrize("path", ["../labels.npz", "/labels.npz", "x/../../labels", "a\\b"])
def test_contract_rejects_mount_escape(path):
    with pytest.raises(ValueError):
        ScoringDiagnosticContract(kind="classification", format="class_scores_npz", prediction_path="predictions", label_path=path)


def test_shard_duplicate_and_missing_ids_are_rejected(tmp_path):
    private = tmp_path / "labels.npz"
    np.savez(private, ids=np.array(["a", "b"]), y=np.array([0, 1]))
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    np.savez(predictions / "predictions-0.npz", ids=np.array(["a"]), predictions=np.array([[1., 0.]]))
    contract = {"format": "class_scores_npz"}
    with pytest.raises(ValueError, match="omit"):
        arrays(predictions, private, contract)
    np.savez(predictions / "predictions-1.npz", ids=np.array(["a"]), predictions=np.array([[1., 0.]]))
    with pytest.raises(ValueError, match="duplicate"):
        arrays(predictions, private, contract)


def test_actual_trusted_worker_process_executes_frozen_numpy_math(tmp_path):
    """A fixed trusted worker subprocess, not a candidate or container execution."""
    import subprocess
    import sys
    store, _, _, _, pipeline, contract = setup(tmp_path)
    try:
        source = tmp_path / "worker"
        source.mkdir()
        for name, text in _source_files().items():
            (source / name).write_text(text)
        (source / "contract.json").write_bytes(canonical(contract))
        output = tmp_path / "worker-result.json"
        subprocess.run([sys.executable, str(source / "diagnostic_worker.py"), "--contract", str(source / "contract.json"),
            "--predictions", str(tmp_path / "predictions/predictions"), "--labels", str(pipeline.private_dir / "labels.npz"),
            "--output", str(output)], check=True, timeout=15, capture_output=True)
        result = json.loads(output.read_text())
        assert result["summary"]["accuracy"] == pytest.approx(2 / 3)
        assert result["arrays"]["confusion_matrix"] == [[1, 1], [0, 1]]
    finally:
        store.close()
