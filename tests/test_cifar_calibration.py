"""Explicitly synthetic calibration traces; no Torch, containers or actual CIFAR."""

import json
from types import SimpleNamespace

import pytest

from scientisttwo.artifacts import ArtifactSpec, freeze_bundle
from scientisttwo.benchmarks.models import NativeResources, NativeSpec
from scientisttwo.benchmarks.runner import NativeRunner
from scientisttwo.live.adapters import implementation_identity
from scientisttwo.live.models import DevelopmentSpec, LiveSpec, Observation
from scientisttwo.live.runtime import Budget, LiveBudgetExhausted
from scientisttwo.live.source_tree import candidate_snapshot, freeze_source_tree
from scientisttwo.process import ProcessResult
from scientisttwo.store import IntegrityError, Store, canonical, digest
from scientisttwo.studies.calibration import _validate_execution, run_calibration, verify_calibration
from scientisttwo.studies.cifar_recipe import (BASELINE_CONFIGURATION, FIXED_CONFIGURATION,
                                               write_study_sources)


class Adapter:
    def __init__(self, root, identity):
        self.root, self.snapshot_ref = root, None
        self.manifest = {"source_identity": identity}

    def identity(self):
        return {"synthetic_adapter": "cifar-calibration-test"}


class Runner:
    def __init__(self, store, config, adapter, outcomes):
        self.store, self.config, self.adapter = store, config, adapter
        self.budget, self.outcomes = Budget(store, config), iter(outcomes)
        self.executions, self.preflights = [], 0
        self.bad_steps = False
        self.protocol_seed = 0
        self.grade_offset = 0

    def preflight(self):
        self.preflights += 1
        self.adapter.snapshot_ref = freeze_source_tree(self.store, self.adapter.root / "reference",
                                                       ["training_config.json"])
        self.store.set("live_images", {"execution": "sha256:" + "a" * 64,
                                        "grading": "sha256:" + "b" * 64})

    def receipt(self, image, *, seconds):
        return self.store.put_json({"image_id": image, "returncode": 0, "timed_out": False,
                                    "limit_exceeded": None, "seconds": seconds,
                                    "execution_enforcement": None,
                                    "job_ref": self.store.put_json({"seed": 0}),
                                    "log_ref": self.store.put(b"SYNTHETIC execution", "log")})

    def execute(self, key, candidate, phase, seeds):
        assert phase == "full" and seeds == [0]
        images = self.store.get("live_images")
        pipeline = {"inference": {"image": images["execution"]}, "score": {"image": images["grading"]}}
        protocol = {"adapter": self.adapter.identity(), "phase": phase, "seeds": [self.protocol_seed],
                    "evaluation": self.config.development.model_dump(mode="json"), "data": {},
                    "runtime": {"images": images, "resources": self.config.native.resources.model_dump(mode="json"),
                                "runtime": self.config.native.runtime}, "pipeline": pipeline}
        protocol_hash = digest(canonical(protocol))
        request = {"candidate": candidate.model_dump(), "protocol": protocol_hash}
        previous = self.budget.reserve("experiment/" + key, "experiment", request)
        if previous is not None:
            if previous["result_ref"]:
                return Observation.model_validate(self.store.json(previous["result_ref"]))
            return Observation(id=key, status="interrupted", phase=phase, metric="accuracy",
                               protocol_hash=protocol_hash, protocol_ref=self.store.put_json(protocol),
                               code_ref=self.store.put_json(candidate))
        self.executions.append(candidate)
        score = next(self.outcomes)
        if isinstance(score, BaseException):
            raise score
        source = self.store.put_json(candidate_snapshot(self.store, candidate))
        if score is None:
            result = Observation(id=key, status="execution_failed", phase=phase, metric="accuracy",
                                 protocol_hash=protocol_hash, protocol_ref=self.store.put_json(protocol),
                                 code_ref=source, source_tree_ref=source, seconds=1.0,
                                 reason="Synthetic failure")
        else:
            outputs = self.store.root / "fake-outputs" / key
            checkpoint = outputs / "checkpoint"
            checkpoint.mkdir(parents=True)
            overrides = json.loads(candidate.files[0].content) if candidate.files else {}
            configuration = {"seed": 0, **FIXED_CONFIGURATION, **BASELINE_CONFIGURATION, **overrides}
            (checkpoint / "config.json").write_bytes(canonical(configuration))
            (checkpoint / "training.json").write_bytes(canonical({
                "task": "cifar10", "seed": 0, "steps": 1 if self.bad_steps else 3130,
                "epochs_started": 10, "max_steps": 3910, "training_seconds": 12.5,
            }))
            (checkpoint / "weights.pt").write_bytes(b"SYNTHETIC; never load")
            bundle = freeze_bundle(self.store, outputs, [ArtifactSpec(path="checkpoint", kind="directory")])
            artifact_refs = {name: entry["ref"] for name, entry in self.store.json(bundle)["files"].items()}
            predictions = outputs / "predict"
            predictions.mkdir()
            (predictions / "prediction.txt").write_text("SYNTHETIC predictions")
            predictions_ref = freeze_bundle(self.store, predictions, [ArtifactSpec(path="prediction.txt")])
            evaluation = self.store.put_json({
                "metrics": {"accuracy": score}, "input_bundle": bundle,
                "prediction_bundle": predictions_ref, "pipeline": pipeline, "seed": 0,
                "receipts": [self.receipt(images["execution"], seconds=2.5),
                             self.receipt(images["grading"], seconds=0.5)],
                "public_manifest": {},
                "grade_ref": self.store.put_json({"valid_submission": True,
                                                  "metrics": {"accuracy": score + self.grade_offset}}),
            })
            result = Observation(
                id=key, status="success", phase=phase, metric="accuracy", value=score, values=[score],
                protocol_hash=protocol_hash, protocol_ref=self.store.put_json(protocol),
                code_ref=source, source_tree_ref=source, seconds=16.0,
                log_refs=[self.receipt(images["execution"], seconds=13.0)],
                bundle_refs=[bundle], artifact_refs=[artifact_refs],
                evaluation_refs=[evaluation], prediction_refs=[predictions_ref],
            )
        self.budget.finish("experiment/" + key, result.status, result, result.seconds)
        return result


def setup(tmp_path, outcomes, **settings):
    root = tmp_path / "study"
    identity = write_study_sources(root)
    config = LiveSpec(
        native=NativeSpec(name="calibration-test", benchmark="mlagentbench", task_id="cifar10",
                          repository=root, revision="a" * 40, mode="baseline", public_dir=root / "public",
                          data_provenance="upstream_no_data"),
        development=DevelopmentSpec(metric="accuracy", direction="maximize", seeds=[0, 1, 2]),
        model="disabled", max_calls=1, max_experiments=2, campaign_seconds=60,
        implementation=implementation_identity(),
    ).model_copy(update=settings)
    store = Store(tmp_path / "calibration")
    adapter = Adapter(root, identity)
    runner = Runner(store, config, adapter, outcomes)
    return store, config, adapter, runner


@pytest.fixture
def serialized_process_receipt(tmp_path, monkeypatch):
    """Use the production receipt serializer with a synthetic subprocess result.

    This checks the real ProcessResult contract without launching a container.
    """
    import scientisttwo.benchmarks.runner as native_module
    monkeypatch.setattr(native_module, "execution_identity", lambda runtime: None)
    runner = NativeRunner.__new__(NativeRunner)
    runner.store = Store(tmp_path / "process-receipt")
    runner.spec = SimpleNamespace(runtime="docker", resources=NativeResources())
    runner.prefix = ["docker"]
    runner.executor = lambda *args, **kwargs: ProcessResult(0, "SYNTHETIC probe", "", 0.1)
    image = "sha256:" + "a" * 64
    _, ref = runner._launch("probe", image, {"seed": 0}, [], 5)
    try:
        yield runner.store, image, ref
    finally:
        runner.store.close()


def test_production_process_receipt_accepts_null_limit(serialized_process_receipt):
    store, image, ref = serialized_process_receipt
    receipt = _validate_execution(store, ref, image=image, seed=0, require_actual=True)
    assert "limit_exceeded" in receipt and receipt["limit_exceeded"] is None


def test_legacy_explicit_false_limit_receipt_remains_readable(serialized_process_receipt):
    store, image, ref = serialized_process_receipt
    legacy = store.put_json({**store.json(ref), "limit_exceeded": False})
    assert _validate_execution(store, legacy, image=image, seed=0,
                               require_actual=False)["limit_exceeded"] is False


@pytest.mark.parametrize("change", [
    {"limit_exceeded": "Output byte quota exceeded"},
    {"limit_exceeded": ""}, {"limit_exceeded": 0}, {"limit_exceeded": True},
    {"timed_out": True}, {"image_id": "sha256:" + "b" * 64}, {"returncode": 1},
])
def test_production_process_receipt_rejects_failure_or_invalid_limit(serialized_process_receipt, change):
    store, image, ref = serialized_process_receipt
    altered = store.put_json({**store.json(ref), **change})
    with pytest.raises(IntegrityError, match="receipt failed"):
        _validate_execution(store, altered, image=image, seed=0, require_actual=True)


def test_production_process_receipt_requires_limit_field(serialized_process_receipt):
    store, image, ref = serialized_process_receipt
    value = store.json(ref)
    del value["limit_exceeded"]
    with pytest.raises(IntegrityError, match="receipt failed"):
        _validate_execution(store, store.put_json(value), image=image, seed=0, require_actual=True)


def test_initial_floor_pass_no_optional_run_and_synthetic_never_ready(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [0.65, 0.9])
    result = run_calibration(store, config, adapter, runner=runner)
    assert result.status == "pass" and len(runner.executions) == 1
    assert result.measurement_origin == "synthetic_fixture"
    receipt = verify_calibration(store, config=config, adapter=adapter, require_actual=False)
    assert receipt["selected_overrides"] == {}
    assert receipt["selected_configuration"]["learning_rate"] == 0.1
    with pytest.raises(ValueError, match="Synthetic"):
        verify_calibration(store, config=config, adapter=adapter)
    assert not any(row["kind"] == "model" for row in runner.budget.summary())


@pytest.mark.parametrize("first", [None, 0.64])
def test_optional_lr_only_on_invalid_or_low_initial(tmp_path, first):
    store, config, adapter, runner = setup(tmp_path, [first, 0.70])
    result = run_calibration(store, config, adapter, runner=runner)
    assert result.status == "pass" and len(runner.executions) == 2
    receipt = verify_calibration(store, config=config, adapter=adapter, require_actual=False)
    assert receipt["selected_overrides"] == {"learning_rate": 0.03}


def test_no_adequate_baseline_is_not_ready(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [0.63, 0.64])
    result = run_calibration(store, config, adapter, runner=runner)
    assert result.status == "not_ready" and result.selected_candidate_ref is None
    assert len(runner.executions) == 2
    with pytest.raises(ValueError, match="not ready"):
        verify_calibration(store, config=config, adapter=adapter, require_actual=False)


def test_measurements_use_recorded_receipts_not_limits_or_invented_peaks(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [0.7])
    result = run_calibration(store, config, adapter, runner=runner)
    measures = result.resource_measurements
    assert measures["training_seconds"]["value"] == 12.5
    assert measures["inference_seconds"]["value"] == 2.5
    assert measures["total_wall_seconds"]["value"] == 16.0
    assert measures["peak_gpu_memory_bytes"]["status"] == "unverified"
    assert measures["peak_host_memory_bytes"]["status"] == "unverified"
    assert "value" not in measures["peak_gpu_memory_bytes"]


def test_completed_resume_performs_no_operations(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [0.7])
    first = run_calibration(store, config, adapter, runner=runner)
    before = runner.budget.summary()
    assert run_calibration(store, config, adapter, runner=runner) == first
    assert runner.preflights == 1 and len(runner.executions) == 1
    assert runner.budget.summary() == before


def test_verification_allows_only_calibration_pointer_settings_addition(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [0.7])
    result = run_calibration(store, config, adapter, runner=runner)
    development = config.development.model_copy(update={"settings": {
        "calibration_root": str(store.root), "calibration_receipt_ref": result.receipt_ref}})
    assert verify_calibration(store, native=config.native, development=development,
                              adapter=adapter, require_actual=False)["status"] == "pass"
    development.settings["subset_other"] = True
    with pytest.raises(IntegrityError, match="source, study"):
        verify_calibration(store, native=config.native, development=development,
                           adapter=adapter, require_actual=False)


def test_modified_ledger_receipt_or_runtime_cannot_verify(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [0.7])
    run_calibration(store, config, adapter, runner=runner)
    store.db.execute("UPDATE live_operations SET request_hash='forged'")
    store.db.commit()
    with pytest.raises(IntegrityError, match="operation ledger"):
        verify_calibration(store, config=config, adapter=adapter, require_actual=False)


def test_changed_image_binding_rejected(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [0.7])
    run_calibration(store, config, adapter, runner=runner)
    native = config.native.model_copy(update={"image": "changed-image"})
    with pytest.raises(IntegrityError, match="source, study"):
        verify_calibration(store, native=native, development=config.development,
                           adapter=adapter, require_actual=False)


@pytest.mark.parametrize("setting", ["protocol_seed", "grade_offset"])
def test_wrong_protocol_seed_or_fabricated_grade_rejected(tmp_path, setting):
    store, config, adapter, runner = setup(tmp_path, [0.7])
    setattr(runner, setting, 1)
    with pytest.raises(IntegrityError):
        run_calibration(store, config, adapter, runner=runner)


def test_shortened_training_does_not_pass_calibration(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [0.7])
    runner.bad_steps = True
    result = run_calibration(store, config, adapter, runner=runner)
    assert result.status == "not_ready" and "ten-epoch" in result.reason


def test_experiment_budget_exhaustion_is_explicit_not_ready(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [0.63, 0.7], max_experiments=1)
    result = run_calibration(store, config, adapter, runner=runner)
    assert result.status == "not_ready" and "limit exhausted" in result.reason
    assert len(runner.executions) == 1


def test_model_operation_disqualifies_calibration(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [0.7])
    runner.budget.reserve("forbidden", "model", {})
    with pytest.raises(IntegrityError, match="without model calls"):
        run_calibration(store, config, adapter, runner=runner)


def test_origin_cannot_be_promoted_by_receipt_label(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [0.7])
    result = run_calibration(store, config, adapter, runner=runner)
    receipt = store.json(result.receipt_ref)
    receipt["measurement_origin"] = "actual_execution"
    forged = store.put_json(receipt)
    store.set("calibration_receipt_ref", forged)
    with pytest.raises(ValueError, match="Synthetic"):
        verify_calibration(store, config=config, adapter=adapter)


def test_interrupted_attempt_is_not_repeated_or_replaced_with_sanity_run(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [KeyboardInterrupt(), 0.7])
    with pytest.raises(KeyboardInterrupt):
        run_calibration(store, config, adapter, runner=runner)
    result = run_calibration(store, config, adapter, runner=runner)
    assert result.status == "not_ready" and "Unknown calibration" in result.reason
    assert runner.preflights == 1 and len(runner.executions) == 1


def test_completed_resume_rechecks_the_committed_operation_ledger(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [0.7])
    run_calibration(store, config, adapter, runner=runner)
    store.db.execute("UPDATE live_operations SET seconds=999")
    store.db.commit()
    with pytest.raises(IntegrityError, match="operation ledger"):
        run_calibration(store, config, adapter, runner=runner)
    assert len(runner.executions) == 1


def test_committed_wall_budget_failure_remains_in_attempt_history(tmp_path):
    store, config, adapter, runner = setup(tmp_path, [None])
    execute = runner.execute

    def timed_out(*args, **kwargs):
        execute(*args, **kwargs)
        raise LiveBudgetExhausted("Synthetic exhausted wall-clock budget")

    runner.execute = timed_out
    result = run_calibration(store, config, adapter, runner=runner)
    assert result.status == "not_ready" and len(result.attempts) == 1
    assert store.json(result.attempts[0]["observation_ref"])["status"] == "execution_failed"
    assert "wall-clock" in result.reason
