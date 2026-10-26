"""Bounded, development-only calibration of the fixed CIFAR study reference.

Calibration never constructs an agent backend or accesses final partitions. A
synthetic/injected execution trace can exercise this module but cannot produce a
receipt accepted by the live-readiness verifier.
"""

import json
import math
from pathlib import Path
import re
from typing import Literal

from pydantic import Field

from ..artifacts import load_bundle
from ..live.adapters import implementation_identity
from ..live.models import Candidate, Observation
from ..live.runtime import Budget, DevelopmentRunner, LiveBudgetExhausted
from ..models import Record
from ..process import run_process
from ..store import IntegrityError, canonical, digest, file_digest
from .cifar_baseline.training_contract import FIXED_CONFIGURATION
from .cifar_recipe import (VARIANT, baseline_candidate, calibration_plan, recipe_candidate,
                           validate_training_receipt)
from .runner import StudyDevelopmentRunner, validate_observation_receipts


class CalibrationResult(Record):
    status: Literal["pass", "not_ready"]
    reason: str
    measurement_origin: Literal["actual_execution", "synthetic_fixture"]
    selected_candidate_ref: str | None = None
    selected_observation_ref: str | None = None
    attempts: list[dict] = Field(default_factory=list)
    resource_measurements: dict = Field(default_factory=dict)
    receipt_ref: str
    live_model_calls: Literal[0] = 0
    final_access: Literal[False] = False


def _development_identity(development):
    value = development.model_dump(mode="json")
    for key in ("calibration_root", "calibration_receipt_ref"):
        value["settings"].pop(key, None)
    return value


def _binding(native, development, adapter):
    return {"variant": VARIANT, "native": native.model_dump(mode="json"),
            "development": _development_identity(development), "adapter": adapter.identity(),
            "study_source_identity": adapter.manifest["source_identity"],
            "implementation": implementation_identity(),
            "calibration_source_sha256": file_digest(Path(__file__)),
            "plan": calibration_plan()}


def _number(value):
    return (type(value) in {int, float} and math.isfinite(value) and value >= 0)


def _measurements(store, observation):
    missing = {"status": "unverified", "reason": "No recorded measurement"}
    result = {name: dict(missing) for name in calibration_plan()["resource_measurements"]}
    if _number(observation.seconds):
        result["total_wall_seconds"] = {"status": "measured", "value": observation.seconds,
                                        "source": "Observation.seconds; complete development operation"}
    for artifacts in observation.artifact_refs:
        training_ref = artifacts.get("checkpoint/training.json")
        if training_ref:
            receipt = store.json(training_ref)
            if _number(receipt.get("training_seconds")):
                result["training_seconds"] = {"status": "measured", "value": receipt["training_seconds"],
                                                "source_ref": training_ref,
                                                "scope": "Frozen worker training timer, including checkpoint writing"}
    for ref in observation.evaluation_refs:
        evaluated = store.json(ref)
        if evaluated.get("pipeline", {}).get("inference") and evaluated.get("receipts"):
            execution_ref = evaluated["receipts"][0]
            receipt = store.json(execution_ref)
            if _number(receipt.get("seconds")):
                result["inference_seconds"] = {"status": "measured", "value": receipt["seconds"],
                                               "source_ref": execution_ref,
                                               "scope": "Isolated inference process including startup"}
    # Resource limits and nvidia-smi free memory are never substituted for peaks.
    for ref in observation.log_refs:
        receipt = store.json(ref)
        enforcement = receipt.get("execution_enforcement") or {}
        evidence_ref = enforcement.get("evidence_ref")
        if enforcement.get("status") != "verified" or not evidence_ref:
            continue
        evidence = store.json(evidence_ref)
        measurements = evidence.get("resource_measurements", {})
        for name in ("peak_gpu_memory_bytes", "peak_host_memory_bytes"):
            if _number(measurements.get(name)):
                result[name] = {"status": "measured", "value": measurements[name],
                                "source_ref": evidence_ref, "scope": "Trusted execution enforcement"}
    return result


def _adequate(observation):
    return (observation.status == "success" and observation.phase == "full"
            and observation.metric == "accuracy" and len(observation.values) == 1
            and observation.value == observation.values[0]
            and calibration_plan()["accuracy_floor"] <= observation.value <= 1)


def _validate_execution(store, ref, *, image, seed, require_actual):
    value = store.json(ref)
    # ProcessResult uses None for no quota failure; older synthetic receipts
    # serialized an explicit False. Require the field and reject every other
    # value, including an empty error string or numeric zero.
    limit_clear = ("limit_exceeded" in value
                   and (value["limit_exceeded"] is None or value["limit_exceeded"] is False))
    if (value.get("image_id") != image or value.get("returncode") != 0
            or value.get("timed_out") is not False or not limit_clear
            or value.get("execution_enforcement_error")):
        raise IntegrityError("Calibration execution receipt failed or differs from the declared image")
    if store.json(value["job_ref"]).get("seed") != seed:
        raise IntegrityError("Calibration execution receipt has a different seed")
    store.read(value["log_ref"])
    if require_actual and not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
        raise IntegrityError("Actual calibration requires immutable container image digests")
    enforcement = value.get("execution_enforcement")
    if enforcement:
        if enforcement.get("status") != "verified":
            raise IntegrityError("Calibration execution enforcement was not verified")
        evidence = store.json(enforcement["evidence_ref"])
        if evidence.get("status") != "completed" or not evidence.get("cleanup", {}).get("confirmed_absent"):
            raise IntegrityError("Calibration execution cleanup evidence is incomplete")
    return value


def _verify_runtime_probe(store):
    hardware, probe_ref = store.get("runtime_hardware", {}), store.get("runtime_probe_ref")
    if hardware.get("status") != "verified" or not probe_ref:
        raise IntegrityError("Calibration runtime/device probe is unverified")
    probe = store.json(probe_ref)
    if probe.get("report", {}).get("status") != "pass":
        raise IntegrityError("Calibration runtime/device probe failed")
    _validate_execution(store, probe["execution_ref"], image=store.get("live_images")["execution"],
                        seed=0, require_actual=True)


def _verify_observation(store, attempt, binding, *, require_actual):
    candidate = Candidate.model_validate(store.json(attempt["candidate_ref"]))
    observation = Observation.model_validate(store.json(attempt["observation_ref"]))
    if (observation.id != attempt["key"] or observation.phase != "full"
            or observation.metric != "accuracy" or observation.protocol_ref is None):
        raise IntegrityError("Calibration observation is not the declared full-development trial")
    protocol = store.json(observation.protocol_ref)
    if (digest(canonical(protocol)) != observation.protocol_hash or protocol.get("phase") != "full"
            or protocol.get("seeds") != [0] or protocol.get("adapter") != binding["adapter"]
            or protocol.get("evaluation") != binding["development"]):
        raise IntegrityError("Calibration protocol, phase, seed or adapter binding changed")
    images = store.get("live_images")
    runtime = protocol.get("runtime", {})
    if (runtime.get("images") != images or runtime.get("resources") != binding["native"]["resources"]
            or runtime.get("runtime") != binding["native"]["runtime"]):
        raise IntegrityError("Calibration runtime or resource identity changed")
    if require_actual:
        prepared_ref = store.get("prepared/full")
        if (not prepared_ref or store.json(prepared_ref)["files"] != protocol.get("data")
                or runtime.get("hardware") != store.get("runtime_hardware")):
            raise IntegrityError("Calibration development-data or hardware binding changed")
    request = {"candidate": candidate.model_dump(), "protocol": observation.protocol_hash}
    row = store.db.execute("SELECT * FROM live_operations WHERE key=?",
                           ("experiment/" + attempt["key"],)).fetchone()
    if (row is None or row["kind"] != "experiment" or row["status"] != observation.status
            or row["request_hash"] != digest(canonical(request)) or not row["result_ref"]
            or store.json(row["result_ref"]) != observation.model_dump(mode="json")
            or row["seconds"] != observation.seconds):
        raise IntegrityError("Calibration observation does not match its charged operation ledger")
    expected = baseline_candidate(candidate.base_snapshot_ref) if attempt["name"] == "initial" else recipe_candidate(
        candidate.base_snapshot_ref, {"learning_rate": 0.03}, description="Predeclared calibration sanity run")
    if candidate != expected:
        raise IntegrityError("Calibration candidate differs from the predeclared recipe")
    base = store.json(candidate.base_snapshot_ref)
    base_files = load_bundle(store, base["bundle_ref"]).files
    manifest = {name: {"sha256": value.sha256, "size": value.size} for name, value in base_files.items()}
    if manifest != binding["study_source_identity"]["reference_files"]:
        raise IntegrityError("Calibration baseline source differs from the frozen study")
    if observation.status != "success":
        return observation
    if len(observation.values) != 1 or observation.value != observation.values[0]:
        raise IntegrityError("Calibration requires exactly one seed score")
    if not (len(observation.log_refs) == len(observation.artifact_refs)
            == len(observation.bundle_refs) == len(observation.evaluation_refs)
            == len(observation.prediction_refs) == 1):
        raise IntegrityError("Calibration training/inference/scoring evidence is incomplete")
    source = store.json(observation.source_tree_ref)
    if (source.get("base_snapshot_ref") != candidate.base_snapshot_ref
            or store.json(source["proposal_ref"]) != candidate.model_dump(mode="json")
            or source.get("command") != candidate.command or observation.code_ref != observation.source_tree_ref):
        raise IntegrityError("Calibration execution source differs from its candidate")
    materialized = load_bundle(store, source["bundle_ref"])
    edits = {file.path: file.content.encode() for file in candidate.files}
    if set(materialized.files) != set(base_files):
        raise IntegrityError("Calibration materialized source inventory changed")
    for name, original in base_files.items():
        expected_hash = digest(edits[name]) if name in edits else original.sha256
        if materialized.files[name].sha256 != expected_hash:
            raise IntegrityError("Calibration materialized source changed: " + name)
    _validate_execution(store, observation.log_refs[0], image=images["execution"], seed=0,
                        require_actual=require_actual)
    artifacts = observation.artifact_refs[0]
    frozen = load_bundle(store, observation.bundle_refs[0])
    if artifacts != {name: item.ref for name, item in frozen.files.items()}:
        raise IntegrityError("Calibration checkpoint artifact index changed")
    configuration = store.json(artifacts["checkpoint/config.json"])
    expected_configuration = {"seed": 0, **FIXED_CONFIGURATION, **calibration_plan()[
        "initial" if attempt["name"] == "initial" else "optional_sanity_check"]}
    if configuration != expected_configuration:
        raise IntegrityError("Calibration training configuration differs from the frozen recipe")
    validate_training_receipt(store.json(artifacts["checkpoint/training.json"]), configuration, phase="full")
    evaluation = store.json(observation.evaluation_refs[0])
    pipeline = protocol.get("pipeline", {})
    if (evaluation.get("pipeline") != pipeline or evaluation.get("seed") != 0
            or evaluation.get("input_bundle") != observation.bundle_refs[0]
            or evaluation.get("prediction_bundle") != observation.prediction_refs[0]
            or evaluation.get("public_manifest") != protocol.get("data")
            or len(evaluation.get("receipts", [])) != 2):
        raise IntegrityError("Calibration isolated evaluation differs from the execution protocol")
    load_bundle(store, evaluation["prediction_bundle"])
    for name, ref in zip(("inference", "score"), evaluation["receipts"], strict=True):
        _validate_execution(store, ref, image=pipeline[name]["image"], seed=0,
                            require_actual=require_actual)
    grade = store.json(evaluation["grade_ref"])
    if (grade.get("valid_submission") is not True
            or grade.get("metrics") != evaluation.get("metrics")
            or evaluation["metrics"].get("accuracy") != observation.value):
        raise IntegrityError("Calibration trusted score differs from the reported accuracy")
    validate_observation_receipts(store, observation, expected_seeds=[0])
    return observation


def verify_calibration(store, receipt_ref=None, *, native=None, development=None, adapter,
                       config=None, require_actual=True):
    """Read back a sealed calibration, rejecting synthetic traces for readiness.

    Returns the receipt plus selected candidate/configuration/overrides, allowing
    an adapter to rebind the exact selected edits to its own equivalent snapshot.
    No backend, framework, container or candidate code is invoked.
    """
    if config is not None:
        native, development = config.native, config.development
    if native is None or development is None:
        raise ValueError("Calibration verification requires native and development specifications")
    receipt_ref = receipt_ref or store.get("calibration_receipt_ref")
    if not receipt_ref or receipt_ref != store.get("calibration_receipt_ref"):
        raise IntegrityError("Calibration receipt is absent or is not the committed result")
    if store.verify():
        raise IntegrityError("Calibration artifact integrity check failed")
    receipt = store.json(receipt_ref)
    binding = _binding(native, development, adapter)
    if (receipt.get("binding") != binding or store.get("calibration_binding") != binding
            or receipt.get("live_model_calls") != 0 or receipt.get("final_access") is not False):
        raise IntegrityError("Calibration source, study, specification or declared access changed")
    if require_actual and (receipt.get("measurement_origin") != "actual_execution"
                           or receipt.get("runner_origin") not in {
                               "DevelopmentRunner/run_process", "StudyDevelopmentRunner/run_process"}):
        raise ValueError("Synthetic calibration cannot establish live readiness")
    operations = list(store.db.execute("SELECT * FROM live_operations"))
    if any(row["kind"] == "model" or row["key"].startswith("final/") or row["status"] == "running"
           for row in operations):
        raise IntegrityError("Calibration contains model calls, final access or unknown operations")
    attempts = receipt.get("attempts", [])
    if (not 1 <= len(attempts) <= 2 or [a["name"] for a in attempts] != ["initial", "sanity"][:len(attempts)]
            or {row["key"] for row in operations if row["kind"] == "experiment"}
            != {"experiment/" + a["key"] for a in attempts}):
        raise IntegrityError("Calibration attempts differ from the bounded predeclared schedule")
    observations = [_verify_observation(store, a, binding, require_actual=require_actual) for a in attempts]
    if len(observations) == 2 and (_adequate(observations[0]) or observations[0].status == "interrupted"):
        raise IntegrityError("Optional calibration was not permitted by the initial result")
    selected = observations[-1]
    if receipt.get("status") != "pass" or not _adequate(selected):
        raise ValueError("Calibration baseline is not ready")
    selected_attempt = attempts[-1]
    if (receipt.get("selected_observation_ref") != selected_attempt["observation_ref"]
            or receipt.get("selected_candidate_ref") != selected_attempt["candidate_ref"]):
        raise IntegrityError("Calibration selected a different or unexecuted candidate")
    if require_actual:
        if receipt.get("runtime_probe_ref") != store.get("runtime_probe_ref"):
            raise IntegrityError("Calibration runtime probe differs from its frozen receipt")
        _verify_runtime_probe(store)
    measures = _measurements(store, selected)
    if measures != receipt.get("resource_measurements"):
        raise IntegrityError("Calibration resource report differs from recorded measurements")
    candidate = Candidate.model_validate(store.json(receipt["selected_candidate_ref"]))
    overrides = json.loads(candidate.files[0].content) if candidate.files else {}
    return {**receipt, "receipt_ref": receipt_ref, "selected_candidate": candidate.model_dump(mode="json"),
            "selected_observation": selected.model_dump(mode="json"), "selected_overrides": overrides,
            "selected_configuration": {"seed": 0, **FIXED_CONFIGURATION,
                                       **calibration_plan()["initial"], **overrides}}


def run_calibration(store, config, adapter, *, runner=None, budget=None):
    """Run at most two declared seed-zero development trials, with no agent calls."""
    if (config.development.metric != "accuracy" or config.development.direction != "maximize"
            or config.development.seeds != [0, 1, 2]):
        raise ValueError("CIFAR calibration requires accuracy/maximize and study seeds [0,1,2]")
    if config.implementation != implementation_identity():
        raise IntegrityError("Calibration requires the currently frozen implementation")
    budget = budget or (runner.budget if runner is not None else Budget(store, config))
    if budget.store is not store or budget.config != config:
        raise ValueError("Calibration must use its declared store and resource ledger")
    runner = runner or StudyDevelopmentRunner(store, config, adapter, budget)
    if runner.budget is not budget:
        raise ValueError("Calibration executions must share the declared budget")
    actual = type(runner) in {DevelopmentRunner, StudyDevelopmentRunner} and runner.raw_executor is run_process
    origin = "actual_execution" if actual else "synthetic_fixture"
    binding = _binding(config.native, config.development, adapter)
    binding["plan"] = calibration_plan()
    if store.get("calibration_binding") not in (None, binding):
        raise IntegrityError("Calibration inputs changed; do not reuse this run")
    store.set("calibration_binding", binding)
    saved = store.cached("calibration/result", {"binding": binding, "origin": origin})
    if saved is not None:
        if store.verify():
            raise IntegrityError("Calibration evidence changed")
        if saved["status"] == "pass":
            verify_calibration(store, config=config, adapter=adapter, require_actual=actual)
        return CalibrationResult.model_validate(saved)
    if store.db.execute("SELECT 1 FROM live_operations WHERE kind='model' OR key LIKE 'final/%'").fetchone():
        raise IntegrityError("Calibration requires a store without model calls or final evaluation")
    attempts, chosen, reason = [], None, "Baseline did not meet the predeclared sanity threshold"
    key = None
    try:
        if store.db.execute("SELECT 1 FROM live_operations WHERE status='running'").fetchone():
            raise ValueError("Unknown calibration operation completion; no automatic retry")
        runner.preflight()
        if actual:
            _verify_runtime_probe(store)
        initial = baseline_candidate(adapter.snapshot_ref)
        for name, candidate in (
            ("initial", initial),
            ("sanity", recipe_candidate(adapter.snapshot_ref, {"learning_rate": 0.03},
                                         description="Predeclared calibration sanity run")),
        ):
            key = "calibration/" + name
            observation = runner.execute(key, candidate, "full", seeds=[0])
            attempt = {"key": key, "name": name,
                       "candidate_ref": store.put_json(candidate, "calibration-candidate"),
                       "observation_ref": store.put_json(observation, "calibration-observation")}
            attempts.append(attempt)
            if observation.status == "interrupted":
                reason = "Unknown calibration completion; no automatic retry"
                break
            _verify_observation(store, attempt, binding, require_actual=actual)
            if _adequate(observation):
                chosen, reason = attempt, "Frozen predeclared recipe passed the development sanity floor"
                break
    except (LiveBudgetExhausted, ValueError, OSError) as error:
        reason = str(error)
        # DevelopmentRunner commits a timeout before propagating wall-clock
        # exhaustion. Preserve that actual failed attempt in the calibration report.
        if key and not any(item["key"] == key for item in attempts):
            row = store.db.execute("SELECT result_ref FROM live_operations WHERE key=?",
                                   ("experiment/" + key,)).fetchone()
            if row is not None and row["result_ref"]:
                observed = Observation.model_validate(store.json(row["result_ref"]))
                attempts.append({"key": key, "name": name,
                                 "candidate_ref": store.put_json(candidate, "calibration-candidate"),
                                 "observation_ref": store.put_json(observed, "calibration-observation")})
    measurements = _measurements(store, Observation.model_validate(store.json(chosen["observation_ref"]))) if chosen else {}
    receipt = {"schema_version": 1, "binding": binding, "measurement_origin": origin,
               "runner_origin": type(runner).__name__ + "/run_process" if actual else "injected_test_runner",
               "status": "pass" if chosen else "not_ready", "reason": reason,
               "selected_candidate_ref": chosen["candidate_ref"] if chosen else None,
               "selected_observation_ref": chosen["observation_ref"] if chosen else None,
               "attempts": attempts, "resource_measurements": measurements,
               "runtime_probe_ref": store.get("runtime_probe_ref"), "images": store.get("live_images"),
               "live_model_calls": 0, "final_access": False}
    receipt_ref = store.put_json(receipt, "calibration-receipt")
    store.set("calibration_receipt_ref", receipt_ref)
    result = CalibrationResult(**{key: receipt[key] for key in CalibrationResult.model_fields
                                  if key != "receipt_ref"}, receipt_ref=receipt_ref)
    store.cache("calibration/result", {"binding": binding, "origin": origin}, result)
    store.event("calibration/complete", result.status, reason)
    return result
