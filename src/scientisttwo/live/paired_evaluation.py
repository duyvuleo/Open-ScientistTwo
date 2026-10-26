"""Checkpointed paired-latency evaluation extension for trusted task adapters."""

import inspect
import math
from pathlib import Path
import statistics
import time

from ..artifacts import ArtifactFile, load_bundle, verified_file
from ..benchmarks.files import file_hash, verify_tree
from ..runtime_contracts import runtime_identity
from ..process import BoundedOutputCapture
from ..store import IntegrityError, canonical, digest
from .evaluation import aggregate_scores, normalize_scores
from .models import Observation
from .source_tree import candidate_snapshot, compare_source_trees, restore_source_tree
from .timing import PairedTimingReport


def _source_record(runner, key, candidate):
    candidate, code_ref = runner.materialize(key, candidate)
    tree_ref = runner.store.put_json(
        candidate_snapshot(runner.store, candidate), "materialized-source-tree"
    )
    return candidate, code_ref, tree_ref


def _combined_worker_logs(store, receipts, max_bytes):
    """Produce the standard controller log receipt from real bounded worker logs."""
    if set(receipts) != {"baseline", "candidate"}:
        raise ValueError("Paired execution requires both worker receipts")
    limit = min(max_bytes, 16 * 1024**2)
    capture = BoundedOutputCapture(limit)
    references = {}
    try:
        for arm in ("baseline", "candidate"):
            worker = store.json(receipts[arm])
            log_ref = worker["log_ref"]
            record = store.db.execute(
                "SELECT sha,size FROM artifacts WHERE path=?", (log_ref,)
            ).fetchone()
            if record is None:
                raise IntegrityError("Paired worker log is not registered evidence")
            path = verified_file(
                store, ArtifactFile(ref=log_ref, sha256=record["sha"], size=record["size"])
            )
            references[arm] = log_ref
            capture.append(("[" + arm + " worker log]\n").encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(64 * 1024), b""):
                    capture.append(block)
            capture.append(b"\n")
        content = capture.text().encode("utf-8")
        truncated = capture.total > limit or len(content) > limit
        if len(content) > limit:
            # Invalid UTF-8 replacement may expand bytes; keep the final artifact bounded too.
            head, tail = (limit + 1) // 2, limit // 2
            content = content[:head] + (content[-tail:] if tail else b"")
        return {
            "log_ref": store.put(content, "log", "paired-execution-log"),
            "worker_log_refs": references,
            "log_bytes": capture.total,
            "log_truncated": truncated,
        }
    finally:
        capture.stream.close()


def execute_paired(runner, key, candidate, phase="full", seeds=None, *, run_pair):
    """Run an adapter-supplied isolated paired executor with campaign recovery rules.

    Baseline observations represent the reference's identity ratio (1.0); candidate
    observations use candidate/base medians from the same paired measurement. Raw
    timings and timing dispersion remain in the sealed report, never discarded.
    """
    from .runtime import LiveBudgetExhausted, validate_prepared

    store, config = runner.store, runner.config.development
    if config.metric != "latency_ratio" or config.direction != "minimize":
        raise ValueError("Paired timing requires latency_ratio with direction=minimize")
    seeds = seeds or (config.seeds[:1] if phase == "subset" else config.seeds)
    if phase == "final" and not store.get("live_frozen"):
        raise ValueError("Protected final timing requires a frozen campaign")
    partition = "final" if phase == "final" else "subset" if phase == "subset" else "full"
    prepared = store.json(store.get("prepared/" + partition))
    validate_prepared(
        prepared,
        store.root / "final-data" if phase == "final" else store.root / "development" / partition,
    )
    baseline, _, baseline_ref = _source_record(
        runner, "paired-reference/" + phase + "/" + key, runner.adapter.baseline()
    )
    baseline_tree = store.json(baseline_ref)
    executor_source = inspect.getsourcefile(run_pair)
    if executor_source is None:
        raise ValueError("Paired executor requires inspectable trusted implementation provenance")
    protocol_spec = {
        "adapter": runner.adapter.identity(),
        "data": prepared["files"],
        "seeds": seeds,
        "evaluation": config.model_dump(mode="json"),
        "phase": phase,
        "runtime": runtime_identity(runner.spec, store.get("live_images"), runner.hardware),
        "paired_timing": {
            "baseline_source_bundle": baseline_tree["bundle_ref"],
            "baseline_command": baseline.command,
            "executor": run_pair.__module__ + ":" + run_pair.__qualname__,
            "executor_source_sha256": file_hash(Path(executor_source)),
            "warmups": 5,
            "pairs": 10,
            "selection": "paired_median_latency_ratio",
        },
    }
    protocol = digest(canonical(protocol_spec))
    request = {"candidate": candidate.model_dump(), "protocol": protocol}
    cache_key = ("final-paired/" if phase == "final" else "development/") + key
    operation_key = ("final/" if phase == "final" else "experiment/") + key
    cached = store.cached(cache_key, request)
    if cached:
        return Observation.model_validate(cached)
    result = Observation(
        id=key,
        phase=phase,
        status="invalid",
        protocol_hash=protocol,
        protocol_ref=store.put_json(protocol_spec, "paired-evaluation-protocol"),
        code_ref=store.put_json(candidate, "candidate-source"),
        metric=config.metric,
    )
    prior = runner.budget.reserve(operation_key, "experiment", request)
    if prior:
        if prior["result_ref"]:
            return Observation.model_validate(store.json(prior["result_ref"]))
        result.status = "interrupted"
        result.reason = "Uncommitted paired timing outcome; attempt will not be repeated"
        return result
    started = time.monotonic()
    budget_error = None
    try:
        candidate, result.code_ref, result.source_tree_ref = _source_record(
            runner, ("final-paired/" if phase == "final" else "development/") + key, candidate
        )
        same_baseline = (
            not compare_source_trees(store, baseline_ref, result.source_tree_ref)
            and candidate.command == baseline.command
        )
        scores = []
        for seed in seeds:
            runner.budget.remaining()
            stage = (
                store.root / "paired" / digest((phase + "/" + key + "/" + str(seed)).encode())[:24]
            )
            baseline_dir, candidate_dir = stage / "baseline-source", stage / "candidate-source"
            restore_source_tree(store, baseline_ref, baseline_dir)
            restore_source_tree(store, result.source_tree_ref, candidate_dir)
            report, receipts = run_pair(
                store,
                runner.spec,
                runner.image,
                baseline_dir,
                candidate_dir,
                prepared,
                stage / "execution",
                timeout=min(runner.spec.resources.seconds, runner.budget.remaining()),
                seed=seed,
            )
            report = PairedTimingReport.model_validate(report)
            _validate_report(report)
            if report.baseline.median_seconds <= 0:
                raise ValueError("Paired baseline median must be positive")
            ratio = (
                1.0
                if same_baseline
                else report.candidate.median_seconds / report.baseline.median_seconds
            )
            score = normalize_scores(config, {"latency_ratio": ratio})
            selected = [
                sample
                for sample in report.samples
                if sample.phase == "measurement" and sample.arm == "candidate"
            ][-1]
            bundle_ref = selected.correctness["bundle_ref"]
            prediction_ref = selected.correctness["prediction_ref"]
            artifacts = {
                path: file.ref for path, file in load_bundle(store, bundle_ref).files.items()
            }
            if prediction_ref not in artifacts.values():
                raise IntegrityError("Timing prediction is not bound to its artifact bundle")
            report_ref = store.put_json(report, "paired-timing-report")
            result.timing_refs.append(
                store.put_json(
                    {
                        "report_ref": report_ref,
                        "baseline_source_tree_ref": baseline_ref,
                        "candidate_source_tree_ref": result.source_tree_ref,
                        "baseline_identity": same_baseline,
                        "seed": seed,
                        "metric": "latency_ratio",
                        "value": ratio,
                        "receipts": receipts,
                    },
                    "paired-timing-evaluation",
                )
            )
            result.log_refs.append(
                store.put_json(
                    {
                        "returncode": 0,
                        "timed_out": False,
                        "seconds": report.wall_seconds,
                        "report_ref": report_ref,
                        "receipts": receipts,
                        **_combined_worker_logs(
                            store, receipts, runner.spec.resources.log_max_bytes
                        ),
                    },
                    "paired-timing-execution",
                )
            )
            result.prediction_refs.append(prediction_ref)
            result.artifact_refs.append(artifacts)
            result.bundle_refs.append(bundle_ref)
            scores.append(score)
        result.metrics, result.metric_values = aggregate_scores(config, scores)
        result.status = "success"
        result.value, result.values = (
            result.metrics[config.metric],
            result.metric_values[config.metric],
        )
        result.reason = "Independent correctness and warmed interleaved paired timing accepted"
        verify_tree(prepared["public"], prepared["files"])
    except TimeoutError:
        result.status, result.reason = (
            "timeout",
            "Paired timing exceeded its bounded execution budget",
        )
    except (ValueError, OSError, KeyError, SyntaxError, IntegrityError) as error:
        result.status, result.reason = "invalid", str(error)
    except RuntimeError as error:
        if isinstance(error, LiveBudgetExhausted):
            result.status, result.reason = "timeout", str(error)
            budget_error = error
        else:
            result.status, result.reason = "execution_failed", str(error)
    if result.status != "success":
        result.value, result.values, result.metrics, result.metric_values = None, [], {}, {}
    result.seconds = time.monotonic() - started
    result = Observation.model_validate(result.model_dump())
    runner.budget.finish(operation_key, result.status, result, result.seconds)
    store.cache(cache_key, request, result)
    if budget_error is not None:
        raise budget_error
    return result


def execute_paired_final(runner, key, candidate, *, run_pair):
    """Evaluate one frozen final pair, entirely outside development context/cache keys."""
    from .runtime import validate_prepared

    store = runner.store
    if not store.get("live_frozen"):
        raise ValueError("Final timing is unavailable before the research method is frozen")
    destination = store.root / "final-data"
    ref = store.get("prepared/final")
    if ref is None:
        prepared = validate_prepared(runner.adapter.prepare("final", destination), destination)
        store.set("prepared/final", store.put_json(prepared, "protected-final-partition"))
    else:
        validate_prepared(store.json(ref), destination)
    result = execute_paired(
        runner,
        key,
        candidate,
        "final",
        [runner.spec.final_seed],
        run_pair=run_pair,
    )
    observation_ref = store.put_json(result, "paired-final-observation")
    return {
        "status": "completed" if result.status == "success" else result.status,
        "reason": result.reason,
        "elapsed_seconds": result.seconds,
        "metric": result.metric,
        "direction": "minimize",
        "score": result.value,
        "metrics": result.metrics,
        "observation_ref": observation_ref,
        "protocol_hash": result.protocol_hash,
        "live_model_calls": 0,
        "leaderboard_eligible": False,
        "evaluation_track": "audited-v1",
    }


def _validate_report(report):
    """Recalculate timing summaries from the complete ordered trusted timer samples."""
    if report.warmup_rounds != 5 or report.measurement_pairs != 10 or report.payload_count < 1:
        raise ValueError("Paired timing report differs from the frozen 5/10 protocol")
    expected = []
    for phase, rounds in (("warmup", 5), ("measurement", 10)):
        for index in range(rounds):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            for position, arm in enumerate(order):
                for payload in range(report.payload_count):
                    expected.append((phase, index, position, arm, payload))
    actual = [(s.phase, s.round, s.position, s.arm, s.payload_index) for s in report.samples]
    if actual != expected or any(s.correctness.get("valid") is not True for s in report.samples):
        raise ValueError("Paired timing report has missing, reordered or unchecked samples")
    for arm in ("baseline", "candidate"):
        values = [
            sum(
                s.seconds
                for s in report.samples
                if s.phase == "measurement" and s.round == index and s.arm == arm
            )
            for index in range(10)
        ]
        summary = getattr(report, arm)
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        if summary.seconds != values or any(
            not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            for a, b in (
                (summary.median_seconds, statistics.median(values)),
                (summary.q1_seconds, q1),
                (summary.q3_seconds, q3),
                (summary.iqr_seconds, q3 - q1),
            )
        ):
            raise ValueError("Paired timing summaries differ from their recorded samples")


def audit_paired_observation(store, result, config, adapter, prepared):
    """Replay output correctness and timing arithmetic; no model or experiment calls."""
    development = config.development if hasattr(config, "development") else config
    callback = getattr(adapter, "audit_timing", None)
    if not callable(callback):
        raise ValueError("Paired observations require an independent adapter audit callback")
    if not (
        len(result.timing_refs)
        == len(result.values)
        == len(result.log_refs)
        == len(result.prediction_refs)
    ):
        raise ValueError("Paired metric/evidence cardinality mismatch")
    protocol = store.json(result.protocol_ref)
    if (
        digest(canonical(protocol)) != result.protocol_hash
        or protocol["adapter"] != adapter.identity()
        or protocol["evaluation"] != development.model_dump(mode="json")
        or protocol["data"] != prepared["files"]
        or protocol["phase"] != result.phase
        or len(protocol["seeds"]) != len(result.values)
    ):
        raise ValueError("Paired observation protocol differs from its declared evaluation")
    scores = []
    for index, ref in enumerate(result.timing_refs):
        evidence = store.json(ref)
        if (
            evidence["candidate_source_tree_ref"] != result.source_tree_ref
            or evidence["seed"] != protocol["seeds"][index]
        ):
            raise ValueError("Paired timing source or seed binding differs")
        baseline = store.json(evidence["baseline_source_tree_ref"])
        candidate = store.json(evidence["candidate_source_tree_ref"])
        if baseline["bundle_ref"] != protocol["paired_timing"]["baseline_source_bundle"]:
            raise ValueError("Paired timing reference source differs from the protocol")
        same = (
            not compare_source_trees(
                store, evidence["baseline_source_tree_ref"], evidence["candidate_source_tree_ref"]
            )
            and baseline["command"] == candidate["command"]
        )
        if same != evidence["baseline_identity"]:
            raise ValueError("Paired timing baseline identity flag is invalid")
        report = PairedTimingReport.model_validate(store.json(evidence["report_ref"]))
        _validate_report(report)
        callback(store, prepared, report)
        if report.baseline.median_seconds <= 0:
            raise ValueError("Paired baseline median must be positive")
        ratio = 1.0 if same else report.candidate.median_seconds / report.baseline.median_seconds
        if not math.isclose(evidence["value"], ratio, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Paired timing ratio differs from sealed host measurements")
        final = [s for s in report.samples if s.phase == "measurement" and s.arm == "candidate"][-1]
        if (
            final.correctness["prediction_ref"] != result.prediction_refs[index]
            or final.correctness["bundle_ref"] != result.bundle_refs[index]
        ):
            raise ValueError("Paired observation predictions differ from its sealed report")
        log = store.json(result.log_refs[index])
        if log["report_ref"] != evidence["report_ref"] or log["returncode"] or log["timed_out"]:
            raise ValueError("Paired observation execution receipt differs")
        log_limit = (
            protocol.get("runtime", {}).get("resources", {}).get("log_max_bytes", 16 * 1024**2)
        )
        expected_logs = _combined_worker_logs(store, evidence["receipts"], log_limit)
        if any(log.get(name) != value for name, value in expected_logs.items()):
            raise ValueError("Paired execution logs differ from their sealed worker evidence")
        scores.append({"latency_ratio": ratio})
    metrics, values = aggregate_scores(development, scores)
    if result.metrics != metrics or result.metric_values != values:
        raise ValueError("Paired aggregate metrics differ from sealed timing evidence")
    return {
        "experiment": result.id,
        "predictions": "pass",
        "paired_timing": "pass",
        "metric": "pass",
    }
