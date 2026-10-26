"""Typed empirical timing evidence and the native final-summary binding.

No candidate code runs here. Timing arithmetic is recomputed from sealed host
samples; independent output-oracle replay remains the native audit's job.
"""
import math
import statistics

from ..live.models import Observation
from ..live.paired_evaluation import _validate_report
from ..live.source_tree import compare_source_trees
from ..live.timing import PairedTimingReport
from ..store import IntegrityError, canonical, digest


TIMING_EVIDENCE_VERSION = "paired-timing-measurements-v2"
LEGACY_TIMING_EVIDENCE_VERSION = "paired-timing-measurements-v1"


def timing_measurements(store, observation, observation_ref, protocol, *, contract=TIMING_EVIDENCE_VERSION):
    """Project actual arm/seed statistics, never normalized identity ratios as time."""
    if not observation.timing_refs:
        return {}
    if contract not in {TIMING_EVIDENCE_VERSION, LEGACY_TIMING_EVIDENCE_VERSION}:
        raise IntegrityError("Unknown timing evidence projection contract")
    # Quartiles and all raw samples remain sealed and independently checked;
    # the current writer projection needs only median and dispersion, avoiding
    # redundant derived comparisons for each quartile. Keep old ledgers verifiable.
    statistics_fields = (("median_seconds", "q1_seconds", "q3_seconds", "iqr_seconds")
                         if contract == LEGACY_TIMING_EVIDENCE_VERSION else ("median_seconds", "iqr_seconds"))
    from ..live.audit import _observation_binding

    final = observation.phase == "final"
    row = store.db.execute("SELECT * FROM calls WHERE key=?",
        (("final-paired/" if final else "development/") + observation.id,)).fetchone()
    if row is None or canonical(store.json(row["output_ref"])) != canonical(observation):
        raise IntegrityError("Timing measurements require a committed observation")
    _observation_binding(store, row, observation, final=final)
    if (digest(canonical(protocol)) != observation.protocol_hash
            or protocol.get("paired_timing", {}).get("warmups") != 5
            or protocol.get("paired_timing", {}).get("pairs") != 10
            or len(observation.timing_refs) != len(protocol["seeds"])):
        raise IntegrityError("Timing measurement protocol/cardinality differs")
    measurements = {(arm, name): [] for arm in ("baseline", "candidate") for name in statistics_fields}
    ratios = []
    for index, ref in enumerate(observation.timing_refs):
        evidence = store.json(ref)
        seed = protocol["seeds"][index]
        if (evidence["seed"] != seed
                or evidence["candidate_source_tree_ref"] != observation.source_tree_ref):
            raise IntegrityError("Timing measurement source or seed binding differs")
        baseline = store.json(evidence["baseline_source_tree_ref"])
        candidate = store.json(evidence["candidate_source_tree_ref"])
        if (baseline["bundle_ref"] != protocol["paired_timing"]["baseline_source_bundle"]
                or baseline["command"] != protocol["paired_timing"]["baseline_command"]):
            raise IntegrityError("Timing baseline source differs from its frozen protocol")
        same = (not compare_source_trees(store, evidence["baseline_source_tree_ref"],
                    evidence["candidate_source_tree_ref"])
                and baseline["command"] == candidate["command"])
        if same != evidence["baseline_identity"]:
            raise IntegrityError("Timing baseline identity flag differs")
        report = PairedTimingReport.model_validate(store.json(evidence["report_ref"]))
        _validate_report(report)
        if any(not math.isfinite(sample.seconds) for sample in report.samples):
            raise IntegrityError("Timing samples must be finite")
        if report.baseline.median_seconds <= 0:
            raise IntegrityError("Timing baseline median must be positive")
        raw_ratio = report.candidate.median_seconds / report.baseline.median_seconds
        ratio = 1.0 if same else raw_ratio
        if (evidence.get("metric") != "latency_ratio"
                or not math.isclose(evidence["value"], ratio, rel_tol=1e-12, abs_tol=1e-12)
                or not math.isclose(observation.values[index], ratio, rel_tol=1e-12, abs_tol=1e-12)):
            raise IntegrityError("Timing ratio differs from its sealed measurements")
        ratios.append(ratio)
        log = store.json(observation.log_refs[index])
        if (log.get("report_ref") != evidence["report_ref"]
                or log.get("receipts") != evidence["receipts"]
                or log.get("returncode") != 0 or log.get("timed_out")):
            raise IntegrityError("Timing report differs from the committed execution receipt")
        for arm in ("baseline", "candidate"):
            # Read the receipt to validate its content hash, without exporting logs.
            store.json(evidence["receipts"][arm])
            for name in statistics_fields:
                measurements[arm, name].append({"seed": seed,
                    "value": getattr(getattr(report, arm), name),
                    "timing_ref": ref, "report_ref": evidence["report_ref"],
                    "source_ref": evidence[arm + "_source_tree_ref"],
                    "receipt_ref": evidence["receipts"][arm],
                    "measurement_scope": report.measurement_scope})
    if (not math.isclose(observation.value, statistics.mean(ratios), rel_tol=1e-12, abs_tol=1e-12)
            or observation.metrics != {"latency_ratio": observation.value}
            or observation.metric_values != {"latency_ratio": observation.values}):
        raise IntegrityError("Timing aggregate differs from committed per-seed ratios")
    entries = {}
    for (arm, name), values in measurements.items():
        metric = "paired." + arm + "." + name
        key = "measurement:" + digest(canonical([observation.id, metric]))[:24]
        entries[key] = {"kind": "measured", "observation_id": observation.id,
            "metric": metric, "arm": arm, "statistic": name, "unit": "seconds",
            "value": statistics.mean(item["value"] for item in values), "per_seed": values,
            "aggregate": "arithmetic_mean_of_per_seed_statistics_not_pooled_latency",
            "seed_semantics": "timing_repeat_seed_not_workload_generation_seed",
            "protocol_hash": observation.protocol_hash, "phase": observation.phase,
            "observation_ref": observation_ref, "source_ref": values[0]["source_ref"],
            "numeric_provenance": contract,
            "verification": "sealed_host_sample_arithmetic_recomputed; output_correctness_requires_native_audit; timing_not_rerun"}
    return entries


def verify_final_result_binding(store, label, result, *, summary_ref=None):
    """Verify a legacy result or paired summary without changing its operation."""
    operation = store.db.execute("SELECT * FROM live_operations WHERE key=?",
        ("final/" + label,)).fetchone()
    if operation is None or operation["kind"] != "experiment" or not operation["result_ref"]:
        raise IntegrityError("Final result lacks a committed experiment operation")
    if summary_ref is not None and canonical(store.json(summary_ref)) != canonical(result):
        raise IntegrityError("Final summary differs from its sealed artifact")
    committed = store.json(operation["result_ref"])
    if canonical(committed) == canonical(result):
        if summary_ref is not None and summary_ref != operation["result_ref"]:
            raise IntegrityError("Legacy final summary differs from its exact operation artifact")
        return {"kind": "legacy_exact_summary", "operation_ref": operation["result_ref"]}
    if not result.get("observation_ref"):
        raise IntegrityError("Final result differs from its immutable operation output")
    observation = Observation.model_validate(store.json(result["observation_ref"]))
    if canonical(observation) != canonical(committed) or observation.id != label or observation.phase != "final":
        raise IntegrityError("Final summary points to a different committed observation")
    from ..live.audit import _observation_binding
    row = store.db.execute("SELECT * FROM calls WHERE key=?", ("final-paired/" + label,)).fetchone()
    if row is None or not store.get("live_frozen"):
        raise IntegrityError("Final paired summary requires frozen committed evidence")
    _observation_binding(store, row, observation, final=True)
    expected = {"status": "completed" if observation.status == "success" else observation.status,
        "reason": observation.reason, "elapsed_seconds": observation.seconds,
        "metric": observation.metric, "direction": "minimize", "score": observation.value,
        "metrics": observation.metrics, "observation_ref": result["observation_ref"],
        "protocol_hash": observation.protocol_hash, "live_model_calls": 0,
        "leaderboard_eligible": False, "evaluation_track": "audited-v1"}
    if canonical(result) != canonical(expected):
        raise IntegrityError("Final paired summary fields differ from the committed observation")
    if observation.status == "success":
        if not observation.timing_refs:
            raise IntegrityError("Final paired observation lacks timing evidence")
        timing_measurements(store, observation, result["observation_ref"], store.json(observation.protocol_ref))
    return {"kind": "paired_observation_summary", "operation_ref": operation["result_ref"],
        "observation_ref": result["observation_ref"]}
