"""Versioned trusted diagnostics; labels never enter controller evidence.

Array functions are pure and are intended for the isolated scoring boundary.
The controller entry point consumes committed observations, timing reports, or
already sealed scorer outputs. It never opens a dataset or imports candidate code.
"""

from typing import Annotated, Literal
from pathlib import Path

import numpy as np
from pydantic import Field, model_validator

from ..models import Record
from ..store import IntegrityError, canonical, digest


Version = Literal[1]
Seed = Annotated[int, Field(strict=True, ge=0)]
_KINDS = {
    "paired_differences": ("committed_observations", 1),
    "paired_interval": ("committed_observations", 3),
    "classification": ("trusted_scoring", 1),
    "regression_residuals": ("trusted_scoring", 1),
    "timing_distribution": ("committed_timing", 1),
    "input_consistency": ("trusted_scoring", 1),
}


class DiagnosticDescriptor(Record):
    schema_version: Version = 1
    kind: str
    input_kind: str
    minimum_seeds: int
    output_kinds: list[str] = ["summary", "table", "figure_inputs"]
    scope: str = "Development evidence only; execution does not resolve scientific concerns"


class DiagnosticRequest(Record):
    schema_version: Version = 1
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_/-]+$")
    kind: str = Field(min_length=1, max_length=100)
    metric: str | None = None
    expected_seeds: list[Seed] = Field(default_factory=list, max_length=32)
    concern_ids: list[str] = Field(default_factory=list, max_length=32)
    parameters: dict = Field(default_factory=dict)
    evaluation_split: Literal["development", "final"] = "development"

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.expected_seeds)) != len(self.expected_seeds):
            raise ValueError("Diagnostic training seeds must be distinct")
        if len(set(self.concern_ids)) != len(self.concern_ids):
            raise ValueError("Diagnostic concern IDs must be distinct")
        canonical(self.parameters)
        return self


class DiagnosticInputs(Record):
    observation_refs: list[str] = Field(default_factory=list, max_length=32)
    # Scoring outputs are admitted only by a controller-owned committed call.
    trusted_result_refs: list[str] = Field(default_factory=list, max_length=32)
    declared_kinds: list[str] = Field(default_factory=list)


class DiagnosticResult(Record):
    schema_version: Version = 1
    request: DiagnosticRequest
    status: Literal["success", "unfulfilled"]
    reasons: list[str] = Field(default_factory=list)
    input_refs: list[str] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)
    tables: dict[str, list[dict]] = Field(default_factory=dict)
    arrays: dict[str, list] = Field(default_factory=dict)
    summary: dict = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    artifact_ref: str | None = None


def diagnostic_capabilities(declared_kinds=(), *, input_kinds=()):
    """Registry membership alone does not advertise runnable task support."""
    return [DiagnosticDescriptor(kind=kind, input_kind=_KINDS[kind][0],
                                 minimum_seeds=_KINDS[kind][1])
            for kind in sorted(set(declared_kinds) & _KINDS.keys())
            if _KINDS[kind][0] in input_kinds]


def _finite(value, *, dimensions=None):
    array = np.asarray(value)
    if (array.dtype.kind not in "iuf" or not array.size or array.size > 10_000_000
            or not np.isfinite(array).all()):
        raise ValueError("Diagnostic arrays must be nonempty bounded finite numeric arrays")
    if dimensions is not None and array.ndim not in dimensions:
        raise ValueError("Diagnostic array dimensions differ from the declared contract")
    return array.astype(np.float64)


def _histogram(values):
    low, high = float(np.min(values)), float(np.max(values))
    width = max(abs(low), abs(high), 1.) * 1e-12
    # Near-constant floating values may not admit twenty distinct bin edges.
    # Widen only the display range; measured values and interval stay untouched.
    if high - low < width:
        return np.histogram(values, bins=20, range=(low - width, high + width))
    return np.histogram(values, bins=20)


def paired_differences(baseline, candidate, seeds):
    left, right = _finite(baseline, dimensions={1}), _finite(candidate, dimensions={1})
    if (left.shape != right.shape or len(seeds) != len(left) or len(seeds) > 32
            or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("Paired diagnostics require distinct matched seeds and equal lengths")
    delta = right - left
    return {"tables": {"paired_differences": [
        {"seed": seed, "baseline": float(a), "candidate": float(b), "difference": float(d)}
        for seed, a, b, d in zip(seeds, left, right, delta, strict=True)]},
        "arrays": {"differences": delta.tolist()},
        "summary": {"seed_count": len(seeds), "mean_difference": float(delta.mean()),
                    "difference_definition": "candidate minus baseline"},
        "limitations": ["A numerical difference is not a statistical-significance claim"]}


def paired_interval(baseline, candidate, seeds, *, confidence=0.95,
                    resamples=10_000, diagnostic_seed=1729):
    """Percentile bootstrap over matched training seeds, never dataset rows."""
    if resamples != 10_000 or diagnostic_seed != 1729:
        raise ValueError("Version 1 requires exactly 10000 resamples and diagnostic RNG seed 1729")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between zero and one")
    output = paired_differences(baseline, candidate, seeds)
    if len(seeds) < 3:
        raise ValueError("Paired uncertainty requires at least three distinct matched training seeds")
    delta = np.asarray(output["arrays"]["differences"])
    rng = np.random.default_rng(diagnostic_seed)
    means = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    counts, edges = _histogram(means)
    output["summary"].update(confidence=confidence, lower=float(low), upper=float(high),
                             resamples=resamples, diagnostic_rng_seed=diagnostic_seed,
                             resampling_unit="distinct matched training seed",
                             interval_method="paired percentile bootstrap")
    output["arrays"].update(bootstrap_histogram_counts=counts.tolist(),
                           bootstrap_histogram_edges=edges.tolist())
    output["limitations"].append("Small seed counts yield unstable uncertainty estimates; no population guarantee")
    return output


def classification_diagnostics(targets, predictions, *, classes=None):
    """Hard class IDs; probability/logit argmax belongs to the declared scorer."""
    actual, predicted = _finite(targets, dimensions={1}), _finite(predictions, dimensions={1})
    if actual.shape != predicted.shape or np.any(actual != actual.astype(int)) or np.any(predicted != predicted.astype(int)):
        raise ValueError("Classification diagnostics require matched integer class IDs")
    classes = sorted(set(actual.tolist()) | set(predicted.tolist())) if classes is None else classes
    labels = _finite(classes, dimensions={1})
    if len(labels) > 4096 or len(set(labels)) != len(labels) or np.any(labels != labels.astype(int)):
        raise ValueError("Class vocabulary must be distinct bounded integer IDs")
    lookup = {int(value): index for index, value in enumerate(labels)}
    if any(int(value) not in lookup for value in np.concatenate([actual, predicted])):
        raise ValueError("Class ID outside declared vocabulary")
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(matrix, ([lookup[int(x)] for x in actual], [lookup[int(x)] for x in predicted]), 1)
    rows = []
    for index, label in enumerate(labels):
        tp = int(matrix[index, index])
        support, proposed = int(matrix[index].sum()), int(matrix[:, index].sum())
        precision, recall = tp / proposed if proposed else 0., tp / support if support else 0.
        rows.append({"class_id": int(label), "support": support, "predicted_count": proposed,
                     "precision": precision, "recall": recall,
                     "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.})
    return {"tables": {"per_class": rows},
            "arrays": {"confusion_matrix": matrix.tolist(), "class_ids": labels.astype(int).tolist()},
            "summary": {"example_count": len(actual), "accuracy": float(np.mean(actual == predicted)),
                        "macro_f1": float(np.mean([row["f1"] for row in rows])),
                        "confusion_axes": "rows=true, columns=predicted", "zero_division": 0},
            "limitations": ["Aggregate development-label diagnostics; no individual labels exported"]}


def regression_residuals(targets, predictions, *, target_names=None):
    actual, predicted = _finite(targets, dimensions={1, 2}), _finite(predictions, dimensions={1, 2})
    if actual.shape != predicted.shape:
        raise ValueError("Regression predictions and targets must have equal shapes")
    actual = actual.reshape(len(actual), -1)
    predicted = predicted.reshape(len(predicted), -1)
    names = target_names or ["target_" + str(i) for i in range(actual.shape[1])]
    if len(names) != actual.shape[1] or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise ValueError("Target names must match distinct regression columns")
    residuals = predicted - actual
    rows, arrays = [], {}
    for index, name in enumerate(names):
        values = residuals[:, index]
        counts, edges = _histogram(values)
        arrays[name + "/histogram_counts"] = counts.tolist()
        arrays[name + "/histogram_edges"] = edges.tolist()
        rows.append({"target": name, "count": len(values), "bias": float(values.mean()),
                     "mae": float(np.abs(values).mean()), "rmse": float(np.sqrt(np.mean(values ** 2))),
                     "median": float(np.median(values)), "q1": float(np.quantile(values, .25)),
                     "q3": float(np.quantile(values, .75))})
    return {"tables": {"per_target_residuals": rows}, "arrays": arrays,
            "summary": {"example_count": len(actual), "mean_target_rmse": float(np.mean([r["rmse"] for r in rows])),
                        "residual_definition": "prediction minus target"},
            "limitations": ["Residuals are aggregate summaries; protected labels and row residuals are not exported"]}


def timing_distribution(baseline, candidate):
    left, right = _finite(baseline, dimensions={1}), _finite(candidate, dimensions={1})
    if left.shape != right.shape or len(left) < 2 or np.any(left <= 0) or np.any(right < 0):
        raise ValueError("Timing requires at least two matched pairs, positive baseline, nonnegative candidate")
    rows = []
    for arm, values in (("baseline", left), ("candidate", right)):
        median = float(np.median(values))
        q1, q3 = np.quantile(values, [.25, .75])
        rows.append({"arm": arm, "count": len(values), "median_seconds": median,
                     "q1_seconds": float(q1), "q3_seconds": float(q3),
                     "iqr_seconds": float(q3 - q1), "mad_seconds": float(np.median(np.abs(values - median))),
                     "p95_seconds": float(np.quantile(values, .95))})
    return {"tables": {"timing": rows}, "arrays": {"baseline_seconds": left.tolist(), "candidate_seconds": right.tolist()},
            "summary": {"paired_count": len(left), "median_latency_ratio": rows[1]["median_seconds"] / rows[0]["median_seconds"]},
            "limitations": ["Resident request latency includes transport overhead; startup is reported separately"]}


class ConsistencyDeclaration(Record):
    id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    expectation: Literal["equal", "allclose"]
    rtol: float = Field(default=0, ge=0)
    atol: float = Field(default=0, ge=0)
    source_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def exact(self):
        if self.expectation == "equal" and (self.rtol or self.atol):
            raise ValueError("Exact consistency cannot carry approximate tolerances")
        return self


def input_consistency(original, transformed, declaration: ConsistencyDeclaration):
    declaration = ConsistencyDeclaration.model_validate(declaration.model_dump())
    left, right = _finite(original), _finite(transformed)
    if left.shape != right.shape:
        raise ValueError("Consistency predictions must have matching shapes")
    matched = (left == right) if declaration.expectation == "equal" else np.isclose(
        left, right, rtol=declaration.rtol, atol=declaration.atol)
    return {"summary": {"declaration_id": declaration.id, "task_id": declaration.task_id,
                        "element_count": int(left.size), "matched_count": int(matched.sum()),
                        "all_matched": bool(matched.all()), "max_absolute_difference": float(np.abs(right-left).max()),
                        "declaration": declaration.model_dump()},
            "tables": {}, "arrays": {},
            "limitations": ["Tests only the task-declared transformation; does not establish generic invariance"]}


def committed_observation(store, ref):
    """Reject forged registered JSON, final results, and modified commitments."""
    from ..live.models import Observation
    raw = store.json(ref)
    if raw.get("phase") == "final":
        raise ValueError("Final-test evidence is forbidden in research diagnostics")
    observation = Observation.model_validate(raw)
    row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("development/" + observation.id,)).fetchone()
    if not row or row[0] != ref:
        raise IntegrityError("Diagnostic observation is not its committed development record")
    if observation.status != "success" or not observation.protocol_ref:
        raise ValueError("Diagnostic needs successful committed observations with a protocol")
    protocol = store.json(observation.protocol_ref)
    if digest(canonical(protocol)) != observation.protocol_hash or protocol.get("phase") != observation.phase:
        raise IntegrityError("Diagnostic protocol identity differs from the observation")
    seeds = protocol.get("seeds")
    if (not isinstance(seeds, list) or len(seeds) != len(observation.values)
            or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds)):
        raise IntegrityError("Diagnostic protocol has invalid seed/metric binding")
    return observation, protocol


def _paired(store, request, inputs):
    if len(inputs.observation_refs) != 2:
        raise ValueError("Paired diagnostics require baseline then candidate observation references")
    left, lp = committed_observation(store, inputs.observation_refs[0])
    right, rp = committed_observation(store, inputs.observation_refs[1])
    if left.protocol_hash != right.protocol_hash or lp != rp:
        raise ValueError("Paired diagnostics require matching complete protocols")
    if not request.expected_seeds or request.expected_seeds != lp["seeds"]:
        raise ValueError("Observed seeds do not match the exact diagnostic request")
    metric = request.metric
    def values(observation):
        metrics = observation.metric_values or {observation.metric: observation.values}
        if metric not in metrics:
            raise ValueError("Requested metric is absent from committed evidence")
        return metrics[metric]
    if request.kind == "paired_differences":
        if request.parameters:
            raise ValueError("Paired differences have no optional parameters")
        return paired_differences(values(left), values(right), request.expected_seeds)
    if set(request.parameters) - {"confidence", "resamples", "diagnostic_seed"}:
        raise ValueError("Unsupported paired uncertainty parameter")
    return paired_interval(values(left), values(right), request.expected_seeds, **request.parameters)


def _timing(store, request, inputs):
    from ..live.paired_evaluation import _validate_report
    from ..live.timing import PairedTimingReport
    from ..live.source_tree import compare_source_trees
    if len(inputs.observation_refs) != 1 or request.parameters:
        raise ValueError("Timing diagnostics need one observation and no additional parameters")
    observation, protocol = committed_observation(store, inputs.observation_refs[0])
    if request.expected_seeds != protocol["seeds"] or len(observation.timing_refs) != len(request.expected_seeds):
        raise ValueError("Timing report seed coverage differs from the exact request")
    output = {"tables": {}, "arrays": {}, "summary": {"seed_count": len(request.expected_seeds)}, "limitations": []}
    per_seed = []
    for index, (seed, ref) in enumerate(zip(request.expected_seeds, observation.timing_refs, strict=True)):
        evidence = store.json(ref)
        if evidence.get("seed") != seed or evidence.get("candidate_source_tree_ref") != observation.source_tree_ref:
            raise IntegrityError("Timing diagnostic source or seed binding differs")
        baseline = store.json(evidence["baseline_source_tree_ref"])
        candidate = store.json(evidence["candidate_source_tree_ref"])
        if baseline["bundle_ref"] != protocol.get("paired_timing", {}).get("baseline_source_bundle"):
            raise IntegrityError("Timing diagnostic baseline differs from the protocol")
        same = (not compare_source_trees(store, evidence["baseline_source_tree_ref"], observation.source_tree_ref)
                and baseline["command"] == candidate["command"])
        if same != evidence.get("baseline_identity"):
            raise IntegrityError("Timing diagnostic baseline identity is invalid")
        if index >= len(observation.log_refs):
            raise IntegrityError("Timing diagnostic lacks bound worker receipt")
        receipt = store.json(observation.log_refs[index])
        if receipt.get("report_ref") != evidence["report_ref"] or receipt.get("receipts") != evidence.get("receipts"):
            raise IntegrityError("Timing diagnostic receipt differs from the scored measurement")
        for worker_ref in evidence["receipts"].values():
            store.json(worker_ref)
        report = PairedTimingReport.model_validate(store.json(evidence["report_ref"]))
        _validate_report(report)
        detail = timing_distribution(report.baseline.seconds, report.candidate.seconds)
        ratio = 1. if same else detail["summary"]["median_latency_ratio"]
        if not np.isclose(ratio, observation.values[index], rtol=1e-12, atol=1e-12) or not np.isclose(ratio, evidence["value"], rtol=1e-12, atol=1e-12):
            raise IntegrityError("Timing diagnostic ratio differs from committed scoring")
        per_seed.append({"seed": seed, **detail["summary"], "wall_seconds": report.wall_seconds})
        output["tables"][f"seed_{seed}/timing"] = detail["tables"]["timing"]
        output["arrays"].update({f"seed_{seed}/" + key: value for key, value in detail["arrays"].items()})
        output["limitations"] = detail["limitations"] + [report.measurement_scope,
            "A baseline identity observation is scored as 1.0; independent timing variation may differ",
            "This diagnostic verifies recorded timing arithmetic; workload correctness remains an adapter audit"]
    output["tables"]["per_seed"] = per_seed
    return output


def run_diagnostic(store, request: DiagnosticRequest, inputs: DiagnosticInputs) -> DiagnosticResult:
    """Persist an exact, repeatable diagnostic or an explicit unfulfilled request."""
    request = DiagnosticRequest.model_validate(request.model_dump())
    inputs = DiagnosticInputs.model_validate(inputs.model_dump())
    envelope = {"request": request.model_dump(), "inputs": inputs.model_dump()}
    key = "research-diagnostic/" + request.id
    # Verify inputs even on resume: a cached result must not conceal tampering.
    for ref in inputs.observation_refs + inputs.trusted_result_refs:
        store.read(ref)
    payload, reasons = {}, []
    try:
        if request.evaluation_split != "development":
            raise ValueError("Final-test diagnostics cannot feed research refinement")
        if request.kind not in inputs.declared_kinds or request.kind not in _KINDS:
            raise ValueError("Diagnostic is not supported by the declared task capability")
        if request.kind in {"paired_differences", "paired_interval"}:
            payload = _paired(store, request, inputs)
        elif request.kind == "timing_distribution":
            payload = _timing(store, request, inputs)
        else:
            if len(inputs.trusted_result_refs) != 1:
                raise ValueError("Label-dependent diagnostics require an isolated trusted scoring result")
            ref = inputs.trusted_result_refs[0]
            row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("trusted-diagnostic/" + request.id,)).fetchone()
            if row is None or row[0] != ref:
                raise IntegrityError("Diagnostic output is not bound to a trusted scoring operation")
            sealed = store.json(ref)
            if sealed.get("request") != request.model_dump() or sealed.get("observation_refs") != inputs.observation_refs:
                raise IntegrityError("Trusted diagnostic result does not match the exact request and evidence")
            for observation_ref in inputs.observation_refs:
                committed_observation(store, observation_ref)
            for child_ref in sealed.get("result_refs", []) + sealed.get("receipt_refs", []):
                store.read(child_ref)
            payload = sealed["output"]
        canonical(payload)
    except ValueError as error:
        reasons.append(str(error))
    result = DiagnosticResult(request=request, status="unfulfilled" if reasons else "success",
                              reasons=reasons, input_refs=inputs.observation_refs + inputs.trusted_result_refs,
                              provenance={"implementation": "scientisttwo.research.diagnostics", "version": 1,
                                          "implementation_sha256": digest(Path(__file__).read_bytes()),
                                          "numpy_version": np.__version__,
                                          "input_hash": digest(canonical(envelope)), "split": "development"},
                              **({} if reasons else payload))
    cached = store.cached(key, envelope)
    if cached is not None and cached != result.model_dump():
        raise IntegrityError("Diagnostic replay differs from the committed result")
    stored = store.cache(key, envelope, result)
    ref = store.db.execute("SELECT output_ref FROM calls WHERE key=?", (key,)).fetchone()[0]
    return DiagnosticResult.model_validate(stored).model_copy(update={"artifact_ref": ref})
