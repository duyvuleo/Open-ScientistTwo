"""Read-only acceptance for the bounded live pilot, not general research quality.

The harness registers the actual research audit and before/after resume snapshots
as artifacts. This checker reads their immutable references and rechecks ledger,
source, measurements and role bindings; it never invokes agents or experiments.
"""

from collections import Counter
from dataclasses import asdict
import math
from pathlib import Path
from statistics import fmean

from ..catalog import CLASSIFICATION, REGRESSION, OPTIMIZATION
from ..models import AgentResult, EvalSpec, ExperimentResult, TaskSpec
from ..live.models import LiveSpec, Observation
from ..store import IntegrityError, canonical, digest, file_digest
from ..studies import media_relay as relay
from ..tasks import load_task_adapter
from .pilot_bridge import _aggregate


MANDATORY_ROLES = frozenset({
    "limitations", "verify_limitations", "initial_idea", "additional_idea", "novelty_checker",
    "baseline_plan", "implement", "critic", "full_set_coder", "selection_review", "evolve",
    "ablation_planner", "ablation_coder", "control_reviewer", "ablation_critic",
    "paperorchestra_model", "paperorchestra_bindings", "writing_critic", "scientific_review", "meta",
})


def _rows(store, table):
    return [dict(row) for row in store.db.execute("SELECT * FROM " + table + " ORDER BY key")]


def _ledger_files(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError("Direct model ledger is missing or a symlink")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise IntegrityError("Direct model ledger contains a symlink")
        if path.is_file() and path.name != "service.lock":
            files[path.relative_to(root).as_posix()] = file_digest(path)
    return files


def capture_resume_snapshot(store):
    """Capture exact durable work; the caller may register this returned object."""
    result = {"schema_version": "minimal-live-resume-snapshot-v1", "store_root": str(store.root),
            "spec_ref": store.get("live_spec_ref"), "task_ref": store.get("task_ref"),
            "deadline": store.get("live_deadline"), "state": store.get("live_state"),
            "selection_sha256": digest(canonical(store.get("live_frozen"))),
            "final_results_sha256": digest(canonical(store.get("live_final"))),
            "manuscript_bundle_ref": store.get("research_manuscript_bundle"),
            "final_bundle_ref": store.get("research_final_bundle"),
            "model_accounting_ref": store.get("live_pilot_model_accounting"),
            "ledgers": {table: {"count": len(rows), "sha256": digest(canonical(rows))}
                        for table in ("live_operations", "jobs", "calls", "events")
                        for rows in [_rows(store, table)]},
            "model_ledger_files": _ledger_files(store.root / "model-ledger")}
    if store.get("inherited_model_accounting_ref"):
        from .live_pilot import validate_inherited_model_accounting
        config = LiveSpec.model_validate(store.json(store.get("live_spec_ref")))
        inherited = validate_inherited_model_accounting(store, config)
        result["inherited_model_artifacts"] = inherited["artifact_hashes"]
    return result


def _measured(store, value, phase, seeds):
    observation = Observation.model_validate(value)
    if observation.status != "success" or observation.phase != phase:
        raise IntegrityError("Required measured observation is absent or unsuccessful")
    original = ExperimentResult.model_validate(store.json(observation.legacy_result_ref))
    if (original.measurement_origin != "container" or original.status != "success"
            or observation.metric_rows != [row.model_dump(mode="json") for row in original.rows]
            or {row.seed for row in original.rows} != set(seeds)):
        raise IntegrityError("Observation lacks the exact real container measurement grid")
    protocol = store.json(observation.protocol_ref)
    if digest(canonical(protocol)) != observation.protocol_hash or protocol["seeds"] != seeds:
        raise IntegrityError("Observation protocol or seed identity differs")
    task = TaskSpec.model_validate(store.json(store.get("task_ref")))
    spec = EvalSpec.model_validate(store.json(original.spec_ref))
    declared = EvalSpec.for_phase(task, phase)
    if (spec.task != task or spec.phase != phase or spec.seeds != seeds
            or spec.datasets != declared.datasets or spec.train_fraction != declared.train_fraction
            or spec.split != declared.split or protocol.get("evaluation") != spec.model_dump(mode="json")
            or original.protocol_hash != digest(canonical(spec))
            or protocol.get("phase") != phase
            or "datasets" in protocol and protocol["datasets"] != declared.datasets
            or observation.metric != task.primary_metric):
        raise IntegrityError("Observation differs from the frozen task evaluation contract")
    metrics, values = _aggregate(original, task, seeds, declared.datasets)
    if (observation.metrics != metrics or observation.metric_values != values
            or observation.value != metrics[task.primary_metric]
            or observation.values != values[task.primary_metric]):
        raise IntegrityError("Observation aggregates differ from the authoritative metric grid")
    operation = ("final/" if phase == "final" else "experiment/") + observation.id
    row = store.db.execute("SELECT * FROM live_operations WHERE key=?", (operation,)).fetchone()
    if not row or not row["result_ref"] or row["status"] not in {"success", "completed"}:
        raise IntegrityError("Observation has no committed successful execution")
    actual = store.json(row["result_ref"])
    if phase == "final":
        actual = store.json(actual["observation_ref"])
    if canonical(actual) != canonical(observation):
        raise IntegrityError("Observation differs from its committed operation")
    return observation


def _phase_outcome(store, task, candidate, baseline):
    """Apply the task's trusted rule, keeping numerical gain and guardrails distinct."""
    if candidate.phase != baseline.phase or candidate.protocol_hash != baseline.protocol_hash:
        raise IntegrityError("Phase comparison requires identical candidate and baseline protocols")
    original_candidate = ExperimentResult.model_validate(store.json(candidate.legacy_result_ref))
    original_baseline = ExperimentResult.model_validate(store.json(baseline.legacy_result_ref))
    comparison = load_task_adapter(task.adapter, task).compare(original_candidate, original_baseline, task)
    if not comparison.valid:
        raise IntegrityError("Task comparison rejected the measured evidence: " + comparison.reason)
    sign = 1 if task.metric_directions[task.primary_metric] == "maximize" else -1
    delta = sign * (candidate.value - baseline.value)
    datasets = sorted({row.dataset for row in original_baseline.rows})
    per_dataset = {}
    for dataset in datasets:
        per_dataset[dataset] = {}
        for metric, direction in task.metric_directions.items():
            before = fmean(row.metrics[metric] for row in original_baseline.rows if row.dataset == dataset)
            after = fmean(row.metrics[metric] for row in original_candidate.rows if row.dataset == dataset)
            per_dataset[dataset][metric] = {"baseline": before, "selected": after,
                "signed_improvement": (1 if direction == "maximize" else -1) * (after - before)}
    return {"status": "supported" if comparison.improved else "not_demonstrated",
            "phase": candidate.phase, "metric": task.primary_metric,
            "direction": task.metric_directions[task.primary_metric],
            "baseline": baseline.value, "selected": candidate.value,
            "baseline_per_seed": baseline.values, "selected_per_seed": candidate.values,
            "improvement": delta, "epsilon": task.improvement_epsilon,
            "numerically_improved": delta > task.improvement_epsilon,
            "guardrails_pass": comparison.guardrails_pass, "guardrails": task.guardrails,
            "guardrail_scope": "Each dataset mean over matched training seeds",
            "per_dataset": per_dataset, "trusted_comparison": comparison.model_dump(mode="json"),
            "statistical_significance": "not_claimed"}


def _roles(store, config):
    roles = Counter()
    for row in store.db.execute("SELECT * FROM calls WHERE key LIKE 'live-agent/%'"):
        key = row["key"][len("live-agent/"):]
        request = store.json(store.get("request/" + key))
        response = AgentResult.model_validate(store.json(row["output_ref"]))
        if (response.synthetic or response.backend != "codex" or response.model != config.model
                or row["input_hash"] != digest(canonical(request))):
            raise IntegrityError("Agent provenance or exact request binding differs")
        operations = [op for op in store.db.execute("SELECT * FROM live_operations WHERE kind='model'")
                      if op["key"].startswith("model/" + key + "/") and op["status"] == "success"]
        if not any(op["request_hash"] == row["input_hash"] and op["result_ref"] == row["output_ref"]
                   for op in operations):
            raise IntegrityError("Agent response has no successful matching model operation")
        roles[request["role"]] += 1
    return dict(roles)


def _model_accounting(store, config):
    if store.get("inherited_model_accounting_ref"):
        return _composite_model_accounting(store, config)
    root = store.root / "model-ledger"
    summary = store.json(store.get("live_pilot_model_accounting"))
    policy = relay.RelayPolicy.from_mapping(relay.read_json(root / "policy.json", 4096))
    if (summary["ledger"] != str(root) or summary["transport"] != "direct_local_no_queue"
            or canonical(summary["policy"]) != canonical(asdict(policy))
            or policy.model != config.model or policy.source_sha256 != digest(canonical(config.implementation))
            or policy.expires_at != store.get("live_deadline") or policy.max_calls != config.max_calls
            or policy.token_stop_threshold != config.token_stop_threshold):
        raise IntegrityError("Direct-model accounting policy differs from the frozen campaign")
    records = []
    for entry in sorted((root / "claims").iterdir()):
        if (entry.is_symlink() or not entry.is_dir()
                or {p.name for p in entry.iterdir()} != {"claim.json", "dispatch.json", "response.json"}):
            raise IntegrityError("Direct-model completion is partial or unknown")
        claim = relay.read_json(entry / "claim.json", relay.REQUEST_LIMIT + 4096)
        receipt = relay.read_json(entry / "dispatch.json", 16384)
        response = relay.read_json(entry / "response.json", relay.RESPONSE_LIMIT)
        identity = relay.validate_request(claim["request"], policy)
        relay.validate_response(response, claim["request"])
        if (entry.name != claim["request"]["request_id"] or claim["request_sha256"] != identity
                or receipt["request_sha256"] != identity or receipt.get("status") != "completed"
                or receipt.get("usage_status") != "known" or receipt.get("synthetic") is not False
                or receipt.get("model") != config.model or type(receipt.get("observed_tokens")) is not int
                or receipt["observed_tokens"] < 0 or receipt["returncode"] != response["process"]["code"]):
            raise IntegrityError("Direct-model provenance, completion or token usage is unknown")
        records.append(receipt)
    records.sort(key=lambda row: row["started_at"])
    tokens = 0
    for row in records:
        if tokens >= policy.token_stop_threshold:
            raise IntegrityError("A model call started after the observed-token stop threshold")
        tokens += row["observed_tokens"]
    if (not records or len(records) > policy.max_calls or not summary.get("usage_complete")
            or summary.get("unknown_calls") or summary["claimed_calls"] != len(records)
            or summary["completed_receipts"] != len(records) or summary["known_observed_tokens"] != tokens):
        raise IntegrityError("Direct model ledger does not reconcile with its accounting artifact")
    controller_calls, controller_tokens = store.db.execute(
        "SELECT count(*),coalesce(sum(tokens),0) FROM live_operations WHERE kind='model'").fetchone()
    if controller_calls < len(records) or controller_tokens > tokens:
        raise IntegrityError("Controller operations do not cover the actual model dispatch ledger")
    return {"calls": len(records), "observed_tokens": tokens, "unknown_calls": 0,
            "controller_attempts": controller_calls, "controller_accounted_tokens": controller_tokens,
            "known_tokens_not_in_controller_total": tokens - controller_tokens,
            "authoritative_token_source": "durable-local-model-ledger"}


def _composite_model_accounting(store, config):
    """Keep old receipts under the old source epoch, charge each dispatch once."""
    from .live_pilot import validate_inherited_model_accounting, _validated_dispatches
    inherited = validate_inherited_model_accounting(store, config)
    root = store.root / "model-ledger"
    _ledger_files(root)
    policy = relay.RelayPolicy.from_mapping(relay.read_json(root / "policy.json", 4096))
    if (policy.model != config.model or policy.allowed_models != (config.model,)
            or policy.source_sha256 != digest(canonical(config.implementation))
            or policy.relay_sha256 != config.implementation.get("studies/media_relay.py")
            or policy.expires_at != store.get("live_deadline") or policy.call_seconds != config.call_seconds
            or policy.max_calls != config.max_calls - inherited["calls"]
            or policy.token_stop_threshold != config.token_stop_threshold - inherited["tokens"]):
        raise IntegrityError("Current model policy differs from the child source or inherited remaining budget")
    items = []
    for entry in sorted((root / "claims").iterdir()):
        if (entry.is_symlink() or not entry.is_dir()
                or {p.name for p in entry.iterdir()} != {"claim.json", "dispatch.json", "response.json"}):
            raise IntegrityError("Current model completion is partial or unknown")
        claim = relay.read_json(entry / "claim.json", relay.REQUEST_LIMIT + 4096)
        if entry.name != claim.get("request", {}).get("request_id"):
            raise IntegrityError("Current model claim directory differs from its request identity")
        items.append((claim, relay.read_json(entry / "dispatch.json", 16384),
                      relay.read_json(entry / "response.json", relay.RESPONSE_LIMIT)))
    current = _validated_dispatches(items, policy)
    if set(current["request_ids"]) & set(inherited["request_ids"]):
        raise IntegrityError("A model dispatch identity was counted in both source epochs")
    calls, tokens = inherited["calls"] + current["calls"], inherited["tokens"] + current["tokens"]
    summary = store.json(store.get("live_pilot_model_accounting"))
    expected = {"source": "composite-direct-local-model-ledger", "ledger": str(root),
        "transport": "direct_local_no_queue", "policy": asdict(policy), "usage_complete": True,
        "unknown_calls": [], "claimed_calls": calls, "completed_receipts": calls,
        "known_observed_tokens": tokens, "inherited_model_accounting_ref": inherited["manifest_ref"],
        "inherited_claimed_calls": inherited["calls"], "inherited_known_observed_tokens": inherited["tokens"],
        "local_claimed_calls": current["calls"], "local_completed_receipts": current["calls"],
        "local_known_observed_tokens": current["tokens"], "global_max_calls": config.max_calls,
        "global_token_stop_threshold": config.token_stop_threshold}
    if canonical(summary) != canonical(expected) or not calls or calls > config.max_calls:
        raise IntegrityError("Composite model accounting omits, duplicates or changes source-epoch usage")
    controller_calls, controller_tokens = store.db.execute(
        "SELECT count(*),coalesce(sum(tokens),0) FROM live_operations WHERE kind='model'").fetchone()
    if controller_calls < calls or controller_tokens > tokens:
        raise IntegrityError("Composite dispatches do not reconcile with inherited/current controller operations")
    return {"calls": calls, "observed_tokens": tokens, "unknown_calls": 0,
            "inherited_calls": inherited["calls"], "inherited_observed_tokens": inherited["tokens"],
            "current_calls": current["calls"], "current_observed_tokens": current["tokens"],
            "parent_policy": inherited["parent_policy"], "current_policy": asdict(policy),
            "controller_attempts": controller_calls, "controller_accounted_tokens": controller_tokens,
            "known_tokens_not_in_controller_total": tokens - controller_tokens,
            "authoritative_token_source": "separate-epoch-validated-model-ledgers"}


def _audit_bindings(store, execution, task):
    records = execution.get("observations", [])
    expected = {row["key"]: row["output_ref"] for row in store.db.execute(
        "SELECT * FROM calls WHERE key LIKE 'development/%' OR key LIKE 'pilot-final/%'")
        if not row["key"].startswith("development/audit/selected/")}
    actual = {row["key"]: row for row in records}
    if not expected or len(actual) != len(records):
        raise IntegrityError("Independent scoring audit omitted or duplicated observations")
    for key, ref in expected.items():
        record = actual.get(key, {})
        observation = Observation.model_validate(store.json(ref))
        if (record.get("observation_ref") != ref or record.get("integrity") != "pass"
                or record.get("ledger") != "pass"
                or observation.status == "success" and record.get("scoring") != "pass"):
            raise IntegrityError("Independent scoring audit is not bound to the current observations")
    repeated = _measured(store, store.json(execution["reexecution_ref"]), "full", task.seeds)
    frozen = store.get("live_frozen") or {}
    selected = Observation.model_validate(frozen["best"]["result"])
    if (not repeated.id.startswith("audit/selected/") or repeated.protocol_hash != selected.protocol_hash
            or repeated.code_ref != selected.code_ref or len(repeated.values) != len(selected.values)
            or any(not math.isclose(a, b, rel_tol=0, abs_tol=task.quality_tolerance)
                   for a, b in zip(repeated.values, selected.values, strict=True))):
        raise IntegrityError("Development reproduction does not match the frozen selected snapshot")


def verify_minimal_live(store, *, audit_ref, resume_before_ref, resume_after_ref):
    """Assess one real live pilot; missing conditional behavior is never fabricated."""
    report = {"schema_version": "minimal-live-acceptance-v1", "workflow": "fail",
              "measured_improvement": {"status": "unverified"}, "checks": {}, "roles": {},
              "phase_outcomes": {},
              "scope": "One frozen bundled-task pilot; not MLAgentBench readiness or novelty certification",
              "scientific_validity": "unverified", "novelty": "unverified"}
    try:
        if errors := store.verify():
            raise IntegrityError("Registered evidence failed integrity verification: " + str(errors))
        config = LiveSpec.model_validate(store.json(store.get("live_spec_ref")))
        task = TaskSpec.model_validate(store.json(store.get("task_ref")))
        catalog = {"classification": CLASSIFICATION, "regression": REGRESSION, "optimization": OPTIMIZATION}
        if (config.model != "gpt-6-astra" or config.engine_version != "research-v2"
                or config.workflow.writer_backend != "paperorchestra"
                or task.adapter not in catalog or not set(task.datasets) <= set(catalog[task.adapter])
                or config.development.adapter != "bundled-pilot"
                or config.development.settings.get("task_ref") != store.get("task_ref")
                or config.native.task_id != task.name or config.native.metric != task.primary_metric
                or config.native.direction != task.metric_directions[task.primary_metric]
                or config.development.metric != task.primary_metric
                or config.development.direction != task.metric_directions[task.primary_metric]
                or {name: metric.direction for name, metric in config.development.metric_specs.items()} != task.metric_directions
                or config.development.split_seed != task.split_seed
                or config.development.subset_fraction != task.subset_fraction
                or config.development.epsilon != task.improvement_epsilon
                or len(task.seeds) != 3 or config.development.seeds != task.seeds
                or store.get("live_backend") != {"backend": "codex", "synthetic": False, "model": config.model}):
            raise IntegrityError("This acceptance contract requires real Astra roles, PaperOrchestra and an exact frozen bundled task protocol")
        report["task"] = {"task_ref": store.get("task_ref"), "name": task.name, "adapter": task.adapter,
            "datasets": task.datasets, "subset_datasets": task.subset_datasets, "seeds": task.seeds,
            "data_seed": task.data_seed, "split_seed": task.split_seed, "subset_fraction": task.subset_fraction,
            "primary_metric": task.primary_metric, "metric_directions": task.metric_directions,
            "guardrails": task.guardrails, "improvement_epsilon": task.improvement_epsilon}
        report["roles"] = _roles(store, config)
        report["model_accounting"] = _model_accounting(store, config)
        missing = sorted(MANDATORY_ROLES - set(report["roles"]))
        report["checks"]["required_roles"] = {"status": "pass" if not missing else "not_exercised", "missing": missing}
        state = store.get("live_state", {})
        report["checks"]["completed"] = {"status": "pass" if state.get("status") == "complete"
            and state.get("outcome") in {"development_improvement", "no_improvement"} else "fail", "state": state}

        before, after = store.json(resume_before_ref), store.json(resume_after_ref)
        current = capture_resume_snapshot(store)
        if before != after or after != current or after["state"].get("status") != "complete":
            raise IntegrityError("Completed resume changed work, source, deadline or recorded state")
        report["checks"]["completed_resume"] = {"status": "pass", "before_ref": resume_before_ref, "after_ref": resume_after_ref}

        audit = store.json(audit_ref)
        execution = audit.get("execution_audit", {})
        if (audit.get("status") != "pass" or audit.get("errors") or audit.get("live_model_calls") != 0
                or execution.get("errors") or execution.get("scoring") != "pass"
                or execution.get("reexecution") != "pass" or not execution.get("final_included")
                or execution.get("reexecution_measurement_origin") != "container"
                or store.json(execution["audit_ref"]) != {k: v for k, v in execution.items() if k != "audit_ref"}
                or audit.get("checks", {}).get("manuscripts", {}).get("status") != "pass"):
            raise IntegrityError("A complete source-bound scoring, reproduction and manuscript audit is required")
        _audit_bindings(store, execution, task)
        report["checks"]["independent_audit"] = {"status": "pass", "audit_ref": audit_ref}

        baselines = {}
        for phase, seeds in (("subset", task.seeds[:1]), ("full", task.seeds)):
            baseline = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("development/baseline/" + phase,)).fetchone()
            if not baseline:
                raise IntegrityError("Measured baseline screening and full-development evidence are required")
            baselines[phase] = _measured(store, store.json(baseline[0]), phase, seeds)
        report["checks"]["measured_baseline"] = {"status": "pass"}

        trials = [store.json(row[0]) for row in store.db.execute("SELECT output_ref FROM calls WHERE key LIKE 'live-trial/%'")]
        evolved = [trial for trial in trials if trial["id"].startswith("search/1/")
                   and trial.get("parents") and trial["result"].get("phase") == "full"
                   and trial["result"].get("status") == "success"]
        for trial in evolved:
            _measured(store, trial["result"], "full", task.seeds)
        screened = {trial["id"].split("/")[2] for trial in trials
                    if trial["id"].startswith("search/0/") and trial["result"].get("phase") == "subset"}
        report["checks"]["search_and_evolution"] = {"status": "pass" if len(screened) >= 2 and evolved else "not_exercised",
            "screened_seed_ideas": len(screened), "measured_full_evolved_trials": [trial["id"] for trial in evolved]}
        controls = []
        for row in store.db.execute("SELECT key,output_ref FROM calls WHERE key LIKE 'control/%/ablation/%'"):
            control = store.json(row["output_ref"])
            if control.get("review", {}).get("decision") not in {"Accept", "DiagnosticOnly"}:
                continue
            matched = _measured(store, control["matched"], "full", task.seeds)
            removed = _measured(store, control["control"], "full", task.seeds)
            if matched.protocol_hash != removed.protocol_hash or not control.get("source_changes"):
                raise IntegrityError("Ablation does not bind a matched protocol and source change")
            controls.append(row["key"])
        report["checks"]["matched_ablation"] = {"status": "pass" if controls else "not_exercised", "controls": controls}

        bundle = store.json(store.get("research_manuscript_bundle"))
        final_bundle = store.json(store.get("research_final_bundle"))
        for value in (bundle, final_bundle):
            if (value.get("compilation", {}).get("status") != "success" or not value.get("pdf_ref")
                    or not store.read(value["pdf_ref"]).startswith(b"%PDF-")
                    or not store.read(value["latex_ref"]).strip()):
                raise IntegrityError("Authoritative manuscript/final appendix has no compiled PDF")
            checked = [record for record in audit["checks"]["manuscripts"].get("records", [])
                       if record.get("bundle_ref") == value.get("manifest_ref")
                       and record.get("status") == "pass" and record.get("pdf") == "pass"]
            if not checked:
                raise IntegrityError("Manuscript audit is not bound to the current authoritative bundles")
        if (not any(record.get("authoritative_writer", {}).get("authoritative_source_binding") == "pass"
                    for record in audit["checks"]["manuscripts"]["records"]
                    if record.get("bundle_ref") == bundle["manifest_ref"])
                or not any(record.get("authoritative_writer", {}).get("paperorchestra_development_source_preserved") == "pass"
                           for record in audit["checks"]["manuscripts"]["records"]
                           if record.get("bundle_ref") == final_bundle["manifest_ref"])):
            raise IntegrityError("PaperOrchestra source preservation audit is missing")
        manuscript = store.json(bundle["manuscript_ref"])
        if manuscript.get("synthetic_agents") is not False:
            raise IntegrityError("Manuscript declares synthetic agents")
        report["checks"]["authoritative_manuscript"] = {"status": "pass", "bundle_ref": store.get("research_manuscript_bundle"),
                                                       "final_bundle_ref": store.get("research_final_bundle")}
        finals = store.get("live_final", {})
        frozen = store.get("live_frozen")
        if not frozen or not store.db.execute("SELECT 1 FROM events WHERE key='live/freeze'").fetchone():
            raise IntegrityError("Protected final evidence lacks its frozen candidate checkpoint")
        selected_development = _measured(store, frozen["best"]["result"], "full", task.seeds)
        report["phase_outcomes"]["development"] = _phase_outcome(
            store, task, selected_development, baselines["full"])
        development = report["phase_outcomes"]["development"]
        if not development["guardrails_pass"]:
            raise IntegrityError("Frozen development selection violates the task guardrails")
        if state.get("outcome") == "development_improvement" and development["status"] != "supported":
            raise IntegrityError("Frozen development evidence does not support the recorded improvement outcome")
        if state.get("outcome") == "no_improvement" and development["status"] == "supported":
            raise IntegrityError("Recorded no-improvement outcome conflicts with the frozen development selection")
        final_observations = {}
        for name, value in finals.items():
            if value.get("status") != "completed":
                raise IntegrityError("A protected final evaluation did not complete")
            observed = _measured(store, store.json(value["observation_ref"]), "final", task.seeds)
            final_observations[name] = observed
            operation = store.db.execute("SELECT result_ref FROM live_operations WHERE key=?", ("final/" + name,)).fetchone()
            if (not operation or canonical(store.json(operation[0])) != canonical(value)
                    or value.get("score") != observed.value or value.get("per_seed_values") != observed.values):
                raise IntegrityError("Final summary differs from its committed measured result")
            expected = frozen["baseline"] if name == "baseline" else frozen["best"]["candidate"]
            if canonical(store.json(observed.code_ref)) != canonical(expected):
                raise IntegrityError("Protected final source differs from the frozen selection")
        final_row = store.db.execute("SELECT min(rowid) FROM live_operations WHERE key LIKE 'final/%'").fetchone()[0]
        if final_row is None or store.db.execute("SELECT 1 FROM live_operations WHERE kind='model' AND rowid>?", (final_row,)).fetchone():
            raise IntegrityError("Final barrier missing or model work followed protected evaluation")
        report["checks"]["protected_finals"] = {"status": "pass" if {"baseline", "selected"} <= set(finals) else "not_exercised",
                                                "evaluated": sorted(finals)}
        if {"baseline", "selected"} <= set(finals):
            report["phase_outcomes"]["final"] = _phase_outcome(
                store, task, final_observations["selected"], final_observations["baseline"])
            report["measured_improvement"] = report["phase_outcomes"]["final"]
        report["conditional_branches"] = {
            "scientific_rejection": "observed" if any(t["decision"]["decision"] == "Bad" for t in trials) else "not_exercised",
            "engineering": "observed" if report["roles"].get("engineer") else "not_exercised",
            "rebuttal": "observed" if report["roles"].get("rebuttal_planner") else "not_exercised",
            "negative_outcome": "observed" if store.get("negative_research_outcome") else "not_exercised"}
        report["workflow"] = "pass" if all(item["status"] == "pass" for item in report["checks"].values()) else "partial"
    except (IntegrityError, ValueError, OSError, TypeError, KeyError) as error:
        report["error"] = str(error)
    return report
