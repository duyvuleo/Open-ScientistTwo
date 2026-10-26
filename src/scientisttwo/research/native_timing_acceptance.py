"""Read-only acceptance of one canonical Vectorization audited-v1 live campaign.

This is deliberately separate from the bundled TaskSpec pilot checker. Recorded
timing arithmetic and every saved output can be audited without rerunning timing;
that does not establish latency reproducibility or scientific novelty.
"""

import math
from pathlib import Path
from statistics import fmean

from ..artifacts import load_bundle
from ..live.models import LiveSpec, Observation
from ..live.paired_evaluation import _validate_report
from ..live.timing import PairedTimingReport
from ..store import IntegrityError, canonical, digest
from ..testbed.catalog import RECIPE_VERSION, UPSTREAM_REVISION, get_recipe
from .minimal_live_acceptance import (
    MANDATORY_ROLES, _model_accounting, _roles, capture_resume_snapshot,
)


def capture_native_resume_snapshot(store):
    """Use the shared source/ledger snapshot; it has no TaskSpec dependency."""
    return capture_resume_snapshot(store)


def _prepared_contract(store, config):
    """Resolve benchmark identity from the frozen recipe, not WorkspaceSpec labels.

    The prepared adapter intentionally uses WorkspaceSpec for execution. Its
    canonical benchmark provenance belongs to manifest.json and adapter identity.
    """
    from ..live.adapters import load_development
    from ..live.workspace_models import WorkspaceSpec
    from ..testbed.preparation import manifest_hash, verify_prepared

    root_value = config.development.settings.get("prepared_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise IntegrityError("Native acceptance requires the exact prepared task directory")
    root = Path(root_value).resolve()
    manifest = verify_prepared(root)
    recipe = get_recipe("vectorization")
    identity = store.get("live_adapter_identity", {})
    if (not isinstance(config.native, WorkspaceSpec) or config.development.adapter != "testbed"
            or manifest.get("task_id") != "vectorization" or config.native.task_id != manifest["task_id"]
            or manifest.get("upstream_revision") != UPSTREAM_REVISION
            or manifest.get("recipe_version") != RECIPE_VERSION or RECIPE_VERSION != "audited-v1"
            or manifest.get("variant") != "audited-v1" or manifest.get("provenance") != "actual_data"
            or manifest.get("recipe_sha256") != recipe.identity
            or manifest.get("reference_baseline") != recipe.reference_baseline
            or identity.get("recipe") != recipe.identity
            or identity.get("prepared_manifest") != manifest_hash(manifest)
            or config.native.data_provenance != "user_prepared"
            or (config.native.metric, config.native.direction) != (recipe.metric, recipe.direction)
            or (config.development.metric, config.development.direction) != (recipe.metric, recipe.direction)
            or config.development.seeds != recipe.seeds):
        raise IntegrityError("Canonical Vectorization recipe, actual prepared manifest or adapter binding differs")
    paths = {"public_dir": "phases/final/public", "candidate_dir": "reference", "evaluator_dir": "evaluator"}
    if any(Path(getattr(config.native, name)).resolve() != root / relative for name, relative in paths.items()):
        raise IntegrityError("Workspace execution paths differ from the pinned prepared task")
    if load_development(config.native, config.development).identity() != identity:
        raise IntegrityError("Prepared adapter source closure differs from the campaign identity")
    return {"benchmark": "mlagentbench", "task_id": manifest["task_id"], "variant": manifest["variant"],
        "execution_contract": config.native.benchmark, "upstream_revision": manifest["upstream_revision"],
        "prepared_manifest_sha256": manifest_hash(manifest), "recipe_sha256": manifest["recipe_sha256"],
        "original_starter_status": manifest.get("original_starter_status"),
        "reference_baseline": manifest["reference_baseline"], "adapter_identity": identity,
        "workload_generation": manifest.get("split_metadata", {})}


def _observation(store, raw, phase, seeds):
    """Bind a native measurement to its operation, protocol and timing samples."""
    observed = Observation.model_validate(raw)
    namespace = "final-paired/" if phase == "final" else "development/"
    row = store.db.execute("SELECT * FROM calls WHERE key=?", (namespace + observed.id,)).fetchone()
    if row is None or canonical(store.json(row["output_ref"])) != canonical(observed):
        raise IntegrityError("Native timing observation is not its committed result")
    from ..live.audit import _observation_binding
    _observation_binding(store, row, observed, final=phase == "final")
    protocol = store.json(observed.protocol_ref)
    if (observed.phase != phase or observed.status != "success"
            or observed.metric != "latency_ratio" or protocol.get("phase") != phase
            or protocol.get("seeds") != seeds or digest(canonical(protocol)) != observed.protocol_hash
            or protocol.get("paired_timing", {}).get("warmups") != 5
            or protocol.get("paired_timing", {}).get("pairs") != 10
            or not len(observed.timing_refs) == len(observed.values) == len(seeds)):
        raise IntegrityError("Native timing result differs from its frozen phase/seed protocol")
    details = []
    for index, (seed, ref) in enumerate(zip(seeds, observed.timing_refs, strict=True)):
        evidence = store.json(ref)
        report = PairedTimingReport.model_validate(store.json(evidence["report_ref"]))
        _validate_report(report)
        if (evidence.get("seed") != seed or evidence.get("candidate_source_tree_ref") != observed.source_tree_ref
                or report.device != "cpu" or report.gpu_synchronization != "not_applicable"
                or report.baseline.median_seconds <= 0):
            raise IntegrityError("Native timing evidence source, seed or device differs")
        raw_ratio = report.candidate.median_seconds / report.baseline.median_seconds
        ratio = 1.0 if evidence.get("baseline_identity") else raw_ratio
        if not (math.isclose(ratio, observed.values[index], rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(ratio, evidence["value"], rel_tol=1e-12, abs_tol=1e-12)):
            raise IntegrityError("Native latency ratio differs from sealed medians")
        details.append({"seed": seed, "timing_evidence_ref": ref, "report_ref": evidence["report_ref"],
            "latency_ratio": ratio, "raw_median_latency_ratio": raw_ratio,
            "baseline_identity": evidence["baseline_identity"],
            "baseline_median_seconds": report.baseline.median_seconds,
            "baseline_iqr_seconds": report.baseline.iqr_seconds,
            "candidate_median_seconds": report.candidate.median_seconds,
            "candidate_iqr_seconds": report.candidate.iqr_seconds,
            "baseline_measurements_seconds": report.baseline.seconds,
            "candidate_measurements_seconds": report.candidate.seconds,
            "wall_seconds": report.wall_seconds, "sample_count": len(report.samples),
            "measurement_scope": report.measurement_scope})
    if (not math.isclose(observed.value, fmean(observed.values), rel_tol=1e-12, abs_tol=1e-12)
            or observed.metrics != {"latency_ratio": observed.value}
            or observed.metric_values != {"latency_ratio": observed.values}):
        raise IntegrityError("Native timing aggregate differs from its per-seed ratios")
    return observed, {"observation_ref": row["output_ref"], "protocol_hash": observed.protocol_hash,
        "phase": phase, "latency_ratio": observed.value, "per_seed": details}


def _audit_bindings(store, execution):
    """Require a separate native oracle-recomputation receipt for every success."""
    verification = execution.get("verification", {})
    if (execution.get("status") != "pass" or execution.get("errors")
            or execution.get("live_model_calls") != 0 or verification.get("complete") is not True):
        raise IntegrityError("A complete native correctness and metric audit is required")
    expected = {}
    for row in store.db.execute("SELECT key,output_ref FROM calls WHERE key LIKE 'development/%' OR key LIKE 'final-paired/%'"):
        observed = Observation.model_validate(store.json(row["output_ref"]))
        if observed.status == "success":
            identifier = row["key"].replace("final-paired/", "final/", 1)
            expected[identifier] = (row["output_ref"], observed)
    proofs = verification.get("paired_correctness_recomputations", [])
    if not expected or len(proofs) != len(expected) or {item.get("id") for item in proofs} != set(expected):
        raise IntegrityError("Native audit omits, duplicates or adds successful observations")
    for item in proofs:
        ref, observed = expected[item["id"]]
        proof = store.json(item["evidence_ref"])
        detail = proof.get("detail", {})
        if (proof.get("verification_kind") != "paired_output_recomputation"
                or proof.get("id") != item["id"] or proof.get("status") != "pass"
                or proof.get("observation_ref") != ref or proof.get("phase") != observed.phase
                or proof.get("metrics") != observed.metrics or item.get("metrics") != observed.metrics
                or any(detail.get(name) != "pass" for name in ("predictions", "paired_timing", "metric"))):
            raise IntegrityError("Native oracle audit is not bound to the measured observation")
    return {"status": "pass", "recomputed_observations": len(expected),
        "scope": "Independent oracle replay on every saved warm-up/measured output and timing arithmetic; no timing rerun"}


def _frozen_source(store, observed, candidate):
    """Compare full-file Vectorization candidates without materializing or executing."""
    if candidate.get("base_snapshot_ref"):
        raise IntegrityError("This Vectorization acceptance contract expects full-file candidates")
    tree = store.json(observed.source_tree_ref)
    bundle = load_bundle(store, tree["bundle_ref"])
    files = {item["path"]: item["content"].encode() for item in candidate["files"]}
    if (tree["command"] != candidate["command"] or set(bundle.files) != set(files)
            or any(bundle.files[path].sha256 != digest(content) for path, content in files.items())):
        raise IntegrityError("Protected final code differs from its frozen candidate")


def _finals(store, config, frozen):
    from .timing_evidence import verify_final_result_binding
    finals, details = store.get("live_final", {}), {}
    expected = {"baseline"} if frozen["best"]["id"] == "baseline" else {"baseline", "selected"}
    rows = list(store.db.execute("SELECT key FROM calls WHERE key LIKE 'final-paired/%'"))
    operations = list(store.db.execute("SELECT key FROM live_operations WHERE key LIKE 'final/%'"))
    if (set(finals) != expected or {row["key"] for row in rows} != {"final-paired/" + key for key in expected}
            or {row["key"] for row in operations} != {"final/" + key for key in expected}):
        raise IntegrityError("Protected final evaluations are missing or duplicated")
    for name, value in finals.items():
        if verify_final_result_binding(store, name, value)["kind"] != "paired_observation_summary":
            raise IntegrityError("Native paired final must bind its committed timing observation")
        observed, detail = _observation(store, store.json(value["observation_ref"]), "final", [config.native.final_seed])
        if (observed.id != name or value.get("status") != "completed"
                or value.get("metric") != observed.metric or value.get("score") != observed.value
                or value.get("metrics") != observed.metrics or value.get("protocol_hash") != observed.protocol_hash
                or value.get("evaluation_track") != "audited-v1" or value.get("live_model_calls") != 0
                or value.get("leaderboard_eligible") is not False):
            raise IntegrityError("Protected final summary differs from its committed measurement")
        _frozen_source(store, observed, frozen["baseline"] if name == "baseline" else frozen["best"]["candidate"])
        details[name] = detail
    final_row = store.db.execute("SELECT min(rowid) FROM live_operations WHERE key LIKE 'final/%'").fetchone()[0]
    if final_row is None or store.db.execute("SELECT 1 FROM live_operations WHERE kind='model' AND rowid>?", (final_row,)).fetchone():
        raise IntegrityError("Model invocation followed protected final evaluation")
    if not store.db.execute("SELECT 1 FROM events WHERE key='live/freeze'").fetchone():
        raise IntegrityError("Protected final measurements lack a freeze transition")
    return {"status": "pass", "final_timing_seeds": [config.native.final_seed], "arms": details,
        "scope": "One final evaluation per frozen arm; each evaluation contains five warm-ups and ten paired measurements"}


def _manuscripts(store, audit):
    records = audit.get("checks", {}).get("manuscripts", {})
    if records.get("status") != "pass":
        raise IntegrityError("Manuscript audit did not pass")
    bundles = {}
    for key in ("research_manuscript_bundle", "research_final_bundle"):
        ref = store.get(key)
        bundle = store.json(ref)
        matches = [row for row in records.get("records", []) if row.get("bundle_ref") == bundle.get("manifest_ref")]
        if (bundle.get("compilation", {}).get("status") != "success" or not bundle.get("pdf_ref")
                or not store.read(bundle["pdf_ref"]).startswith(b"%PDF-") or not store.read(bundle["latex_ref"]).strip()
                or len(matches) != 1 or matches[0].get("status") != "pass" or matches[0].get("pdf") != "pass"):
            raise IntegrityError("Authoritative manuscript or final appendix lacks audited PDF evidence")
        binding = "authoritative_source_binding" if key == "research_manuscript_bundle" else "paperorchestra_development_source_preserved"
        if matches[0].get("authoritative_writer", {}).get(binding) != "pass":
            raise IntegrityError("PaperOrchestra source preservation audit is missing")
        bundles[key] = ref
    manuscript = store.json(store.json(bundles["research_manuscript_bundle"])["manuscript_ref"])
    if manuscript.get("synthetic_agents") is not False:
        raise IntegrityError("The manuscript does not identify actual live agents")
    return {"status": "pass", **bundles}


def verify_native_timing_live(store, *, audit_ref, resume_before_ref, resume_after_ref):
    """Verify native evidence only; never dispatch models, containers or experiments."""
    report = {"schema_version": "native-timing-live-acceptance-v1", "workflow": "fail", "checks": {},
        "scope": "One MLAgentBench Vectorization audited-v1 campaign; not published-score reproduction or full-suite readiness",
        "latency_reproducibility": "unverified; no fresh timing rerun or tolerance is claimed",
        "method_code_consistency": "unverified; numerical output checks do not exclude caching or other benchmark shortcuts",
        "scientific_validity": "unverified", "novelty": "unverified"}
    try:
        if errors := store.verify():
            raise IntegrityError("Registered evidence failed integrity verification: " + str(errors))
        config = LiveSpec.model_validate(store.json(store.get("live_spec_ref")))
        prepared_contract = _prepared_contract(store, config)
        if (config.engine_version != "research-v2" or config.workflow.writer_backend != "paperorchestra"
                or config.mode != "research" or config.development.adapter != "testbed"
                or store.get("live_backend") != {"backend": "codex", "synthetic": False, "model": config.model}):
            raise IntegrityError("Expected canonical Vectorization audited-v1 native protocol and actual live agents")
        report["task"] = {**prepared_contract, "model": config.model, "development_seeds": config.development.seeds,
            "metric": "latency_ratio", "direction": "minimize"}
        report["roles"] = _roles(store, config)
        report["model_accounting"] = _model_accounting(store, config)
        operations = [dict(row) for row in store.db.execute("SELECT * FROM live_operations")]
        report["resource_use"] = {"experiment_operations": sum(row["kind"] == "experiment" for row in operations),
            "experiment_seconds": sum(row["seconds"] for row in operations if row["kind"] == "experiment"),
            "experiment_limit": config.max_experiments, "model_call_limit": config.max_calls,
            "observed_token_stop_threshold": config.token_stop_threshold,
            "operation_unit": "One phase evaluation containing its declared seeds; each timing seed runs two resident arms"}
        report["execution_failures"] = [{"key": row["key"], "status": row["status"], "result_ref": row["result_ref"]}
            for row in operations if row["kind"] == "experiment" and row["status"] != "success"]
        missing = sorted(MANDATORY_ROLES - set(report["roles"]))
        report["checks"]["actual_research_roles"] = {"status": "pass" if not missing else "not_exercised", "missing": missing}
        state = store.get("live_state", {})
        report["checks"]["completed"] = {"status": "pass" if state.get("status") == "complete"
            and state.get("outcome") in {"development_improvement", "no_improvement"} else "fail", "state": state}
        before, after = store.json(resume_before_ref), store.json(resume_after_ref)
        if before != after or after != capture_native_resume_snapshot(store) or after["state"].get("status") != "complete":
            raise IntegrityError("Completed resume changed work, source, deadline or state")
        report["checks"]["completed_resume"] = {"status": "pass", "before_ref": resume_before_ref, "after_ref": resume_after_ref}
        audit = store.json(audit_ref)
        if audit.get("status") != "pass" or audit.get("errors") or audit.get("live_model_calls") != 0:
            raise IntegrityError("A complete independent research audit is required")
        report["checks"]["independent_audit"] = {**_audit_bindings(store, audit.get("execution_audit", {})), "audit_ref": audit_ref}
        baselines = {}
        for phase, seeds in (("subset", config.development.seeds[:1]), ("full", config.development.seeds)):
            row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("development/baseline/" + phase,)).fetchone()
            if row is None:
                raise IntegrityError("Screening and full-development baseline measurements are required")
            _, baselines[phase] = _observation(store, store.json(row[0]), phase, seeds)
        report["checks"]["measured_baseline"] = {"status": "pass", "phases": baselines}
        trials = [store.json(row[0]) for row in store.db.execute("SELECT output_ref FROM calls WHERE key LIKE 'live-trial/%'")]
        screened = {trial["id"].split("/")[2] for trial in trials if trial["id"].startswith("search/0/")
            and trial["result"].get("phase") == "subset"}
        evolved = []
        for trial in trials:
            if (trial["id"].startswith("search/1/") and trial.get("parents") and trial["result"].get("phase") == "full"
                    and trial["result"].get("status") == "success"):
                _observation(store, trial["result"], "full", config.development.seeds)
                evolved.append(trial["id"])
        report["checks"]["search_and_evolution"] = {"status": "pass" if len(screened) >= 2 and evolved else "not_exercised",
            "screened_seed_ideas": len(screened), "measured_evolved_trials": evolved}
        controls = []
        for row in store.db.execute("SELECT key,output_ref FROM calls WHERE key LIKE 'control/%/ablation/%'"):
            control = store.json(row["output_ref"])
            if control.get("review", {}).get("decision") not in {"Accept", "DiagnosticOnly"}:
                continue
            matched, _ = _observation(store, control["matched"], "full", config.development.seeds)
            removed, _ = _observation(store, control["control"], "full", config.development.seeds)
            if matched.protocol_hash != removed.protocol_hash or not control.get("source_changes"):
                raise IntegrityError("Ablation lacks a matched protocol and declared source change")
            controls.append(row["key"])
        report["checks"]["matched_ablation"] = {"status": "pass" if controls else "not_exercised", "controls": controls}
        report["checks"]["authoritative_manuscript"] = _manuscripts(store, audit)
        frozen = store.get("live_frozen")
        if not frozen:
            raise IntegrityError("Frozen final selection is missing")
        selected, selected_detail = _observation(store, frozen["best"]["result"], "full", config.development.seeds)
        if selected.protocol_hash != baselines["full"]["protocol_hash"]:
            raise IntegrityError("Selected development result is incomparable to the baseline")
        improvement = baselines["full"]["latency_ratio"] - selected.value > config.development.epsilon
        if (state.get("outcome") == "development_improvement") != improvement or bool(frozen["improved"]) != improvement:
            raise IntegrityError("Recorded development outcome conflicts with frozen timing evidence")
        report["selected_development"] = selected_detail
        report["checks"]["protected_finals"] = _finals(store, config, frozen)
        report["conditional_branches"] = {
            "scientific_rejection": "observed" if any(t["decision"]["decision"] == "Bad" for t in trials) else "not_exercised",
            "engineering": "observed" if report["roles"].get("engineer") else "not_exercised",
            "rebuttal_experiments": "observed" if report["roles"].get("rebuttal_coder") else "not_exercised"}
        report["workflow"] = "pass" if all(item["status"] == "pass" for item in report["checks"].values()) else "partial"
    except (IntegrityError, ValueError, OSError, TypeError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report
