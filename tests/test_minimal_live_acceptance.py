"""Synthetic evidence contracts only; no experiment, PDF engine or model runs.

The positive fixture deliberately constructs a complete receipt graph to test
the checker. Its acceptance is not evidence of live or scientific capability.
"""

from dataclasses import asdict
import json
from pathlib import Path
from statistics import fmean
import time

import pytest

from scientisttwo.live.adapters import implementation_identity
from scientisttwo.live.models import DevelopmentSpec, LiveSpec, MetricSpec, Observation
from scientisttwo.live.runtime import Budget
from scientisttwo.live.workspace_models import WorkspaceSpec
from scientisttwo.models import AgentResult, EvalSpec, ExperimentResult, MetricRow, TaskSpec
from scientisttwo.research.minimal_live_acceptance import (
    MANDATORY_ROLES, capture_resume_snapshot, verify_minimal_live,
)
from scientisttwo.research.policy import ResearchPolicy
from scientisttwo.store import IntegrityError, Store, canonical, digest
from scientisttwo.studies import media_relay as relay


@pytest.fixture
def evidence(tmp_path, request):
    fixture_options = getattr(request, "param", {})
    store = Store(tmp_path)
    classification = fixture_options.get("classification", False)
    task = (TaskSpec(name="fixture", datasets=fixture_options.get("datasets", ["moons"]),
                     subset_datasets=["moons"]) if classification else
            TaskSpec(name="fixture", adapter="regression", datasets=[fixture_options.get("regression_dataset", "quadratic")],
                     subset_datasets=[fixture_options.get("regression_dataset", "quadratic")],
                     primary_metric="mse", metric_directions={"mse": "minimize"}, guardrails={}))
    metric, direction = task.primary_metric, task.metric_directions[task.primary_metric]
    task_ref = store.put_json(task)
    base_score = .8 if classification else 6.0
    candidate_score = .9 if classification else 1.0
    native = WorkspaceSpec(name="fixture", task_id="fixture", public_dir=tmp_path / "public",
        private_dir=tmp_path / "private", candidate_dir=tmp_path / "candidate", evaluator_dir=tmp_path / "evaluator",
        command=["python", "/worker.py"], evaluator_command=["trusted-score"], metric=metric, direction=direction,
        image="sha256:" + "1" * 64, grader_image="sha256:" + "1" * 64, data_provenance="user_prepared")
    config = LiveSpec(native=native, development=DevelopmentSpec(adapter="bundled-pilot", metric=metric,
        direction=direction, seeds=task.seeds, split_seed=task.split_seed,
        settings={"task_ref": task_ref},
        metric_specs={name: MetricSpec(direction=value) for name, value in task.metric_directions.items()}),
        engine_version="research-v2", mode="research", model="gpt-6-astra", max_calls=100,
        token_stop_threshold=1_500_000, campaign_seconds=7200, call_seconds=600,
        workflow=ResearchPolicy(writer_backend="paperorchestra", paperorchestra_root=str(tmp_path / "upstream"),
                                compiler_image="sha256:" + "2" * 64), implementation=implementation_identity())
    Budget(store, config)
    store.set("live_spec_ref", store.put_json(config))
    store.set("task_ref", task_ref)
    store.set("live_deadline", time.time() + 7200)
    store.set("live_backend", {"backend": "codex", "synthetic": False, "model": config.model})
    ledger = relay.private_directory(store.root / "model-ledger")
    claims = relay.private_directory(ledger / "claims")
    policy = relay.RelayPolicy(run_id="test-evidence", expires_at=store.get("live_deadline"),
        model=config.model, allowed_models=(config.model,), source_sha256=digest(canonical(config.implementation)),
        relay_sha256=digest(Path(relay.__file__).read_bytes()), max_calls=100, token_stop_threshold=1_500_000,
        call_seconds=600)
    relay.publish_new(ledger / "policy.json", asdict(policy))

    def operation(key, kind, result, request=None, status="success", tokens=0):
        ref = store.put_json(result)
        with store.db:
            store.db.execute("INSERT INTO live_operations VALUES(?,?,?,?,?,?,?)",
                (key, kind, digest(canonical(request or {})), status, .1, tokens, ref))
        return ref

    for index, role in enumerate(sorted(MANDATORY_ROLES)):
        key, request = "fixture/" + role, {"role": role, "inputs": {"fixture": True}}
        result = AgentResult(payload={}, backend="codex", model=config.model, synthetic=False,
                             usage={"input_tokens": 1, "output_tokens": 1})
        store.set("request/" + key, store.put_json(request))
        operation("model/" + key + "/0", "model", result, request, tokens=2)
        store.cache("live-agent/" + key, request, result)
        request_id = f"{index:032x}"
        entry = relay.private_directory(claims / request_id)
        now = time.time()
        wire = {"schema_version": 2, "request_id": request_id, "policy_sha256": policy.identity,
            "run_id": policy.run_id, "model": config.model, "role": role, "created_at": now,
            "deadline": now + 300, "timeout": 300, "source_sha256": policy.source_sha256, "media": [],
            "prompt": json.dumps({"role": role, "prompt_version": "scientisttwo-v1",
                                  "instructions": relay.INSTRUCTIONS, "inputs": {"fixture": True}}),
            "schema": {"type": "object"}}
        request_hash = digest(canonical(wire))
        response = relay.response_envelope(wire, relay._result(0), "{}")
        relay.publish_new(entry / "claim.json", {"request": wire, "request_sha256": request_hash})
        relay.publish_new(entry / "dispatch.json", {"request_sha256": request_hash, "status": "completed",
            "usage_status": "known", "synthetic": False, "model": config.model, "observed_tokens": 2,
            "returncode": 0, "started_at": now})
        relay.publish_new(entry / "response.json", response)
    store.set("live_pilot_model_accounting", store.put_json({
        "ledger": str(ledger), "transport": "direct_local_no_queue", "policy": asdict(policy),
        "usage_complete": True, "unknown_calls": [], "claimed_calls": len(MANDATORY_ROLES),
        "completed_receipts": len(MANDATORY_ROLES), "known_observed_tokens": 2 * len(MANDATORY_ROLES)}))
    candidate = {"files": [{"path": "candidate.py", "content": "def build(seed): return None"}],
                 "command": ["python", "/worker.py"], "description": "Synthetic contract source"}

    def measurement(key, phase="full", score=None):
        score = candidate_score if score is None else score
        seeds = [0] if phase == "subset" else task.seeds
        spec = EvalSpec.for_phase(task, phase).model_copy(update={"split_manifest_hash": "a" * 64})
        # PilotRunner binds datasets inside its authoritative EvalSpec, without
        # requiring a duplicate top-level alias.
        protocol = {"seeds": seeds, "phase": phase, "evaluation": spec.model_dump(mode="json")}
        if "dataset_alias" in fixture_options:
            protocol["datasets"] = fixture_options["dataset_alias"]
        rows = []
        for dataset in spec.datasets:
            values = {metric: score}
            if classification:
                values["accuracy"] = .9 if key.startswith("baseline") else .93
                values["accuracy"] = fixture_options.get("accuracy", {}).get(key, {}).get(dataset, values["accuracy"])
            rows.extend(MetricRow(dataset=dataset, seed=seed, metrics=values,
                                  training_seconds=.1, inference_seconds=.01, count=10) for seed in seeds)
        series = {name: [fmean(row.metrics[name] for row in rows if row.seed == seed) for seed in seeds]
                  for name in task.metric_directions}
        aggregate = {name: fmean(values) for name, values in series.items()}
        original = ExperimentResult(id=key, idea_id="fixture", status="success", protocol_hash=digest(canonical(spec)),
            code_ref=store.put_json(candidate), spec_ref=store.put_json(spec), rows=rows,
            observations_ref=store.put_json({"fixture": "prediction receipt"}), measurement_origin="container")
        observed = Observation(id=key, phase=phase, status="success", protocol_hash=digest(canonical(protocol)),
            protocol_ref=store.put_json(protocol), code_ref=store.put_json(candidate), metric=metric, value=aggregate[metric],
            values=series[metric], metrics=aggregate, metric_values=series,
            metric_rows=[row.model_dump(mode="json") for row in rows],
            legacy_result_ref=store.put_json(original), prediction_refs=[original.observations_ref])
        output = observed.model_dump(mode="json")
        if phase == "final":
            output = {"status": "completed", "observation_ref": store.put_json(observed), "score": observed.value,
                      "per_seed_values": observed.values, "metric": metric}
        operation(("final/" if phase == "final" else "experiment/") + key, "experiment", output,
                  status="completed" if phase == "final" else "success")
        if phase != "final":
            store.cache("development/" + key, {}, observed)
        else:
            store.cache("pilot-final/" + key, {}, observed)
        return observed.model_dump(mode="json"), output

    measurement("baseline/subset", "subset", base_score)
    measurement("baseline/full", "full", base_score)
    for prefix in ("search/0/0", "search/0/1"):
        obs, _ = measurement(prefix + "/subset/0", "subset")
        store.cache("live-trial/" + obs["id"], {}, {"id": obs["id"], "result": obs,
                    "parents": [], "decision": {"decision": "Good"}})
    obs, _ = measurement("search/1/0/full/0")
    selected_observation = obs
    store.cache("live-trial/" + obs["id"], {}, {"id": obs["id"], "result": obs,
                "parents": ["search/0/0/subset/0"], "decision": {"decision": "Good"}})
    matched, _ = measurement("method/0/ablation/0/matched")
    control, _ = measurement("method/0/ablation/0/0/control", score=.8 if classification else 3.0)
    store.cache("control/method/0/ablation/0/0", {}, {"review": {"decision": "Accept"},
        "matched": matched, "control": control, "source_changes": {"changed": ["candidate.py"]}})
    bundle = {"manuscript_ref": store.put_json({"synthetic_agents": False}),
              "manifest_ref": store.put_json({"fixture_manifest": "development"}),
              "latex_ref": store.put(b"Synthetic TeX test bytes", "tex"),
              "pdf_ref": store.put(b"%PDF-Synthetic contract fixture", "pdf"),
              "compilation": {"status": "success"}}
    store.set("research_manuscript_bundle", store.put_json(bundle))
    final_bundle = {**bundle, "manifest_ref": store.put_json({"fixture_manifest": "final"})}
    store.set("research_final_bundle", store.put_json(final_bundle))
    store.set("live_frozen", {"baseline": candidate, "best": {"candidate": candidate, "result": selected_observation}})
    store.event("live/freeze", "freeze", "Synthetic fixture freeze")
    selected_final = fixture_options.get("selected_final", .92 if classification else .02)
    finals = {label: measurement(label, "final", score)[1] for label, score in (("baseline", base_score), ("selected", selected_final))}
    store.set("live_final", finals)
    store.set("live_state", {"status": "complete", "stage": "complete", "outcome": "development_improvement"})
    records = [{"key": row["key"], "observation_ref": row["output_ref"], "integrity": "pass", "ledger": "pass", "scoring": "pass"}
               for row in store.db.execute("SELECT * FROM calls WHERE key LIKE 'development/%' OR key LIKE 'pilot-final/%'")]
    repeated, _ = measurement("audit/selected/fixture")
    execution = {"errors": [], "scoring": "pass", "reexecution": "pass", "final_included": True,
                 "reexecution_measurement_origin": "container", "observations": records,
                 "reexecution_ref": store.put_json(repeated)}
    execution["audit_ref"] = store.put_json(execution)
    audit = {"status": "pass", "errors": [], "live_model_calls": 0, "execution_audit": execution,
             "checks": {"manuscripts": {"status": "pass", "records": [
                 {"bundle_ref": bundle["manifest_ref"], "status": "pass", "pdf": "pass",
                  "authoritative_writer": {"authoritative_source_binding": "pass"}},
                 {"bundle_ref": final_bundle["manifest_ref"], "status": "pass", "pdf": "pass",
                  "authoritative_writer": {"paperorchestra_development_source_preserved": "pass"}},
             ]}}}
    refs = {"audit_ref": store.put_json(audit)}
    refs["resume_before_ref"] = refs["resume_after_ref"] = store.put_json(capture_resume_snapshot(store))
    yield store, refs, measurement, operation
    store.close()


def test_complete_bound_fixture_accepts_workflow_separately_from_gain(evidence):
    store, refs, _, _ = evidence
    before = capture_resume_snapshot(store)
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "pass", result
    assert result["measured_improvement"]["status"] == "supported"
    assert result["conditional_branches"]["scientific_rejection"] == "not_exercised"
    assert result["scientific_validity"] == result["novelty"] == "unverified"
    assert capture_resume_snapshot(store) == before


@pytest.mark.parametrize("evidence", [{"classification": True}], indirect=True)
def test_production_protocol_dataset_nesting_requires_no_redundant_alias(evidence):
    store, refs, _, _ = evidence
    observation = store.json(store.db.execute(
        "SELECT output_ref FROM calls WHERE key='development/baseline/full'").fetchone()[0])
    protocol = store.json(observation["protocol_ref"])
    assert "datasets" not in protocol and protocol["evaluation"]["datasets"] == ["moons"]
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "pass", result


@pytest.mark.parametrize("evidence", [{"classification": True, "dataset_alias": ["moons"]}], indirect=True)
def test_optional_consistent_dataset_alias_is_accepted(evidence):
    store, refs, _, _ = evidence
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "pass", result


@pytest.mark.parametrize("alias", [["circles"], None])
def test_optional_inconsistent_dataset_alias_is_rejected(evidence, alias):
    from scientisttwo.research.minimal_live_acceptance import _measured

    store, _, _, _ = evidence
    original = store.json(store.db.execute(
        "SELECT output_ref FROM calls WHERE key='development/baseline/full'").fetchone()[0])
    protocol = store.json(original["protocol_ref"])
    protocol["datasets"] = alias
    altered = {**original, "protocol_ref": store.put_json(protocol), "protocol_hash": digest(canonical(protocol))}
    with pytest.raises(IntegrityError, match="frozen task evaluation contract"):
        _measured(store, altered, "full", [0, 1, 2])


def test_plain_success_boolean_cannot_replace_registered_audit(evidence):
    store, refs, _, _ = evidence
    refs["audit_ref"] = store.put_json({"status": "pass"})
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "fail" and "audit" in result["error"]


@pytest.mark.parametrize("change", ["unknown_usage", "partial_dispatch", "synthetic_agent", "changed_resume", "final_model"])
def test_incomplete_or_misrepresented_execution_never_passes(evidence, change):
    store, refs, _, operation = evidence
    if change in {"unknown_usage", "partial_dispatch"}:
        path = next((store.root / "model-ledger/claims").iterdir()) / "dispatch.json"
        if change == "partial_dispatch":
            path.unlink()
        else:
            receipt = relay.read_json(path, 16384)
            receipt.update(usage_status="unknown", observed_tokens=None)
            path.write_bytes(relay.canonical(receipt))
    elif change == "synthetic_agent":
        store.set("live_backend", {"backend": "fixture", "synthetic": True, "model": "scripted"})
    elif change == "changed_resume":
        operation("experiment/unexpected-resume", "experiment", {"fixture": True})
    else:
        operation("model/after-final", "model", {"fixture": True})
        refs["resume_before_ref"] = refs["resume_after_ref"] = store.put_json(capture_resume_snapshot(store))
    assert verify_minimal_live(store, **refs)["workflow"] == "fail"


def test_missing_evolution_is_partial_not_manufactured(evidence):
    store, refs, _, _ = evidence
    with store.db:
        store.db.execute("DELETE FROM calls WHERE key='live-trial/search/1/0/full/0'")
    refs["resume_before_ref"] = refs["resume_after_ref"] = store.put_json(capture_resume_snapshot(store))
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "partial"
    assert result["checks"]["search_and_evolution"]["status"] == "not_exercised"


def test_manuscript_pdf_missing_is_failure(evidence):
    store, refs, _, _ = evidence
    bundle = store.json(store.get("research_final_bundle"))
    bundle["pdf_ref"] = store.put(b"Not a PDF", "pdf")
    store.set("research_final_bundle", store.put_json(bundle))
    assert verify_minimal_live(store, **refs)["workflow"] == "fail"


def test_unsupported_improvement_claim_is_separate_from_workflow(evidence):
    store, refs, _, _ = evidence
    finals = store.get("live_final")
    finals["selected"]["score"] = finals["baseline"]["score"]
    store.set("live_final", finals)
    # A detached aggregate must not be accepted as a measured final comparison.
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "fail"


@pytest.mark.parametrize("evidence", [{"selected_final": 6.0}], indirect=True)
def test_valid_workflow_does_not_require_a_positive_final_result(evidence):
    store, refs, _, _ = evidence
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "pass", result
    assert result["measured_improvement"]["status"] == "not_demonstrated"


def test_missing_required_actual_role_remains_partial(evidence):
    store, refs, _, _ = evidence
    with store.db:
        store.db.execute("DELETE FROM calls WHERE key='live-agent/fixture/meta'")
    refs["resume_before_ref"] = refs["resume_after_ref"] = store.put_json(capture_resume_snapshot(store))
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "partial"
    assert result["checks"]["required_roles"]["missing"] == ["meta"]


def test_stale_audit_pass_cannot_cover_a_different_observation(evidence):
    store, refs, _, _ = evidence
    audit = store.json(refs["audit_ref"])
    audit["execution_audit"]["observations"][0]["observation_ref"] = store.put_json({"different": True})
    execution = audit["execution_audit"]
    execution["audit_ref"] = store.put_json({k: v for k, v in execution.items() if k != "audit_ref"})
    refs["audit_ref"] = store.put_json(audit)
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "fail" and "current observations" in result["error"]


def test_unbound_manuscript_pass_does_not_validate_current_pdf(evidence):
    store, refs, _, _ = evidence
    audit = store.json(refs["audit_ref"])
    audit["checks"]["manuscripts"]["records"] = []
    refs["audit_ref"] = store.put_json(audit)
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "fail" and "authoritative bundles" in result["error"]


@pytest.mark.parametrize("evidence", [{"classification": True}], indirect=True)
def test_moons_contract_reports_maximize_metrics_and_separate_development_final(evidence):
    store, refs, _, _ = evidence
    before = capture_resume_snapshot(store)
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "pass", result
    assert result["task"]["adapter"] == "classification" and result["task"]["datasets"] == ["moons"]
    assert result["task"]["guardrails"] == {"accuracy": .01}
    development, final = (result["phase_outcomes"][phase] for phase in ("development", "final"))
    assert development["phase"] == "full" and development["selected"] == .9
    assert final["phase"] == "final" and final["selected"] == .92
    assert final["direction"] == "maximize" and final["metric"] == "macro_f1"
    assert final["status"] == "supported" and final["guardrails_pass"]
    assert final["selected_per_seed"] == [.92] * 3
    assert final["per_dataset"]["moons"]["accuracy"]["selected"] == .93
    assert capture_resume_snapshot(store) == before


@pytest.mark.parametrize("evidence", [{"classification": True,
    "accuracy": {"selected": {"moons": .88}}}], indirect=True)
def test_final_macro_f1_gain_with_failed_accuracy_guardrail_is_not_supported(evidence):
    store, refs, _, _ = evidence
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "pass", result  # A valid held-out negative result is retained.
    assert result["phase_outcomes"]["development"]["status"] == "supported"
    final = result["measured_improvement"]
    assert final["numerically_improved"] and not final["guardrails_pass"]
    assert final["status"] == "not_demonstrated"
    assert final["trusted_comparison"]["improved"] is False


@pytest.mark.parametrize("evidence", [{"classification": True, "datasets": ["moons", "circles"],
    "accuracy": {"selected": {"moons": .88, "circles": .98}}}], indirect=True)
def test_guardrails_remain_per_dataset_even_when_global_accuracy_improves(evidence):
    store, refs, _, _ = evidence
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "pass", result
    final = result["measured_improvement"]
    assert fmean(row["accuracy"]["selected"] for row in final["per_dataset"].values()) > .9
    assert not final["guardrails_pass"] and final["status"] == "not_demonstrated"


@pytest.mark.parametrize("evidence", [{"classification": True,
    "accuracy": {"search/1/0/full/0": {"moons": .88}, "audit/selected/fixture": {"moons": .88}}}], indirect=True)
def test_development_selection_cannot_promote_a_guardrail_violation(evidence):
    store, refs, _, _ = evidence
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "fail" and "development selection violates" in result["error"]


@pytest.mark.parametrize("evidence", [{"classification": True, "selected_final": .79}], indirect=True)
def test_classification_final_regression_does_not_rewrite_the_supported_development_outcome(evidence):
    store, refs, _, _ = evidence
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "pass", result
    assert result["phase_outcomes"]["development"]["status"] == "supported"
    assert result["measured_improvement"]["status"] == "not_demonstrated"
    assert result["measured_improvement"]["improvement"] < 0


def test_negative_outcome_cannot_hide_a_numerically_promoted_frozen_candidate(evidence):
    store, refs, _, _ = evidence
    state = store.get("live_state")
    state["outcome"] = "no_improvement"
    store.set("live_state", state)
    refs["resume_before_ref"] = refs["resume_after_ref"] = store.put_json(capture_resume_snapshot(store))
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "fail" and "no-improvement outcome conflicts" in result["error"]


@pytest.mark.parametrize("evidence", [{"regression_dataset": "sine"}], indirect=True)
def test_other_declared_bundled_regression_family_is_not_hardcoded_to_quadratic(evidence):
    store, refs, _, _ = evidence
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "pass", result
    assert result["task"]["datasets"] == ["sine"]
    assert result["measured_improvement"]["direction"] == "minimize"


@pytest.mark.parametrize("evidence", [{"classification": True}], indirect=True)
@pytest.mark.parametrize("change", ["metric", "direction", "split", "subset", "epsilon", "adapter", "task", "task_ref", "native_task", "native_metric"])
def test_frozen_task_and_development_protocol_drift_cannot_receive_a_pass(evidence, change):
    store, refs, _, _ = evidence
    spec = store.json(store.get("live_spec_ref"))
    if change == "metric":
        spec["development"]["metric"] = "accuracy"
    elif change == "direction":
        spec["development"]["direction"] = "minimize"
        spec["development"]["metric_specs"]["macro_f1"]["direction"] = "minimize"
    elif change == "split":
        spec["development"]["split_seed"] += 1
    elif change == "subset":
        spec["development"]["subset_fraction"] = .25
    elif change == "epsilon":
        spec["development"]["epsilon"] = .2
    elif change == "adapter":
        spec["development"]["adapter"] = "vectorization"
    elif change == "task_ref":
        spec["development"]["settings"]["task_ref"] = store.put_json({"not": "task"})
    elif change == "native_task":
        spec["native"]["task_id"] = "different-task"
    elif change == "native_metric":
        spec["native"]["metric"] = "accuracy"
    else:
        task = store.json(store.get("task_ref"))
        task["datasets"] = task["subset_datasets"] = ["circles"]
        store.set("task_ref", store.put_json(task))
    store.set("live_spec_ref", store.put_json(spec))
    refs["resume_before_ref"] = refs["resume_after_ref"] = store.put_json(capture_resume_snapshot(store))
    result = verify_minimal_live(store, **refs)
    assert result["workflow"] == "fail", result
