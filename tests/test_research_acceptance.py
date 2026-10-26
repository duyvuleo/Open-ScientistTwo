"""Acceptance arithmetic uses explicit synthetic receipts, never real-run claims."""

import json

import pytest
from pydantic import ValidationError

from scientisttwo.research.acceptance import (
    AcceptanceBudget,
    AcceptancePlan,
    AttemptRecord,
    EngineIdentity,
    EvidenceFile,
    REQUIRED_BRANCHES,
    REQUIRED_LIVE_ROLES,
    REQUIRED_ROLES,
    TASKS,
    TaskBinding,
    VerificationProof,
    assess_acceptance,
    estimate_acceptance,
    identity,
    prepare_acceptance_plan,
    validate_fresh_manifest,
)


@pytest.fixture
def evidence(tmp_path):
    # These are controlled contract traces. This test never claims that these
    # receipts establish real experimental execution outside the unit test.
    path = tmp_path / "synthetic-test-receipt.json"
    content = b'{"provenance":"synthetic unit-test trace"}'
    path.write_bytes(content)
    import hashlib
    return EvidenceFile(path=str(path), sha256=hashlib.sha256(content).hexdigest())


@pytest.fixture
def engine():
    return EngineIdentity(implementation_sha256="a" * 64)


@pytest.fixture
def bindings():
    return tuple(
        TaskBinding(
            family=family, task_id=task, prepared_manifest_sha256="b" * 64,
            protocol_sha256="c" * 64, runtime_sha256="d" * 64,
            data_provenance="bundled_real_data" if family == "cpu_frames" else "actual_data",
            datasets=("iris", "wine", "digits") if family == "cpu_frames" else (task,),
        ) for family, task in TASKS.items()
    )


@pytest.fixture
def proof(engine, evidence):
    return VerificationProof(
        engine=engine, status="pass", implemented_roles=REQUIRED_ROLES,
        controlled_roles=REQUIRED_ROLES, controlled_branches=REQUIRED_BRANCHES,
        evidence=(evidence,),
    )


@pytest.fixture
def plan(engine, bindings, proof):
    return prepare_acceptance_plan(engine_identity=engine, bindings=bindings,
                                   implementation_proof=proof)


def passing_records(plan, evidence, *, missing_live=()):
    records = []
    for slot in plan.slots:
        if slot.conditional or not slot.mandatory:
            continue
        binding = next(v for v in plan.bindings if v.family == slot.family)
        live = slot.scope.startswith("live")
        records.append(AttemptRecord(
            slot_id=slot.id, attempt_id=slot.id + "/attempt-1", plan_sha256=plan.sha256,
            engine=plan.engine, binding_sha256=identity(binding), scope=slot.scope, status="pass",
            model=slot.budget.model, data_provenance=binding.data_provenance,
            execution_origin="isolated_real", datasets=binding.datasets,
            model_calls=10 if live else 0, observed_tokens=1000 if live else 0,
            elapsed_seconds=30, natural_live_roles=tuple(
                v for v in REQUIRED_LIVE_ROLES if v not in missing_live
            ) if slot.scope == "live_full" else (),
            real_experiments="pass", manuscript_pdf="pass", final_evaluation="pass",
            independent_audit="pass", completed_resume="pass", runtime_probe="pass",
            workflow_complete=True, scientific_outcome="supported_candidate", evidence=(evidence,),
        ))
    return records


def change(records, slot_id, **changes):
    return [v.model_copy(update=changes) if v.slot_id == slot_id else v for v in records]


def test_plan_is_frozen_roundtrippable_and_has_schema(plan):
    assert AcceptancePlan.model_validate_json(plan.model_dump_json()) == plan
    assert AcceptancePlan.model_json_schema()["properties"]["schema_version"]["const"] == 1
    assert plan.sha256 == identity(plan)
    with pytest.raises(ValidationError):
        plan.engine = EngineIdentity(implementation_sha256="e" * 64)
    with pytest.raises(ValidationError):
        plan.slots[0].budget.max_calls = 10
    assert isinstance(plan.slots, tuple) and isinstance(plan.bindings, tuple)


def test_exact_bounded_defaults_without_execution_or_authorization(plan):
    slots = {v.id: v for v in plan.slots}
    full = slots["live-checkpoint_pipeline"]
    assert (full.budget.max_calls, full.budget.observed_token_stop, full.budget.seconds) == (120, 1_500_000, 14_400)
    for family in ("cpu_frames", "paired_timing"):
        budget = slots[f"live-{family}"].budget
        assert (budget.max_calls, budget.observed_token_stop, budget.seconds) == (20, 250_000, 3600)
    challenge = slots["live-conditional-challenge"]
    assert challenge.conditional and challenge.budget.max_calls == 20
    assert challenge.budget.observed_token_stop == 300_000 and challenge.budget.seconds == 5400
    assert all(v.budget.model == "gpt-6-astra" for v in plan.slots if v.scope.startswith("live"))
    assert not plan.automatic_execution
    assert plan.authorization == "separate_explicit_live_and_worker_configuration_required"
    assert not slots["compatibility-ogbn"].mandatory


def test_model_free_preparation_with_no_assets_is_blocked(engine):
    plan = prepare_acceptance_plan(engine_identity=engine)
    assert len(plan.bindings) == 3
    assert any("missing frozen runtime" in v for v in plan.blockers)
    report = assess_acceptance(plan, [])
    assert not report["release_ready"]
    assert all(v["status"] == "not_exercised" for v in report["gates"].values())
    assert report["accounting"]["model_calls"] == 0


def test_estimate_has_stage_arithmetic_reserves_and_uncertainty(plan):
    full = next(v for v in plan.slots if v.scope == "live_full")
    estimate = full.estimate
    assert estimate.estimated_calls == sum(v.calls for v in estimate.stage_calls)
    assert estimate.estimated_tokens == estimate.estimated_calls * 15000
    assert estimate.reserved_closing_calls == 24
    assert estimate.reserved_closing_tokens == 300000
    assert estimate.reserved_closing_seconds == 2880
    assert estimate.fits_limits
    assert any("not calibrated" in text for text in estimate.assumptions)


def test_paper_ceiling_is_not_mislabeled_as_affordable(engine, bindings, proof):
    plan = prepare_acceptance_plan(engine_identity=engine, bindings=bindings,
                                   implementation_proof=proof,
                                   profiles={"checkpoint_pipeline": "paper-methodology"})
    assert any("profile estimate exceeds" in v for v in plan.blockers)
    full = next(v for v in plan.slots if v.scope == "live_full")
    assert not full.estimate.fits_limits


def test_negative_stage_and_unbounded_live_budgets_are_rejected():
    with pytest.raises(ValidationError):
        AcceptanceBudget(max_calls=1, observed_token_stop=2, seconds=30)
    with pytest.raises(ValidationError):
        AcceptanceBudget(max_calls=0, observed_token_stop=0, seconds=30, model="gpt-6-astra")
    with pytest.raises(ValidationError):
        estimate_acceptance(AcceptanceBudget(max_calls=0, observed_token_stop=0, seconds=30),
                            stage_calls={"misleading": -1})


def test_complete_controlled_receipts_pass_only_declared_gate_scope(plan, evidence):
    report = assess_acceptance(plan, passing_records(plan, evidence))
    assert report["release_ready"], report["errors"]
    assert report["status"] == "pass" and not report["challenge_required"]
    assert report["gates"]["live-conditional-challenge"]["status"] == "not_exercised"
    assert report["all_13_readiness"] == "not_assessed_separate_release_gate"
    assert report["scientific_validity"] == "requires_independent_judgment"


@pytest.mark.parametrize("field,value", [
    ("execution_origin", "synthetic"), ("data_provenance", "synthetic"),
    ("independent_audit", "not_exercised"), ("real_experiments", "blocked"),
    ("runtime_probe", "fail"), ("completed_resume", "not_exercised"),
    ("final_evaluation", "fail"), ("workflow_complete", False),
    ("evidence", ()), ("engine", EngineIdentity(implementation_sha256="f" * 64)),
    ("plan_sha256", "f" * 64), ("binding_sha256", "f" * 64),
])
def test_missing_fake_or_stale_evidence_cannot_pass(plan, evidence, field, value):
    records = change(passing_records(plan, evidence), "scripted-checkpoint_pipeline", **{field: value})
    report = assess_acceptance(plan, records)
    assert not report["release_ready"]
    assert report["gates"]["scripted-checkpoint_pipeline"]["status"] == "fail"
    assert report["gates"]["live-checkpoint_pipeline"]["status"] == "blocked"


def test_full_cannot_be_replaced_with_probe(plan, evidence):
    records = change(passing_records(plan, evidence), "live-checkpoint_pipeline", scope="live_probe")
    report = assess_acceptance(plan, records)
    assert not report["release_ready"]
    assert any("scope" in v for v in report["attempts"][-1]["errors"])


def test_missing_one_real_bundled_dataset_blocks_cpu_gate(plan, evidence):
    records = change(passing_records(plan, evidence), "scripted-cpu_frames", datasets=("iris", "wine"))
    assert not assess_acceptance(plan, records)["release_ready"]


def test_synthetic_agents_are_not_live_model_coverage(plan, evidence):
    records = change(passing_records(plan, evidence), "scripted-cpu_frames",
                     model_calls=1, natural_live_roles=REQUIRED_LIVE_ROLES)
    report = assess_acceptance(plan, records)
    assert not report["release_ready"]
    assert any("Scripted validation" in item for item in report["attempts"][0]["errors"])


def test_live_response_overshoot_is_reported_not_falsely_a_hard_token_limit(plan, evidence):
    records = change(passing_records(plan, evidence), "live-checkpoint_pipeline", observed_tokens=1_500_001)
    report = assess_acceptance(plan, records)
    assert report["release_ready"]
    assert report["attempts"][-1]["token_threshold_exceeded"]


@pytest.mark.parametrize("field,value", [("model_calls", 121), ("elapsed_seconds", 14401),
                                        ("model", "different-model"), ("model_calls", 0)])
def test_scope_limits_and_model_are_enforced(plan, evidence, field, value):
    records = change(passing_records(plan, evidence), "live-checkpoint_pipeline", **{field: value})
    assert not assess_acceptance(plan, records)["release_ready"]


def test_no_improvement_is_allowed_but_not_a_shortcut_around_required_outputs(plan, evidence):
    records = change(passing_records(plan, evidence), "live-checkpoint_pipeline", scientific_outcome="no_improvement")
    assert assess_acceptance(plan, records)["release_ready"]
    records = change(records, "live-checkpoint_pipeline", manuscript_pdf="not_exercised")
    assert not assess_acceptance(plan, records)["release_ready"]


def test_unexercised_rebuttal_needs_real_preregistered_challenge(plan, evidence):
    records = passing_records(plan, evidence, missing_live=("rebuttal_planner",))
    report = assess_acceptance(plan, records)
    assert report["challenge_required"] and not report["release_ready"]
    full = records[-1]
    challenge = full.model_copy(update={
        "slot_id": "live-conditional-challenge", "attempt_id": "challenge/1", "scope": "live_challenge",
        "natural_live_roles": (), "challenge_live_roles": ("rebuttal_planner",),
    })
    report = assess_acceptance(plan, [*records, challenge])
    assert report["release_ready"], report["errors"]
    assert report["coverage"]["missing_natural_live_roles"] == ["rebuttal_planner"]
    assert report["coverage"]["challenge_live_roles"] == ["rebuttal_planner"]
    assert not report["coverage"]["missing_live_roles"]
    bad = challenge.model_copy(update={"execution_origin": "synthetic"})
    assert not assess_acceptance(plan, [*records, bad])["release_ready"]


def test_challenge_cannot_be_called_natural_live_work(plan, evidence):
    full = passing_records(plan, evidence)[-1]
    challenge = full.model_copy(update={"slot_id": "live-conditional-challenge", "attempt_id": "challenge/1",
                                        "scope": "live_challenge"})
    report = assess_acceptance(plan, [*passing_records(plan, evidence), challenge])
    assert report["gates"]["live-conditional-challenge"]["status"] == "fail"


def test_failed_attempts_stay_in_denominator_and_retries_need_authorization(plan, evidence):
    records = passing_records(plan, evidence)
    original = records[-1]
    failed = original.model_copy(update={"status": "fail", "reason": "GPU OOM", "model_calls": 8})
    retry = original.model_copy(update={"attempt_id": "retry/2", "supersedes_attempt_id": original.attempt_id})
    records[-1] = failed
    report = assess_acceptance(plan, [*records, retry])
    assert not report["release_ready"]
    authorized = retry.model_copy(update={"retry_authorization": evidence})
    report = assess_acceptance(plan, [*records, authorized])
    assert report["release_ready"], report["errors"]
    assert report["accounting"]["failed_attempts"] == 1
    assert report["accounting"]["attempts"] == 7
    assert report["accounting"]["model_calls"] == 38
    assert not report["automatic_retry"]


def test_unknown_completion_cannot_be_retried_even_with_authorization(plan, evidence):
    records = passing_records(plan, evidence)
    original = records[-1]
    records[-1] = original.model_copy(update={"status": "unknown"})
    retry = original.model_copy(update={"attempt_id": "retry/2", "supersedes_attempt_id": original.attempt_id,
                                       "retry_authorization": evidence})
    report = assess_acceptance(plan, [*records, retry])
    assert not report["release_ready"]
    assert any("completion is unresolved" in e for e in report["attempts"][-1]["errors"])


def test_duplicate_and_unscheduled_attempts_rejected(plan, evidence):
    records = passing_records(plan, evidence)
    with pytest.raises(ValueError, match="Duplicate"):
        assess_acceptance(plan, [*records, records[0]])
    with pytest.raises(ValueError, match="Unscheduled"):
        assess_acceptance(plan, [records[0].model_copy(update={"slot_id": "hidden-retry"})])


def test_modified_evidence_cannot_retain_green_proof(plan, evidence):
    records = passing_records(plan, evidence)
    from pathlib import Path
    Path(evidence.path).write_bytes(b"rewritten")
    report = assess_acceptance(plan, records)
    assert not report["release_ready"]
    assert any("Evidence modified" in e for e in report["errors"])


def test_old_engine_and_dataset_proofs_are_blocked(engine, proof, bindings):
    old = proof.model_copy(update={"engine": EngineIdentity(implementation_sha256="f" * 64)})
    plan = prepare_acceptance_plan(engine_identity=engine, bindings=bindings, implementation_proof=old)
    assert any("Stale proof" in v for v in plan.blockers)
    mismatched = proof.model_copy(update={"binding_sha256": "f" * 64})
    plan = prepare_acceptance_plan(engine_identity=engine, bindings=bindings, implementation_proof=proof,
                                   prerequisite_proofs=(mismatched,))
    assert any("does not bind a current task" in v for v in plan.blockers)


def test_current_manifest_change_requires_new_plan(plan, bindings):
    updated = [v.model_copy(update={"runtime_sha256": "f" * 64}) for v in bindings]
    assert validate_fresh_manifest(plan, bindings=updated)
    assert validate_fresh_manifest(plan, engine_identity={"new_source.py": "changed"})


def test_prepared_metadata_is_observed_without_running_or_claiming_readiness(tmp_path, engine):
    root = tmp_path / "registry" / "cifar10" / "prepared"
    root.mkdir(parents=True)
    value = {"task_id": "cifar10", "provenance": "actual_data", "arbitrary": [1, 2]}
    (root / "manifest.json").write_text(json.dumps(value))
    plan = prepare_acceptance_plan(engine_identity=engine, registry=tmp_path / "registry")
    binding = next(v for v in plan.bindings if v.task_id == "cifar10")
    assert binding.prepared_manifest_sha256 == identity(value)
    assert binding.runtime_sha256 is None and plan.blockers
    (root / "manifest.json").write_text(json.dumps({**value, "arbitrary": [3]}))
    assert any("manifest changed" in v for v in validate_fresh_manifest(plan))


def test_symlink_evidence_is_not_a_sealed_artifact(tmp_path, evidence):
    link = tmp_path / "link.json"
    link.symlink_to(evidence.path)
    ref = evidence.model_copy(update={"path": str(link)})
    assert "symlink" in ref.verify()


def test_complete_role_implementation_does_not_replace_controlled_branch_coverage(engine, bindings, proof, evidence):
    proof = proof.model_copy(update={"controlled_branches": ()})
    plan = prepare_acceptance_plan(engine_identity=engine, bindings=bindings, implementation_proof=proof)
    report = assess_acceptance(plan, passing_records(plan, evidence))
    assert not report["release_ready"]
    assert set(report["coverage"]["missing_controlled_branches"]) == set(REQUIRED_BRANCHES)


def test_compatibility_slot_does_not_replace_required_cifar_gate(plan, evidence):
    records = passing_records(plan, evidence)
    full = records.pop()
    records.append(full.model_copy(update={"slot_id": "compatibility-ogbn", "scope": "compatibility"}))
    report = assess_acceptance(plan, records)
    assert not report["release_ready"]
    assert report["gates"]["live-checkpoint_pipeline"]["status"] == "not_exercised"


@pytest.mark.parametrize("status", ["scheduled", "running", "skipped", "unknown", "blocked"])
def test_nonterminal_skipped_or_blocked_attempts_stay_visible(plan, evidence, status):
    records = change(passing_records(plan, evidence), "live-paired_timing", status=status)
    report = assess_acceptance(plan, records)
    assert not report["release_ready"]
    assert report["accounting"]["attempts"] == 6
    assert report["gates"]["live-paired_timing"]["status"] == "blocked"
    row = next(v for v in report["attempts"] if v["slot_id"] == "live-paired_timing")
    assert row["reported_status"] == status


def test_controlled_roles_cannot_fill_live_coverage(plan, evidence):
    records = passing_records(plan, evidence, missing_live=("rebuttal_planner",))
    records = change(records, "live-checkpoint_pipeline", controlled_roles=("rebuttal_planner",))
    report = assess_acceptance(plan, records)
    assert report["coverage"]["missing_live_roles"] == ["rebuttal_planner"]
    assert not report["release_ready"]


def test_malformed_metadata_produces_a_blocker_not_a_ready_skip(tmp_path, engine):
    (tmp_path / "manifest.json").write_text("[]")
    plan = prepare_acceptance_plan(engine_identity=engine, paths={"checkpoint_pipeline": tmp_path})
    assert any("JSON object" in v for v in plan.blockers)


def test_wrong_representative_and_unknown_family_are_errors(engine, bindings):
    wrong = bindings[0].model_copy(update={"task_id": "house-price"})
    with pytest.raises(ValueError, match="representative binding"):
        prepare_acceptance_plan(engine_identity=engine, bindings=(wrong,))
    with pytest.raises(ValueError, match="declared representative"):
        prepare_acceptance_plan(engine_identity=engine, profiles={"typo": "research-pilot"})
