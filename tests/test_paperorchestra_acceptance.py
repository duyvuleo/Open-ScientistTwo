"""Writer-aware immutable acceptance plans; receipts are synthetic contract data."""

import pytest
from pydantic import ValidationError

from scientisttwo.research.acceptance import (
    AcceptancePlan, REQUIRED_LIVE_ROLES, REQUIRED_ROLES, assess_acceptance,
    identity, prepare_acceptance_plan, required_role_sets, validate_fresh_manifest,
)

import test_research_acceptance as contracts

bindings, engine = contracts.bindings, contracts.engine
evidence, proof = contracts.evidence, contracts.proof
passing_records = contracts.passing_records


def test_paperorchestra_plan_uses_actual_controller_writer_roles(engine, bindings, proof):
    roles, live = required_role_sets("paperorchestra")
    plan = prepare_acceptance_plan(engine_identity=engine, bindings=bindings,
        implementation_proof=proof, writer_backend="paperorchestra")
    assert plan.writer_backend == "paperorchestra"
    assert plan.required_roles == roles and plan.required_live_roles == live
    assert {"paperorchestra_model", "paperorchestra_bindings"} <= set(roles)
    assert not {"initial_drafter", "draft_enhancer"} & set(roles)
    assert set(REQUIRED_ROLES) - {"initial_drafter", "draft_enhancer"} <= set(roles)
    assert plan.model_dump()["writer_backend"] == "paperorchestra"
    assert AcceptancePlan.model_validate_json(plan.model_dump_json()) == plan


def test_canonical_default_preserves_legacy_schema_and_hash(engine, bindings, proof):
    plan = prepare_acceptance_plan(engine_identity=engine, bindings=bindings,
                                   implementation_proof=proof)
    assert plan.writer_backend == "canonical"
    assert plan.required_roles == REQUIRED_ROLES
    assert plan.required_live_roles == REQUIRED_LIVE_ROLES
    legacy = plan.model_dump(mode="json")
    assert "writer_backend" not in legacy
    assert identity(legacy) == plan.sha256 == identity(plan)
    restored = AcceptancePlan.model_validate(legacy)
    assert restored == plan and restored.sha256 == identity(legacy)


@pytest.mark.parametrize("field", ["required_roles", "required_live_roles"])
def test_supplied_frozen_roles_cannot_weaken_writer_coverage(engine, bindings, proof, field):
    plan = prepare_acceptance_plan(engine_identity=engine, bindings=bindings,
        implementation_proof=proof, writer_backend="paperorchestra")
    value = plan.model_dump()
    value[field] = tuple(role for role in value[field] if role != "paperorchestra_model")
    with pytest.raises(ValidationError, match="selected writer backend"):
        AcceptancePlan.model_validate(value)
    bypassed = plan.model_copy(update={field: value[field]})
    assert any("selected writer backend" in error for error in validate_fresh_manifest(bypassed))
    with pytest.raises(ValidationError, match="selected writer backend"):
        assess_acceptance(bypassed, [])


def test_canonical_proof_and_role_receipts_cannot_establish_paperorchestra_acceptance(
    engine, bindings, proof, evidence,
):
    plan = prepare_acceptance_plan(engine_identity=engine, bindings=bindings,
        implementation_proof=proof, writer_backend="paperorchestra")
    report = assess_acceptance(plan, passing_records(plan, evidence))
    assert report["writer_backend"] == "paperorchestra"
    assert not report["release_ready"]
    for field in ("missing_implemented_roles", "missing_controlled_roles", "missing_live_roles"):
        assert set(report["coverage"][field]) == {"paperorchestra_model", "paperorchestra_bindings"}


def test_explicit_paperorchestra_proof_and_actual_role_records_fill_writer_gate(
    engine, bindings, proof, evidence,
):
    roles, live = required_role_sets("paperorchestra")
    writer_proof = proof.model_copy(update={"implemented_roles": roles, "controlled_roles": roles})
    plan = prepare_acceptance_plan(engine_identity=engine, bindings=bindings,
        implementation_proof=writer_proof, writer_backend="paperorchestra")
    records = [record.model_copy(update={"natural_live_roles": live})
               if record.scope == "live_full" else record
               for record in passing_records(plan, evidence)]
    report = assess_acceptance(plan, records)
    assert not report["coverage"]["missing_implemented_roles"]
    assert not report["coverage"]["missing_controlled_roles"]
    assert not report["coverage"]["missing_live_roles"]


def test_plan_auto_derives_roles_only_when_not_supplied(engine):
    plan = AcceptancePlan(engine=engine, bindings=(), slots=(), writer_backend="paperorchestra")
    assert plan.required_roles == required_role_sets("paperorchestra")[0]
    with pytest.raises(ValueError, match="Unknown acceptance writer backend"):
        prepare_acceptance_plan(engine_identity=engine, writer_backend="unknown")


def test_writer_costs_include_inner_calls_without_expanding_authorized_limits(engine):
    canonical = prepare_acceptance_plan(engine_identity=engine)
    paper = prepare_acceptance_plan(engine_identity=engine, writer_backend="paperorchestra")
    before = next(slot for slot in canonical.slots if slot.scope == "live_full")
    after = next(slot for slot in paper.slots if slot.scope == "live_full")
    assert before.budget == after.budget
    assert after.estimate.estimated_calls > before.estimate.estimated_calls
    assert not after.estimate.fits_limits
    assert any("exceeds frozen limits" in reason for reason in paper.blockers)
    assert any("PaperOrchestra structural estimate" in note for note in after.estimate.assumptions)
    assert all(slot.estimate.estimated_calls == 0 for slot in paper.slots if slot.budget.model is None)
