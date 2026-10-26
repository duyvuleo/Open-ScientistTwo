"""Synthetic role/transport cases; measured diagnostic arithmetic is real."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scientisttwo.live.models import Candidate, DevelopmentSpec, SourceFile
from scientisttwo.live.supplementary import (Concern, SupplementaryDiagnostic, SupplementaryExecution,
    SupplementaryPlan, SupplementaryPlans, negotiate, summarize_coverage)
from scientisttwo.research.diagnostic_stages import DiagnosticStages
from scientisttwo.store import Store
from scientisttwo.testbed.adapter import PreparedDevelopment
from scientisttwo.testbed.preparation import prepare
from scientisttwo.testbed.validation import create_validation_spec
from test_research_diagnostics import observation


class Harness(DiagnosticStages):
    def __init__(self, store, *, kinds=("paired_interval",)):
        self.store = store
        self.adapter = SimpleNamespace(diagnostic_capabilities=lambda: list(kinds))
        self.config = SimpleNamespace(development=DevelopmentSpec(metric="accuracy", direction="maximize"))
        self.research_diagnostics = []


def plan(kind="paired_interval", *, seed_count=3, execution_kind="paired_control", parameters=None):
    return SupplementaryPlan(id="analysis", concern_ids=["concern"], purpose="Synthetic uncertainty concern",
        seed_count=seed_count, execution_kind=execution_kind, evaluation_split="development",
        diagnostics=[SupplementaryDiagnostic(kind=kind, metric="accuracy", parameters=parameters or {})],
        changed_file="train.py" if execution_kind == "paired_control" else None,
        candidate=Candidate(files=[SourceFile(path="train.py", content="# synthetic fixture")], command=["python", "train.py"], description="Synthetic control"))


def negotiation(value, supported=None):
    return negotiate(SupplementaryPlans(plans=[value], unaddressed_concerns=[]),
        [Concern(id="concern", review_key="r", text="Synthetic concern")], list(range(value.seed_count)), ["accuracy"],
        supported_diagnostics=supported)


def test_optional_capabilities_do_not_change_legacy_default_and_require_exact_uncertainty_support():
    assert negotiation(plan()).decisions[0].status == "unfulfilled"
    assert negotiation(plan(), ["metric", "paired_interval"]).decisions[0].status == "accepted"
    assert negotiation(plan(seed_count=2), ["paired_interval"]).decisions[0].status == "unfulfilled"
    assert negotiation(plan(execution_kind="repeat"), ["paired_interval"]).decisions[0].status == "unfulfilled"
    assert negotiation(plan(parameters={"resamples": 100}), ["paired_interval"]).decisions[0].status == "unfulfilled"


def test_real_paired_artifacts_required_for_coverage_and_terminal_hydration(tmp_path):
    store = Store(tmp_path / "campaign")
    try:
        baseline, _ = observation(store, "base")
        candidate, _ = observation(store, "control", (.7, .8, .9))
        controller = Harness(store)
        value = plan()
        complete, refs = controller.research_supplementary_diagnostics("rebuttal/0", value, baseline, candidate)
        assert complete == value.diagnostics and len(refs) == 1
        executed = SupplementaryExecution(plan_id=value.id, status="success", seeds=[0, 1, 2], evaluation_count=6,
            completed_diagnostics=complete, observation_ids=[baseline.id, candidate.id], diagnostic_refs=refs)
        record = negotiation(value, ["metric", "paired_interval"])
        assert summarize_coverage(record, [executed], store=store)[0].status == "executed"
        no_proof = executed.model_copy(update={"diagnostic_refs": []})
        assert summarize_coverage(record, [no_proof], store=store)[0].status != "executed"
        assert summarize_coverage(record, [executed])[0].status != "executed"
        saved = list(controller.research_diagnostics)
        controller.restore_research_diagnostics()
        assert controller.research_diagnostics == saved
        count = store.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        assert controller.research_supplementary_diagnostics("rebuttal/0", value, baseline, candidate) == (complete, refs)
        assert len(controller.research_diagnostics) == 1
        assert store.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == count
    finally:
        store.close()


def test_registry_membership_without_adapter_support_does_not_advertise_capability(tmp_path):
    store = Store(tmp_path / "campaign")
    try:
        controller = Harness(store, kinds=["classification", "input_consistency", "unknown"])
        assert controller.research_diagnostic_capabilities() == ["metric"]
    finally:
        store.close()


@pytest.mark.parametrize("task,kind", [("house-price", "regression_residuals"), ("feedback", "regression_residuals"), ("spaceship-titanic", "classification")])
def test_cpu_adapter_exposes_frozen_development_diagnostic_contract(tmp_path, task, kind):
    from test_testbed_preparation import raw_data
    source = raw_data(tmp_path, task)
    prepared_root = tmp_path / "prepared"
    prepare(task, source, prepared_root, provenance="synthetic_fixture")
    spec = create_validation_spec(prepared_root, image="synthetic:cpu", level="full")
    adapter = PreparedDevelopment(spec.native, spec.development)
    prepared = adapter.prepare("full", tmp_path / "stage")
    declared = adapter.diagnostics_contract(kind, "rebuttal", prepared=prepared)
    assert declared["contract"]["format"] == "cpu_frames"
    assert declared["contract"]["targets"] == adapter.recipe.targets
    assert Path(declared["pipeline"]["private_dir"]) == prepared_root / "phases/full/private"
    assert declared["pipeline"]["score"]["resources"]["gpus"] is None
    with pytest.raises(ValueError, match="Final labels"):
        adapter.diagnostics_contract(kind, "final", prepared=prepared)


def test_cifar_adapter_binds_diagnostics_to_owned_copied_phase_labels(tmp_path):
    from test_testbed_families import family_bundle
    source = family_bundle(tmp_path / "source")
    root = tmp_path / "prepared"
    prepare("cifar10", source, root, provenance="synthetic_fixture", source_format="bundle")
    spec = create_validation_spec(root, image="synthetic:torch", grader_image="synthetic:cpu", level="full", gpus="device=0")
    adapter = PreparedDevelopment(spec.native, spec.development)
    prepared = adapter.prepare("subset", tmp_path / "stage")
    declared = adapter.diagnostics_contract("classification", "subset", prepared=prepared)
    assert declared["contract"]["format"] == "class_scores_npz"
    assert declared["pipeline"] == adapter.evaluation(prepared).model_dump(mode="json")
    (Path(prepared["private"]) / "labels.npz").write_bytes(b"altered")
    with pytest.raises(ValueError, match="differ"):
        adapter.diagnostics_contract("classification", "subset", prepared=prepared)


def test_legacy_cpu_prediction_bundle_gets_new_isolated_diagnostic_without_label_export(tmp_path):
    import pandas as pd
    from test_testbed_preparation import raw_data
    from test_research_diagnostic_runner import SyntheticTransport
    from scientisttwo.artifacts import ArtifactSpec, freeze_bundle
    from scientisttwo.live.models import Observation
    from scientisttwo.live.pipeline import EvaluationPipeline
    from scientisttwo.research.diagnostic_runner import ScoringDiagnosticContract, run_scoring_diagnostic
    from scientisttwo.research.diagnostics import DiagnosticRequest
    from scientisttwo.store import canonical, digest
    source = raw_data(tmp_path, "house-price")
    root = tmp_path / "prepared"
    prepare("house-price", source, root, provenance="synthetic_fixture")
    spec = create_validation_spec(root, image="synthetic:cpu", level="full")
    adapter = PreparedDevelopment(spec.native, spec.development)
    prepared = adapter.prepare("full", tmp_path / "stage")
    store = Store(tmp_path / "campaign")
    try:
        store.set("prepared/full", store.put_json(prepared))
        predictions = tmp_path / "predicted"
        predictions.mkdir()
        # These explicitly synthetic fixture labels are already in legacy prepared metadata.
        frame = pd.DataFrame(prepared["labels"])
        frame["SalePrice"] += 2
        frame.to_csv(predictions / "submission.csv", index=False)
        bundle = freeze_bundle(store, predictions, [ArtifactSpec(path="submission.csv")])
        protocol = {"phase": "full", "seeds": [0, 1, 2], "adapter": adapter.identity(), "data": prepared["files"]}
        observed = Observation(id="legacy-cpu", phase="full", status="success", protocol_hash=digest(canonical(protocol)),
            protocol_ref=store.put_json(protocol), code_ref=store.put_json({"synthetic": True}), metric="mae", value=2., values=[2., 2., 2.],
            prediction_refs=[bundle]*3, bundle_refs=[bundle]*3)
        store.cache("development/legacy-cpu", {}, observed)
        ref = store.put_json(observed)
        request = DiagnosticRequest(id="residuals", kind="regression_residuals", expected_seeds=[0, 1, 2])
        declared = adapter.diagnostics_contract(request.kind, "full", prepared=prepared)
        runner = SyntheticTransport(store)
        runner.adapter = adapter
        result_ref = run_scoring_diagnostic(runner, request, ref, EvaluationPipeline.model_validate(declared["pipeline"]),
                                          ScoringDiagnosticContract.model_validate(declared["contract"]))
        result = store.json(result_ref)
        assert result["output"]["tables"]["per_seed"][0]["mean_target_rmse"] == 2.
        assert len(runner.launches) == 3
        assert "labels" not in result["output"]
        assert "SalePrice" not in result["output"]["arrays"]
    finally:
        store.close()
