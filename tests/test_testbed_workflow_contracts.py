import pytest

from scientisttwo.testbed.catalog import recipes
from scientisttwo.testbed import export
from scientisttwo.testbed.workflow_contracts import verified_workflow_contracts, register_workflow_contract


def test_workflow_requirements_follow_reusable_interfaces_not_task_count(tmp_path):
    families = {recipe.adapter_family for recipe in recipes()}
    assert families == {"cpu_frames", "checkpoint_pipeline", "paired_timing"}
    contracts = verified_workflow_contracts(tmp_path)
    assert set(contracts) == families
    assert all(value["status"] == "missing" for value in contracts.values())


def test_missing_real_workflow_evidence_cannot_register_green_contract(tmp_path):
    with pytest.raises(OSError):
        register_workflow_contract(tmp_path / "root", tmp_path / "missing")
    assert not (tmp_path / "root/workflow-contracts").exists()


def test_export_does_not_enable_an_unvalidated_task(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "verify_validation", lambda *args: {"task_id": "house-price", "actual_task_validated": False})
    with pytest.raises(ValueError, match="fully validated actual"):
        export.export_live("house-price", tmp_path, tmp_path / "live.json", model="gpt-6.1-sol",
                           max_calls=10, campaign_seconds=300)
    assert not (tmp_path / "live.json").exists()
