import json

from typer.testing import CliRunner

from scientisttwo.audit import audit_campaign
from scientisttwo.cli import app
from scientisttwo.reporting import evidence_audit
from scientisttwo.runner import PrerequisiteError
from test_controller import campaign


def test_reexecution_audit_and_corrupted_metric_detection(tmp_path):
    controller, runner, store, adapter = campaign(tmp_path)
    controller.run()
    result = audit_campaign(store, adapter, runner)
    assert result["reexecution"] == "pass"
    assert result["reference_existence"] == "unverified"
    job = store.db.execute(
        "SELECT key,result_ref FROM jobs WHERE key LIKE 'search/%/full/%' LIMIT 1"
    ).fetchone()
    forged = store.json(job["result_ref"])
    forged["rows"][0]["metrics"]["loss"] = 999
    ref = store.put_json(forged)
    with store.db:
        store.db.execute("UPDATE jobs SET result_ref=? WHERE key=?", (ref, job["key"]))
    assert evidence_audit(store, adapter)["score_recalculation"] == "fail"


def test_negative_campaign_audits_baseline_without_inventing_winner(tmp_path):
    controller, runner, store, adapter = campaign(tmp_path, reject=True)
    controller.run()
    result = audit_campaign(store, adapter, runner)
    assert result["reexecution"] == "pass"
    assert result["reexecution_target"] == "baseline_only_no_selected_candidate"
    assert store.get("best") is None


def test_screening_audit_preserves_negative_campaign_and_checks_rejected_candidate(tmp_path):
    controller, runner, store, adapter = campaign(tmp_path, reject=True)
    controller.run()
    previous = store.get("state")
    result = audit_campaign(store, adapter, runner, screening_sample_size=1)
    assert len(result["screening_checks"]) == 1
    assert result["screening_checks"][0]["comparison"]["improved"]
    assert store.get("state") == previous
    assert store.get("best") is None


def test_doctor_unavailable_runtime_is_explicit(monkeypatch):
    def unavailable(self):
        raise PrerequisiteError("daemon unavailable")

    monkeypatch.setattr("scientisttwo.cli.ContainerRunner.preflight", unavailable)
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["container"] == "unavailable"


def test_cli_blocked_campaign_and_inspect(tmp_path, monkeypatch):
    def unavailable(self):
        raise PrerequisiteError("daemon unavailable")

    monkeypatch.setattr("scientisttwo.cli.ContainerRunner.preflight", unavailable)
    cli = CliRunner()
    result = cli.invoke(app, ["run", "pilot", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    inspected = cli.invoke(app, ["inspect", payload["directory"]])
    assert inspected.exit_code == 0
    assert json.loads(inspected.stdout)["jobs"] == []
    resumed = cli.invoke(app, ["resume", payload["directory"]])
    assert resumed.exit_code == 2


def test_plugin_registration(monkeypatch):
    from scientisttwo.tasks import load_task_adapter
    from helpers import ScalarTask

    class Entry:
        def load(self):
            return ScalarTask

    monkeypatch.setattr("scientisttwo.tasks.entry_points", lambda **kwargs: [Entry()])
    assert load_task_adapter("a-new-research-domain").name == "scalar-test"
