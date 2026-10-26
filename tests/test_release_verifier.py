"""Release verification must accept upstream logs without masking bad results."""

import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "verify_release", Path(__file__).resolve().parents[1] / "scripts" / "verify_release.py"
)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


@pytest.mark.parametrize("prefix", ["", "Writing paper...\n", '{"progress": 1}\nReviewing...\n'])
def test_cli_reads_final_complete_object_after_progress(prefix):
    expected = {"status": "complete", "nested": {"message": 'Literal { braces } and "quotes"'}}
    assert verifier.parse_trailing_json_object(prefix + json.dumps(expected, indent=2) + "\n \t") == expected


@pytest.mark.parametrize("output", [
    "", "Progress only\n", '{"status":', "[]", "null", '[{"status": "complete"}]',
    '{"status": "complete"}\nUnexpected trailing output',
    '{"status": "complete"}\n{"broken":',
])
def test_cli_rejects_missing_incomplete_or_nonterminal_json(output):
    with pytest.raises(ValueError, match="does not end with a complete JSON object"):
        verifier.parse_trailing_json_object(output)


def test_cli_preserves_raw_logs_and_writes_only_terminal_object(tmp_path, monkeypatch):
    stdout = 'PaperOrchestra stage 1\n{"status": "complete"}\n'
    monkeypatch.setattr(verifier.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=stdout, stderr="diagnostic\n"))
    assert verifier.cli(["run"], cwd=tmp_path, output=tmp_path / "run") == {"status": "complete"}
    assert (tmp_path / "run.stdout.txt").read_text() == stdout
    assert (tmp_path / "run.stderr.txt").read_text() == "diagnostic\n"
    assert json.loads((tmp_path / "run.json").read_text()) == {"status": "complete"}


def test_cli_keeps_failed_parse_logs_without_creating_result(tmp_path, monkeypatch):
    stdout = '{"status": "complete"}\ntrailing garbage'
    monkeypatch.setattr(verifier.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=stdout, stderr=""))
    with pytest.raises(ValueError, match="complete JSON object"):
        verifier.cli(["run"], cwd=tmp_path, output=tmp_path / "run")
    assert (tmp_path / "run.stdout.txt").read_text() == stdout
    assert not (tmp_path / "run.json").exists()


def test_completed_campaign_helper_never_launches_initial_run(tmp_path, monkeypatch):
    work = tmp_path / "work"
    campaign = work / "runs" / "completed-campaign"
    campaign.mkdir(parents=True)
    task = tmp_path / "task.json"
    task.write_text('{"name": "synthetic-verifier-test"}')
    run = {"status": "complete", "outcome": "no_improvement", "directory": str(campaign)}
    inspected = {"engine_version": "research-v2", "backend": {"synthetic": True},
                 "live_calls": False, "live_model_calls": 0}
    audit = {"status": "pass", "live_model_calls": 0, "execution_audit": {
        "reexecution": "pass", "reexecution_measurement_origin": "container", "scoring": "pass",
        "observations": [{"measurement_origin": "container"}],
    }}
    commands = []

    def cli(arguments, **kwargs):
        commands.append(arguments[0])
        return {"inspect": inspected, "audit": audit, "resume": run}[arguments[0]]

    monkeypatch.setattr(verifier, "cli", cli)
    monkeypatch.setattr(verifier, "campaign_snapshot", lambda root: {"stable": True})
    args = SimpleNamespace(task=task, engine="docker", writer_backend="canonical",
                           compiler_image=None, max_prompt_chars=200_000)
    result = verifier.verify_completed_campaign(args, work, run)
    assert result["status"] == "pass"
    assert commands == ["inspect", "audit", "resume", "inspect"]
    assert json.loads((work / "run.json").read_text()) == run


@pytest.fixture
def snapshot_campaign(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    with sqlite3.connect(root / "campaign.sqlite") as db:
        db.executescript('''
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata VALUES('a', 'one');
            INSERT INTO metadata VALUES('b', 'two');
            CREATE TABLE live_operations(key TEXT PRIMARY KEY, kind TEXT NOT NULL);
            INSERT INTO live_operations VALUES('development/first', 'experiment');
            INSERT INTO live_operations VALUES('final/baseline', 'experiment');
        ''')
    (root / "artifact.txt").write_text("immutable fixture evidence")
    return root


def test_snapshot_ignores_same_value_metadata_replace_and_row_reordering(snapshot_campaign):
    root = snapshot_campaign
    before = verifier.campaign_snapshot(root)
    with sqlite3.connect(root / "campaign.sqlite") as db:
        db.execute("INSERT OR REPLACE INTO metadata VALUES('a', 'one')")
    assert verifier.campaign_snapshot(root) == before
    with sqlite3.connect(root / "campaign.sqlite") as db:
        rows = db.execute("SELECT * FROM metadata ORDER BY key DESC").fetchall()
        db.execute("DELETE FROM metadata")
        db.executemany("INSERT INTO metadata VALUES(?,?)", rows)
    assert verifier.campaign_snapshot(root) == before


@pytest.mark.parametrize("mutation", ["value", "operation", "operation_order", "schema", "artifact"])
def test_snapshot_detects_meaningful_ledger_or_evidence_changes(snapshot_campaign, mutation):
    root = snapshot_campaign
    before = verifier.campaign_snapshot(root)
    if mutation == "artifact":
        (root / "artifact.txt").write_text("modified evidence")
    else:
        with sqlite3.connect(root / "campaign.sqlite") as db:
            if mutation == "value":
                db.execute("UPDATE metadata SET value='changed' WHERE key='a'")
            elif mutation == "operation":
                db.execute("INSERT INTO live_operations VALUES('new/model', 'model')")
            elif mutation == "operation_order":
                db.execute("INSERT OR REPLACE INTO live_operations VALUES('development/first', 'experiment')")
            else:
                db.execute("CREATE TABLE unexpected(key TEXT)")
    assert verifier.campaign_snapshot(root) != before
