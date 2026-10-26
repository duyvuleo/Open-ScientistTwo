"""Source-epoch accounting contracts using an executable fake CLI; no live calls."""
from copy import deepcopy
import shutil
import sqlite3
import time
from types import SimpleNamespace

import pytest

from scientisttwo.backends import CodexBackend, BackendError
from scientisttwo.live.runtime import Budget
from scientisttwo.process import ProcessResult, run_process
from scientisttwo.research.controller import ResearchController
from scientisttwo.research import live_pilot as live
from scientisttwo.research.minimal_live_acceptance import _model_accounting, _roles, capture_resume_snapshot
from scientisttwo.store import Store, IntegrityError, canonical, digest
from scientisttwo.studies import media_relay as relay
import test_study_media_relay as relay_tests
from test_study_media_relay import Answer
from test_live_pilot import setup, options

fake_cli = relay_tests.fake_cli


def controller(store, config, direct):
    budget = Budget(store, config)
    backend = CodexBackend(store.root / "agent", config.model, allow_live=True, timeout=30, executor=direct)
    return ResearchController(store, config, SimpleNamespace(context=lambda: {"synthetic_fixture": True}),
                              backend, SimpleNamespace(budget=budget), budget=budget)


@pytest.fixture
def inherited(tmp_path, fake_cli):
    cli, counter = fake_cli
    parent, parent_config, _, _, _ = setup(tmp_path / "parent", options(max_calls=6, token_stop_threshold=120))
    parent.set("live_spec_ref", parent.put_json(parent_config))
    deadline = time.time() + 600
    parent.set("live_deadline", deadline)
    calls = []

    def execute(*args, **kwargs):
        raw = run_process(*args, **kwargs)
        calls.append(True)
        # A known-cost CLI failure still consumes the second dispatch budget.
        return raw if len(calls) == 1 else ProcessResult(1, raw.stdout, "", raw.seconds)

    direct = live.DirectPilotExecutor(parent.root / "model-ledger", parent_config, deadline,
        executable=str(cli), executor=execute, allow_live=True, accounting_store=parent)
    agent = controller(parent, parent_config, direct)
    agent.ask("old-success", "critic", {"synthetic": "old success"}, Answer)
    with pytest.raises(BackendError):
        agent.ask("old-failure", "critic", {"synthetic": "known failed cost"}, Answer)
    summary_ref = parent.put_json(direct.accounting())
    parent.set("live_pilot_model_accounting", summary_ref)
    parent_spec_ref = parent.get("live_spec_ref")
    snapshot = tmp_path / "parent-snapshot.sqlite"
    backup = sqlite3.connect(snapshot)
    parent.db.backup(backup)
    backup.close()
    child_root = tmp_path / "child"
    child_root.mkdir()
    shutil.copyfile(snapshot, child_root / "campaign.sqlite")
    shutil.copytree(parent.root / "artifacts", child_root / "artifacts")
    child = Store(child_root)
    config = parent_config.model_copy(update={"implementation": {**parent_config.implementation,
                                      "synthetic-child-source-epoch": "a" * 64}})
    child.set("live_spec_ref", child.put_json(config))
    inherited_claims = []
    for entry in sorted((parent.root / "model-ledger/claims").iterdir()):
        inherited_claims.append({key + "_ref": child.put_file(entry / (key + ".json"))
                                for key in ("claim", "dispatch", "response")})
    operations = [dict(row) for row in parent.db.execute("SELECT * FROM live_operations ORDER BY key")]
    manifest = {"schema_version": live.INHERITED_ACCOUNTING_VERSION, "parent_root": str(parent.root),
        "parent_spec_ref": parent_spec_ref,
        "parent_policy_ref": child.put_file(parent.root / "model-ledger/policy.json"),
        "claims": inherited_claims, "parent_deadline": deadline, "parent_summary_ref": summary_ref,
        "parent_operations_ref": child.put_json(operations), "parent_snapshot_ref": child.put_file(snapshot),
        "source_epoch": "parent", "includes_prior_epochs": False}
    child.set("inherited_model_accounting_ref", child.put_json(manifest))
    yield SimpleNamespace(parent=parent, store=child, config=config, parent_config=parent_config,
                          manifest=manifest, cli=cli, counter=counter, deadline=deadline)
    child.close()
    parent.close()


def open_child(value, *, executor=run_process):
    direct = live.DirectPilotExecutor(value.store.root / "model-ledger", value.config, value.deadline,
        executable=str(value.cli), executor=executor, allow_live=True, accounting_store=value.store)
    return direct, controller(value.store, value.config, direct)


def finish(value, direct):
    value.store.set("live_pilot_model_accounting", value.store.put_json(direct.accounting()))


def test_separate_source_epochs_charge_known_success_and_failure_once(inherited):
    item = inherited
    prior = live.validate_inherited_model_accounting(item.store, item.config)
    assert prior["calls"] == 2 and prior["tokens"] == 24
    direct, agent = open_child(item)
    assert direct.policy.max_calls == 4 and direct.policy.token_stop_threshold == 96
    assert direct.policy.expires_at == item.deadline
    assert direct.policy.source_sha256 != prior["parent_policy"]["source_sha256"]
    agent.ask("new-review", "critic", {"synthetic": "new epoch"}, Answer)
    finish(item, direct)
    checked = _model_accounting(item.store, item.config)
    assert checked["calls"] == 3 and checked["observed_tokens"] == 36
    assert checked["inherited_calls"] == 2 and checked["current_calls"] == 1
    assert checked["known_tokens_not_in_controller_total"] == 12
    assert _roles(item.store, item.config) == {"critic": 2}
    before = capture_resume_snapshot(item.store)
    agent.ask("new-review", "critic", {"synthetic": "new epoch"}, Answer)
    assert capture_resume_snapshot(item.store) == before
    assert item.manifest["parent_snapshot_ref"] in before["inherited_model_artifacts"]
    assert len(item.counter.read_text().splitlines()) == 3


@pytest.mark.parametrize("damage", ["missing_claim", "duplicate_claim", "summary_high", "summary_low",
    "unknown_usage", "wrong_parent_source", "changed_operation", "deadline", "nested_parent"])
def test_inherited_omission_forgery_unknown_and_budget_reset_fail_before_dispatch(inherited, damage):
    item = inherited
    manifest = deepcopy(item.manifest)
    if damage == "missing_claim":
        manifest["claims"].pop()
    elif damage == "duplicate_claim":
        manifest["claims"].append(manifest["claims"][0])
    elif damage.startswith("summary_"):
        value = item.store.json(manifest["parent_summary_ref"])
        value["known_observed_tokens"] += 1 if damage == "summary_high" else -1
        manifest["parent_summary_ref"] = item.store.put_json(value)
    elif damage == "unknown_usage":
        value = item.store.json(manifest["claims"][0]["dispatch_ref"])
        value.update(usage_status="unknown", observed_tokens=None)
        manifest["claims"][0]["dispatch_ref"] = item.store.put_json(value)
    elif damage == "wrong_parent_source":
        value = item.store.json(manifest["parent_policy_ref"])
        value["source_sha256"] = digest(canonical(item.config.implementation))
        manifest["parent_policy_ref"] = item.store.put_json(value)
    elif damage == "changed_operation":
        item.store.db.execute("UPDATE live_operations SET tokens=0 WHERE kind='model'")
        item.store.db.commit()
    elif damage == "deadline":
        manifest["parent_deadline"] += 1
    else:
        manifest["includes_prior_epochs"] = True
    item.store.set("inherited_model_accounting_ref", item.store.put_json(manifest))
    with pytest.raises((IntegrityError, relay.RelayError)):
        open_child(item)
    assert not (item.store.root / "model-ledger").exists()
    assert len(item.counter.read_text().splitlines()) == 2


def test_inherited_store_cannot_silently_use_a_fresh_unaccounted_executor(inherited):
    item = inherited
    with pytest.raises(IntegrityError, match="requires its verified"):
        live.DirectPilotExecutor(item.store.root / "model-ledger", item.config, item.deadline,
            executable=str(item.cli), allow_live=True)
    assert not (item.store.root / "model-ledger").exists()


def test_unknown_current_call_keeps_parent_cost_and_blocks_every_next_call(inherited):
    invoked = []
    def failed(*args, **kwargs):
        invoked.append(True)
        return ProcessResult(1, '{"type":"turn.failed"}\n', "", .1)
    direct, agent = open_child(inherited, executor=failed)
    with pytest.raises(BackendError):
        agent.ask("unknown", "critic", {}, Answer)
    summary = direct.accounting()
    assert summary["claimed_calls"] == 3 and summary["known_observed_tokens"] == 24
    assert summary["unknown_calls"] and not summary["usage_complete"]
    with pytest.raises(relay.UnknownDispatch):
        agent.ask("must-not-dispatch", "critic", {}, Answer)
    assert len(invoked) == 1


def test_imported_artifact_mutation_breaks_snapshot_and_completed_resume(inherited, monkeypatch):
    item = inherited
    direct, _ = open_child(item)
    finish(item, direct)
    before = capture_resume_snapshot(item.store)
    assert before["inherited_model_artifacts"]
    ref = item.manifest["claims"][0]["dispatch_ref"]
    (item.store.root / ref).write_text('{"modified":true}')
    with pytest.raises(IntegrityError):
        capture_resume_snapshot(item.store)


def test_child_cannot_reset_local_policy_to_the_full_global_allowance(inherited):
    item = inherited
    direct, agent = open_child(item)
    agent.ask("new-review", "critic", {}, Answer)
    finish(item, direct)
    policy_path = item.store.root / "model-ledger/policy.json"
    policy = relay.read_json(policy_path, 4096)
    policy["max_calls"] = item.config.max_calls
    policy_path.write_bytes(canonical(policy))
    with pytest.raises(IntegrityError, match="remaining budget"):
        _model_accounting(item.store, item.config)


def test_current_dispatch_cannot_reuse_an_inherited_request_identity(inherited):
    item = inherited
    direct, agent = open_child(item)
    agent.ask("new-review", "critic", {}, Answer)
    finish(item, direct)
    inherited_id = live.validate_inherited_model_accounting(item.store, item.config)["request_ids"][0]
    entry = next(direct.service.claims.iterdir())
    claim = relay.read_json(entry / "claim.json", relay.REQUEST_LIMIT + 4096)
    receipt = relay.read_json(entry / "dispatch.json", 16384)
    response = relay.read_json(entry / "response.json", relay.RESPONSE_LIMIT)
    claim["request"]["request_id"] = inherited_id
    claim["request_sha256"] = digest(canonical(claim["request"]))
    receipt["request_sha256"] = claim["request_sha256"]
    response = relay.response_envelope(claim["request"], response["process"], response["response"])
    for name, value in (("claim.json", claim), ("dispatch.json", receipt), ("response.json", response)):
        (entry / name).write_bytes(canonical(value))
    entry.rename(entry.with_name(inherited_id))
    with pytest.raises(IntegrityError, match="both source epochs"):
        _model_accounting(item.store, item.config)


def test_completed_child_resume_does_not_construct_backend_or_change_ledgers(inherited, monkeypatch):
    item = inherited
    direct, agent = open_child(item)
    agent.ask("new-review", "critic", {}, Answer)
    finish(item, direct)
    item.store.set("live_state", {"status": "complete", "stage": "complete", "outcome": "development_improvement"})
    before = capture_resume_snapshot(item.store)
    monkeypatch.setattr(live, "implementation_identity", lambda: item.config.implementation)
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed resume cannot construct a model executor")
    monkeypatch.setattr(live, "DirectPilotExecutor", forbidden)
    live._run_prepared(item.store, item.config, SimpleNamespace(), SimpleNamespace(),
                       allow_live=True, executable=str(item.cli), process_executor=forbidden)
    assert capture_resume_snapshot(item.store) == before
