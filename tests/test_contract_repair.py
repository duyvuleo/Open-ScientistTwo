"""Synthetic repair-fork records only; no model, experiment or container calls."""

from dataclasses import asdict
from pathlib import Path
import time

import pytest

from scientisttwo.live.models import LiveSpec
from scientisttwo.live.runtime import Budget
from scientisttwo.models import Profile, TaskSpec
from scientisttwo.research import contract_repair as repair
from scientisttwo.research.live_pilot import LivePilotOptions, validate_inherited_model_accounting
from scientisttwo.research.pilot_bridge import create_pilot_campaign
from scientisttwo.research.policy import ResearchPolicy
from scientisttwo.store import IntegrityError, Store, canonical, digest
from scientisttwo.studies import media_relay as relay


@pytest.fixture
def parent(tmp_path, monkeypatch, request):
    root = tmp_path / "parent"
    store = Store(root)
    options = LivePilotOptions(model="gpt-6-astra", max_calls=100, token_stop_threshold=2_000_000,
        campaign_seconds=7200, call_seconds=600,
        workflow=ResearchPolicy(writer_backend="paperorchestra", paperorchestra_root=str(tmp_path / "upstream"),
                                compiler_image="sha256:" + "2" * 64))
    task = TaskSpec(name="fixture", datasets=["moons"], subset_datasets=["moons"])
    config, adapter, _ = create_pilot_campaign(store, task, Profile(), image="sha256:" + "1" * 64, live_options=options)
    current = repair.implementation_identity()
    monkeypatch.setattr(repair, "implementation_identity", lambda: current)
    package = Path(repair.__file__).parents[1]
    old_source = (package / "research/reviews.py").read_bytes() + b"\n# Synthetic previous source epoch\n"
    old_identity = {**current, "research/reviews.py": digest(old_source)}
    old_identity.pop("research/contract_repair.py", None)
    config = LiveSpec.model_validate({**config.model_dump(mode="json"), "implementation": old_identity})
    spec_ref = store.put_json(config)
    store.set("live_spec_ref", spec_ref)
    store.set("pilot_live_spec_ref", spec_ref)
    source_index = {"research/reviews.py": store.put(old_source, "py")}
    store.set("live_sources_ref", store.put_json(source_index))
    store.set("adapter_sources_ref", store.put_json({name: store.put_file(path) for name, path in adapter.source_files().items()}))
    store.set("live_adapter_identity", adapter.identity())
    store.set("live_backend", {"backend": "codex", "synthetic": False, "model": config.model})
    store.set("engine_version", "research-v2")
    store.set("research_manuscript", store.put_json({"explicit_synthetic_test": True}))
    store.set("research_manuscript_bundle", store.put_json({"explicit_synthetic_test": True, "compiled": True}))
    store.set("live_state", {"status": "blocked", "stage": "review", "outcome": "invalid_agent_output",
                             "reason": "Synthetic review contract mismatch"})
    store.set("live_deadline", time.time() + 3600)
    task_source_text = getattr(request, "param", {}).get("task_source_text", "{'settings': {'b': 2, 'a': 1}, 'enabled': True}")
    request = {"role": "limitations", "inputs": {"sources": [{"id": "task", "text": task_source_text}]}}
    response_ref = store.put_json({"explicit_synthetic_test": True})
    with store.db:
        store.db.execute("INSERT INTO live_operations VALUES(?,?,?,?,?,?,?)",
            ("model/original-review/0", "model", digest(canonical(request)), "success", 1., 7, response_ref))
        store.db.execute("INSERT INTO live_operations VALUES(?,?,?,?,?,?,?)",
            ("experiment/baseline/full", "experiment", digest(canonical(request)), "success", 1., 0, response_ref))
        store.db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)",
            ("pilot/experiment/baseline/full", digest(canonical(request)), "success", time.time(), 60., 1., response_ref))
    store.set("request/limitations/0", store.put_json(request))
    store.cache("live-agent/limitations/0", request, {"explicit_synthetic_test": True})
    store.event("live/method/0/review/0", "review", "Synthetic already-committed transition")
    ledger = relay.private_directory(root / "model-ledger")
    claims = relay.private_directory(ledger / "claims")
    policy = relay.RelayPolicy(run_id="synthetic-parent", expires_at=store.get("live_deadline"), model=config.model,
        allowed_models=(config.model,), source_sha256=digest(canonical(old_identity)),
        relay_sha256=old_identity["studies/media_relay.py"], max_calls=config.max_calls,
        token_stop_threshold=config.token_stop_threshold, call_seconds=config.call_seconds)
    relay.publish_new(ledger / "policy.json", asdict(policy))
    now = time.time()
    wire = {"schema_version": 2, "request_id": "a" * 32, "policy_sha256": policy.identity,
        "run_id": policy.run_id, "model": config.model, "role": "limitations", "created_at": now,
        "deadline": now + 300, "timeout": 300, "source_sha256": policy.source_sha256, "media": [],
        "prompt": canonical({"role": "limitations", "prompt_version": "scientisttwo-v1",
                             "instructions": relay.INSTRUCTIONS, "inputs": request["inputs"]}).decode(),
        "schema": {"type": "object"}}
    item = relay.private_directory(claims / wire["request_id"])
    identity = digest(canonical(wire))
    relay.publish_new(item / "claim.json", {"request": wire, "request_sha256": identity})
    relay.publish_new(item / "dispatch.json", {"request_sha256": identity, "status": "completed", "usage_status": "known",
        "synthetic": False, "model": config.model, "observed_tokens": 7, "returncode": 0, "started_at": now})
    event = canonical({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}).decode() + "\n"
    relay.publish_new(item / "response.json", relay.response_envelope(wire, {**relay._result(0), "stdout": event}, "{}"))
    store.set("live_pilot_model_accounting", store.put_json({"ledger": str(ledger), "transport": "direct_local_no_queue",
        "policy": asdict(policy), "claimed_calls": 1, "completed_receipts": 1,
        "known_observed_tokens": 7, "usage_complete": True, "unknown_calls": []}))
    store.close()
    (root / ".campaign.lock").touch()
    return root


def fork(parent, tmp_path):
    child = tmp_path / "child"
    ref = repair.fork_review_contract(parent, child, repair_id="score-scale-v1",
                                     authorization_ref="Synthetic fixture existing authorization")
    return Store(child), ref


def test_fork_preserves_parent_epoch_deadline_rows_and_artifacts(parent, tmp_path):
    before = repair._tree(parent)
    store, ref = fork(parent, tmp_path)
    try:
        manifest = store.json(ref)
        assert repair._tree(parent) == before
        assert manifest["source_changes"].keys() == {"research/reviews.py", "research/contract_repair.py"}
        assert store.get("live_state")["stage"] == "review" and store.get("live_state")["status"] == "running"
        assert not (store.root / "model-ledger").exists()
        assert store.get("live_deadline") == manifest["original_deadline"]
        assert store.json(store.get("live_spec_ref"))["implementation"] != manifest["parent_implementation"]
        config = LiveSpec.model_validate(store.json(store.get("live_spec_ref")))
        inherited = validate_inherited_model_accounting(store, config)
        assert inherited["calls"] == 1 and inherited["tokens"] == 7
        original = next(iter(manifest["historical_rows"]["artifacts"]))
        assert (store.root / original).stat().st_ino != (parent / original).stat().st_ino
        repair.verify_review_contract(store)
    finally:
        store.close()


def test_only_review_roles_are_namespaced_and_no_new_work_can_precede_the_gate(parent, tmp_path):
    store, _ = fork(parent, tmp_path)
    try:
        config = LiveSpec.model_validate(store.json(store.get("live_spec_ref")))
        budget = repair.ContractRepairBudget(Budget(store, config))
        old = store.db.execute("SELECT request_hash FROM live_operations WHERE key='experiment/baseline/full'").fetchone()
        assert old is not None
        saved = store.json(store.get("request/limitations/0"))
        assert budget.reserve("experiment/baseline/full", "experiment", saved)["status"] == "success"
        for kind in ("model", "experiment", "probe"):
            with pytest.raises(IntegrityError, match="New operation"):
                budget.reserve("new/" + kind, kind, {})
        with pytest.raises(IntegrityError, match="New model dispatch"):
            repair.require_repair_dispatch_gate(store)
        assert repair.repair_invocation_key(store, "writer", "paperorchestra_model") == "writer"
        with pytest.raises(IntegrityError, match="must precede"):
            repair.repair_invocation_key(store, "rebuttal", "rebuttal_planner")
        key = repair.repair_invocation_key(store, "method/0/review/0", "scientific_review")
        assert key == "method/0/review/0/contract-repair/score-scale-v1"
        gate = store.json(store.get("contract_repair_replay_gate_ref"))
        assert gate["added_model_calls"] == gate["added_experiments"] == 0
        repair.require_repair_dispatch_gate(store)
        assert repair.repair_invocation_key(store, "rebuttal", "rebuttal_planner").endswith("/contract-repair/score-scale-v1")
        assert budget.reserve(key, "model", {}) is None
    finally:
        store.close()


@pytest.mark.parametrize("key", ["task_ref", "profile_ref", "evaluator", "adapter_identity", "live_deadline"])
def test_protected_metadata_remains_frozen_after_review_gate(parent, tmp_path, key):
    store, _ = fork(parent, tmp_path)
    try:
        repair.repair_invocation_key(store, "review", "scientific_review")
        store.set(key, "changed")
        with pytest.raises(IntegrityError):
            repair.require_repair_dispatch_gate(store)
    finally:
        store.close()


def test_added_or_modified_historical_work_blocks_replay_gate(parent, tmp_path):
    store, _ = fork(parent, tmp_path)
    try:
        with store.db:
            store.db.execute("UPDATE live_operations SET tokens=8 WHERE kind='model'")
        with pytest.raises(IntegrityError, match="Inherited evidence row"):
            repair.repair_invocation_key(store, "review", "scientific_review")
    finally:
        store.close()


def test_dry_replay_stops_before_reservation_or_dispatch(parent, tmp_path):
    store, _ = fork(parent, tmp_path)
    try:
        budget = repair.ContractRepairBudget(Budget(store, LiveSpec.model_validate(store.json(store.get("live_spec_ref")))),
                                              stop_at_review=True)
        before = repair._rows(store.db)
        corrected = repair.repair_invocation_key(store, "method/0/review/0", "scientific_review")
        with pytest.raises(repair.ReviewReplayReady):
            budget.reserve("model/" + corrected + "/0", "model", {"scope": "corrected review"})
        assert repair._rows(store.db)["live_operations"] == before["live_operations"]
        assert repair._rows(store.db)["calls"] == before["calls"]
        assert not (store.root / "model-ledger").exists()
    finally:
        store.close()


def test_stale_known_summary_cannot_hide_unknown_original_receipt(parent, tmp_path):
    path = next((parent / "model-ledger/claims").iterdir()) / "dispatch.json"
    receipt = relay.read_json(path, 16384)
    receipt.update(usage_status="unknown", observed_tokens=None)
    path.write_bytes(canonical(receipt))
    with pytest.raises(IntegrityError, match="unknown or inconsistent"):
        fork(parent, tmp_path)
    assert not (tmp_path / "child").exists()


@pytest.mark.parametrize("change", ["expired", "final", "running", "nested", "unknown", "worker"])
def test_ineligible_parent_rejected_before_child_creation(parent, tmp_path, change):
    store = Store(parent)
    if change == "expired":
        store.set("live_deadline", time.time() - 1)
    elif change == "final":
        store.set("live_frozen", {"fixture": True})
    elif change == "running":
        with store.db:
            store.db.execute("UPDATE live_operations SET status='running' WHERE kind='model'")
    elif change == "nested":
        store.set("inherited_model_accounting_ref", "earlier-epoch")
    elif change == "unknown":
        summary = store.json(store.get("live_pilot_model_accounting"))
        summary.update(usage_complete=False, unknown_calls=["unknown-fixture"])
        store.set("live_pilot_model_accounting", store.put_json(summary))
    else:
        identity = store.get("adapter_identity")
        identity["worker.py"] = "0" * 64
        store.set("adapter_identity", identity)
    store.close()
    before = repair._tree(parent)
    with pytest.raises(IntegrityError):
        fork(parent, tmp_path)
    assert not (tmp_path / "child").exists() and repair._tree(parent) == before


def test_symlinked_parent_payload_is_not_followed(parent, tmp_path):
    (parent / "external-link").symlink_to(tmp_path / "unrelated")
    with pytest.raises(IntegrityError, match="symlinks"):
        fork(parent, tmp_path)
    assert not (tmp_path / "child").exists()


def test_non_review_controller_changes_are_rejected(parent, tmp_path):
    store = Store(parent)
    spec = store.json(store.get("live_spec_ref"))
    source = Path(repair.__file__).with_name("controller.py").read_text()
    changed = source.replace("Screen, scale and evolve hypotheses using measured evidence", "Changed scientific selection")
    assert source != changed
    spec["implementation"]["research/controller.py"] = digest(changed.encode())
    ref = store.put_json(spec)
    store.set("live_spec_ref", ref)
    store.set("pilot_live_spec_ref", ref)
    index = store.json(store.get("live_sources_ref"))
    index["research/controller.py"] = store.put(changed.encode(), "py")
    store.set("live_sources_ref", store.put_json(index))
    store.close()
    with pytest.raises(IntegrityError, match="non-review workflow"):
        fork(parent, tmp_path)
    assert not (tmp_path / "child").exists()


def test_exact_original_task_text_is_reused_only_for_equal_typed_context(parent, tmp_path):
    store, _ = fork(parent, tmp_path)
    try:
        original = "{'settings': {'b': 2, 'a': 1}, 'enabled': True}"
        current = {"enabled": True, "settings": {"a": 1, "b": 2}}
        assert str(current) != original
        assert repair.recorded_task_source_text(store, current) == original
        # A failed child replay's new request pointer cannot replace the snapshot.
        store.set("request/limitations/0", store.put_json({"unrelated": True}))
        assert repair.recorded_task_source_text(store, current) == original
        for changed in ({"enabled": 1, "settings": {"a": 1, "b": 2}},
                        {"enabled": True, "settings": {"a": 1, "b": 3}}):
            with pytest.raises(IntegrityError, match="representation-only equivalent"):
                repair.recorded_task_source_text(store, changed)
    finally:
        store.close()


@pytest.mark.parametrize("parent", [
    {"task_source_text": "__import__('os').getcwd()"},
    {"task_source_text": "{'a':1,'a':2}"},
    {"task_source_text": "{'tuple': (1, 2)}"},
    {"task_source_text": "x" * 100001},
], indirect=True)
def test_unbounded_ambiguous_or_executable_task_text_is_rejected(parent, tmp_path):
    store, _ = fork(parent, tmp_path)
    try:
        with pytest.raises(IntegrityError):
            repair.recorded_task_source_text(store, {"a": 2})
    finally:
        store.close()


def test_task_source_ast_normalization_does_not_allow_other_init_or_run_changes():
    current = Path(repair.__file__).with_name("controller.py").read_text()
    start = current.index("    def _task_source_text(self):")
    end = current.index("    def transition(", start)
    previous = current[:start] + current[end:]
    previous = previous.replace('"text": self._task_source_text()', '"text": str(adapter.context())')
    previous = previous.replace('self.sources[0]["text"] = self._task_source_text()',
                                'self.sources[0]["text"] = str(self.adapter.context())')
    assert repair._review_ast(previous, "research/controller.py") == repair._review_ast(current, "research/controller.py")
    altered_init = current.replace('self.sources = [', 'self.sources = tuple([').replace(
        '*[s.model_dump() for s in config.sources],\n        ]', '*[s.model_dump() for s in config.sources],\n        ])', 1)
    assert repair._review_ast(previous, "research/controller.py") != repair._review_ast(altered_init, "research/controller.py")
    altered_run = current.replace('self.sources[0]["text"] = self._task_source_text()', 'self.sources[0]["text"] = "changed"')
    with pytest.raises(IntegrityError, match="outside representation repair"):
        repair._review_ast(altered_run, "research/controller.py")
    for altered in (current.replace('"text": self._task_source_text()', '"text": "verified-task-source-representation"'),
                    current.replace('self.sources[0]["text"] = self._task_source_text()',
                                    'self.sources[0]["text"] = "verified-task-source-representation"')):
        with pytest.raises(IntegrityError, match="outside representation repair"):
            repair._review_ast(altered, "research/controller.py")
    bad_helper = current.replace('context = self.adapter.context()\n        if self.store.get("contract_repair_manifest_ref"):',
                                 'context = {}\n        if self.store.get("contract_repair_manifest_ref"):')
    with pytest.raises(IntegrityError, match="representation-only adapter"):
        repair._review_ast(bad_helper, "research/controller.py")
