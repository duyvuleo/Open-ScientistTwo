"""Lazy local replay of frozen source dependencies; no models or candidate execution."""

from types import SimpleNamespace

import pytest

from scientisttwo.artifacts import load_bundle, verified_file
from scientisttwo.live.models import Candidate, Observation
from scientisttwo.live.parent_edits import parent_context
from scientisttwo.live.runtime import Budget
from scientisttwo.live.source_tree import candidate_snapshot, freeze_source_tree
from scientisttwo.models import AgentResult
from scientisttwo.research.audit import _ReplayStore, _handoffs, audit_research
from scientisttwo.research.contracts import FullSetHandoff
from scientisttwo.store import IntegrityError, Store, digest
from scientisttwo.testbed.validation import sealed_evidence


@pytest.fixture
def frozen_handoff(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("from utility import value\nprint(value)\n")
    (source / "utility.py").write_text("value = 1\n")
    original = Store(tmp_path / "original")
    base = freeze_source_tree(original, source, ["train.py"])
    candidate = Candidate(base_snapshot_ref=base, command=["python", "train.py"],
                          description="Synthetic frozen checkpoint task source")
    tree = candidate_snapshot(original, candidate)
    tree_ref = original.put_json(tree)
    subset = Observation(id="search/0/0/subset/0", phase="subset", status="success",
                         code_ref=tree_ref, source_tree_ref=tree_ref,
                         protocol_hash="0" * 64, metric="accuracy", value=.5, values=[.5],
                         prediction_refs=[original.put_json({"synthetic_prediction": True})])
    full = subset.model_copy(update={"id": "search/0/0/full/0", "phase": "full"})
    for observation in (subset, full):
        original.cache("development/" + observation.id, {"synthetic": True}, observation)
    context = parent_context(original, candidate, tree_ref, editable_paths=["train.py"])
    context["trial_id"] = subset.id
    key = "search/0/0/full-handoff"
    request = {"role": "full_set_coder", "inputs": {"parent": context,
               "subset_result": subset.model_dump(), "scaling_paths": []}}
    handoff = FullSetHandoff(action="ReuseSnapshot", reason="Synthetic immutable snapshot reuse")
    result = AgentResult(payload=handoff.model_dump(), backend="fixture", model="fixture", synthetic=True)
    budget = Budget(original, SimpleNamespace(campaign_seconds=60, max_calls=1,
                                             max_experiments=1, token_stop_threshold=None))
    budget.reserve("model/" + key + "/0", "model", request)
    budget.finish("model/" + key + "/0", "success", result, 0)
    original.cache("live-agent/" + key, request, result)
    original.set("request/" + key, original.put_json(request))
    original.set("full-handoff/" + key, original.put_json(handoff))
    # An unrelated model artifact must not be eagerly copied into scratch.
    checkpoint = tmp_path / "unrelated-checkpoint.bin"
    with checkpoint.open("wb") as stream:
        for _ in range(8):
            stream.write(b"x" * 1024 * 1024)
    checkpoint_ref = original.put_file(checkpoint, "checkpoint")
    adapter = SimpleNamespace(editable_paths=lambda: ["train.py"], validate=lambda value: None)
    replay = _ReplayStore(tmp_path / "replay", original)
    yield original, replay, adapter, tree, checkpoint_ref
    replay.close()
    original.close()


def test_frozen_handoff_lazily_copies_source_payloads_without_mutating_original(frozen_handoff, monkeypatch):
    original, replay, adapter, tree, checkpoint_ref = frozen_handoff
    before_rows = list(original.db.iterdump())
    before_files = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns, digest(p.read_bytes()))
                    for p in (original.root / "artifacts").iterdir()}
    payloads = load_bundle(original, tree.bundle_ref).files
    assert all(not (replay.root / item.ref).exists() for item in payloads.values())
    original_read = original.read

    def no_payload_read(name):
        assert name not in {item.ref for item in payloads.values()}
        assert name != checkpoint_ref
        return original_read(name)

    monkeypatch.setattr(original, "read", no_payload_read)
    # Reproduce the old metadata-only overlay: read() can resolve source JSON,
    # but bundle verification still fails without a local-payload hook.
    with monkeypatch.context() as previous_behavior:
        previous_behavior.setattr(replay, "materialize_artifact", lambda name: None)
        with pytest.raises(IntegrityError, match="Missing or modified bundle artifact"):
            _handoffs(original, replay, adapter)
    report = _handoffs(original, replay, adapter)
    assert report[0]["status"] == "pass" and report[0]["full_execution"] == "recorded"
    for item in payloads.values():
        copied = verified_file(replay, item)
        assert copied.read_bytes() == (original.root / item.ref).read_bytes()
        assert copied.stat().st_ino != (original.root / item.ref).stat().st_ino
    assert not (replay.root / checkpoint_ref).exists()
    assert list(original.db.iterdump()) == before_rows
    assert {p.name: (p.stat().st_ino, p.stat().st_mtime_ns, digest(p.read_bytes()))
            for p in (original.root / "artifacts").iterdir()} == before_files
    assert _handoffs(original, replay, adapter) == report


@pytest.mark.parametrize("fault", ["same_size_corruption", "wrong_size", "original_symlink",
                                  "scratch_symlink", "local_corruption", "registration"])
def test_lazy_replay_preserves_bundle_integrity_checks(frozen_handoff, fault):
    original, replay, _, tree, _ = frozen_handoff
    item = next(iter(load_bundle(original, tree.bundle_ref).files.values()))
    source, target = original.root / item.ref, replay.root / item.ref
    if fault == "same_size_corruption":
        source.write_bytes(b"z" * item.size)
    elif fault == "wrong_size":
        source.write_bytes(b"longer corrupted bytes" * item.size)
    elif fault == "original_symlink":
        backup = source.with_suffix(".fixture")
        source.rename(backup)
        source.symlink_to(backup)
    elif fault == "scratch_symlink":
        target.symlink_to(source)
    elif fault == "local_corruption":
        target.write_bytes(b"z" * item.size)
    else:
        with replay.db:
            replay.db.execute("UPDATE artifacts SET size=size+1 WHERE path=?", (item.ref,))
    with pytest.raises(IntegrityError):
        verified_file(replay, item)


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "artifacts/../outside"])
def test_lazy_replay_rejects_unsafe_registered_paths(frozen_handoff, name):
    _, replay, *_ = frozen_handoff
    with replay.db:
        replay.db.execute("INSERT INTO artifacts VALUES(?,?,?,?)", (name, "a" * 64, "fixture", 1))
    with pytest.raises((IntegrityError, ValueError)):
        replay.materialize_artifact(name)


def test_streaming_copy_detects_source_change_after_initial_verification(frozen_handoff, monkeypatch):
    from scientisttwo.research import audit
    original, replay, _, tree, _ = frozen_handoff
    item = next(iter(load_bundle(original, tree.bundle_ref).files.values()))
    checked = audit.verified_file

    def change_after_check(store, artifact):
        path = checked(store, artifact)
        path.write_bytes(b"z" * artifact.size)
        return path

    monkeypatch.setattr(audit, "verified_file", change_after_check)
    with pytest.raises(IntegrityError, match="changed during copying"):
        verified_file(replay, item)
    assert not (replay.root / item.ref).exists()
    assert not list((replay.root / "artifacts").glob("replay-*.tmp"))


def test_checkpoint_research_audit_keeps_existing_seal_and_database_unchanged(tmp_path, monkeypatch):
    from scientisttwo.live import audit as live_audit
    from scientisttwo.research import audit as research_audit
    from test_research_workflow import ResearchStagedAdapter, campaign

    controller, store, runner, agents, _ = campaign(tmp_path)

    class FrozenAdapter(ResearchStagedAdapter):
        snapshot_ref = None

        def source_tree(self):
            return {"path": str(self.native.candidate_dir), "editable_paths": ["custom.py"]}

        def bind_source(self, ref):
            self.snapshot_ref = ref

        def baseline(self):
            candidate = super().baseline()
            if self.snapshot_ref:
                return candidate.model_copy(update={"base_snapshot_ref": self.snapshot_ref,
                    "files": [item for item in candidate.files if item.path == "custom.py"]})
            return candidate

        def editable_paths(self):
            return ["custom.py"]

    adapter = FrozenAdapter(controller.config.native)
    controller.adapter = runner.adapter = agents.legacy_response.adapter = adapter
    agents.store = store
    monkeypatch.setattr(live_audit, "load_development", lambda *args: adapter)
    monkeypatch.setattr(research_audit, "load_development", lambda *args: adapter)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        assert store.get("authorized_source_snapshot")
        standard = live_audit.audit_live(store.root)
        assert standard["status"] == "pass", standard["errors"]
        store.db.execute("PRAGMA wal_checkpoint(FULL)")
        before = sealed_evidence(store.root)
        rows = list(store.db.iterdump())
        report = audit_research(store.root)
        assert report["status"] == "pass", report["errors"]
        assert report["checks"]["full_stage_handoffs"]["status"] == "pass"
        store.db.execute("PRAGMA wal_checkpoint(FULL)")
        assert list(store.db.iterdump()) == rows
        assert sealed_evidence(store.root) == before
    finally:
        store.close()
