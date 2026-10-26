"""Synthetic evidence fixtures: no model, GPU, or real research claims.

The staged transport runs the real controller/runner evidence bookkeeping but
emits scripted train/inference/score outputs. V3-shaped cases distinguish seeds
0,1,2 from rebuttal seeds 3,4 and equal checkpoint bytes from separate launches.
"""
from copy import deepcopy
import json

import pytest

from scientisttwo.artifacts import load_bundle
from scientisttwo.live.audit import audit_live
from scientisttwo.live.evidence_context import build_evidence_context
from scientisttwo.live.models import Candidate, SourceFile
from scientisttwo.store import IntegrityError, canonical
from test_live import setup
from test_staged_evaluation import StagedTransportDouble, staged_development


def execute_staged(tmp_path, monkeypatch, *, seeds=(0, 1, 2)):
    store, runner, adapter = staged_development(
        tmp_path, monkeypatch, executor=StagedTransportDouble()
    )
    runner.config.development.seeds = list(seeds)
    store.set("live_spec_ref", store.put_json(runner.config))
    original = adapter.baseline()
    result = runner.execute("search/0/full/0", original)
    return store, runner, adapter, original, result


def test_pipeline_prediction_refs_point_to_actual_scored_bundle_not_checkpoint(tmp_path, monkeypatch):
    store, _, _, _, result = execute_staged(tmp_path, monkeypatch, seeds=(0,))
    try:
        evaluated = store.json(result.evaluation_refs[0])
        assert result.prediction_refs == [evaluated["prediction_bundle"]]
        assert result.prediction_refs != result.bundle_refs
        prediction = load_bundle(store, result.prediction_refs[0])
        assert set(prediction.files) == {"predictions.npy"}
        assert all(spec.role == "prediction" for spec in prediction.specs)
        assert audit_live(store.root)["status"] == "pass"
    finally:
        store.close()


def test_audit_rejects_training_config_relabelled_as_predictions(tmp_path, monkeypatch):
    store, _, _, _, result = execute_staged(tmp_path, monkeypatch, seeds=(0,))
    try:
        # Reproduce the pre-fix semantic alias with fully registered new bytes;
        # content hashing alone must not certify that this is a prediction.
        result.prediction_refs[0] = load_bundle(store, result.bundle_refs[0]).files[
            "checkpoint/tokenizer/config.json"].ref
        replacement = store.put_json(result)
        with store.db:
            store.db.execute("UPDATE calls SET output_ref=? WHERE key=?", (
                replacement, "development/" + result.id))
            store.db.execute("UPDATE live_operations SET result_ref=? WHERE key=?", (
                replacement, "experiment/" + result.id))
        # Archived legacy aliases can still be explained honestly: the bound
        # evaluation is the authority, and the inconsistent alias is visible.
        presented = build_evidence_context(store, [result])["observations"][0]
        prediction = presented["prediction_evidence"][0]
        assert prediction["ref"] == store.json(result.evaluation_refs[0])["prediction_bundle"]
        assert prediction["observation_reference_status"].startswith("legacy_or_incorrect_alias")
        checked = audit_live(store.root)
        assert checked["status"] == "fail"
        assert any("binding" in error for error in checked["errors"])
    finally:
        store.close()


def test_v3_shaped_seeds_and_same_checkpoint_bytes_show_distinct_execution_receipts(tmp_path, monkeypatch):
    store, runner, _, original, full = execute_staged(tmp_path, monkeypatch)
    try:
        extra = runner.execute("rebuttal/0/matched", original, "rebuttal", seeds=[3, 4])
        package = build_evidence_context(store, [full, extra])
        first, second = package["observations"]
        assert first["protocol"]["seeds"] == [0, 1, 2]
        assert second["protocol"]["seeds"] == [3, 4]
        assert [row["seed"] for row in second["per_seed_metrics"]] == [3, 4]
        executions = first["training_executions"] + second["training_executions"]
        assert len({row["receipt_ref"] for row in executions}) == 5
        assert len({row["execution_id"] for row in executions}) == 5
        assert all(row["status"] == "recorded_success" for row in executions)
        # Scripted checkpoint bytes are identical across all five launches.
        assert len(set(full.bundle_refs + extra.bundle_refs)) == 1
        assert all("checkpoint" in row["roles"] for row in first["training_output_bundles"])
        assert all("prediction" in row["roles"] for row in first["prediction_evidence"])
        assert {"phase", "seeds"} <= set(package["protocol_differences"][0]["different_components"])
        assert any("hashes describe equal bytes" in rule for rule in package["interpretation_rules"])
        assert any("distinct seeds/receipts do not certify" in rule for rule in package["interpretation_rules"])
        assert package == build_evidence_context(store, [full, extra])
    finally:
        store.close()


def test_control_evidence_supplies_actual_frozen_editable_source_and_byte_difference(tmp_path, monkeypatch):
    store, runner, _, original, matched = execute_staged(tmp_path, monkeypatch, seeds=(0,))
    try:
        edited_text = original.files[1].content + "\n# SYNTHETIC DECLARED CONTROL CHANGE\n"
        control_candidate = Candidate(
            files=[original.files[0], SourceFile(path="custom.py", content=edited_text)],
            command=original.command, description="Synthetic source-only control"
        )
        control = runner.execute("ablation/0/control", control_candidate)
        store.cache("control/ablation/0", {"synthetic": True}, {
            "matched": matched.model_dump(), "control": control.model_dump(),
            "method_isolation": "unverified", "file_isolation": "pass",
        })
        package = build_evidence_context(store, [matched, control])
        pair = package["matched_controls"][0]
        assert pair["changed_paths"] == ["custom.py"]
        assert pair["exact_protocol_match"] and pair["command_unchanged"]
        assert pair["method_isolation"] == "unverified"
        source = next(row for row in package["sources"]
                      if row["source_tree_ref"] == control.source_tree_ref)
        file = next(file for file in source["files"] if file["path"] == "custom.py")
        assert file["text"] == edited_text and file["text_complete"]
        assert file["sha256"] == store.db.execute(
            "SELECT sha FROM artifacts WHERE path=?", (file["ref"],)).fetchone()[0]
        # Source comes from the materialized tree, not caller-supplied prose/code.
        assert canonical(package).decode().count("SYNTHETIC DECLARED CONTROL CHANGE") == 1
    finally:
        store.close()


def test_context_excludes_final_observations_labels_host_paths_and_raw_requests(tmp_path, monkeypatch):
    store, _, _, _, result = execute_staged(tmp_path, monkeypatch, seeds=(0,))
    try:
        original_read = store.read

        def no_model_payloads(ref):
            assert not ref.endswith((".npy", ".pt", ".bin", ".safetensors"))
            return original_read(ref)

        monkeypatch.setattr(store, "read", no_model_payloads)
        final = result.model_copy(update={"id": "SECRET_FINAL_ID", "phase": "final",
                                          "value": 0.123456789, "values": [0.123456789]})
        # Final should be discarded before any read; its references need not exist.
        final.protocol_ref = "protected/final-protocol.json"
        final.source_tree_ref = "protected/final-source.json"
        package = build_evidence_context(store, [result, final])
        text = canonical(package).decode()
        assert "SECRET_FINAL_ID" not in text and "0.123456789" not in text
        assert str(tmp_path) not in text
        assert "private_dir" not in text and "source_dir" not in text
        assert "argv" not in text and "job_ref" not in text and "request_ref" not in text
        assert "FINAL_STAGED_LABELS" not in text and "labels.npy" not in text
        assert '"stdout"' not in text and '"stderr"' not in text
        assert '"candidate_manifest"' not in text and '"public_manifest"' not in text
    finally:
        store.close()


def test_source_excerpt_limit_is_total_exact_and_read_only(tmp_path, monkeypatch):
    store, _, _, _, result = execute_staged(tmp_path, monkeypatch, seeds=(0,))
    try:
        before = list(store.db.iterdump())
        package = build_evidence_context(store, [result], max_source_chars=13)
        after = list(store.db.iterdump())
        assert before == after
        files = package["sources"][0]["files"]
        assert sum(len(f.get("excerpt_head", "")) + len(f.get("excerpt_tail", "")) for f in files) == 13
        assert package["source_excerpt_characters"] == 13
        assert all(f["omitted_characters"] > 0 and not f["text_complete"] for f in files)
        assert all(f["sha256"] for f in files)
    finally:
        store.close()


def test_synthetic_missing_proofs_are_explicitly_unverified(tmp_path):
    controller, store, _, runner = setup(tmp_path)
    try:
        result = runner.execute("synthetic", controller.adapter.baseline())
        package = build_evidence_context(store, [result])
        row = package["observations"][0]
        assert row["commitment"] == "recorded"
        assert row["protocol"]["status"] == "unverified"
        assert row["source"]["status"] == "unverified"
        assert row["training_executions"][0]["status"] == "unverified"
        assert row["per_seed_metrics"][0]["seed"] is None
        uncommitted = result.model_copy(update={"id": "not-stored"})
        missing = build_evidence_context(store, [uncommitted])["observations"][0]
        assert missing["commitment"] == "unverified" and "metrics" not in missing
    finally:
        store.close()


@pytest.mark.parametrize("damage", ["observation", "protocol", "source", "receipt"])
def test_integrity_errors_are_not_relabelled_unverified(tmp_path, monkeypatch, damage):
    store, _, _, _, result = execute_staged(tmp_path, monkeypatch, seeds=(0,))
    try:
        if damage == "observation":
            result.reason = "forged offered result"
        else:
            ref = {"protocol": result.protocol_ref, "source": result.source_tree_ref,
                   "receipt": result.log_refs[0]}[damage]
            (store.root / ref).write_text("{}")
        with pytest.raises(IntegrityError):
            build_evidence_context(store, [result])
    finally:
        store.close()


def test_receipt_seed_binding_cannot_claim_a_new_training_run(tmp_path, monkeypatch):
    store, _, _, _, result = execute_staged(tmp_path, monkeypatch, seeds=(0,))
    try:
        receipt = store.json(result.log_refs[0])
        job = store.json(receipt["job_ref"])
        receipt["job_ref"] = store.put_json({**job, "seed": 99})
        result.log_refs[0] = store.put_json(receipt)
        replacement = store.put_json(result)
        with store.db:
            store.db.execute("UPDATE calls SET output_ref=? WHERE key=?", (
                replacement, "development/" + result.id))
        with pytest.raises(IntegrityError, match="different observation or seed"):
            build_evidence_context(store, [result])
    finally:
        store.close()


def test_meta_attempt_summary_preserves_failures_rejection_and_lineage_without_source_prose(tmp_path):
    controller, store, _, runner = setup(tmp_path)
    try:
        candidate = controller.adapter.baseline()
        observation = runner.execute("synthetic-search", candidate)
        trial = {"id": "synthetic-search", "result": observation.model_dump(),
                 "candidate": candidate.model_dump(), "parents": [],
                 "decision": {"decision": "Good", "reason": "Synthetic acceptance"}}
        store.cache("live-trial/" + trial["id"], {"synthetic": True}, trial)
        trials = [trial]
        package = build_evidence_context(store, [], trials=trials)
        assert len(package["search_attempts"]) == len(trials)
        assert all(row["commitment"] == "recorded" for row in package["search_attempts"])
        assert {row["decision"] for row in package["search_attempts"]} == {"Good"}
        final_trial = deepcopy(trials[0])
        final_trial["result"]["phase"] = "final"
        assert not build_evidence_context(store, [], trials=[final_trial])["search_attempts"]
        fake = deepcopy(trials[0])
        fake["id"] = "synthetic-failure"
        fake["result"]["status"] = "execution_failed"
        fake["result"]["reason"] = "Cannot read /home/example/private/labels.npy"
        fake["decision"] = {"decision": "Engineer", "reason": "Execution failed"}
        fake["parents"] = [trials[0]["id"]]
        store.cache("live-trial/" + fake["id"], {"synthetic": True}, fake)
        summary = build_evidence_context(store, [], trials=[fake])["search_attempts"][0]
        assert summary["execution_status"] == "execution_failed" and summary["decision"] == "Engineer"
        assert summary["parents"] == fake["parents"] and "[path omitted]" in summary["execution_reason"]
        assert "/home/example" not in json.dumps(summary)
        fake["decision"]["reason"] = "Changed after commitment"
        with pytest.raises(IntegrityError, match="committed decision"):
            build_evidence_context(store, [], trials=[fake])
    finally:
        store.close()
