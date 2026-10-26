"""Synthetic, model-free checks of evidence-preserving context presentation."""

from copy import deepcopy

import pytest

from scientisttwo.live.context import (
    ContextProtocolError,
    compact_trial_context,
    input_characters,
    present_inputs,
    validate_presentation,
)
from scientisttwo.live.models import LimitationAssessment
from scientisttwo.store import Store, canonical, digest
from test_live import candidate, idea, setup


def trial(name, code, *, status="success", parents=()):
    return {
        "id": name,
        "idea": idea("Synthetic hypothesis").model_dump(),
        "candidate": code,
        "result": {
            "status": status,
            "metrics": {"quality": 0.8, "latency": 12.0},
            "metric_values": {"quality": [0.7, 0.9], "latency": [11.0, 13.0]},
            "value": 0.8,
            "values": [0.7, 0.9],
            "protocol_hash": "immutable-protocol",
            "code_ref": "artifacts/executed-code.json",
            "reason": "Synthetic observation",
        },
        "decision": {
            "decision": "Good" if status == "success" else "Engineer",
            "reason": "Synthetic scientific/methodological rationale",
        },
        "parents": list(parents),
    }


def test_history_retains_all_results_failures_lineage_and_distinct_source_evidence(tmp_path):
    store = Store(tmp_path / "store")
    code = candidate(8).model_dump()
    code["files"][0]["content"] = "HEAD" + "x" * 10000 + "TAIL"
    second = deepcopy(code)
    second["files"].append({"path": "helper.py", "content": "y" * 20000})
    original = [
        trial("subset", code),
        trial("full", code, parents=["subset"]),
        trial("failed", second, status="execution_failed", parents=["full"]),
    ]
    before = deepcopy(original)
    compact, proof = compact_trial_context(
        original, code_chars=120, archive_candidate=store.put_json
    )
    assert original == before
    assert len(compact) == 3 and proof["trials_preserved"] == 3
    assert proof["distinct_proposals"] == 2 and proof["source_excerpt_characters"] == 120
    for old, new in zip(original, compact, strict=True):
        for key in ("id", "idea", "result", "decision", "parents"):
            assert old[key] == new[key]
        assert store.json(new["candidate"]["proposal_artifact_ref"]) == old["candidate"]
        assert new["candidate"]["proposal_sha256"] == digest(canonical(old["candidate"]))
    assert compact[1]["candidate"]["candidate_context_trial"] == "subset"
    assert "files" not in compact[1]["candidate"]
    files = [f for t in compact for f in t["candidate"].get("files", [])]
    assert sum(len(f["excerpt_head"]) + len(f["excerpt_tail"]) for f in files) == 120
    assert all(f["omitted_characters"] > 0 for f in files)
    repeated, _ = compact_trial_context(original, code_chars=120, archive_candidate=store.put_json)
    assert repeated == compact
    store.close()


def test_zero_source_allowance_keeps_file_digests_and_all_observations(tmp_path):
    store = Store(tmp_path / "store")
    original = [trial("one", candidate(8).model_dump())]
    compact, proof = compact_trial_context(original, code_chars=0, archive_candidate=store.put_json)
    file = compact[0]["candidate"]["files"][0]
    assert file["excerpt_head"] == file["excerpt_tail"] == ""
    assert file["sha256"] and file["omitted_characters"] == file["characters"]
    assert compact[0]["result"] == original[0]["result"]
    assert proof["source_excerpt_characters"] == 0
    store.close()


def test_log_tail_budget_is_total_and_omissions_are_explicit(tmp_path):
    store = Store(tmp_path / "store")
    inputs = {
        "evidence": {
            "observation": {"values": [1.0, 2.0], "log_refs": ["a", "b"]},
            "execution_logs": ["prefix-one-last1", "prefix-two-last2"],
        },
        "contract": {"max_latency": 100, "epsilon": 0.0001},
    }
    original = deepcopy(inputs)
    presented, changes = present_inputs(
        "critic",
        inputs,
        history_code_chars=100,
        log_tail_chars=10,
        archive_candidate=store.put_json,
    )
    assert inputs == original
    assert presented["evidence"]["execution_logs"] == ["last1", "last2"]
    assert presented["evidence"]["observation"] == original["evidence"]["observation"]
    assert presented["contract"] == original["contract"]
    assert len(changes) == 2 and all(c["omitted_characters"] > 0 for c in changes)
    assert "omitted" in presented["context_presentation"]["changes"][0]["note"]
    store.close()


def test_character_cap_does_not_truncate_metrics_and_is_not_a_token_estimate():
    inputs = {"values": list(range(1000)), "unicode": "研究"}
    original = deepcopy(inputs)
    exact = len(canonical(inputs).decode("utf-8"))
    assert input_characters(inputs) == exact
    assert validate_presentation(inputs, max_prompt_chars=exact) == exact
    with pytest.raises(ContextProtocolError, match="not a model token-context guarantee"):
        validate_presentation(inputs, max_prompt_chars=exact - 1)
    assert inputs == original


def test_full_raw_request_and_presented_request_are_both_sealed(tmp_path):
    from scientisttwo.live.parent_edits import MethodChange
    controller, store, backend, _ = setup(tmp_path, history_code_chars=80, log_tail_chars=10)
    code = candidate(8).model_dump()
    code["files"][0]["content"] = "x" * 10000
    inputs = {
        "attempts": [trial("subset", code), trial("full", code)],
        "contract": {"guardrail": 0.01},
        "sources": [],
    }
    controller.ask("synthetic-evolve", "evolve", inputs, MethodChange)
    raw = store.json(store.get("raw_request/synthetic-evolve"))
    presented = store.json(store.get("request/synthetic-evolve"))
    proof = store.json(store.get("context/synthetic-evolve"))
    assert raw["inputs"]["attempts"][0]["candidate"] == code
    assert presented["inputs"] == backend.seen[-1][1]
    assert (
        proof["status"] == "pass"
        and proof["presented_input_characters"] < proof["raw_input_characters"]
    )
    assert proof["raw_request_ref"] != proof["presented_request_ref"]
    assert proof["presented_input_characters"] == input_characters(presented["inputs"])
    count = len(backend.seen)
    controller.ask("synthetic-evolve", "evolve", inputs, MethodChange)
    assert len(backend.seen) == count  # Identical presentation is resumable without another call.
    store.close()


def test_oversize_request_fails_before_reservation_and_records_evidence(tmp_path):
    controller, store, backend, _ = setup(tmp_path, max_prompt_chars=1000)
    with pytest.raises(ContextProtocolError, match="exceeding"):
        controller.ask(
            "oversize", "limitations", {"measurements": list(range(1000))}, LimitationAssessment
        )
    assert not backend.seen and not controller.budget.summary()
    proof = store.json(store.get("context/oversize"))
    assert proof["status"] == "blocked"
    assert store.json(proof["raw_request_ref"])["inputs"]["measurements"] == list(range(1000))
    assert store.json(proof["presented_request_ref"])["inputs"]["measurements"] == list(range(1000))
    store.close()


def test_protected_host_paths_are_blocked_before_a_model_call(tmp_path):
    controller, store, backend, _ = setup(tmp_path)
    private = tmp_path / "private-final-labels"
    controller.config.native.private_dir = private
    with pytest.raises(ContextProtocolError, match="protected host"):
        controller.ask(
            "private", "limitations", {"text": f"Read {private}/labels.csv"}, LimitationAssessment
        )
    assert not backend.seen and not controller.budget.summary()
    assert store.json(store.get("context/private"))["status"] == "blocked"
    store.close()


def test_campaign_reports_context_protocol_failure_without_model_or_experiment(tmp_path):
    controller, store, backend, runner = setup(tmp_path, max_prompt_chars=1000)
    controller.adapter.context = lambda: {"contract": "SYNTHETIC " * 1000}
    result = controller.run()
    assert result["status"] == "blocked" and result["outcome"] == "context_protocol_failed"
    assert not backend.seen and not runner.executions
    assert not controller.budget.summary()
    store.close()


def test_feedback_archives_all_log_content_before_presentation(tmp_path):
    from scientisttwo.live.models import Observation

    controller, store, _, _ = setup(tmp_path, log_tail_chars=20)
    content = "SYNTHETIC FULL LOG " * 1000
    refs = [
        store.put_json({"log_ref": store.put((content + str(i)).encode(), "log")}) for i in range(4)
    ]
    observation = Observation(
        id="synthetic",
        phase="full",
        status="execution_failed",
        protocol_hash="test",
        code_ref="artifacts/test.json",
        metric="process_seconds",
        log_refs=refs,
    )
    feedback = controller.feedback(observation)
    assert len(feedback["execution_logs"]) == 4
    assert all(log.startswith(content) for log in feedback["execution_logs"])
    assert feedback["observation"]["log_refs"] == refs
    store.close()
