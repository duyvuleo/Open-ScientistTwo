"""Synthetic staged integration of the actual agent request wiring.

A scripted transport supplies checkpoint/prediction/score artifacts. The tests
verify provenance/context contracts, not empirical quality or scientific gain.
"""
from copy import deepcopy
import json

import pytest

from scientisttwo.backends import FixtureBackend
from scientisttwo.live.controller import LiveController
from scientisttwo.live.models import Candidate, SourceFile
from scientisttwo.models import CriticDecision
from scientisttwo.process import ProcessResult
from scientisttwo.store import IntegrityError
from test_live import concern_responses, idea
from test_staged_evaluation import StagedTransportDouble, staged_development


class FailingOnceTransport(StagedTransportDouble):
    fail_next_training = False

    def __call__(self, argv, timeout, **kwargs):
        if (self.fail_next_training and "run" in argv
                and any(":/workspace:rw" in value for value in argv)):
            self.fail_next_training = False
            return ProcessResult(7, "SYNTHETIC training failure", "", 0.1)
        return super().__call__(argv, timeout, **kwargs)


class EvidenceAgents(FixtureBackend):
    def __init__(self, baseline):
        self.seen = []
        self.baseline = baseline
        super().__init__(self.respond)

    def control(self):
        return Candidate(
            files=[self.baseline.files[0], SourceFile(
                path="custom.py", content=self.baseline.files[1].content + "\n# SYNTHETIC_STAGED_CONTROL\n")],
            command=list(self.baseline.command), description="Synthetic source-only control",
        )

    def respond(self, role, inputs):
        self.seen.append((role, deepcopy(inputs)))
        if role == "implement":
            if "parent" not in inputs:
                return self.baseline.model_dump()
            parent = inputs["parent"]
            text = next(item["content"] for item in parent["files"] if item["path"] == "custom.py")
            return {"parent_source_ref": parent["parent_source_ref"],
                    "edits": [{"path": "custom.py", "before": text,
                               "after": text + "\nSYNTHETIC_REVISION = True\n", "reason": "Synthetic explicit change"}],
                    "description": "Synthetic parent-bound revision"}
        if role == "critic":
            return {"decision": "Bad", "reason": "Synthetic non-improving candidate"}
        if role == "ablations":
            return {"plans": [{"purpose": "Synthetic file-isolation check",
                               "changed_file": "custom.py", "candidate": self.control().model_dump()}]}
        if role == "ablation_critic":
            return {"decision": "Good", "reason": "Synthetic proceed-to-review branch"}
        if role == "draft":
            observation = inputs["evidence"][0]
            return {"title": "Synthetic evidence integration", "source_ids": ["task"],
                    "claims": [{"experiment_id": observation["id"], "metric": observation["metric"],
                                "value": observation["value"]}],
                    "limitations": ["Scripted transport; not empirical validation"],
                    "concern_responses": concern_responses(inputs)}
        if role == "review":
            return {"score": 8 if inputs["supplementary_coverage"] else 7,
                    "concerns": ["Inspect exactly two additional training seeds"]}
        if role == "rebuttal":
            return {"plans": [{"id": "repeat-two", "concern_ids": [row["id"] for row in inputs["concerns"]],
                        "purpose": "Inspect the explicitly declared repeated measurements",
                        "seed_count": 2, "execution_kind": "repeat", "evaluation_split": "development",
                        "diagnostics": [{"kind": "metric", "metric": "mse"}], "changed_file": None,
                        "candidate": self.baseline.model_dump()}], "unaddressed_concerns": []}
        if role == "meta":
            return {"decision": "Accept", "reason": "Synthetic evidence-scoped acceptance"}
        if role == "refine":
            return {"idea": idea("synthetic revision").model_dump(), "change_plan": {
                "changed_files": ["custom.py"], "reason": "Synthetic explicit revision"}}
        raise AssertionError(role)


@pytest.fixture
def evidence_cycle(tmp_path, monkeypatch):
    transport = FailingOnceTransport()
    store, runner, adapter = staged_development(tmp_path, monkeypatch, executor=transport)
    config = runner.config
    config.model = "scripted-fixture-v1"
    config.mode = "research"
    config.max_calls = 40
    config.development.seeds = [0, 1, 2]
    config.ablation_refinements = 0
    config.review_rounds = 1
    store.set("live_spec_ref", store.put_json(config))
    store.set("live_state", {"stage": "search", "status": "running"})
    backend = EvidenceAgents(adapter.baseline())
    controller = LiveController(store, config, adapter, backend, runner, budget=runner.budget)
    baseline = adapter.baseline()
    controller.baselines = {phase: runner.execute("baseline/" + phase, baseline, phase)
                            for phase in ("subset", "full")}
    transport.fail_next_training = True
    failed = runner.execute("search/synthetic-failed/subset/0", baseline, "subset")
    assert failed.status == "execution_failed"
    controller.record(failed.id, idea("failed"), baseline, failed,
                      CriticDecision(decision="Engineer", reason="Synthetic failed launch"))
    rejected = runner.execute("search/synthetic-rejected/full/0", baseline, "full")
    controller.record(rejected.id, idea("rejected"), baseline, rejected,
                      CriticDecision(decision="Bad", reason="Synthetic unsupported mechanism"))
    chosen = runner.execute("search/synthetic-selected/full/0", baseline, "full")
    best = controller.record(chosen.id, idea("fixture incumbent"), baseline, chosen,
                             CriticDecision(decision="Good", reason="Synthetic downstream branch setup"))
    controller.set_best(best)
    # Final evidence is stored elsewhere and never enters development inputs.
    final_ref = store.put_json({"protected_final_canary": "DO_NOT_PRESENT_FINAL_VALUE", "accuracy": 0.123456789})
    store.set("synthetic-final-proof", final_ref)
    yield controller, store, backend, runner, best
    store.close()


def test_actual_critic_and_downstream_requests_receive_failed_search_history_and_scored_artifacts(evidence_cycle):
    controller, store, backend, _, best = evidence_cycle
    assert controller.evaluate("search/another", idea("another")) is None
    selected, meta = controller.downstream("method/0", best)
    assert selected == best and meta.decision == "Accept"
    roles = {role for role, _ in backend.seen}
    assert {"critic", "ablations", "ablation_critic", "draft", "review", "rebuttal", "meta"} <= roles
    for role, inputs in backend.seen:
        if role not in {"critic", "ablations", "ablation_critic", "draft", "review", "rebuttal", "meta"}:
            continue
        package = inputs["evidence_package"]
        history = {row["id"]: row for row in package["search_attempts"]}
        assert history["search/synthetic-failed/subset/0"]["execution_status"] == "execution_failed", role
        assert history["search/synthetic-rejected/full/0"]["decision"] == "Bad", role
        assert all(row["commitment"] == "recorded" for row in history.values())
        for observation in package["observations"]:
            assert observation["protocol"]["status"] == "recorded"
            for predictions in observation["prediction_evidence"]:
                assert predictions["kind"] == "scored_prediction_bundle"
                assert predictions["roles"]["prediction"]["files"] == 1
                assert predictions["observation_reference_status"] == "matches"
            assert all(receipt["status"] == "recorded_success" for receipt in observation["training_executions"])
        text = json.dumps(inputs)
        assert "DO_NOT_PRESENT_FINAL_VALUE" not in text and "0.123456789" not in text
        assert str(controller.config.native.private_dir) not in text
        assert str(controller.config.native.evaluator_dir) not in text
        assert "private_dir" not in json.dumps(package)
        assert "request_ref" not in json.dumps(package)
    # Verify exactly what was sealed for the actual review invocation, not only
    # the helper output called directly in a unit test.
    saved = store.json(store.get("request/method/0/review/1"))["inputs"]
    assert saved == next(inputs for role, inputs in backend.seen if role == "review" and inputs["supplementary_coverage"])
    assert saved["supplementary_coverage"][0]["status"] == "executed"
    extra = [row for row in saved["evidence_package"]["observations"] if row["phase"] == "rebuttal"]
    assert len(extra) == 1
    assert extra[0]["protocol"]["seeds"] == [3, 4]
    assert len(extra[0]["training_executions"]) == 2
    assert len({row["receipt_ref"] for row in extra[0]["training_executions"]}) == 2
    assert any("SYNTHETIC_STAGED_CONTROL" in json.dumps(source)
               for source in saved["evidence_package"]["sources"])
    pair = saved["evidence_package"]["matched_controls"][0]
    assert pair["changed_paths"] == ["custom.py"] and pair["exact_protocol_match"]
    assert pair["method_isolation"] == "unverified"


def test_refinement_receives_executed_control_source_and_rejection_history_preserves_incumbent(evidence_cycle):
    controller, _, backend, _, best = evidence_cycle
    controller.downstream("method/0", best)
    refined, improved = controller.refine("meta/revision", best, {
        "meta": {"decision": "Refine", "reason": "Synthetic revision test"},
        "evidence": [result.model_dump() for result in controller.downstream_evidence],
    })
    assert not improved and refined == best and controller.best == best
    request = next(inputs for role, inputs in backend.seen if role == "refine")
    package = request["evidence_package"]
    assert package["matched_controls"][0]["changed_paths"] == ["custom.py"]
    assert any(row["decision"] == "Bad" for row in package["search_attempts"])
    assert any(row["execution_status"] == "execution_failed" for row in package["search_attempts"])
    assert request["parent"]["files"]
    assert all("content" in file for file in request["parent"]["files"])
    assert any("SYNTHETIC_STAGED_CONTROL" in json.dumps(source) for source in package["sources"])
    critique = [inputs for role, inputs in backend.seen if role == "critic"][-1]
    assert any(row["decision"] == "Bad" for row in critique["evidence_package"]["search_attempts"])


def test_final_freeze_blocks_any_additional_review_call(evidence_cycle):
    controller, store, backend, _, _ = evidence_cycle
    store.set("live_frozen", {"synthetic": True})
    before = len(backend.seen)
    from scientisttwo.live.models import LiveReview
    with pytest.raises(IntegrityError, match="after final candidate freeze"):
        controller.ask("illegal-review", "review", {"draft": {}}, LiveReview)
    assert len(backend.seen) == before
