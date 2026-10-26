"""Explicitly synthetic research traces; no models, containers, or measurements."""

from collections import Counter
from copy import deepcopy

import pytest
from pydantic import ValidationError

from scientisttwo.live.controller import LiveController
from scientisttwo.live.models import LiveSpec
from scientisttwo.store import IntegrityError

from test_live import Agents, setup


EDITORIAL_REASON = "Clarify that this synthetic result does not isolate a causal mechanism."


class EditorialAgents(Agents):
    """Choose review routing deliberately, rather than fabricate empirical success."""

    def __init__(self, decisions, *, unchanged=False, invented_claim=False, omit_concerns=False):
        super().__init__()
        self.decisions = iter(decisions)
        self.unchanged = unchanged
        self.invented_claim = invented_claim
        self.omit_concerns = omit_concerns

    def respond(self, role, inputs):
        if role == "meta":
            self.seen.append((role, deepcopy(inputs)))
            return {"decision": next(self.decisions), "reason": EDITORIAL_REASON}
        if role == "draft" and "manuscript_revision" in inputs:
            self.seen.append((role, deepcopy(inputs)))
            prior = deepcopy(inputs["manuscript_revision"]["previous_draft"])
            if not self.unchanged:
                prior["limitations"].append(EDITORIAL_REASON)
            if self.invented_claim:
                prior["claims"][0]["value"] += 100
            if self.omit_concerns:
                prior["concern_responses"] = []
            return prior
        return super().respond(role, inputs)


def campaign(tmp_path, backend, **settings):
    limits = {
        "search_rounds": 1,
        "success_target": 1,
        "ablation_refinements": 0,
        "review_rounds": 0,
    }
    limits.update(settings)
    return setup(tmp_path, backend, **limits)


def editorial_inputs(backend):
    return [
        inputs
        for role, inputs in backend.seen
        if role == "draft" and "manuscript_revision" in inputs
    ]


def operations(store):
    return [
        tuple(row)
        for row in store.db.execute(
            "SELECT key,kind,status,result_ref FROM live_operations ORDER BY rowid"
        )
    ]


@pytest.mark.parametrize("method_allowance", [0, 1])
def test_editorial_feedback_reaches_writer_without_changing_method_or_retraining(
    tmp_path, method_allowance
):
    original, original_store, original_backend, original_runner = campaign(
        tmp_path / "original", EditorialAgents(["Accept"]), meta_refinements=method_allowance
    )
    corrected, corrected_store, corrected_backend, corrected_runner = campaign(
        tmp_path / "corrected",
        EditorialAgents(["ReviseManuscript", "Accept"]),
        meta_refinements=method_allowance,
    )
    try:
        assert original.run()["status"] == corrected.run()["status"] == "complete"
        inputs = editorial_inputs(corrected_backend)
        assert len(inputs) == 1
        previous = inputs[0]["manuscript_revision"]["previous_draft"]
        first_meta = next(value for role, value in corrected_backend.seen if role == "meta")
        assert previous == first_meta["draft"]
        assert inputs[0]["manuscript_revision"]["feedback"]["meta"] == {
            "decision": "ReviseManuscript",
            "reason": EDITORIAL_REASON,
        }
        assert inputs[0]["manuscript_revision"]["feedback"]["review"] == first_meta["review"]
        assert inputs[0]["evidence"] == first_meta["evidence"]
        assert (
            corrected_store.get("live_frozen")["best"] == original_store.get("live_frozen")["best"]
        )
        assert corrected_runner.executions == original_runner.executions
        assert corrected_runner.finals == original_runner.finals == ["baseline", "selected"]
        extra_roles = Counter(role for role, _ in corrected_backend.seen)
        extra_roles.subtract(role for role, _ in original_backend.seen)
        assert +extra_roles == Counter(draft=1, review=1, meta=1)
        final_draft = corrected_store.json(corrected_store.get("live_draft"))
        assert EDITORIAL_REASON in final_draft["limitations"]
        assert final_draft["claims"] == previous["claims"]
        assert "123" not in str(corrected_backend.seen)  # protected final scores stay excluded
        completed = operations(corrected_store)
        calls = len(corrected_backend.seen)
        assert corrected.run()["status"] == "complete"
        assert operations(corrected_store) == completed
        assert len(corrected_backend.seen) == calls
    finally:
        original_store.close()
        corrected_store.close()


@pytest.mark.parametrize("allowance", [0, 1, 2])
def test_editorial_limit_stops_without_rerouting_to_method_changes(tmp_path, allowance):
    backend = EditorialAgents(["ReviseManuscript"] * (allowance + 1))
    controller, store, _, runner = campaign(tmp_path, backend, manuscript_revisions=allowance)
    try:
        result = controller.run()
        assert result["status"] == "blocked"
        assert result["outcome"] == "manuscript_revision_limit"
        assert len(editorial_inputs(backend)) == allowance
        assert not any(role == "refine" for role, _ in backend.seen)
        assert not any("manuscript" in key or "meta" in key for key in runner.executions)
        assert not store.get("live_frozen") and not runner.finals
        assert store.get("live_best")["result"]["value"] == 8
        old_operations = operations(store)
        old_calls = len(backend.seen)
        assert controller.run() == result
        assert operations(store) == old_operations
        assert len(backend.seen) == old_calls
    finally:
        store.close()


@pytest.mark.parametrize("completed_role", ["draft", "review", "meta"])
def test_editorial_resume_reuses_committed_calls_and_experiments(tmp_path, completed_role):
    backend = EditorialAgents(["ReviseManuscript", "Accept"])
    controller, store, _, runner = campaign(tmp_path, backend)
    original_ask = controller.ask
    interrupted = False

    def interrupt_after_commit(key, role, *args, **kwargs):
        nonlocal interrupted
        result = original_ask(key, role, *args, **kwargs)
        if key.endswith(f"/manuscript/0/{completed_role}") and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return result

    controller.ask = interrupt_after_commit
    try:
        with pytest.raises(KeyboardInterrupt):
            controller.run()
        recorded = operations(store)
        measured = list(runner.executions)
        calls = list(backend.seen)
        resumed = LiveController(
            store, controller.config, controller.adapter, backend, runner, budget=controller.budget
        )
        assert resumed.run()["status"] == "complete"
        assert operations(store)[: len(recorded)] == recorded
        assert runner.executions == measured
        assert backend.seen[: len(calls)] == calls
        assert len(editorial_inputs(backend)) == 1
        assert Counter(role for role, _ in backend.seen)["meta"] == 2
        completed = operations(store)
        assert resumed.run()["status"] == "complete"
        assert operations(store) == completed
    finally:
        store.close()


def test_editorial_rewrite_cannot_add_unsupported_empirical_claim(tmp_path):
    backend = EditorialAgents(["ReviseManuscript"], invented_claim=True)
    controller, store, _, runner = campaign(tmp_path, backend)
    try:
        with pytest.raises(IntegrityError, match="Unsupported manuscript claim"):
            controller.run()
        assert not store.get("live_frozen") and not runner.finals
        assert not any(role == "refine" for role, _ in backend.seen)
        assert not any("manuscript" in key for key in runner.executions)
    finally:
        store.close()


def test_editorial_rewrite_keeps_supplementary_concern_coverage_required(tmp_path):
    backend = EditorialAgents(["ReviseManuscript"], omit_concerns=True)
    controller, store, _, runner = campaign(tmp_path, backend, review_rounds=1)
    try:
        result = controller.run()
        assert result["status"] == "blocked" and result["outcome"] == "invalid_agent_output"
        assert "concern" in result["reason"].lower()
        assert editorial_inputs(backend)[0]["supplementary_coverage"]
        assert not store.get("live_frozen") and not runner.finals
    finally:
        store.close()


def test_unchanged_draft_does_not_satisfy_manuscript_revision(tmp_path):
    backend = EditorialAgents(["ReviseManuscript"], unchanged=True)
    controller, store, _, runner = campaign(tmp_path, backend)
    try:
        result = controller.run()
        assert result["status"] == "blocked" and result["outcome"] == "invalid_agent_output"
        assert not store.get("live_frozen") and not runner.finals
        assert not any(role == "refine" for role, _ in backend.seen)
    finally:
        store.close()


def test_method_revision_still_retests_and_rebuilds_downstream_evidence(tmp_path):
    backend = EditorialAgents(["ReviseMethod", "ReviseManuscript", "Accept"])
    controller, store, _, runner = campaign(tmp_path, backend, manuscript_revisions=1)
    try:
        assert controller.run()["status"] == "complete"
        assert Counter(role for role, _ in backend.seen)["refine"] == 1
        assert "meta/0" in runner.executions
        assert any(key.startswith("method/1/ablation/") for key in runner.executions)
        assert store.get("live_frozen")["best"]["result"]["value"] == 6
        assert len(editorial_inputs(backend)) == 1
        assert editorial_inputs(backend)[0]["incumbent"]["result"]["value"] == 6
        assert not any("manuscript" in key for key in runner.executions)
    finally:
        store.close()


def test_comment_only_method_request_is_rejected_before_retraining(tmp_path):
    """Replay the v4 routing failure structurally, without using its measured scores."""

    class CommentOnlyMethod(EditorialAgents):
        def respond(self, role, inputs):
            if role == "implement" and "parent" in inputs:
                self.seen.append((role, deepcopy(inputs)))
                self.executions_before_patch = list(self.runner.executions)
                source = inputs["parent"]["files"][0]["content"]
                return {
                    "parent_source_ref": inputs["parent"]["parent_source_ref"],
                    "edits": [
                        {
                            "path": "train.py",
                            "before": source,
                            "after": source + "\n# Clarify that the result is synthetic.\n",
                            "reason": "Clarify manuscript interpretation in a source comment",
                        }
                    ],
                    "description": "Synthetic comment-only patch",
                }
            return super().respond(role, inputs)

    backend = CommentOnlyMethod(["ReviseMethod"])
    controller, store, _, runner = campaign(tmp_path, backend)
    backend.runner = runner
    try:
        result = controller.run()
        assert result["status"] == "blocked" and result["outcome"] == "invalid_agent_output"
        assert not any(key.startswith("meta/") for key in runner.executions)
        assert runner.executions == backend.executions_before_patch
        assert store.get("live_best")["result"]["value"] == 8
        assert not runner.finals and not store.get("live_frozen")
    finally:
        store.close()


def test_manuscript_allowance_is_campaign_wide_across_method_revisions(tmp_path):
    backend = EditorialAgents(["ReviseManuscript", "ReviseMethod", "ReviseManuscript"])
    controller, store, _, runner = campaign(tmp_path, backend, manuscript_revisions=1)
    try:
        result = controller.run()
        assert result["status"] == "blocked" and result["outcome"] == "manuscript_revision_limit"
        assert len(editorial_inputs(backend)) == 1
        assert Counter(role for role, _ in backend.seen)["refine"] == 1
        assert store.get("live_best")["result"]["value"] == 6
        assert not runner.finals and not store.get("live_frozen")
    finally:
        store.close()


def test_live_meta_review_requires_explicit_routing_and_bounded_editorial_allowance(tmp_path):
    from scientisttwo.live.models import LiveMetaReview

    for decision in ("Accept", "ReviseManuscript", "ReviseMethod"):
        assert LiveMetaReview(decision=decision, reason="Synthetic").decision == decision
    with pytest.raises(ValidationError):
        LiveMetaReview(decision="Refine", reason="Ambiguous legacy routing")
    controller, store, _, _ = campaign(tmp_path, EditorialAgents(["Accept"]))
    try:
        assert controller.config.manuscript_revisions == 1
        for value in (-1, 3):
            invalid = controller.config.model_dump()
            invalid["manuscript_revisions"] = value
            with pytest.raises(ValidationError):
                LiveSpec.model_validate(invalid)
    finally:
        store.close()
