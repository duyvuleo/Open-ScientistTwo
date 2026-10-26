"""Synthetic review contract traces; no live model or container execution."""
from copy import deepcopy

from scientisttwo.research.reviews import (
    ScientificReview, fixture_review, validate_live_review_scale,
)
from test_research_workflow import campaign


def test_legacy_unspecified_scale_is_preserved_and_never_rescaled():
    raw = fixture_review(score=4.3).model_dump()
    raw.pop("score_scale")
    raw["summary"] = "Synthetic legacy opinion on a five-point scale."
    before = deepcopy(raw)
    review = ScientificReview.model_validate(raw)
    assert review.score == 4.3 and review.score_scale is None
    assert validate_live_review_scale(review)
    assert raw == before
    assert not validate_live_review_scale(fixture_review(score=8.6))


def test_controller_requests_explicit_scale_correction_without_new_science(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path)
    controller.config.agent_output_retries = 1
    original = agents.respond
    seen = []

    def respond(role, inputs):
        value = original(role, inputs)
        if role == "scientific_review":
            assert inputs["score_contract"]["score_scale"] == 10
            assert inputs["score_contract"]["early_stop_threshold"] == 8
            assert set(inputs["reference_contract"]["evidence_ids"]) == set(inputs["ledger"]["entries"])
            count = store.db.execute("SELECT count(*) FROM live_operations WHERE kind='experiment'").fetchone()[0]
            seen.append((deepcopy(inputs), count))
            if len(seen) == 1:
                value.update(score=4.3, score_scale=None, summary="Synthetic legacy five-point response.")
            else:
                value.update(score=8.6, score_scale=10, summary="New synthetic assessment on the requested 1–10 scale.")
        return value

    agents.responder = respond
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        assert len(seen) == 2 and seen[0][1] == seen[1][1]
        assert "score_scale=10" in seen[1][0]["output_correction"]["errors"][0]
        rows = store.db.execute("SELECT key,output_ref FROM calls WHERE key LIKE '%/review/0%' ORDER BY rowid").fetchall()
        responses = [store.json(row["output_ref"])["payload"] for row in rows if row["key"].startswith("live-agent/")]
        assert responses[0]["score"] == 4.3 and responses[0]["score_scale"] is None
        assert responses[1]["score"] == 8.6 and responses[1]["score_scale"] == 10
    finally:
        store.close()


def test_honest_low_score_with_empty_typed_concerns_routes_meta_without_fake_work(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path)
    controller.config.review_rounds = 1
    original = agents.respond

    def respond(role, inputs):
        value = original(role, inputs)
        if role == "scientific_review":
            value.update(score=4.3, score_scale=10, concerns=[], decision="Accept",
                         summary="Synthetic low opinion with no actionable scientific concern.")
        return value

    agents.responder = respond
    try:
        state = controller.run()
        assert state["status"] == "blocked" and state["outcome"] == "manuscript_revision_limit", state
        assert "rebuttal_planner" not in [role for role, _ in agents.seen]
        assert not store.db.execute("SELECT 1 FROM live_operations WHERE key LIKE '%rebuttal%'").fetchone()
        routing = store.json(store.get("review-routing/method/0/0"))
        assert routing["score"] == 4.3 and routing["route"] == "meta_review"
        assert not routing["new_experiments"]
        assert store.get("research_review_gate")["accepted"] is False
        meta_inputs = next(inputs for role, inputs in agents.seen if role == "meta")
        assert meta_inputs["review"]["score"] == 4.3
    finally:
        store.close()


def test_unknown_rebuttal_id_gets_bounded_correction_before_coding(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, rebuttal=True)
    controller.config.agent_output_retries = 1
    original = agents.respond
    seen = []

    def respond(role, inputs):
        value = original(role, inputs)
        if role == "rebuttal_planner":
            seen.append(deepcopy(inputs))
            assert inputs["reference_contract"]["concern_ids"] == sorted(c["id"] for c in inputs["concerns"])
            if len(seen) == 1:
                value["plans"][0]["concern_ids"] = ["invented-score-concern"]
        return value

    agents.responder = respond
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        assert len(seen) == 2
        assert "unknown" in seen[1]["output_correction"]["errors"][0].lower()
        assert seen[1]["output_correction"]["rejected_output"]["plans"][0]["concern_ids"] == ["invented-score-concern"]
        assert controller.supplementary_coverage()
    finally:
        store.close()
