"""End-to-end research-v2 contracts on explicitly synthetic staged transports.

These tests execute controller, storage, isolated-stage command construction and
trusted arithmetic over fixture predictions. They run no model/container and
cannot establish real benchmark performance or autonomous research quality.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scientisttwo.live.adapters import implementation_identity
from scientisttwo.live.models import Candidate, DevelopmentSpec, Hypothesis, LiveSpec, Source
from scientisttwo.live.runtime import Budget, DevelopmentRunner
from scientisttwo.research.controller import ResearchController
from scientisttwo.research.fixtures import ResearchFixtures
from scientisttwo.research.manuscript import ManuscriptRecord, verify_evidence_ledger
from scientisttwo.research.policy import ResearchPolicy
from scientisttwo.store import Store

from test_staged_evaluation import StagedAdapter, StagedTransportDouble, make_staged_task


MARKER = "SYNTHETIC_TEST_OFFSET = "


def with_offset(candidate, offset):
    value = Candidate.model_validate(candidate)
    files = []
    for source in value.files:
        if source.path == "custom.py":
            lines = [line for line in source.content.splitlines() if not line.startswith(MARKER)]
            source = source.model_copy(update={"content": "\n".join(lines) + f"\n{MARKER}{offset!r}\n"})
        files.append(source)
    return value.model_copy(update={"files": files, "description": "Synthetic staged contract candidate"})


class ResearchStagedAdapter(StagedAdapter):
    def baseline(self):
        return with_offset(super().baseline(), 1.)

    def context(self):
        return {**super().context(), "baseline_modes": ["prepared_reference", "reconstruct_from_sources"],
                "scaling_paths": []}


class ResearchStagedTransport(StagedTransportDouble):
    """Fake candidate-dependent inference; prediction values are synthetic fixtures."""
    def __call__(self, argv, timeout, **kwargs):
        result = super().__call__(argv, timeout, **kwargs)
        if "run" in argv:
            mounts = {value.rsplit(":", 2)[1]: Path(value.rsplit(":", 2)[0])
                      for index, value in enumerate(argv) if index and argv[index - 1] == "-v"}
            if "/artifacts" in mounts:
                source = (mounts["/candidate"] / "custom.py").read_text()
                offset = float(next(line.removeprefix(MARKER) for line in source.splitlines() if line.startswith(MARKER)))
                path = mounts["/output"] / "predictions.npy"
                np.save(path, np.load(path, allow_pickle=False) + offset)
        return result


class LegacyFixtureRoles:
    def __init__(self, adapter, *, reject=False, refine=False, editorial=False, rebuttal=False):
        self.adapter, self.reject, self.refine, self.editorial = adapter, reject, refine, editorial
        self.rebuttal = rebuttal
        self.seen = []
        self.meta_count = 0

    def __call__(self, role, inputs):
        self.seen.append((role, deepcopy(inputs)))
        if role in {"limitations", "verify_limitations"}:
            return {"limitations": ["Synthetic baseline offset can be changed"], "relevant": True,
                    "reason": "Synthetic scientific-role trace only"}
        if role == "seeds":
            return {"ideas": [Hypothesis(title=f"fixture-seed-{i}", hypothesis="Reduce synthetic prediction offset",
                              novelty_score=0, novelty_scope="Synthetic control; no novelty claim",
                              source_ids=["task"]).model_dump() for i in range(inputs["count"])]}
        if role == "implement":
            if "parent" in inputs:
                before = next(item["content"] for item in inputs["parent"]["files"] if item["path"] == "custom.py")
                after = "\n".join(line for line in before.splitlines() if not line.startswith(MARKER)) + f"\n{MARKER}0.25\n"
                return {"parent_source_ref": inputs["parent"]["parent_source_ref"],
                        "edits": [{"path": "custom.py", "before": before, "after": after,
                                   "reason": "Synthetic method refinement"}], "description": "Synthetic parent edit"}
            return with_offset(self.adapter.baseline(), .5).model_dump()
        if role == "refine":
            return {"idea": Hypothesis(title="ablation-scientific-revision", hypothesis="Refine the synthetic offset",
                    novelty_score=0, novelty_scope="Synthetic; no novelty claim", source_ids=["task"]).model_dump(),
                    "change_plan": {"changed_files": ["custom.py"], "reason": "Synthetic offset refinement"}}
        if role == "critic":
            return {"decision": "Bad" if self.reject else "Good", "reason": "Explicitly scripted fixture decision"}
        if role == "ablations":
            return {"plans": [{"purpose": "Synthetic component removal", "changed_file": "custom.py",
                               "candidate": with_offset(inputs["incumbent"]["candidate"], .9).model_dump()}]}
        if role == "ablation_critic":
            return {"decision": "Refine" if self.refine else "Good", "reason": "Synthetic interpretation"}
        if role == "review":
            if self.rebuttal and not inputs.get("has_rebuttal"):
                return {"score": 7, "concerns": ["Require a synthetic matched repeat"]}
            return {"score": 8, "concerns": []}
        if role == "rebuttal":
            return {"plans": [{"id": "synthetic-repeat", "concern_ids": [item["id"] for item in inputs["concerns"]],
                    "purpose": "Synthetic matched control", "seed_count": 2, "execution_kind": "paired_control",
                    "evaluation_split": "development", "diagnostics": [{"kind": "metric", "metric": "mse"}],
                    "changed_file": "custom.py", "candidate": with_offset(inputs["incumbent"]["candidate"], .9).model_dump()}],
                    "unaddressed_concerns": []}
        if role == "meta":
            self.meta_count += 1
            return {"decision": "ReviseManuscript" if self.editorial and self.meta_count == 1 else "Accept",
                    "reason": "Clarify synthetic evidence scope"}
        raise AssertionError("Unexpected synthetic legacy role: " + role)


class WorkflowFixtures(ResearchFixtures):
    def __init__(self, legacy, *, reject_controls=False, no_change=False, handoff=None, unresolved_writing=False):
        super().__init__(legacy)
        self.reject_controls, self.no_change, self.handoff = reject_controls, no_change, handoff
        self.unresolved_writing = unresolved_writing

    def respond(self, role, inputs):
        if role == "scientific_review" and self.unresolved_writing:
            value = super().respond(role, inputs)
            value["concerns"] = [{**item, "resolution": "unresolved", "resolution_reason": "Synthetic objection remains unresolved"}
                                 for item in inputs["previous_concerns"]]
            return value
        if role == "writing_critic" and self.unresolved_writing:
            self.seen.append((role, deepcopy(inputs)))
            return {"decision": "ReviseManuscript", "summary": "Synthetic unresolved writing objection", "confidence": 1,
                    "concerns": [{"id": "writing-blocker", "category": "claim_support", "severity": "blocking",
                                  "text": "Synthetic unresolved claim-support objection", "requested_action": "revise_manuscript",
                                  "confidence": 1, "evidence_ids": [], "claim_ids": []}]}
        if role == "full_set_coder" and self.handoff:
            self.seen.append((role, deepcopy(inputs)))
            parent = self._parent_candidate(inputs)
            before = next(item.content for item in parent.files if item.path == "custom.py")
            after = (before + "# Purely editorial comment\n" if self.handoff == "comment" else
                     next(item.content for item in with_offset(parent, .25).files if item.path == "custom.py"))
            return {"action": "ScalingPatch" if self.handoff == "scaling" else "ScientificChange",
                    "reason": "Explicit synthetic phase handoff", "change_plan": {
                        "changed_files": ["custom.py"], "reason": "Change synthetic method offset"},
                    "patch": {"parent_source_ref": inputs["parent"]["parent_source_ref"],
                              "edits": [{"path": "custom.py", "before": before, "after": after,
                                         "reason": "Declared synthetic change"}], "description": "Synthetic phase edit"},
                    "revised_idea": None if self.handoff == "scaling" else Hypothesis(
                        title="full-stage-scientific-revision", hypothesis="Further reduce the synthetic method offset",
                        novelty_score=0, novelty_scope="Synthetic fixture; no novelty claim", source_ids=["task"]).model_dump()}
        if role == "control_reviewer" and self.reject_controls:
            self.seen.append((role, deepcopy(inputs)))
            return {"decision": "Reject", "reason": "Synthetic invalid control", "confounds": ["Deliberate test rejection"]}
        if role == "method_disposition" and self.no_change:
            self.seen.append((role, deepcopy(inputs)))
            return {"action": "NoMethodChange", "reason": "No justified scientific change remains"}
        return super().respond(role, inputs)


def campaign(tmp_path, **options):
    native = make_staged_task(tmp_path / "inputs")
    adapter = ResearchStagedAdapter(native)
    config = LiveSpec(
        engine_version="research-v2", mode="research", native=native,
        development=DevelopmentSpec(adapter="staged-fixture", metric="mse", direction="minimize", seeds=[0]),
        workflow=ResearchPolicy(seed_ideas=2, novelty_attempts=1, writing_refinements=options.pop("writing_refinements", 1),
                                baseline_mode=options.pop("baseline_mode", "prepared_reference")),
        model="scripted-fixture-v1", max_calls=200, max_experiments=50, campaign_seconds=600,
        max_prompt_chars=1_000_000, search_rounds=1, success_target=1,
        ablation_refinements=1 if options.get("refine") else 0, review_rounds=1 if options.get("rebuttal") else 0, meta_refinements=0,
        model_retries=0, agent_output_retries=0,
        sources=[Source(id="fixture-paper", title="Synthetic supplied source", text="Fixture literature passage.")],
        implementation=implementation_identity(),
    )
    store = Store(tmp_path / "campaign")
    budget = Budget(store, config)
    transport = ResearchStagedTransport()
    runner = DevelopmentRunner(store, config, adapter, budget, executor=transport)
    legacy = LegacyFixtureRoles(adapter, reject=options.get("reject", False),
                                refine=options.get("refine", False), editorial=options.get("editorial", False),
                                rebuttal=options.get("rebuttal", False))
    agents = WorkflowFixtures(legacy, reject_controls=options.get("reject_controls", False),
                              no_change=options.get("no_change", False), handoff=options.get("handoff"),
                              unresolved_writing=options.get("unresolved_writing", False))
    controller = ResearchController(store, config, adapter, agents, runner, budget=budget)
    return controller, store, runner, agents, transport


def test_research_v2_runs_complete_roles_and_binds_writer_evidence(tmp_path):
    controller, store, runner, agents, transport = campaign(tmp_path)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        assert state["outcome"] == "development_improvement"
        roles = [role for role, _ in agents.seen]
        assert {"limitations", "verify_limitations", "initial_idea", "additional_idea", "novelty_checker",
                "baseline_plan", "implement", "critic", "full_set_coder", "selection_review",
                "ablation_planner", "ablation_coder", "control_reviewer", "ablation_critic",
                "initial_drafter", "writing_critic", "draft_enhancer", "scientific_review", "meta"} <= set(roles)
        assert "draft" not in roles and "review" not in roles and "ablations" not in roles
        assert roles.index("ablation_planner") < roles.index("ablation_coder") < roles.index("control_reviewer")
        assert roles.index("initial_drafter") < roles.index("draft_enhancer") < roles.index("scientific_review")
        manuscript = ManuscriptRecord.model_validate(store.json(store.get("research_manuscript")))
        bundle = store.json(store.get("research_manuscript_bundle"))
        ledger = store.json(bundle["ledger_ref"])
        verify_evidence_ledger(store, ledger)
        assert ledger["includes_final"] is False
        assert manuscript.synthetic_agents and manuscript.claims
        assert all(claim.evidence_ids[0] in ledger["entries"] for claim in manuscript.claims)
        final_draft = store.json(store.get("live_draft"))
        assert final_draft["manuscript_ref"] == bundle["manuscript_ref"]
        checks = store.json(store.get("research_manuscript_validation"))["checks"]
        assert checks["citation_id_resolution"] == "pass"
        assert checks["citation_existence"] == "unverified"
        assert checks["citation_entailment"] == "unverified"
        assert "FINAL_STAGED_LABELS_MUST_NOT_ENTER_REFINEMENT" not in str(agents.seen)
        assert not any("phase': 'final'" in str(inputs) for _, inputs in agents.seen)
        assert store.get("live_final")["selected"]["status"] == "completed"
        assert transport.stages and not store.verify()
        before_operations, before_calls, before_stages = controller.budget.summary(), len(agents.seen), len(transport.stages)
        assert controller.run() == state
        assert controller.budget.summary() == before_operations
        assert len(agents.seen) == before_calls and len(transport.stages) == before_stages
    finally:
        store.close()


def test_baseline_reconstruction_is_separate_from_prepared_provenance(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, baseline_mode="reconstruct_from_sources")
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        assert any(role == "baseline_coder" for role, _ in agents.seen)
        provenance = store.json(store.get("baseline_provenance"))
        assert provenance["plan"]["mode"] == "reconstruct_from_sources"
        assert provenance["scientific_reproduction"] != "verified"
    finally:
        store.close()


def test_rejected_control_is_recorded_without_executing_it(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, reject_controls=True)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        assert any(role == "control_reviewer" for role, _ in agents.seen)
        keys = [row["key"] for row in controller.budget.summary() if row["kind"] == "experiment"]
        assert not any("/control" in key or "/matched" in key for key in keys)
        proof = store.json(store.get("ablation-design/method/0/ablation/0/0"))
        assert proof["review"]["decision"] == "Reject"
    finally:
        store.close()


def test_no_change_disposition_preserves_incumbent_without_revision_experiments(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, refine=True, no_change=True)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        assert any(role == "method_disposition" for role, _ in agents.seen)
        assert not any(role == "refine" for role, _ in agents.seen)
        keys = [row["key"] for row in controller.budget.summary() if row["kind"] == "experiment"]
        assert not any("/revision" in key for key in keys)
        assert store.get("live_best")["id"] == store.get("live_frozen")["best"]["id"]
    finally:
        store.close()


def test_negative_search_has_explicit_outcome_and_no_success_invention(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, reject=True)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        assert state["outcome"] == "no_improvement"
        assert store.get("live_best") is None
        assert store.get("live_frozen")["best"]["id"] == "baseline"
        assert not any(role == "selection_review" for role, _ in agents.seen)
        assert set(store.get("live_final")) == {"baseline"}
        assert store.get("negative_research_outcome")["method_revision_not_started"] is True
        assert store.get("research_manuscript") and store.get("research_manuscript_bundle")
        manuscript = ManuscriptRecord.model_validate(store.json(store.get("research_manuscript")))
        assert manuscript.results.text and manuscript.limitations.text
        roles = [role for role, _ in agents.seen]
        assert "initial_drafter" in roles and "scientific_review" in roles and "meta" in roles
        assert "ablation_planner" not in roles and "refine" not in roles
        bundle = store.json(store.get("research_manuscript_bundle"))
        ledger = store.json(bundle["ledger_ref"])
        verify_evidence_ledger(store, ledger)
        assert any(identifier.startswith("search/") for identifier in ledger["observations"])
        assert ledger["includes_final"] is False
        writer = next(inputs for role, inputs in agents.seen if role == "initial_drafter")
        assert writer["search_outcome"] == "no_approved_improvement"
        assert writer["scientific_history"]
        assert all(item["decision"]["decision"] == "Bad" for item in writer["scientific_history"])
    finally:
        store.close()


@pytest.mark.parametrize("interrupted_role", ["initial_drafter", "draft_enhancer", "scientific_review"])
def test_research_replay_preserves_chronological_writer_context(tmp_path, interrupted_role):
    controller, store, runner, agents, transport = campaign(tmp_path, editorial=True)
    original_ask = controller.ask
    interrupted = False
    def interrupt_after_commit(key, role, *args, **kwargs):
        nonlocal interrupted
        result = original_ask(key, role, *args, **kwargs)
        if role == interrupted_role and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return result
    controller.ask = interrupt_after_commit
    try:
        with pytest.raises(KeyboardInterrupt):
            controller.run()
        before_calls, before_operations = deepcopy(agents.seen), controller.budget.summary()
        resumed = ResearchController(store, controller.config, controller.adapter, agents, runner, budget=controller.budget)
        result = resumed.run()
        assert result["status"] == "complete", result
        assert agents.seen[:len(before_calls)] == before_calls
        assert resumed.budget.summary()[:len(before_operations)] == before_operations
        editorial = [inputs for role, inputs in agents.seen if role == "draft_enhancer" and "manuscript_revision" in inputs]
        assert editorial
        previous = editorial[0]["manuscript_revision"]["previous_draft"]["manuscript_ref"]
        assert editorial[0]["previous_manuscript"] == store.json(previous)
        assert "FINAL_STAGED_LABELS_MUST_NOT_ENTER_REFINEMENT" not in str(agents.seen)
        after, calls, stages = resumed.budget.summary(), len(agents.seen), len(transport.stages)
        assert resumed.run() == result
        assert resumed.budget.summary() == after and len(agents.seen) == calls and len(transport.stages) == stages
    finally:
        store.close()


def test_scientific_full_handoff_rescreens_rechecks_novelty_and_preserves_lineage(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, handoff="scientific")
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        reviewed_titles = [inputs["idea"]["title"] for role, inputs in agents.seen if role == "novelty_checker"]
        assert "full-stage-scientific-revision" in reviewed_titles
        operations = controller.budget.summary()
        assert any(row["key"] == "experiment/search/0/0/full-handoff/rescreen" for row in operations)
        final_trial = store.get("live_best")
        assert "search/0/0/full-handoff/rescreen" in final_trial["parents"]
    finally:
        store.close()


@pytest.mark.parametrize("handoff", ["scaling", "comment"])
def test_invalid_phase_handoff_cannot_start_full_training(tmp_path, handoff):
    controller, store, _, _, _ = campaign(tmp_path, handoff=handoff)
    try:
        state = controller.run()
        assert state["status"] == "blocked" and state["outcome"] == "invalid_agent_output", state
        operations = controller.budget.summary()
        assert not any(row["key"].startswith("experiment/search/0/0/full") for row in operations)
        assert store.get("live_frozen") is None
    finally:
        store.close()


def test_ablation_method_revision_receives_independent_novelty_assessment(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, refine=True)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        reviewed_titles = [inputs["idea"]["title"] for role, inputs in agents.seen if role == "novelty_checker"]
        assert "ablation-scientific-revision" in reviewed_titles
        assert store.get("live_best")["idea"]["title"] == "ablation-scientific-revision"
        assert any(row["key"].startswith("experiment/method/0/ablation/0/revision") for row in controller.budget.summary())
    finally:
        store.close()


def test_rebuttal_executes_exact_seeds_and_preserves_execution_vs_resolution(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, rebuttal=True)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        roles = [role for role, _ in agents.seen]
        assert "rebuttal_planner" in roles and "rebuttal_coder" in roles
        coverage = controller.supplementary_coverage()
        assert coverage and all(item["status"] == "executed" for item in coverage)
        manuscript = ManuscriptRecord.model_validate(store.json(store.get("research_manuscript")))
        assert manuscript.concern_responses and manuscript.concern_responses[0].status == "executed"
        assert "scientific resolution remains unverified" in manuscript.concern_responses[0].explanation
        refs = store.db.execute("SELECT output_ref FROM calls WHERE key LIKE 'development/method/0/rebuttal/%'").fetchall()
        assert refs
        for row in refs:
            result = store.json(row[0])
            assert len(store.json(result["protocol_ref"])["seeds"]) == 2
    finally:
        store.close()


def test_unresolved_writing_critic_is_bound_and_forwarded_to_scientific_review(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, unresolved_writing=True, writing_refinements=0)
    try:
        state = controller.run()
        assert state["status"] == "blocked" and state["outcome"] == "manuscript_revision_limit", state
        scientific_inputs = [inputs for role, inputs in agents.seen if role == "scientific_review"]
        assert len(scientific_inputs) == 2
        for inputs in scientific_inputs:
            assert inputs["writing_review"]["decision"] == "ReviseManuscript"
            assert inputs["writing_review"]["concerns"][0]["id"] == "writing-blocker"
            assert any(item["id"] == "writing-blocker" for item in inputs["previous_concerns"])
            manuscript_ref = store.put_json(inputs["manuscript"])
            bound = store.get("writing-review-for/" + manuscript_ref)
            assert store.json(bound) == inputs["writing_review"]
        assert store.get("research_review_gate")["accepted"] is False
        assert any("writing-blocker" in reason for reason in store.get("research_review_gate")["reasons"])
        assert not store.get("live_frozen") and not store.get("live_final")
        revision = store.json(store.get("manuscript-revision/method/0/manuscript/0"))
        assert revision["no_new_experiments"] is True
        assert not any(row["kind"] == "experiment" and "manuscript" in row["key"] for row in controller.budget.summary())
    finally:
        store.close()


def test_peer_rebuttal_enhances_exact_reviewed_manuscript_with_actual_evidence(tmp_path):
    controller, store, _, agents, _ = campaign(tmp_path, rebuttal=True, writing_refinements=0)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        roles = [role for role, _ in agents.seen]
        assert roles.count("initial_drafter") == 1
        assert roles.count("draft_enhancer") == 1
        enhanced = next(inputs for role, inputs in agents.seen if role == "draft_enhancer")
        revision = enhanced["manuscript_revision"]
        assert revision["kind"] == "peer_rebuttal"
        prior = store.json(revision["previous_draft"]["manuscript_ref"])
        assert enhanced["previous_manuscript"] == prior
        assert revision["feedback"]["review"]["concerns"] == ["Require a synthetic matched repeat"]
        assert enhanced["supplementary_coverage"]
        assert all(item["status"] == "executed" for item in enhanced["supplementary_coverage"])
        assert any("/rebuttal/" in identifier for identifier in enhanced["ledger"]["observations"])
        assert len(store.json(store.get("research_manuscript"))["concern_responses"]) == 1
    finally:
        store.close()
