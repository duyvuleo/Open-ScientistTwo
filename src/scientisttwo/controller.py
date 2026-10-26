"""General ScientistTwo state machine. Task-specific policy lives in TaskAdapter."""

from dataclasses import dataclass
import time

from .backends import AgentBackend
from .models import (
    AgentResult,
    CampaignState,
    CodeProposal,
    CriticDecision,
    EvalSpec,
    ExperimentResult,
    Idea,
    IdeaBatch,
    Limitations,
    ManuscriptDraft,
    MetaReview,
    Novelty,
    Plans,
    Profile,
    Refinement,
    Review,
    Selection,
    Stage,
    TaskSpec,
)
from .runner import BudgetExhausted, PrerequisiteError, evaluator_identity
from .store import IntegrityError, Store
from .tasks import TaskAdapter


ALLOWED = {
    Stage.CREATED: {Stage.LIMITATIONS},
    Stage.LIMITATIONS: {Stage.BASELINE, Stage.COMPLETE},
    Stage.BASELINE: {Stage.SEARCH, Stage.COMPLETE},
    Stage.SEARCH: {Stage.ABLATION, Stage.COMPLETE},
    Stage.ABLATION: {Stage.DRAFT, Stage.COMPLETE},
    Stage.DRAFT: {Stage.REVIEW, Stage.COMPLETE},
    Stage.REVIEW: {Stage.META, Stage.COMPLETE},
    Stage.META: {Stage.ABLATION, Stage.FINAL, Stage.COMPLETE},
    Stage.FINAL: {Stage.COMPLETE},
    Stage.COMPLETE: set(),
}


@dataclass
class Trial:
    idea: Idea
    result: ExperimentResult
    decision: CriticDecision

    def payload(self):
        return {
            "idea": self.idea.model_dump(mode="json"),
            "result": self.result.model_dump(mode="json"),
            "decision": self.decision.model_dump(mode="json"),
        }


class Controller:
    def __init__(
        self,
        store: Store,
        task: TaskSpec,
        profile: Profile,
        adapter: TaskAdapter,
        backend: AgentBackend,
        runner,
        *,
        literature=None,
        writer=None,
    ):
        self.store, self.task, self.profile = store, task, profile
        self.adapter, self.backend, self.runner = adapter, backend, runner
        if hasattr(adapter, "configure"):
            adapter.configure(task)
        from .literature import LocalLiterature
        from .reporting import EvidenceWriter

        self.literature = (
            literature if literature is not None else LocalLiterature(adapter.sources())
        )
        self.writer = writer if writer is not None else EvidenceWriter()
        if not profile.live_calls and not backend.synthetic:
            raise ValueError("This profile forbids live model calls")
        self.traces: list[Trial] = []
        self.best: Trial | None = None
        self.baselines: dict[str, ExperimentResult] = {}

    def check_budget(self):
        deadline = self.store.get("deadline")
        if deadline is not None and time.time() >= deadline:
            raise BudgetExhausted("Campaign wall-clock budget exhausted")

    def transition(self, key, stage, reason):
        if self.store.db.execute("SELECT 1 FROM events WHERE key=?", (key,)).fetchone():
            return
        state = self.store.state()
        if stage not in ALLOWED[state.stage] and stage != state.stage:
            raise IntegrityError(f"Illegal transition: {state.stage} -> {stage}")
        # Commit event and state together, so resume cannot observe half a transition.
        from .store import canonical

        state.stage, state.reason, state.status = stage, reason, "running"
        with self.store.db:
            self.store.db.execute(
                "INSERT INTO events VALUES(?,?,?,?)", (key, time.time(), stage, reason)
            )
            self.store.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES('state',?)", (canonical(state).decode(),)
            )

    def ask(self, key, role, inputs, schema):
        request = {
            "role": role,
            "inputs": inputs,
            "schema": schema.model_json_schema(),
            "backend": self.backend.name,
            "model": getattr(self.backend, "model", "scripted-fixture-v1"),
            "prompt_version": "scientisttwo-v1",
        }
        cached = self.store.cached("agent/" + key, request)
        if cached is None:
            self.check_budget()
            result = self.backend.invoke(role, inputs, schema)
            if result.synthetic != self.backend.synthetic:
                raise IntegrityError("Agent provenance mismatch")
            schema.model_validate(result.payload)
            cached = self.store.cache("agent/" + key, request, result)
        return schema.model_validate(AgentResult.model_validate(cached).payload)

    def experiment(self, key, idea, phase, spec=None):
        spec = spec or self.adapter.evaluation(self.task, phase)
        from .splits import bind_spec

        spec = bind_spec(spec, self.store.json(self.store.get("split_manifest_ref")))
        proposal = self.ask(
            key + "/code",
            "implement",
            {"idea": idea.model_dump(), "spec": spec.model_dump()},
            CodeProposal,
        )
        return self.runner.execute(key, idea, proposal.source, spec)

    def baseline(self, phase):
        idea = self.adapter.baseline()
        result = None
        for attempt in range(self.profile.engineering_rounds + 1):
            result = self.experiment(f"baseline/{phase}/{attempt}", idea, phase)
            if result.status == "success":
                self.baselines[phase] = result
                return result
        return result

    def record(self, key, trial):
        self.traces.append(trial)
        self.store.cache("trial/" + key, {"key": key}, trial.payload())
        self.store.event("decision/" + key, self.store.state().stage, trial.decision.reason)

    def evaluate_idea(self, key, idea):
        trial = None
        for phase in ("subset", "full"):
            for attempt in range(self.profile.engineering_rounds + 1):
                step = f"{key}/{phase}/{attempt}"
                result = self.experiment(step, idea, phase)
                comparison = self.adapter.compare(result, self.baselines[phase], self.task)
                decision = self.ask(
                    step + "/critic",
                    "critic",
                    {
                        "idea": idea.model_dump(),
                        "result": result.model_dump(),
                        "comparison": comparison.model_dump(),
                        "can_engineer": attempt < self.profile.engineering_rounds,
                    },
                    CriticDecision,
                )
                if decision.decision == "Good" and not (
                    comparison.valid and comparison.improved and comparison.guardrails_pass
                ):
                    decision = CriticDecision(
                        decision="Bad",
                        reason="Controller veto: invalid evidence or unmet comparison policy",
                    )
                if decision.decision == "Engineer" and attempt == self.profile.engineering_rounds:
                    decision = CriticDecision(
                        decision="Bad", reason="Engineering budget exhausted; " + decision.reason
                    )
                trial = Trial(idea, result, decision)
                self.record(step, trial)
                if decision.decision == "Good":
                    break
                if decision.decision == "Bad":
                    return None
                revised = self.ask(
                    step + "/engineer",
                    "engineer",
                    {
                        "idea": idea.model_dump(),
                        "feedback": decision.model_dump(),
                        "result": result.model_dump(),
                        "new_id": f"{idea.id}-eng-{phase}-{attempt}",
                    },
                    Idea,
                )
                idea = self.validate_child(revised, [idea.id])
        return trial

    @staticmethod
    def validate_child(idea, parents):
        if idea.id in parents or not set(parents) <= set(idea.parent_ids):
            raise IntegrityError("Refined idea must have a new ID and preserve parent lineage")
        return idea

    def set_best(self, trial):
        self.best = trial
        self.store.set("best", trial.payload())
        state = self.store.state()
        state.best_idea_id = trial.idea.id
        self.store.set("state", state)

    def search(self, seeds):
        successful = []
        for round_index in range(self.profile.search_rounds):
            if round_index == 0:
                candidates = seeds[:2]
            else:
                traces = [t.payload() for t in self.traces]
                evolved = self.ask(
                    f"search/{round_index}/evolve",
                    "evolve",
                    {"traces": traces, "new_id": f"evolved-{round_index}"},
                    Idea,
                )
                known = {t.idea.id for t in self.traces}
                if (
                    not evolved.parent_ids
                    or not set(evolved.parent_ids) <= known
                    or evolved.id in known
                ):
                    raise IntegrityError("Evolution must reference actual historic trials")
                candidates = seeds[round_index + 1 : round_index + 2] + [evolved]
            for index, idea in enumerate(candidates):
                trial = self.evaluate_idea(f"search/{round_index}/{index}", idea)
                if trial is not None:
                    successful.append(trial)
                if len(successful) >= self.profile.success_target:
                    break
            if len(successful) >= self.profile.success_target:
                break
        if not successful:
            return None
        selection = self.ask(
            "select",
            "select",
            {"candidates": [t.payload() for t in successful], "task": self.task.model_dump()},
            Selection,
        )
        selected = next((t for t in successful if t.idea.id == selection.idea_id), None)
        if selected is None:
            raise IntegrityError("Selector chose a candidate without valid full-benchmark evidence")
        self.set_best(selected)
        return selected

    def refine_full(self, key, incumbent, feedback):
        revised = self.ask(
            key + "/idea",
            "refine",
            {
                "idea": incumbent.idea.model_dump(),
                "feedback": feedback,
                "new_id": incumbent.idea.id + "-" + key.replace("/", "-"),
            },
            Idea,
        )
        self.validate_child(revised, [incumbent.idea.id])
        result = self.experiment(key + "/experiment", revised, "full")
        comparison = self.adapter.compare(result, incumbent.result, self.task)
        decision = CriticDecision(
            decision="Good" if comparison.improved else "Bad",
            reason="Revision comparison: " + comparison.reason,
        )
        trial = Trial(revised, result, decision)
        self.record(key, trial)
        if comparison.valid and comparison.improved and comparison.guardrails_pass:
            self.set_best(trial)
            return trial, True
        return incumbent, False

    def ablate(self, prefix, best):
        evidence = []
        for cycle in range(self.profile.ablation_rounds + 1):
            key = f"{prefix}/ablation/{cycle}"
            plans = self.ask(
                key + "/plans",
                "plan_ablations",
                {"idea": best.idea.model_dump(), "new_id": best.idea.id + f"-ablate-{cycle}"},
                Plans,
            )
            if len(plans.plans) > 8:
                raise ValueError("At most eight ablations per cycle")
            comparisons = []
            for index, plan in enumerate(plans.plans):
                self.adapter.validate_plan(best.idea, plan)
                result = self.experiment(f"{key}/{index}", plan.idea, "full")
                comparison = self.adapter.compare(result, best.result, self.task)
                evidence.append(result)
                comparisons.append(comparison.model_dump())
            critique = self.ask(
                key + "/critic",
                "ablation_critic",
                {
                    "idea": best.idea.model_dump(),
                    "comparisons": comparisons,
                    "results": [r.model_dump() for r in evidence],
                },
                Refinement,
            )
            if critique.decision == "Good" or cycle == self.profile.ablation_rounds:
                break
            best, improved = self.refine_full(key + "/refine", best, critique.model_dump())
            if not improved:
                break
        return best, evidence

    def write_and_review(self, prefix, best, ablations):
        self.transition(prefix + "/draft", Stage.DRAFT, "Draft from executed development evidence")
        evidence = [self.baselines["full"], best.result, *ablations]
        draft = None
        review = None
        for cycle in range(self.profile.peer_rounds + 1):
            key = f"{prefix}/review/{cycle}"
            draft = self.ask(
                key + "/draft",
                "draft",
                {"idea": best.idea.model_dump(), "evidence": [r.model_dump() for r in evidence]},
                ManuscriptDraft,
            )
            from .reporting import validate_claims

            validate_claims(draft, evidence)
            self.transition(
                prefix + "/review",
                Stage.REVIEW,
                "Review draft and plan bounded supplementary experiments",
            )
            review = self.ask(
                key + "/review",
                "review",
                {"draft": draft.model_dump(), "has_rebuttal": cycle > 0},
                Review,
            )
            if review.score >= self.profile.review_threshold or cycle == self.profile.peer_rounds:
                break
            plans = self.ask(
                key + "/plans",
                "plan_rebuttal",
                {
                    "idea": best.idea.model_dump(),
                    "review": review.model_dump(),
                    "new_id": best.idea.id + f"-rebuttal-{cycle}",
                },
                Plans,
            )
            if len(plans.plans) > 8:
                raise ValueError("At most eight rebuttal experiments per review")
            for index, plan in enumerate(plans.plans):
                self.adapter.validate_plan(best.idea, plan)
                spec = self.adapter.evaluation(self.task, "rebuttal")
                # Supplementary repeats use fresh DEVELOPMENT seeds, never final-test observations.
                spec = spec.model_copy(
                    update={
                        "seeds": [
                            max(self.task.seeds) + 1 + cycle * 2,
                            max(self.task.seeds) + 2 + cycle * 2,
                        ]
                    }
                )
                evidence.append(
                    self.experiment(f"{key}/rebuttal/{index}", plan.idea, "rebuttal", spec)
                )
        self.store.set("draft_ref", self.store.put_json(draft, "draft"))
        self.store.set("review_ref", self.store.put_json(review, "review"))
        return draft, review

    def run(self):
        state = self.store.state()
        if state.status == "complete":
            errors = self.store.verify()
            if errors:
                raise IntegrityError(errors[0])
            if not self.store.get("report_ref"):
                self.writer.write(self.store, self.adapter)
            return state
        self.traces, self.baselines, self.best = [], {}, None
        try:
            self.runner.preflight()
            errors = self.store.verify()
            if errors:
                raise IntegrityError(errors[0])
            if (
                self.store.get("evaluator") != evaluator_identity()
                or self.store.get("adapter_identity") != self.adapter.identity()
            ):
                raise IntegrityError("Evaluator or task adapter changed; create a new campaign")
            if self.store.get("deadline") is None:
                self.store.set("deadline", time.time() + self.profile.campaign_seconds)
            self.transition(
                "limitations", Stage.LIMITATIONS, "Extract and verify actionable limitations"
            )
            limitation = None
            for round_index in range(self.profile.limitation_rounds):
                limitation = self.ask(
                    f"limitations/{round_index}/extract",
                    "extract_limitations",
                    {
                        "task": self.task.model_dump(),
                        "previous": limitation.model_dump() if limitation else None,
                    },
                    Limitations,
                )
                limitation = self.ask(
                    f"limitations/{round_index}/verify",
                    "verify_limitations",
                    {"limitations": limitation.model_dump()},
                    Limitations,
                )
                if limitation.sufficient:
                    break
            if not limitation.sufficient:
                return self.finish(
                    "insufficient_limitations", "Limitation-verification budget exhausted"
                )
            batch = self.ask(
                "seeds",
                "seed_ideas",
                {
                    "task": self.task.model_dump(),
                    "limitations": limitation.model_dump(),
                    "count": self.profile.seed_ideas,
                },
                IdeaBatch,
            )
            if len(batch.ideas) != self.profile.seed_ideas or len(
                {i.id for i in batch.ideas}
            ) != len(batch.ideas):
                raise IntegrityError("Seed generator returned an incorrect count or duplicate IDs")
            seeds = []
            for idea in batch.ideas:
                sources = self.store.cached("sources/" + idea.id, {"query": idea.hypothesis})
                if sources is None:
                    self.check_budget()
                    sources = self.store.cache(
                        "sources/" + idea.id,
                        {"query": idea.hypothesis},
                        self.literature.search(idea.hypothesis, limit=2),
                    )
                novelty = self.ask(
                    "novelty/" + idea.id,
                    "novelty",
                    {"idea": idea.model_dump(), "sources": sources},
                    Novelty,
                )
                if not set(novelty.references) <= {s["id"] for s in sources}:
                    raise IntegrityError("Novelty judgment cites a source that was not retrieved")
                seeds.append(
                    idea.model_copy(
                        update={"novelty": novelty.score, "references": novelty.references}
                    )
                )
            seeds.sort(key=lambda idea: (-idea.novelty, idea.id))
            self.transition(
                "baseline", Stage.BASELINE, "Reproduce fixed baselines under matching protocols"
            )
            for phase in ("subset", "full"):
                if self.baseline(phase).status != "success":
                    return self.finish(
                        "baseline_failed", "No valid baseline; scientific comparison unavailable"
                    )
            self.transition(
                "search",
                Stage.SEARCH,
                "Screen seeds, scale validated ideas, and evolve from traces",
            )
            best = self.search(seeds)
            if best is None:
                return self.finish(
                    "no_improvement", "No candidate passed the full development comparison"
                )
            for meta_round in range(self.profile.meta_rounds + 1):
                prefix = f"cycle/{meta_round}"
                self.transition(
                    prefix + "/ablation",
                    Stage.ABLATION,
                    "Test component attribution and bounded method revision",
                )
                best, ablations = self.ablate(prefix, best)
                self.set_best(best)
                draft, review = self.write_and_review(prefix, best, ablations)
                self.transition(
                    prefix + "/meta",
                    Stage.META,
                    "Assess whether deeper method refinement is needed",
                )
                meta = self.ask(
                    prefix + "/meta",
                    "meta_review",
                    {"draft": draft.model_dump(), "review": review.model_dump()},
                    MetaReview,
                )
                self.store.set("meta_review", meta.model_dump())
                if meta.decision == "Accept" or meta_round == self.profile.meta_rounds:
                    break
                revised, improved = self.refine_full(
                    prefix + "/meta-refine", best, meta.model_dump()
                )
                if not improved:
                    break
                best = revised
            self.transition(
                "final", Stage.FINAL, "Freeze candidate and perform one final-test evaluation"
            )
            from .splits import bind_spec

            final_spec = bind_spec(
                self.adapter.evaluation(self.task, "final"),
                self.store.json(self.store.get("split_manifest_ref")),
            )
            freeze = {
                "idea": best.idea.model_dump(),
                "code_ref": best.result.code_ref,
                "development_result": best.result.model_dump(),
                "spec": final_spec.model_dump(),
            }
            self.store.set("freeze_ref", self.store.put_json(freeze, "frozen-final-plan"))
            spec = EvalSpec.model_validate(freeze["spec"])
            for label, idea, source in (
                (
                    "baseline",
                    self.adapter.baseline(),
                    self.store.read(self.baselines["full"].code_ref).decode(),
                ),
                ("candidate", best.idea, self.store.read(best.result.code_ref).decode()),
            ):
                result = self.runner.execute("final/" + label, idea, source, spec)
                self.store.set("final_" + label, result.model_dump())
            final_candidate = ExperimentResult.model_validate(self.store.get("final_candidate"))
            final_baseline = ExperimentResult.model_validate(self.store.get("final_baseline"))
            final_comparison = self.adapter.compare(final_candidate, final_baseline, self.task)
            self.store.set("final_comparison", final_comparison.model_dump())
            if not final_comparison.valid:
                return self.finish(
                    "final_evaluation_failed",
                    "Development candidate retained; final evaluation unavailable",
                )
            if not final_comparison.improved:
                return self.finish(
                    "development_only",
                    "Development improvement did not meet the final-test comparison rule",
                )
            return self.finish(
                "candidate_supported",
                "Candidate met development and final-test comparison rules on this fixture",
            )
        except BudgetExhausted as error:
            return self.finish("budget_exhausted", str(error))
        except PrerequisiteError as error:
            state = self.store.state()
            state.status, state.reason = "blocked", str(error)
            self.store.set("state", state)
            return state
        except KeyboardInterrupt:
            state = self.store.state()
            state.status, state.reason = (
                "interrupted",
                "Interrupted; resume reuses committed evidence",
            )
            self.store.set("state", state)
            raise

    def finish(self, outcome, reason):
        self.transition("complete", Stage.COMPLETE, reason)
        state = self.store.state()
        state.status, state.outcome = "complete", outcome
        self.store.set("state", state)
        self.writer.write(self.store, self.adapter)
        return state


def initialize(store, task, profile, adapter):
    if store.get("state") is not None:
        raise ValueError("Campaign already exists; use resume")
    if task.schema_version != 2:
        raise ValueError(
            "New campaigns require TaskSpec schema_version=2 with fixed task-level partitions"
        )
    if hasattr(adapter, "configure"):
        adapter.configure(task)
    store.set("split_manifest_ref", store.put_json(adapter.split_manifest(task), "split-manifest"))
    store.set("task_ref", store.put_json(task, "task-spec"))
    store.set("profile_ref", store.put_json(profile, "profile"))
    store.set("evaluator", evaluator_identity())
    store.set("adapter_identity", adapter.identity())
    from pathlib import Path

    store.set(
        "implementation_ref",
        store.put_json(
            {
                path.name: store.put(path.read_bytes(), "py", "implementation-source")
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "implementation-manifest",
        ),
    )
    store.set("state", CampaignState(run_id=store.root.name))
