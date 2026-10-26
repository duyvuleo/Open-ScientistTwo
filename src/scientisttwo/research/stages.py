"""Scientific role handoffs for the shared research-v2 engine."""
import math

from ..live.models import Candidate, Hypothesis
from ..live.output_contracts import AgentOutputError
from ..live.parent_edits import (ParentChangePlan, apply_parent_patch, ParentEditError,
                                 validate_executable_parent_change)
from ..live.source_tree import candidate_snapshot, compare_source_trees
from ..live.runtime import LiveBudgetExhausted
from ..store import canonical, IntegrityError
from .contracts import (BaselinePlan, FullSetHandoff, SelectionReview, AblationDesigns,
                        ControlReview, MethodDisposition)
from .literature import LiteratureConfig, NoveltyAssessment, retrieve_literature, validate_novelty
from .diagnostic_stages import DiagnosticStages


ROLE_INSTRUCTIONS = {
    "initial_idea": "Propose one testable seed hypothesis addressing the verified limitations. Use supplied documents only for source_ids. Novelty is independently assessed later; do not certify originality.",
    "additional_idea": "Propose one distinct hypothesis addressing the verified limitations and differences from existing ideas. Respond to prior novelty criticism without inventing references. Novelty is independently assessed.",
    "novelty_checker": "Independently compare the idea with the supplied retrieved passages. Identify overlap and specific distinction, with uncertainty and limitations. Use only bundle source IDs. No global novelty certification; metadata-only records are not fully read papers.",
    "baseline_plan": "Identify the permitted baseline preparation mode and its provenance. Prepared references are immutable. Reconstruction is allowed only when explicitly offered by the task; incomplete starters are not reproduced scientific baselines.",
    "baseline_coder": "Reconstruct the declared baseline within supplied approved files, dependencies and task constraints. Return reproducible candidate code, not model-written scores. Preserve evaluation and data protocols.",
    "baseline_engineer": "Repair the supplied reconstructed baseline using actual execution diagnostics. Preserve the declared reference method, data and scoring protocol; edit only task-permitted source. Do not replace it with an improved research candidate or fabricate scores.",
    "full_set_coder": "Inspect the exact screened snapshot and full-development contract. Return ReuseSnapshot when no edit is needed. A ScalingPatch may change only task-declared scaling paths. ScientificChange requires renewed subset screening; never disguise a method change as scaling.",
    "selection_review": "Explain selection among the offered eligible full-development results. Choose only an eligible ID under the controller-owned comparison policy. Consider recorded methodology critiques and limitations; never override failed constraints.",
    "ablation_planner": "Plan component removal tests, without code. Name the scientific question, changed files, invariants and known confounds. Multiple files may implement one component. Do not equate file isolation with causal isolation.",
    "ablation_coder": "Implement only the declared ablation against the exact supplied parent. Return unique anchored edits; preserve other bytes, command, initialization and protocol unless explicitly in the plan. Never fabricate measurements.",
    "control_reviewer": "Inspect the declared question and materialized source difference before execution. Accept a supported control, Reject an invalid design, or DiagnosticOnly for a useful but confounded comparison. Identify changes to active capacity, initialization, normalization and compute. This is a scientific judgment, not proof of causal isolation.",
    "method_disposition": "Determine whether feedback justifies a new executable method. Return NoMethodChange if only editorial changes or no defensible method change is warranted; do not force a source comment or formatting edit. Otherwise return ChangeMethod with a concrete rationale.",
    "initial_drafter": "Write a complete research manuscript in the canonical schema using the evidence ledger and source passages. Include actual negative/inconclusive results and limitations. Numerical claims and literature assertions need the provided evidence identifiers; no new experiments or fabricated measurements.",
    "writing_critic": "Review the complete manuscript for factual grounding, citation support, missing sections and clarity. Do not invent empirical evidence. Distinguish mechanically verified facts from scientific interpretation.",
    "draft_enhancer": "Revise the complete previous manuscript using supplied writing/peer/meta feedback and current evidence only. Preserve experimental facts; fix or remove unsupported claims. Do not request code execution or invent new results.",
    "paperorchestra_model": "Execute the supplied original PaperOrchestra writing or review prompt using only the supplied evidence and media. Return its complete requested output as text in output. Do not invent measurements, sources or completed experiments. Follow the controller's explicit evidence and revision constraints; do not run tools or experiments.",
    "paperorchestra_bindings": "Index the exact supplied PaperOrchestra manuscript into the requested structured evidence binding schema. Bind the exact source hash, literal sections and claim quotations; do not rewrite the manuscript. Use only supplied evidence/source IDs. Unsupported claims must fail validation, never be concealed or reclassified to make them pass.",
    "scientific_review": "Independently review the submitted manuscript and evidence. Use a 1–10 scale for the overall score and every rubric rating and explicitly return score_scale=10; 8/10 is the early-stop threshold. Do not use a five-point scale or silently convert a prior score. Judge the declared task scope, confidence and stable actionable concerns. Distinguish execution, responses and scientific resolution. Do not invent experiments or concern IDs merely because a score is low. High scores cannot excuse failed factual checks. Review acceptance is internal only.",
    "rebuttal_coder": "Implement the accepted supplementary control through exact-parent anchored edits within its declared file scope. The controller owns exact seeds, scoring and diagnostics.",
    "rebuttal_planner": "Map each review concern to an exact supported supplementary task or an explicit unaddressed disposition. Plan seed count, metrics/diagnostics, repeat or one-file paired control and permitted changes, but no code. Unsupported requests remain unfulfilled; prose cannot request hidden actions.",
}


class ResearchAdmissionDeferred(RuntimeError):
    """An optional block was not admitted so existing evidence can be closed out."""


class ResearchStages(DiagnosticStages):
    def scientific_history(self):
        return [{"id": trial["id"], "idea": trial["idea"],
                 "decision": trial["decision"], "execution_status": trial["result"]["status"],
                 "metrics": trial["result"].get("metrics") or {trial["result"]["metric"]: trial["result"]["value"]},
                 "protocol_hash": trial["result"]["protocol_hash"],
                 "source_tree_ref": trial["result"].get("source_tree_ref")}
                for trial in self.trials]
    def closing_calls(self):
        # A manuscript pass (including its configured writing refinements), an
        # independent scientific review, and meta-review. Conditional rebuttals
        # and editorial loops remain bounded separately, never promised here.
        return self.writing_calls() + 2

    def writing_calls(self):
        if self.config.workflow.writer_backend == "paperorchestra":
            # Every inner upstream call, binding response and independent writing
            # critic consumes the same campaign ledger and admission allowance.
            return ((self.config.workflow.paperorchestra_max_calls + 2)
                    * (1 + self.config.workflow.writing_refinements))
        return 2 + 2 * self.config.workflow.writing_refinements

    def ablation_calls(self):
        return 2 + 2 * self.config.workflow.max_ablation_plans

    def admit_block(self, key, *, model_calls, experiments=0, closing=False, optional=False):
        """Conservative admission estimate; not a hard token or runtime guarantee."""
        deferred = self.store.get("research-deferred/" + key)
        if deferred:
            value = self.store.json(deferred)
            if not any(item["key"] == key for item in self.research_deferrals):
                self.research_deferrals.append(value)
            raise ResearchAdmissionDeferred(value["reason"])
        prior = self.store.get("research-admission/" + key)
        if prior:
            return self.store.json(prior)
        used = self.budget.summary()
        calls = sum(row["kind"] == "model" for row in used)
        tokens = sum(row["tokens"] for row in used)
        exp = sum(row["kind"] == "experiment" for row in used)
        fraction = 0 if closing else self.config.workflow.closing_fraction
        reserve_calls = max(math.ceil(self.config.max_calls * fraction),
                            self.closing_calls() if not closing else 0)
        token_limit = self.config.token_stop_threshold
        reserve_tokens = max(math.ceil((token_limit or 0) * fraction),
                             reserve_calls * self.config.workflow.estimated_tokens_per_call)
        # Writing and rebuttal happen before protected final evaluation too.
        # They may consume reserved model calls but not its experiment slots.
        reserve_experiments = 1 if closing and self.best is None else 2
        estimate = model_calls * self.config.workflow.estimated_tokens_per_call
        # Actual deadline exhaustion is never an optional deferral.
        remaining = self.budget.remaining()
        value = {"key": key, "required_model_calls": model_calls, "required_experiments": experiments,
                 "estimated_tokens": estimate, "reserved_closing_calls": reserve_calls,
                 "reserved_closing_tokens": reserve_tokens,
                 "reserved_final_experiments": reserve_experiments,
                 "scope": "Admission estimates only; every actual operation remains separately charged and bounded"}
        if (calls + model_calls + reserve_calls > self.config.max_calls
                or exp + experiments + reserve_experiments > self.config.max_experiments
                or token_limit and tokens + estimate + reserve_tokens > token_limit
                or remaining <= self.config.campaign_seconds * fraction):
            reason = "Insufficient remaining resources for block and reserved closing work: " + key
            if not optional:
                raise LiveBudgetExhausted(reason)
            value.update(status="deferred", reason=reason, used_model_calls=calls,
                         used_experiments=exp, observed_tokens=tokens,
                         disposition="Preserve measured incumbent/reference and continue available closing work; this block was not completed")
            ref = self.store.put_json(value, "research-budget-deferral")
            self.store.set("research-deferred/" + key, ref)
            self.research_deferrals.append(value)
            self.store.event("research-deferred/" + key,
                             self.store.get("live_state", {}).get("stage", "created"), reason)
            raise ResearchAdmissionDeferred(reason)
        value["status"] = "admitted"
        self.store.set("research-admission/" + key, self.store.put_json(value, "research-admission"))
        return value

    def research_novelty(self, key, idea):
        self.budget.remaining()
        local = [*self.sources, *getattr(self.adapter, "sources", lambda: [])()]
        # Stable local source set prevents a later retrieval from changing an earlier replay.
        original = self.store.get("research_local_sources")
        if original is None:
            original = self.store.put_json(local, "literature-input")
            self.store.set("research_local_sources", original)
        local = self.store.json(original)
        bundle = retrieve_literature(self.store, LiteratureConfig.model_validate(self.config.workflow.literature),
                                     idea.hypothesis, local,
                                     allow_network=self.allow_network_literature and self.config.workflow.network_literature)
        def validate(value):
            try:
                return validate_novelty(value, bundle)
            except ValueError as error:
                raise AgentOutputError(str(error)) from error
        assessment = self.ask(key, "novelty_checker", {
            "idea": {k: v for k, v in idea.model_dump().items() if k not in {"novelty_score", "novelty_scope"}},
            "literature": bundle,
        }, NoveltyAssessment, validator=validate)
        verification = validate_novelty(assessment, bundle)
        result = verification.model_dump(mode="json")
        self.store.set("novelty/" + key, self.store.put_json(result, "novelty-assessment"))
        for source in bundle["sources"]:
            if not any(s["id"] == source["id"] for s in self.sources):
                self.sources.append(source)
        scope = "Independent assessment against retrieved sources; " + result["effective_status"]
        return idea.model_copy(update={"novelty_score": assessment.score, "novelty_scope": scope}), result

    def research_seeds(self, limitations, count):
        ideas = []
        for index in range(count):
            previous = None
            for attempt in range(self.config.workflow.novelty_attempts):
                key = f"seed/{index}/{attempt}"
                role = "initial_idea" if index == 0 and attempt == 0 else "additional_idea"
                idea = self.ask(key + "/generate", role, {
                    "index": index, "attempt": attempt, "count": count,
                    "limitations": limitations.model_dump(), "task": self.adapter.context(),
                    "sources": self.sources, "existing_ideas": [v.model_dump() for v in ideas],
                    "previous_assessment": previous,
                }, Hypothesis)
                if any(v.title == idea.title for v in ideas):
                    raise AgentOutputError("Seed titles must be distinct")
                assessed, previous = self.research_novelty(key + "/novelty", idea)
                # Evidence insufficiency is recorded, not converted to certified novelty.
                if previous["effective_status"] == "verified_scope" or attempt + 1 == self.config.workflow.novelty_attempts:
                    ideas.append(assessed)
                    break
        return sorted(ideas, key=lambda v: (-v.novelty_score, v.title))

    def research_baseline(self):
        baseline = self.adapter.baseline()
        mode = self.config.workflow.baseline_mode
        allowed = self.adapter.context().get("baseline_modes", ["prepared_reference"])
        if mode not in allowed:
            raise ValueError("Task does not authorize the requested baseline preparation mode")
        plan = self.ask("baseline/plan", "baseline_plan", {
            "required_mode": mode, "allowed_modes": allowed, "task": self.adapter.context(),
            "sources": self.sources, "original_starter": baseline.model_dump(),
        }, BaselinePlan)
        if plan.mode != mode or not set(plan.source_ids) <= {s["id"] for s in self.sources}:
            raise AgentOutputError("Baseline plan changes the frozen mode or cites unknown sources")
        original = candidate_snapshot(self.store, baseline)
        if mode == "reconstruct_from_sources":
            baseline = self.ask("baseline/reconstruct", "baseline_coder", {
                "plan": plan.model_dump(), "task": self.adapter.context(),
                "reference": baseline.model_dump(), "sources": self.sources,
            }, Candidate)
            self.adapter.validate(baseline)
        self.store.set("baseline_provenance", self.store.put_json({
            "plan": plan.model_dump(), "original_source_ref": self.store.put_json(original),
            "executable_reference": baseline.model_dump(), "scientific_reproduction": "unverified_until_executed",
        }, "baseline-provenance"))
        return baseline

    def research_handoff(self, key, trial):
        parent, ref, options, context = self.parent_input(trial)
        handoff = self.ask(key, "full_set_coder", {"parent": context,
            "subset_result": trial["result"], "full_contract": self.adapter.context(),
            "scaling_paths": self.adapter.context().get("scaling_paths", []),
        }, FullSetHandoff)
        if handoff.action == "ReuseSnapshot":
            self.store.set("full-handoff/" + key, self.store.put_json(handoff, "full-stage-handoff"))
            return parent, None
        if handoff.action == "ScalingPatch" and not set(handoff.change_plan.changed_files) <= set(self.adapter.context().get("scaling_paths", [])):
            raise AgentOutputError("Scaling edits exceed task-declared scaling paths; classify scientific changes explicitly")
        try:
            candidate, proof = apply_parent_patch(self.store, parent, ref, handoff.patch,
                declared_paths=handoff.change_plan.changed_files, **options)
            if handoff.action == "ScientificChange":
                proof["executable_change_check"] = validate_executable_parent_change(
                    self.store, ref, proof["candidate_source_ref"])
        except ParentEditError as error:
            raise AgentOutputError(str(error)) from error
        self.adapter.validate(candidate)
        if handoff.revised_idea:
            permitted = {s["id"] for s in self.sources}
            offered = {t["id"] for t in self.trials}
            if (not set(handoff.revised_idea.source_ids) <= permitted
                    or not set(handoff.revised_idea.evidence_ids) <= offered):
                raise AgentOutputError("Full-stage hypothesis cites unoffered source or experimental evidence")
        self.store.set("full-handoff/" + key, self.store.put_json({
            "handoff": handoff.model_dump(), "source_proof": proof,
            "subset_recheck_required": handoff.action == "ScientificChange",
        }, "full-stage-handoff"))
        return candidate, handoff.revised_idea

    def research_baseline_result(self, base, phase):
        """Prepared references stay immutable; reconstructed baselines may be repaired."""
        last = None
        for attempt in range(self.config.engineering_retries + 1):
            key = f"baseline/{phase}" if attempt == 0 else f"baseline/{phase}/repair/{attempt}"
            last = self.runner.execute(key, base, phase)
            self.store.set("baseline-attempt/" + key, self.store.put_json(last, "baseline-attempt"))
            if last.status == "success":
                break
            if (self.config.workflow.baseline_mode != "reconstruct_from_sources"
                    or attempt == self.config.engineering_retries):
                break
            self.admit_block(key + "/repair-admission", model_calls=1, experiments=1)
            revised = self.ask(key + "/repair-code", "baseline_engineer", {
                "candidate": base.model_dump(), "feedback": self.feedback(last),
                "task": self.adapter.context(), "sources": self.sources,
            }, Candidate)
            self.adapter.validate(revised)
            materialize = getattr(self.adapter, "materialize", lambda value: value)
            before = self.store.put_json(candidate_snapshot(self.store, materialize(base)))
            after = self.store.put_json(candidate_snapshot(self.store, materialize(revised)))
            paths = getattr(self.adapter, "editable_paths", None)
            allowed = paths() if callable(paths) else self.adapter.context().get("editable_files", [v.path for v in base.files])
            if revised.command != base.command or not set(compare_source_trees(self.store, before, after)) <= set(allowed):
                raise AgentOutputError("Baseline engineering exceeded the declared source/command scope")
            base = revised
        return base, last

    def research_align_baseline(self, base):
        """A repaired full reference must also own the screening measurements.

        Only source bytes, executable metadata and argv determine alignment;
        descriptions and proposal provenance do not require another experiment.
        This is a single verification run, never another engineering loop.
        """
        subset, full = self.baselines["subset"], self.baselines["full"]
        if not subset.source_tree_ref or not full.source_tree_ref:
            raise IntegrityError("Baseline alignment requires both executed source snapshots")
        materialize = getattr(self.adapter, "materialize", lambda value: value)
        target = candidate_snapshot(self.store, materialize(base))
        target_ref = self.store.put_json(target, "baseline-alignment-source")
        full_tree = self.store.json(full.source_tree_ref)
        if (compare_source_trees(self.store, full.source_tree_ref, target_ref)
                or full_tree["command"] != target.command):
            raise IntegrityError("Final baseline proposal differs from its full-development execution")
        subset_tree = self.store.json(subset.source_tree_ref)
        changes = compare_source_trees(self.store, subset.source_tree_ref, full.source_tree_ref)
        command_changed = subset_tree["command"] != full_tree["command"]
        key = "baseline/subset/aligned-with-full"
        proof = {
            "original_subset_id": subset.id, "original_subset_ref": self.store.put_json(subset),
            "full_id": full.id, "full_ref": self.store.put_json(full),
            "original_subset_source_ref": subset.source_tree_ref,
            "full_source_ref": full.source_tree_ref, "target_source_ref": target_ref,
            "changed_paths": changes, "command_changed": command_changed,
            "scope": "One common executed reference implementation across subset and full development; scientific baseline fidelity remains a review judgment",
        }

        def record(status, **updates):
            proof.update(status=status, **updates)
            self.store.set("baseline_alignment", self.store.put_json(proof, "baseline-alignment"))

        if not changes and not command_changed:
            record("not_required")
            return True
        record("pending", alignment_key=key)
        self.store.event("baseline/alignment", "baseline",
                         "Full-development engineering changed the reference; verify its subset before search")
        self.admit_block(key + "/admission", model_calls=0, experiments=1)
        aligned = self.runner.execute(key, base, "subset")
        self.store.set("baseline-attempt/" + key, self.store.put_json(aligned, "baseline-attempt"))
        updates = {"aligned_subset_id": aligned.id,
                   "aligned_subset_ref": self.store.put_json(aligned),
                   "aligned_subset_source_ref": aligned.source_tree_ref}
        if aligned.status != "success":
            record("failed", reason="Final reference failed subset alignment; no additional baseline repair authorized", **updates)
            return False
        if (not aligned.source_tree_ref
                or compare_source_trees(self.store, aligned.source_tree_ref, full.source_tree_ref)
                or self.store.json(aligned.source_tree_ref)["command"] != full_tree["command"]
                or aligned.protocol_hash != subset.protocol_hash):
            record("failed", reason="Subset alignment changed the final source, command or screening protocol", **updates)
            return False
        self.baselines["subset"] = aligned
        record("pass", **updates)
        return True

    def research_selection(self, successful, best):
        from ..live.models import Observation
        eligible = []
        for trial in successful:
            comparison = self.comparison(Observation.model_validate(trial["result"]),
                                         Observation.model_validate(best["result"]))
            if comparison["valid"] and abs(comparison.get("delta", 0)) <= self.config.development.epsilon:
                eligible.append(trial["id"])
        if best["id"] not in eligible:
            eligible.append(best["id"])
        selection = self.ask("selection/review", "selection_review", {
            "candidates": successful, "eligible_ids": eligible,
            "comparison_policy": self.config.development.model_dump(),
            "evidence_package": self.evidence_package([v["result"] for v in successful], trials=self.trials),
        }, SelectionReview)
        if selection.selected_id not in eligible:
            raise AgentOutputError("Selector chose an invalid or numerically inferior candidate")
        chosen = next(v for v in successful if v["id"] == selection.selected_id)
        self.store.set("selection_review", self.store.put_json(selection, "selection-review"))
        self.set_best(chosen)
        return chosen

    def research_ablations(self, key, best):
        self.admit_block(key + "/planning", model_calls=self.ablation_calls(),
                         experiments=1 + self.config.workflow.max_ablation_plans, optional=True)
        plans = self.ask(key + "/plans", "ablation_planner", {
            "incumbent": best, "max_plans": self.config.workflow.max_ablation_plans,
            "task": self.adapter.context(),
            "evidence_package": self.evidence_package([best["result"]], trials=self.trials),
        }, AblationDesigns)
        if len(plans.plans) > self.config.workflow.max_ablation_plans:
            raise AgentOutputError("Ablation plan limit exceeded")
        evidence = []
        matched = None
        for index, plan in enumerate(plans.plans):
            change = ParentChangePlan(changed_files=plan.changed_files, reason=plan.question)
            _, _, _, parent = self.parent_input(best)
            if not set(change.changed_files) <= set(parent["editable_paths"]):
                raise AgentOutputError("Ablation plan exceeds editable parent files")
            idea = Hypothesis.model_validate(best["idea"])
            candidate = self.implement_change(f"{key}/{index}/code", "ablation_coder", idea,
                                              best, change, ablation_plan=plan.model_dump())
            materialize = getattr(self.adapter, "materialize", lambda v: v)
            before = self.store.put_json(candidate_snapshot(self.store, materialize(Candidate.model_validate(best["candidate"]))))
            after = self.store.put_json(candidate_snapshot(self.store, materialize(candidate)))
            changes = compare_source_trees(self.store, before, after)
            review = self.ask(f"{key}/{index}/design-review", "control_reviewer", {
                "plan": plan.model_dump(), "parent": parent, "candidate": candidate.model_dump(),
                "source_changes": changes, "effective_execution": self.evidence_package([best["result"]]),
            }, ControlReview)
            proof = {"plan": plan.model_dump(), "review": review.model_dump(),
                     "source_changes": changes, "semantic_isolation": "review_judgment_not_certified"}
            self.store.set(f"ablation-design/{key}/{index}", self.store.put_json(proof, "ablation-design-review"))
            if review.decision == "Reject":
                continue
            self.admit_block(f"{key}/{index}", model_calls=1,
                             experiments=2 if matched is None else 1)
            if matched is None:
                matched = self.runner.execute(key + "/matched", Candidate.model_validate(best["candidate"]), "full")
                evidence.append(matched)
            control = self.runner.execute(f"{key}/{index}/control", candidate, "full")
            evidence.append(control)
            self.store.cache(f"control/{key}/{index}", {"plan": plan.model_dump()}, {
                **proof, "matched": matched.model_dump(), "control": control.model_dump(),
                "shared_matched_execution": matched.id,
            })
        return evidence

    def research_method_disposition(self, key, best, feedback):
        # Include the method assessment and regenerated ablations, not merely
        # training. A declined admission must not spend the closing reserve on
        # planning a method that cannot be evaluated and written up.
        self.admit_block(key, model_calls=5 + self.ablation_calls(),
                         experiments=2 + self.config.workflow.max_ablation_plans,
                         optional=True)
        decision = self.ask(key + "/disposition", "method_disposition", {
            "incumbent": best, "feedback": feedback,
            "evidence_package": self.evidence_package([best["result"], *feedback.get("evidence", [])], trials=self.trials),
        }, MethodDisposition)
        self.store.set("method-disposition/" + key, self.store.put_json(decision, "method-disposition"))
        return decision.action == "ChangeMethod"

    def research_supplementary_plans(self, key, best, inputs, seeds, metrics):
        from .contracts import SupplementaryDesigns
        from ..live.supplementary import SupplementaryPlan, SupplementaryPlans, negotiate
        original = Candidate.model_validate(best["candidate"])
        from ..live.supplementary import Concern
        concerns = [Concern.model_validate(c) for c in inputs["concerns"]]

        def checked(designs):
            initial = SupplementaryPlans(plans=[SupplementaryPlan(
                **{k: v for k, v in plan.model_dump().items() if k != "change_plan"},
                changed_file=plan.change_plan.changed_files[0] if plan.change_plan else None,
                candidate=original,
            ) for plan in designs.plans], unaddressed_concerns=designs.unaddressed_concerns)
            decision = negotiate(initial, concerns, seeds, metrics,
                                 supported_diagnostics=self.research_diagnostic_capabilities())
            if not decision.contract_valid:
                raise AgentOutputError("; ".join(decision.contract_errors))
            return initial, decision

        designs = self.ask(key + "/plans", "rebuttal_planner", inputs, SupplementaryDesigns,
                           validator=lambda value: checked(value))
        initial, decision = checked(designs)
        plans = []
        for index, (design, resolved) in enumerate(zip(designs.plans, decision.decisions, strict=True)):
            candidate = original
            if resolved.status == "accepted" and design.execution_kind == "paired_control":
                self.admit_block(f"{key}/{index}/code", model_calls=1, experiments=2, closing=True)
                candidate = self.implement_change(f"{key}/{index}/code", "rebuttal_coder",
                    Hypothesis.model_validate(best["idea"]), best, design.change_plan,
                    supplementary_plan=design.model_dump())
            plans.append(initial.plans[index].model_copy(update={"candidate": candidate}))
        output = SupplementaryPlans(plans=plans, unaddressed_concerns=designs.unaddressed_concerns)
        self.store.set("research-supplementary-plan/" + key, self.store.put_json(output, "compiled-supplementary-plan"))
        return output

    def research_draft(self, key, best, evidence, *, manuscript_revision=None):
        from .manuscript import (ManuscriptRecord, build_evidence_ledger, validate_manuscript,
                                 legacy_draft_view, render_manuscript, CompilationPolicy,
                                 freeze_research_history, writer_history_contract)
        from .reviews import WritingCriticReview, validate_review_references
        from .compiler import IsolatedCompiler
        from .writer_comparisons import include_phase_baselines, register_writer_comparisons
        self.admit_block(key + "/writing", model_calls=self.writing_calls(),
                         closing=True)
        evidence = include_phase_baselines(evidence, self.baselines)
        history_request = {"schema_version": "research-history-context-v1"}
        history_snapshot = self.store.cached("writer-history/" + key, history_request)
        if history_snapshot is None:
            history_snapshot = {**history_request, "trial_refs": freeze_research_history(self.store, self.trials)}
            self.store.cache("writer-history/" + key, history_request, history_snapshot)
        if history_snapshot.get("schema_version") != history_request["schema_version"]:
            raise IntegrityError("Unsupported frozen writer-history context")
        ledger = build_evidence_ledger(self.store, evidence, self.sources,
            diagnostics=self.research_diagnostics,
            history_refs=history_snapshot["trial_refs"],
            method_facts={"selected_source": {
                "text": "Selected method source and effective execution settings are recorded; mechanism attribution remains unverified.",
                "evidence_ref": self.store.put_json({
                    "candidate_ref": self.store.put_json(best["candidate"]),
                    "source_tree_ref": best["result"].get("source_tree_ref"),
                    "evidence": self.evidence_package([best["result"]]),
                }, "method-evidence")}})
        if self.config.workflow.writer_backend == "paperorchestra":
            from ..integrations.paperorchestra.writer_context import context_method_fact
            methods = {key.removeprefix("method:"): {name: value for name, value in entry.items()
                       if name not in {"kind", "semantic_support"}}
                       for key, entry in ledger["entries"].items() if entry.get("kind") == "method"}
            methods["writer_context"] = context_method_fact(self.store, ledger)
            ledger = build_evidence_ledger(self.store, evidence, self.sources,
                diagnostics=self.research_diagnostics, history_refs=history_snapshot["trial_refs"],
                method_facts=methods)
        ledger, comparisons = register_writer_comparisons(self.store, ledger, self.baselines)
        comparison_ref = self.store.put_json(comparisons, "writer-comparison-provenance")
        self.store.set("writer-comparisons/" + key, comparison_ref)
        ledger_ref = self.store.put_json(ledger, "manuscript-ledger")
        self.store.set("manuscript-ledger/" + key, ledger_ref)
        previous = None
        if manuscript_revision:
            ref = manuscript_revision["previous_draft"].get("manuscript_ref")
            if not ref:
                raise IntegrityError("Editorial revision requires the exact previous full manuscript")
            previous = self.store.json(ref)
        inputs = {"incumbent": best, "ledger": ledger, "sources": self.sources,
                  "search_outcome": "selected_improvement" if self.best is not None else "no_approved_improvement",
                  "scientific_history": list(ledger["research_history"]["entries"].values()),
                  "writer_evidence_contract": writer_history_contract(ledger),
                  "writer_comparison_contract": {name: value for name, value in comparisons.items() if name != "entries"},
                  "supplementary_coverage": self.supplementary_coverage(),
                  "supplementary_actions": self.supplementary_context(),
                  "synthetic_agents": self.backend.synthetic,
                  "deferred_research_work": self.research_deferrals}
        if previous is not None:
            inputs.update(previous_manuscript=previous, manuscript_revision=manuscript_revision)

        def validate(value):
            result = validate_manuscript(value, ledger, self.sources)
            if result.status != "pass":
                raise AgentOutputError("; ".join(result.errors))
            if value.synthetic_agents != self.backend.synthetic:
                raise AgentOutputError("Manuscript synthetic-agent declaration differs from the actual backend")
            view = legacy_draft_view(value, ledger)
            self.validate_concern_responses(view)
            if previous is not None and canonical(value.model_dump()) == canonical(previous):
                raise AgentOutputError("Manuscript revision returned unchanged content")

        role = "draft_enhancer" if previous is not None else "initial_drafter"
        paperorchestra = self.config.workflow.writer_backend == "paperorchestra"
        rendered = None

        def write(pass_key, role, context):
            nonlocal rendered
            if not paperorchestra:
                return self.ask(pass_key, role, context, ManuscriptRecord, validator=validate)
            from ..integrations.paperorchestra.controller_writer import write_controller_manuscript
            result = write_controller_manuscript(self, pass_key, context, ledger,
                previous_manuscript=context.get("previous_manuscript"),
                feedback={name: context[name] for name in ("writing_feedback", "manuscript_revision")
                          if name in context})
            validate(result.manuscript)
            rendered = result.rendered
            return result.manuscript

        manuscript = write(key + "/manuscript", role, inputs)
        for attempt in range(self.config.workflow.writing_refinements + 1):
            critique = self.ask(f"{key}/writing-critic/{attempt}", "writing_critic", {
                "manuscript": manuscript.model_dump(), "ledger": ledger,
                "sources": self.sources, "pass_index": attempt,
                "mechanical_checks": validate_manuscript(manuscript, ledger, self.sources).model_dump(),
                **self.authoritative_writer_context(rendered),
            }, WritingCriticReview)
            errors = validate_review_references(critique, evidence_ids=set(ledger["entries"]),
                                                claim_ids={c.id for c in manuscript.claims})
            if errors:
                raise AgentOutputError("; ".join(errors))
            self.store.set("writing-review/" + key, self.store.put_json(critique, "writing-critique"))
            if critique.decision == "Accept" or attempt == self.config.workflow.writing_refinements:
                break
            previous = manuscript.model_dump()
            manuscript = write(f"{key}/enhance/{attempt}", "draft_enhancer", {
                **inputs, "previous_manuscript": previous, "writing_feedback": critique.model_dump(),
            })
        ref = self.store.put_json(manuscript, "research-manuscript")
        self.store.set("writing-review-for/" + ref, self.store.put_json(critique, "writing-critique"))
        compiler = None
        options = {}
        if self.config.workflow.compiler_image:
            compiler = IsolatedCompiler(self.runner, self.config.workflow.compiler_image)
            options["compilation_policy"] = CompilationPolicy(image_digest=self.config.workflow.compiler_image)
        from pathlib import Path
        if not paperorchestra:
            rendered = render_manuscript(self.store, manuscript, ledger, self.sources,
                template_path=self.config.workflow.template_directory or Path(__file__).parent / "templates/iclr2025",
                compiler=compiler, **options)
        self.store.set("manuscript/" + key, ref)
        self.store.set("research_manuscript", ref)
        self.store.set("research_manuscript_bundle", self.store.put_json(rendered, "rendered-manuscript"))
        self.store.set("research_manuscript_validation", self.store.put_json(
            validate_manuscript(manuscript, ledger, self.sources), "manuscript-validation"))
        if self.config.workflow.require_pdf and rendered.compilation.status != "success":
            raise ValueError("Required isolated manuscript PDF compilation did not succeed")
        draft = legacy_draft_view(manuscript, ledger).model_copy(update={"manuscript_ref": ref})
        self.store.set("live_draft", self.store.put_json(draft, "structured-manuscript"))
        return draft

    def authoritative_writer_context(self, rendered):
        """Give every reviewer the actual submitted source, not just its index."""
        if rendered is None or self.config.workflow.writer_backend != "paperorchestra":
            return {}
        from ..integrations.paperorchestra.manuscript_binding import verify_bound_manifest
        if hasattr(rendered, "model_dump"):
            rendered = rendered.model_dump(mode="json")
        manifest = self.store.json(rendered["manifest_ref"])
        verify_bound_manifest(self.store, manifest,
                              expected_image_digest=self.config.workflow.compiler_image)
        return {"authoritative_manuscript": {
            "writer": "paperorchestra", "latex_ref": rendered["latex_ref"],
            "latex": self.store.read(rendered["latex_ref"]).decode(),
            "pdf_ref": rendered["pdf_ref"], "manifest_ref": rendered["manifest_ref"],
            "binding": manifest["writer"],
            "instruction": "Review this actual manuscript; the structured manuscript is its evidence index. "
                           "A valid index does not establish that all prose is supported."}}

    def research_review(self, key, draft, evidence, *, manuscript_revision=None):
        from .manuscript import ManuscriptRecord, validate_manuscript
        from .reviews import ScientificReview, review_gate, validate_review_references, validate_live_review_scale
        from ..live.models import LiveReview
        manuscript = ManuscriptRecord.model_validate(self.store.json(draft.manuscript_ref))
        bundle = self.store.json(self.store.get("research_manuscript_bundle"))
        manifest = self.store.json(bundle["manifest_ref"])
        if manifest["manuscript_ref"] != draft.manuscript_ref:
            raise IntegrityError("Review manuscript differs from the submitted bundle")
        ledger = self.store.json(manifest["ledger_ref"])
        prior = [c for r in self.scientific_reviews for c in r.concerns]
        writing_ref = self.store.get("writing-review-for/" + draft.manuscript_ref)
        from .reviews import WritingCriticReview
        writing = WritingCriticReview.model_validate(self.store.json(writing_ref)) if writing_ref else None
        if writing is not None and writing.decision != "Accept":
            prior.extend(writing.concerns)
        latest = {c.id: c for c in prior}
        checks = validate_manuscript(manuscript, ledger, self.sources)
        def validate_review(value):
            errors = validate_live_review_scale(value) + validate_review_references(
                value, evidence_ids=set(ledger["entries"]), claim_ids={c.id for c in manuscript.claims},
                prior_concerns=list(latest.values()))
            if errors:
                raise AgentOutputError("; ".join(errors))

        review = self.ask(key, "scientific_review", {
            "score_contract": {"minimum": 1, "maximum": 10, "score_scale": 10,
                "early_stop_threshold": 8, "acceptance_scope": "internal_workflow_only",
                "instruction": "Use the same 1–10 scale for all ratings. The criterion is faithful evidence and the declared task scope, not invented novelty. A low score is not an executable experiment request; report only real, stable concerns."},
            "manuscript": manuscript.model_dump(), "ledger": ledger, "sources": self.sources,
            "scientific_history": self.scientific_history(),
            "search_outcome": "selected_improvement" if self.best is not None else "no_approved_improvement",
            "writing_review": writing.model_dump() if writing else None,
            "previous_concerns": [c.model_dump() for c in latest.values()],
            "mechanical_checks": checks.model_dump(), "action_contract": self.supplementary_contract(),
            "supplementary_coverage": self.supplementary_coverage(),
            **self.authoritative_writer_context(bundle),
            **({"manuscript_revision": manuscript_revision} if manuscript_revision else {}),
        }, ScientificReview, validator=validate_review)
        gate = review_gate(review, integrity_passed=checks.status == "pass",
                           previous_concerns=list(latest.values()))
        self.scientific_reviews.append(review)
        ref = self.store.put_json(review, "scientific-review")
        self.store.set("research_review/" + key, ref)
        self.store.set("research_review_gate", gate.model_dump())
        result = LiveReview(score=review.score if gate.accepted else min(review.score, 7),
            concerns=[c.text for c in review.concerns if c.resolution != "resolved"] or ([] if gate.accepted else gate.reasons),
            review_ref=ref)
        self.store.set("live_review", self.store.put_json(result, "live-review"))
        return result

    def research_final_appendix(self, results):
        from .manuscript import final_results_appendix, CompilationPolicy
        from .compiler import IsolatedCompiler
        ref = self.store.get("research_manuscript_bundle")
        if not ref:
            return
        compiler, options = None, {}
        if self.config.workflow.compiler_image:
            compiler = IsolatedCompiler(self.runner, self.config.workflow.compiler_image)
            options["compilation_policy"] = CompilationPolicy(image_digest=self.config.workflow.compiler_image)
        final = final_results_appendix(self.store, self.store.json(ref), results,
                                      compiler=compiler, **options)
        self.store.set("research_final_bundle", self.store.put_json(final, "final-manuscript-bundle"))
        if self.config.workflow.require_pdf and final.compilation.status != "success":
            raise ValueError("Required final appendix PDF compilation did not succeed")
