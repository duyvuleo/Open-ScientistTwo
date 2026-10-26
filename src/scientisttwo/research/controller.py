"""Controller-owned research transitions with replayable, budgeted agent roles."""

from copy import deepcopy
import math
from pathlib import Path
import time

from ..backends import BackendError
from ..runner import PrerequisiteError
from ..models import AgentResult, CriticDecision, Refinement
from ..store import IntegrityError, canonical, digest
from ..live.adapters import implementation_identity
from ..live.models import (
    Candidate,
    ControlPlans,
    Draft,
    LimitationAssessment,
    LiveMetaReview,
    LiveReview,
    Seeds,
)
from ..live.runtime import AmbiguousCall, Budget, LiveBudgetExhausted
from ..live.evaluation import compare_observations, observation_metrics, metric_specs
from ..live.evidence_context import build_evidence_context
from ..live.output_contracts import AgentOutputError, applies, invocation_scope, validate_output
from ..live.parent_edits import (
    MethodChange, ParentPatch, ParentEditError, parent_context, apply_parent_patch,
    validate_executable_parent_change,
)
from ..live.supplementary import (
    SupplementaryPlans, SupplementaryExecution, negotiate, normalize_concerns, summarize_coverage,
)
from ..live.source_tree import candidate_snapshot, compare_source_trees
from ..artifacts import load_bundle
from ..live.context import (
    ContextProtocolError,
    input_characters,
    present_inputs,
    validate_presentation,
)
from .stages import ResearchAdmissionDeferred, ResearchStages, ROLE_INSTRUCTIONS


PROMPTS = {
    "limitations": "Extract actionable limitations and hypotheses relevant to the task and permitted development protocol.",
    "verify_limitations": "Independently assess whether limitations are actionable and relevant. Reject unsupported assumptions.",
    "seeds": "Propose exactly count distinct hypotheses. Rank novelty only within supplied sources; do not certify global novelty.",
    "implement": "Implement the hypothesis. When parent is supplied, return only hash-bound anchored edits to its complete source within change_plan; preserve all other bytes including initialization. Otherwise use the file format and argv in the task contract. Never reconstruct a parent from a baseline or historical excerpts. No fabricated predictions or scores.",
    "critic": "Assess the current promotion gate specified in stage_contract. For subset screening, Good permits full-development evaluation; for full development, Good permits incumbent comparison under the controller's policy. Neither decision certifies a mechanism or final-test generalization. A proposed explanation remains a hypothesis until the relevant controls execute. Do not reject solely because later full seeds, ablations or protected final results have not run. Reject invalid current execution, incompatible protocols, failed hard constraints, non-improvement, or unsupported assertions presented as established findings. Engineer requests a permitted bounded code repair, not experiments belonging to later stages. Comparison delta is an absolute metric difference, not a percentage.",
    "engineer": "Repair the supplied candidate using actual execution diagnostics. Preserve the hypothesis and evaluation protocol.",
    "evolve": "Propose a hypothesis and explicit changed_files plan for the controller-selected parent, using BOTH successful and failed attempts. The exact parent source is supplied separately from historical excerpts. Code will be edited in a separate invocation. source_ids name documents; evidence_ids name offered non-final experiments. Do not mix namespaces.",
    "refine": "Plan a revised hypothesis and explicit changed_files for the exact incumbent parent addressing the supplied evidence and feedback. Code edits occur separately; preserve undeclared parent behavior and evaluation protocol.",
    "ablations": "Plan one to three component removal/control experiments. Each may change exactly one existing source file, named changed_file; all other files and argv must be byte-identical. State what it isolates. These are file-scoped controls; do not claim semantic isolation is certified.",
    "ablation_critic": "Use executed controls to assess mechanism attribution and whether a method revision is warranted.",
    "draft": "Produce evidence-grounded claims exactly matching supplied successful observations. When manuscript_revision is supplied, revise its previous_draft to address the explicit meta and reviewer feedback using only existing evidence; change claims, wording or limitations, never code or experimental results. Do not claim new experiments or resolved concerns without support. Address every supplementary concern with its actual execution coverage; unfulfilled or unsupported work cannot be marked executed. Executed does not mean scientifically resolved. Identical content hashes do not imply checkpoint reuse or independent seeds; consult execution receipts and protocol summaries. Free-form empirical prose is not requested.",
    "review": "Review the draft and evidence package, including executed control source, seed/protocol summaries and distinct execution receipts. Identical content hashes alone prove neither checkpoint reuse nor new independent seeds. Score 1-10 and identify concrete concerns. Consult the supplementary action contract; unsupported diagnostics must remain unresolved. This is internal review, not conference acceptance.",
    "rebuttal": "Map every supplied concern ID to a structured supplementary plan or explicit unaddressed disposition. Specify exact seed_count, split, metric/diagnostics and repeat versus paired_control. Purpose is explanation only and cannot request additional actions. Only the advertised contract executes; unsupported work is recorded unfulfilled, never silently substituted. Preserve argv and all files except one declared changed_file; unchanged repeats use null.",
    "meta": "Choose exactly Accept, ReviseManuscript, or ReviseMethod. Use ReviseManuscript for changes to interpretation, wording, evidence presentation or limitations using existing results; it invokes drafting and review only, never implementation or experiments. Use ReviseMethod only for a justified executable method change requiring new experiments. Never request a source comment as a method change. Accept means an internal evidence-scoped review, not publication acceptance. Respect the supplied revision limits; if a required revision remains unresolved, state it rather than falsely accepting. Consider full search history, rejected and failed attempts, supplementary coverage and unresolved concerns. Content hashes are not execution evidence; use the evidence package.",
}
PROMPTS.update(ROLE_INSTRUCTIONS)

ALLOWED = {
    "created": {"limitations"},
    "limitations": {"baseline", "complete"},
    "baseline": {"search", "complete"},
    "search": {"ablation", "draft", "freeze", "complete"},
    "ablation": {"draft", "ablation", "complete"},
    "draft": {"review", "complete"},
    "review": {"draft", "meta", "complete"},
    "meta": {"draft", "ablation", "freeze", "complete"},
    "freeze": {"final", "complete"},
    "final": {"complete"},
    "complete": set(),
}


class ResearchController(ResearchStages):
    def __init__(self, store, config, adapter, backend, runner, *, budget=None,
                 allow_network_literature=False):
        self.store, self.config, self.adapter = store, config, adapter
        self.backend, self.runner = backend, runner
        self.budget = budget or Budget(store, config)
        self.research_enabled = config.engine_version == "research-v2"
        self.allow_network_literature = allow_network_literature
        if backend.synthetic and config.workflow.network_literature:
            raise ValueError("Scripted/offline campaigns cannot enable live literature access")
        self.trials, self.baselines, self.best = [], {}, None
        self.supplementary_records = {}
        self.manuscript_revisions_used = 0
        self.scientific_reviews = []
        self.research_diagnostics = []
        self.research_deferrals = []
        if self.store.get("live_state", {}).get("status") in {"complete", "blocked"}:
            self.restore_supplementary_records()
        self.sources = [
            {"id": "task", "title": config.native.task_id, "text": self._task_source_text()},
            *[s.model_dump() for s in config.sources],
        ]

    def _task_source_text(self):
        context = self.adapter.context()
        if self.store.get("contract_repair_manifest_ref"):
            from .contract_repair import recorded_task_source_text
            return recorded_task_source_text(self.store, context)
        return str(context)

    def transition(self, key, stage, reason):
        if self.store.db.execute("SELECT 1 FROM events WHERE key=?", ("live/" + key,)).fetchone():
            return
        state = self.store.get("live_state", {"stage": "created"})
        if stage not in ALLOWED[state["stage"]]:
            raise IntegrityError(f"Illegal live transition: {state['stage']} -> {stage}")
        state.update(stage=stage, status="running", reason=reason)
        with self.store.db:
            self.store.db.execute(
                "INSERT INTO events VALUES(?,?,?,?)", ("live/" + key, time.time(), stage, reason)
            )
            self.store.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES('live_state',?)",
                (canonical(state).decode(),),
            )

    def ask(self, key, role, inputs, schema, *, validator=None):
        """Keep paid completion evidence separate from accepted output contracts.

        Every cached or newly completed response is revalidated. A correction
        has its own deterministic operation key, so an interrupted or resumed
        invocation cannot reset its allowance or repeat a possibly paid call.
        """
        if self.store.get("contract_repair_manifest_ref"):
            from .contract_repair import repair_invocation_key
            key = repair_invocation_key(self.store, key, role)
        if self.store.get("live_frozen"):
            raise IntegrityError("No agent invocation after final candidate freeze")
        if not applies(schema) and validator is None:
            return self._ask_once(key, role, inputs, schema)
        scope = invocation_scope(self.store, role, inputs, self.sources)
        original_inputs = deepcopy(inputs)
        if role == "scientific_review":
            original_inputs["reference_contract"] = {
                "source_ids": list(scope.source_ids),
                "evidence_ids": sorted(inputs.get("ledger", {}).get("entries", {})),
                "claim_ids": sorted(c["id"] for c in inputs.get("manuscript", {}).get("claims", [])),
                "meaning": "Review evidence IDs name supplied typed ledger entries; claim IDs name manuscript claims. Do not substitute experiment IDs or invent concerns.",
            }
        elif role == "rebuttal_planner":
            original_inputs["reference_contract"] = {
                "concern_ids": sorted(c["id"] for c in inputs.get("concerns", [])),
                "meaning": "Only these actual review concern IDs may be addressed; an empty list does not authorize invented score-related concerns.",
            }
        else:
            original_inputs["reference_contract"] = scope.prompt()
        rejected, rejection = None, None
        for correction in range(self.config.agent_output_retries + 1):
            call_key = key if correction == 0 else f"{key}/output-correction/{correction}"
            current_inputs = deepcopy(original_inputs)
            if correction:
                current_inputs["output_correction"] = {
                    "attempt": correction,
                    "maximum_corrections": self.config.agent_output_retries,
                    "rejected_output": rejected,
                    "errors": [rejection],
                    "instruction": "Return a complete corrected artifact following the supplied contracts and validation feedback. Preserve the scientific intent; do not invent observations, widen edit scope or claim an unexecuted experiment.",
                }
            parsed = self._ask_once(call_key, role, current_inputs, schema)
            row = self.store.db.execute(
                "SELECT output_ref FROM calls WHERE key=?", ("live-agent/" + call_key,)
            ).fetchone()
            if row is None:
                raise IntegrityError("Completed agent output has no durable response")
            proof = {
                "invocation_key": key, "response_key": call_key,
                "request_ref": self.store.get("request/" + call_key),
                "response_ref": row[0], "reference_contract": original_inputs["reference_contract"],
                "correction": correction, "maximum_corrections": self.config.agent_output_retries,
                "scope": "Reference/seed contracts and declared action/source constraints; not scientific sufficiency or semantic method-code equivalence",
            }
            try:
                if applies(schema):
                    validate_output(parsed, scope)
                if validator is not None:
                    validator(parsed)
            except AgentOutputError as error:
                proof.update(status="rejected", error=str(error))
                self.store.set("agent-output/" + call_key,
                               self.store.put_json(proof, "agent-output-validation"))
                self.store.event("agent-output/" + call_key + "/rejected",
                                 self.store.get("live_state", {"stage": "created"})["stage"], str(error))
                if correction == self.config.agent_output_retries:
                    raise AgentOutputError(f"{role}: {error}; output-correction allowance exhausted") from error
                rejected, rejection = parsed.model_dump(mode="json"), str(error)
            else:
                proof["status"] = "accepted"
                ref = self.store.put_json(proof, "agent-output-validation")
                self.store.set("agent-output/" + call_key, ref)
                self.store.set("accepted-agent-output/" + key, ref)
                return parsed
        raise AssertionError("Output correction loop must return or raise")

    def _ask_once(self, key, role, inputs, schema):
        # Final scores are never input to a role, including after resume.
        if self.store.get("live_frozen"):
            raise IntegrityError("No agent invocation after final candidate freeze")
        raw_inputs = {"instruction": PROMPTS[role], **inputs}
        raw_request = {
            "role": role,
            "inputs": raw_inputs,
            "schema": schema.model_json_schema(),
            "model": self.config.model,
            "prompt_version": "research-v2-complete-workflow" if self.research_enabled else "native-live-v6-typed-manuscript-revisions",
        }
        request_guard = getattr(self.backend, "validate_request_for_storage", None)
        if request_guard is not None:
            request_guard(raw_request)
        raw_ref = self.store.put_json(raw_request, "agent-raw-request")
        self.store.set("raw_request/" + key, raw_ref)
        inputs, changes = present_inputs(
            role,
            raw_inputs,
            history_code_chars=self.config.history_code_chars,
            log_tail_chars=self.config.log_tail_chars,
            archive_candidate=lambda candidate: self.store.put_json(
                candidate, "historical-proposal"
            ),
        )
        request = {**raw_request, "inputs": inputs}
        request_ref = self.store.put_json(request, "agent-request")
        self.store.set("request/" + key, request_ref)
        proof = {
            "raw_request_ref": raw_ref,
            "presented_request_ref": request_ref,
            "raw_input_characters": input_characters(raw_inputs),
            "presented_input_characters": input_characters(inputs),
            "max_prompt_chars": self.config.max_prompt_chars,
            "measurement": "Exact canonical JSON input characters; excludes backend envelope "
            "and output schema, and does not estimate tokens",
            "transformations": changes,
        }
        try:
            validate_presentation(
                inputs,
                max_prompt_chars=self.config.max_prompt_chars,
                protected_paths=[
                    getattr(self.config.native, field, None)
                    for field in ("private_dir", "evaluator_dir")
                ],
            )
        except ContextProtocolError as error:
            proof.update(status="blocked", reason=str(error))
            self.store.set("context/" + key, self.store.put_json(proof, "agent-context-proof"))
            self.store.event(
                "context/" + key,
                self.store.get("live_state", {}).get("stage", "created"),
                str(error),
            )
            raise
        proof["status"] = "pass"
        self.store.set("context/" + key, self.store.put_json(proof, "agent-context-proof"))
        cached = self.store.cached("live-agent/" + key, request)
        if cached:
            return schema.model_validate(AgentResult.model_validate(cached).payload)
        for attempt in range(self.config.model_retries + 1):
            operation = f"model/{key}/{attempt}"
            prior = self.budget.reserve(operation, "model", request)
            if prior:
                if prior["status"] == "running":
                    raise AmbiguousCall(
                        "Model outcome was not committed; refusing to repeat a possibly billed call: "
                        + operation
                    )
                if prior["status"] == "success":
                    result = AgentResult.model_validate(self.store.json(prior["result_ref"]))
                    self.store.cache("live-agent/" + key, request, result)
                    return schema.model_validate(result.payload)
                continue
            self.backend.timeout = min(self.config.call_seconds, self.budget.remaining())
            started = time.monotonic()
            try:
                result = self.backend.invoke(role, inputs, schema)
                if result.synthetic != self.backend.synthetic or result.model != self.config.model:
                    raise IntegrityError("Agent/model provenance mismatch")
                parsed = schema.model_validate(result.payload)
                tokens = sum(
                    v
                    for k, v in result.usage.items()
                    if k in {"input_tokens", "output_tokens"}
                    and isinstance(v, int)
                    and not isinstance(v, bool)
                    and v >= 0
                )
                self.budget.finish(operation, "success", result, time.monotonic() - started, tokens)
                self.store.cache("live-agent/" + key, request, result)
                self.store.set(
                    "events/" + operation,
                    self.store.put_json(getattr(self.backend, "events", []), "model-events"),
                )
                return parsed
            except (BackendError, ValueError, OSError) as error:
                # API completions retain usage even if JSON/domain validation
                # failed. Missing usage is explicit and must block dispatch on
                # resume; never silently convert it into a zero-cost retry.
                from ..providers.backend import APIBackendError
                failure = {"error": str(error)}
                failure_tokens = 0
                if isinstance(error, APIBackendError):
                    failure.update(usage_known=error.usage_known, usage=error.usage,
                                   api_call_id=error.call_id, api_receipt_ref=error.receipt_ref)
                    if error.usage_known:
                        failure_tokens = sum(error.usage.get(k, 0) for k in ("input_tokens", "output_tokens"))
                self.budget.finish(
                    operation, "failed", failure, time.monotonic() - started, failure_tokens
                )
                self.store.event(
                    operation + "/failure",
                    self.store.get("live_state", {"stage": "created"})["stage"], str(error)
                )
                if isinstance(error, APIBackendError) and not error.usage_known:
                    raise
        raise BackendError("Model retry limit exhausted: " + key)

    def comparison(self, result, reference):
        if self.research_enabled and callable(getattr(self.adapter, "compare", None)):
            return self.adapter.compare(result, reference)
        return compare_observations(self.config.development, result, reference)

    def record(self, key, idea, candidate, result, decision, parents=()):
        trial = {
            "id": key,
            "idea": idea.model_dump(),
            "candidate": candidate.model_dump(),
            "result": result.model_dump(),
            "decision": decision.model_dump(),
            "parents": list(parents),
        }
        self.trials.append(trial)
        self.store.cache("live-trial/" + key, {"key": key}, trial)
        self.store.event("decision/" + key, self.store.get("live_state")["stage"], decision.reason)
        return trial

    def feedback(self, result):
        logs = []
        for ref in result.log_refs:
            receipt = self.store.json(ref)
            logs.append(self.store.read(receipt["log_ref"]).decode(errors="replace"))
        return {"observation": result.model_dump(), "execution_logs": logs}

    def evidence_package(self, observations, *, trials=()):
        value = build_evidence_context(self.store, observations, trials=trials)
        if self.research_enabled:
            from .execution_context import build_execution_context
            value["effective_execution"] = build_execution_context(self.store, observations)
        return value

    def parent_input(self, trial):
        proposal = Candidate.model_validate(trial["candidate"])
        materialize = getattr(self.adapter, "materialize", None)
        tree = candidate_snapshot(self.store, materialize(proposal) if materialize else proposal)
        ref = self.store.put_json(tree, "implementation-parent")
        executed = trial.get("result", {}).get("source_tree_ref")
        if executed:
            prior = self.store.json(executed)
            if prior["bundle_ref"] != tree.bundle_ref or prior["command"] != tree.command:
                raise IntegrityError("Implementation parent differs from executed source")
        paths = getattr(self.adapter, "editable_paths", None)
        paths = paths() if callable(paths) else self.adapter.context().get("editable_files")
        options = {"materialize": materialize, "editable_paths": paths}
        try:
            context = parent_context(self.store, proposal, ref, **options)
        except ParentEditError as error:
            raise ContextProtocolError(str(error)) from error
        context["trial_id"] = trial["id"]
        return proposal, ref, options, context

    def critic_evidence(self, result, reference, implementation_parent=None):
        observations = [reference, result]
        extra = {}
        if self.research_enabled and implementation_parent is not None:
            from .parent_comparison import resolve_parent_comparison

            _, source_ref, _, _ = self.parent_input(implementation_parent)
            context, parent_result = resolve_parent_comparison(
                self.store, implementation_parent, source_ref, result, self.comparison
            )
            extra["parent_comparison"] = context
            if parent_result is not None and parent_result.id not in {item.id for item in observations}:
                observations.append(parent_result)
        return {"evidence_package": self.evidence_package(observations, trials=self.trials), **extra}

    def critic_stage_contract(self, result, reference, *, purpose="candidate_search", reference_role="baseline"):
        """Describe the current decision without treating future evidence as measured.

        Protocol seeds come from the frozen observation, alongside the task's
        expected stage seeds. A hypothesis is not converted to a mechanism claim
        by numerical promotion. Actual methodological objections remain valid.
        """
        if result.phase not in {"subset", "full"} or reference.phase != result.phase:
            raise IntegrityError("Scientific critic requires a matched subset or full-development phase")

        def protocol(observation):
            if observation.protocol_ref is None:
                return {"status": "unverified", "ref": None, "hash": observation.protocol_hash,
                        "phase": observation.phase, "seeds": []}
            record = self.store.json(observation.protocol_ref)
            seeds = record.get("seeds")
            if (digest(canonical(record)) != observation.protocol_hash or record.get("phase") != observation.phase
                    or not isinstance(seeds, list) or not seeds
                    or any(type(seed) is not int or seed < 0 for seed in seeds)
                    or len(set(seeds)) != len(seeds)
                    or observation.status == "success" and len(seeds) != len(observation.values)):
                raise IntegrityError("Critic stage contract differs from the frozen observation protocol")
            return {"status": "recorded", "ref": observation.protocol_ref, "hash": observation.protocol_hash,
                    "phase": observation.phase, "seeds": list(seeds)}

        subset = result.phase == "subset"
        deferred = ([{"kind": "full_development", "stage": "full", "seeds": list(self.config.development.seeds)}]
                    if subset else [])
        deferred.extend([
            {"kind": "component_removal_or_matched_control", "stage": "ablation"},
            {"kind": "mechanism_attribution", "stage": "ablation_and_scientific_review"},
            {"kind": "final_test_generalization", "stage": "after_candidate_freeze"},
        ])
        return {
            "schema_version": "scientisttwo-critic-stage-v1", "current_phase": result.phase,
            "evaluation_purpose": purpose, "reference_role": reference_role,
            "candidate_observation_id": result.id, "reference_observation_id": reference.id,
            "candidate_protocol": protocol(result), "reference_protocol": protocol(reference),
            "declared_stage_seeds": list(self.config.development.seeds[:1] if subset else self.config.development.seeds),
            "promotion": {"when_decision_is": "Good", "authorizes": "full_development" if subset else "incumbent_comparison",
                "meaning": "Promising valid screening result; execute the declared full-development protocol next"
                    if subset else "Valid full-development improvement; eligible for the controller's incumbent comparison",
                "certifies_mechanism": False, "certifies_final_test_performance": False},
            "required_current_evidence": ["valid_execution_and_trusted_metrics", "matched_current_phase_protocol",
                "declared_current_stage_seeds", "hard_constraints_and_guardrails", "numerical_improvement",
                "no_unsupported_assertions_presented_as_established_findings"],
            "deferred_evidence": deferred, "missing_deferred_evidence_alone_requires_rejection": False,
            "claim_boundary": "Ideas and proposed causal explanations are hypotheses. Missing later-stage evidence limits claims, not current-stage promotion. Reject a claim presented as established when current evidence contradicts it or cannot support it; do not require a future ablation merely to investigate a hypothesis.",
            "methodology_boundary": "Current execution failures, incomplete current-stage evidence, protocol mismatches, leakage, unsupported reported facts, hard-constraint failures and non-improvement remain grounds for rejection. A comparison against the declared baseline is not automatically a causal control.",
        }

    @staticmethod
    def validate_change_plan(change, parent):
        if not set(change.change_plan.changed_files) <= set(parent["editable_paths"]):
            raise AgentOutputError("Method change plan exceeds the supplied parent's editable files")

    def implement_change(self, key, role, idea, parent_trial, change_plan, **feedback):
        parent, ref, options, context = self.parent_input(parent_trial)

        def apply(patch):
            try:
                paths = change_plan.changed_files
                if role == "engineer":
                    edited = sorted({edit.path for edit in patch.edits})
                    if not set(edited) <= set(paths):
                        raise ParentEditError("Repair exceeds the prior declared file scope")
                    paths = edited
                candidate, proof = apply_parent_patch(self.store, parent, ref, patch,
                                                     declared_paths=paths, **options)
                if role in {"implement", "ablation_coder", "rebuttal_coder"}:
                    proof["executable_change_check"] = validate_executable_parent_change(
                        self.store, ref, proof["candidate_source_ref"]
                    )
                return candidate, proof
            except ParentEditError as error:
                raise AgentOutputError(str(error)) from error

        patch = self.ask(key, role, {
            "idea": idea.model_dump(), "contract": self.adapter.context(),
            "parent": context, "change_plan": change_plan.model_dump(), **feedback,
            "scope_rule": "Repair may edit a subset of the planned files" if role == "engineer" else "Edit exactly the planned files",
        }, ParentPatch, validator=apply)
        candidate, proof = apply(patch)
        accepted = self.store.json(self.store.get("accepted-agent-output/" + key))
        proof.update(parent_trial_id=parent_trial["id"], change_plan=change_plan.model_dump(),
                     parent_observation_id=parent_trial.get("result", {}).get("id"),
                     response_key=accepted["response_key"], request_ref=accepted["request_ref"],
                     response_ref=accepted["response_ref"],
                     parent_candidate_ref=self.store.put_json(parent, "implementation-parent-candidate"),
                     candidate_ref=self.store.put_json(candidate, "parent-edited-candidate"))
        self.store.cache("parent-edit/" + key, {"parent_ref": ref, "patch": patch.model_dump()}, proof)
        return candidate

    def evaluate(self, key, idea, parents=(), *, implementation_parent=None, change_plan=None):
        if self.research_enabled:
            self.admit_block(key, model_calls=5, experiments=2, optional=True)
        candidate = (self.implement_change(key + "/implement", "implement", idea,
                                           implementation_parent, change_plan)
                     if implementation_parent is not None else self.ask(
                         key + "/implement", "implement",
                         {"idea": idea.model_dump(), "contract": self.adapter.context(),
                          **({"baseline": self.baseline_candidate.model_dump()} if self.research_enabled and hasattr(self, "baseline_candidate") else {})}, Candidate))
        trial = None
        for phase in ("subset", "full"):
            if phase == "full" and self.research_enabled:
                if trial is None:
                    raise IntegrityError("Full-stage handoff requires a committed screened trial")
                candidate, revised_idea = self.research_handoff(key + "/full-handoff", trial)
                if revised_idea is not None:
                    idea, _ = self.research_novelty(key + "/full-handoff/novelty", revised_idea)
                    self.admit_block(key + "/full-handoff/rescreen", model_calls=2, experiments=2)
                    recheck = self.runner.execute(key + "/full-handoff/rescreen", candidate, "subset")
                    comparison = self.comparison(recheck, self.baselines["subset"])
                    critique = self.ask(key + "/full-handoff/rescreen/critic", "critic", {
                        "idea": idea.model_dump(), "candidate": candidate.model_dump(),
                        "comparison": comparison, "evidence": self.feedback(recheck),
                        "evidence_package": self.evidence_package([self.baselines["subset"], recheck], trials=self.trials),
                        "stage_contract": self.critic_stage_contract(recheck, self.baselines["subset"],
                            purpose="full_handoff_rescreen"),
                        "can_engineer": False,
                    }, CriticDecision)
                    if not comparison["improved"]:
                        critique = CriticDecision(decision="Bad", reason="Scientific full-stage change failed renewed screening")
                    trial = self.record(recheck.id, idea, candidate, recheck, critique, (trial["id"],))
                    if critique.decision != "Good":
                        return None
                    parents = (trial["id"],)
            for attempt in range(self.config.engineering_retries + 1):
                step = f"{key}/{phase}/{attempt}"
                if self.research_enabled:
                    self.admit_block(step + "/execution", model_calls=1, experiments=1)
                result = self.runner.execute(step, candidate, phase)
                comparison = self.comparison(result, self.baselines[phase])
                decision = self.ask(
                    step + "/critic",
                    "critic",
                    {
                        "idea": idea.model_dump(),
                        "candidate": candidate.model_dump(),
                        "evidence": self.feedback(result),
                        **self.critic_evidence(result, self.baselines[phase], implementation_parent),
                        "stage_contract": self.critic_stage_contract(result, self.baselines[phase]),
                        "comparison": comparison,
                        "can_engineer": attempt < self.config.engineering_retries,
                    },
                    CriticDecision,
                )
                if decision.decision == "Good" and not comparison["improved"]:
                    decision = CriticDecision(
                        decision="Bad", reason="Controller veto: invalid or non-improving evidence"
                    )
                trial = self.record(step, idea, candidate, result, decision, parents)
                if decision.decision == "Good":
                    break
                if decision.decision != "Engineer" or attempt == self.config.engineering_retries:
                    return None
                repair_inputs = {
                    "feedback": self.feedback(result), "critique": decision.model_dump(),
                }
                if implementation_parent is not None:
                    candidate = self.implement_change(
                        step + "/repair", "engineer", idea, trial, change_plan, **repair_inputs
                    )
                    parents = (trial["id"],)
                    continue
                candidate = self.ask(
                    step + "/repair",
                    "engineer",
                    {
                        "idea": idea.model_dump(),
                        "candidate": candidate.model_dump(),
                        "feedback": self.feedback(result),
                        "critique": decision.model_dump(),
                        "contract": self.adapter.context(),
                    },
                    Candidate,
                )
                parents = (trial["id"],)
        return trial

    def set_best(self, trial):
        self.best = trial
        self.store.set("live_best", trial)

    def refine(self, key, best, feedback):
        if self.research_enabled:
            try:
                if not self.research_method_disposition(key, best, feedback):
                    return best, False
            except ResearchAdmissionDeferred:
                return best, False
        _, _, _, parent = self.parent_input(best)
        revision = self.ask(
            key + "/revision",
            "refine",
            {
                "incumbent": best,
                "feedback": feedback,
                "contract": self.adapter.context(),
                "sources": self.sources,
                "parent": parent,
                "evidence_package": self.evidence_package(
                    [best["result"], *feedback.get("evidence", [])], trials=self.trials
                ),
            },
            MethodChange,
            validator=lambda change: self.validate_change_plan(change, parent),
        )
        if self.research_enabled:
            revised_idea, _ = self.research_novelty(key + "/novelty", revision.idea)
            revision = revision.model_copy(update={"idea": revised_idea})
        candidate = self.implement_change(key + "/implement", "implement", revision.idea,
                                          best, revision.change_plan)
        if self.research_enabled:
            self.admit_block(key + "/execution", model_calls=1, experiments=1)
        result = self.runner.execute(key, candidate, "full")
        from ..live.models import Observation

        comparison = self.comparison(result, Observation.model_validate(best["result"]))
        decision = self.ask(
            key + "/critic",
            "critic",
            {
                "idea": revision.idea.model_dump(),
                "candidate": candidate.model_dump(),
                "evidence": self.feedback(result),
                "evidence_package": self.evidence_package([best["result"], result], trials=self.trials),
                "stage_contract": self.critic_stage_contract(result, Observation.model_validate(best["result"]),
                    purpose="method_revision", reference_role="incumbent"),
                "comparison": comparison,
                "can_engineer": False,
            },
            CriticDecision,
        )
        if decision.decision != "Good" or not comparison["improved"]:
            decision = CriticDecision(
                decision="Bad",
                reason="Revision did not pass comparison and methodology review; incumbent preserved",
            )
        trial = self.record(key, revision.idea, candidate, result, decision, (best["id"],))
        if decision.decision == "Good":
            self.set_best(trial)
            return trial, True
        return best, False

    def validate_control(self, original, plan, *, allow_repeat):
        transform = getattr(self.adapter, "materialize", lambda candidate: candidate)
        before_tree = candidate_snapshot(self.store, transform(original))
        after_tree = candidate_snapshot(self.store, transform(plan.candidate))
        before = load_bundle(self.store, before_tree.bundle_ref).files
        after = load_bundle(self.store, after_tree.bundle_ref).files
        changes = set(compare_source_trees(self.store, self.store.put_json(before_tree),
                                           self.store.put_json(after_tree)))
        if (before.keys() != after.keys()
                or changes != ({plan.changed_file} if plan.changed_file else set())
                or plan.candidate.command != original.command):
            raise AgentOutputError("Control must change exactly its declared existing file and preserve argv")
        if plan.changed_file is None and not allow_repeat:
            raise AgentOutputError("Component-removal ablations require a declared changed file")
        return {"before_source_ref": self.store.put_json(before_tree),
                "after_source_ref": self.store.put_json(after_tree),
                "changed_file": plan.changed_file, "method_isolation": "unverified",
                "file_isolation": "pass"}

    def fresh_supplementary_seeds(self, key):
        metadata_key = "supplementary-seeds/" + key
        saved = self.store.get(metadata_key)
        if saved is not None:
            return saved
        import json
        previous = [json.loads(row[0]) for row in self.store.db.execute(
            "SELECT value FROM metadata WHERE key LIKE 'supplementary-seeds/%'")]
        highest = max([*self.config.development.seeds, *[seed for row in previous for seed in row]])
        seeds = list(range(highest + 1, highest + 1 + self.config.supplementary_seed_count))
        self.store.set(metadata_key, seeds)
        return seeds

    def supplementary_contract(self, seeds=None):
        return {"version": 1, "max_seed_count": self.config.supplementary_seed_count,
                "fresh_seeds": seeds, "metrics": list(metric_specs(self.config.development)),
                "supported_diagnostics": ["metric"], "evaluation_split": "development",
                "execution_kinds": ["repeat", "paired_control"],
                "purpose_is_executable": False,
                "unsupported_behavior": "Record the exact request as unfulfilled; never run a smaller substitute",
                "repeat_semantics": "One fresh training execution per selected seed. Same incumbent/seed set is shared across plans in this batch, with explicit provenance; it does not add independent replications.",
                **({"supported_diagnostics": self.research_diagnostic_capabilities()} if self.research_enabled else {}),
                "scientific_resolution": "Execution coverage is verified; adequacy for the concern remains a review judgment"}

    def supplementary_coverage(self):
        return [item for record in self.supplementary_records.values() for item in record["coverage"]]

    def restore_supplementary_records(self):
        import json
        self.supplementary_records = {
            row[0].removeprefix("supplementary/"): self.store.json(json.loads(row[1]))
            for row in self.store.db.execute(
                "SELECT key,value FROM metadata WHERE key LIKE 'supplementary/%' ORDER BY key"
            )
        }

    def supplementary_context(self):
        return [{"key": key, "concerns": record["negotiation"]["concerns"],
                 "decisions": [{**decision, "plan": {k: v for k, v in decision["plan"].items()
                                                      if k != "candidate"}}
                               for decision in record["negotiation"]["decisions"]],
                 "unaddressed_concerns": record["negotiation"]["unaddressed_concerns"],
                 "executions": record["executions"], "coverage": record["coverage"]}
                for key, record in self.supplementary_records.items()]

    def supplementary_controls(self, key, best, feedback):
        original = Candidate.model_validate(best["candidate"])
        concerns = normalize_concerns(key, feedback["concerns"])
        if self.research_enabled and feedback.get("review_ref"):
            from ..live.supplementary import Concern
            review = self.store.json(feedback["review_ref"])
            concerns = [Concern(id=c["id"], review_key=key, text=c["text"])
                        for c in review["concerns"] if c["resolution"] != "resolved"]
        seeds = self.fresh_supplementary_seeds(key)
        metrics = list(metric_specs(self.config.development))
        negotiation_options = ({"supported_diagnostics": self.research_diagnostic_capabilities()}
                               if self.research_enabled else {})

        def validate_request(request):
            proposed = negotiate(request, concerns, seeds, metrics, **negotiation_options)
            if not proposed.contract_valid:
                raise AgentOutputError("; ".join(proposed.contract_errors))
            for decision in proposed.decisions:
                if decision.status == "accepted":
                    self.validate_control(original, decision.plan, allow_repeat=True)

        inputs = {
            "incumbent": best, "feedback": feedback,
            "concerns": [item.model_dump() for item in concerns],
            "action_contract": self.supplementary_contract(seeds),
            "contract": self.adapter.context(),
            "evidence_package": self.evidence_package([best["result"]], trials=self.trials),
        }
        if self.research_enabled:
            plans = self.research_supplementary_plans(key, best, inputs, seeds, metrics)
            validate_request(plans)
        else:
            plans = self.ask(key + "/plans", "rebuttal", inputs, SupplementaryPlans, validator=validate_request)
        negotiation = negotiate(plans, concerns, seeds, metrics, **negotiation_options)
        executions, evidence = [], {}

        def checkpoint():
            record = {"negotiation": negotiation.model_dump(mode="json"),
                      "executions": [row.model_dump(mode="json") for row in executions],
                      "coverage": [row.model_dump(mode="json") for row in summarize_coverage(
                          negotiation, executions, store=self.store if self.research_enabled else None)]}
            self.supplementary_records[key] = record
            self.store.set("supplementary/" + key, self.store.put_json(record, "supplementary-coverage"))

        checkpoint()
        accepted = [(i, d) for i, d in enumerate(negotiation.decisions) if d.status == "accepted"]
        if not accepted:
            # An empty or wholly unsupported request carries no executable work.
            # Preserve concern coverage without reserving resources for fictional
            # experiments or an assumed later six-call rebuttal sequence.
            self.store.set("supplementary-routing/" + key, self.store.put_json({
                "route": "no_accepted_actions", "coverage_ref": self.store.get("supplementary/" + key),
                "new_experiments": False, "execution_admission_required": False,
            }, "supplementary-no-execution"))
            return []
        # Admit the whole block against the count budget before starting any new
        # experiment; normal per-operation time/shared-budget enforcement remains.
        operation_keys = {f"{key}/matched/{d.plan.seed_count}" for _, d in accepted}
        operation_keys.update(f"{key}/{i}/control" for i, d in accepted if d.plan.execution_kind == "paired_control")
        pending = sum(not self.store.db.execute("SELECT 1 FROM live_operations WHERE key=?",
                                               ("experiment/" + item,)).fetchone() for item in operation_keys)
        used = self.store.db.execute("SELECT count(*) FROM live_operations WHERE kind='experiment'").fetchone()[0]
        if self.research_enabled:
            scoring_operations = sum(
                (2 if decision.plan.execution_kind == "paired_control" else 1)
                * sum(item.kind in {"classification", "regression_residuals"} for item in decision.plan.diagnostics)
                for _, decision in accepted
            )
            self.admit_block(key + "/execution", model_calls=6,
                             experiments=pending + scoring_operations, closing=True)
        if used + pending > self.config.max_experiments:
            raise LiveBudgetExhausted("Insufficient experiment operations for the exact supplementary block; no substitute dispatched")
        try:
            for i, decision in accepted:
                plan = decision.plan
                isolation = self.validate_control(original, plan, allow_repeat=True)
                matched = self.runner.execute(f"{key}/matched/{plan.seed_count}", original,
                                              "rebuttal", seeds=decision.accepted_seeds)
                observed = [matched]
                if plan.execution_kind == "paired_control":
                    result = self.runner.execute(f"{key}/{i}/control", plan.candidate,
                                                 "rebuttal", seeds=decision.accepted_seeds)
                    observed.append(result)
                    self.store.cache(f"control/{key}/{i}", {"plan": plan.model_dump()}, {
                        "matched": matched.model_dump(), "control": result.model_dump(), **isolation,
                    })
                for result in observed:
                    evidence[result.id] = result
                complete = all(result.status == "success" for result in observed)
                diagnostics = [diagnostic for diagnostic in plan.diagnostics
                               if complete and all(diagnostic.metric in observation_metrics(self.config.development, result)
                                                   for result in observed)]
                diagnostic_refs = []
                if self.research_enabled:
                    diagnostics, diagnostic_refs = self.research_supplementary_diagnostics(
                        key, plan, matched, observed[1] if len(observed) > 1 else None)
                executions.append(SupplementaryExecution(
                    plan_id=plan.id, status="success" if complete else "failed",
                    seeds=decision.accepted_seeds,
                    evaluation_count=sum(len(result.values) for result in observed),
                    completed_diagnostics=diagnostics, observation_ids=[result.id for result in observed],
                    diagnostic_refs=diagnostic_refs,
                ))
                checkpoint()
        finally:
            checkpoint()
        return list(evidence.values())

    def controls(self, key, best, role, feedback=None):
        if role == "rebuttal":
            return self.supplementary_controls(key, best, feedback)
        if self.research_enabled:
            return self.research_ablations(key, best)
        original = Candidate.model_validate(best["candidate"])
        def validate_plans(plans):
            for plan in plans.plans:
                self.validate_control(original, plan, allow_repeat=False)
        plans = self.ask(key + "/plans", role, {
            "incumbent": best, "feedback": feedback, "contract": self.adapter.context(),
            "evidence_package": self.evidence_package([best["result"]], trials=self.trials),
        }, ControlPlans, validator=validate_plans)
        evidence = []
        for i, plan in enumerate(plans.plans):
            isolation = self.validate_control(original, plan, allow_repeat=False)
            matched = self.runner.execute(f"{key}/{i}/matched", original, "full")
            result = self.runner.execute(f"{key}/{i}/control", plan.candidate, "full")
            evidence += [matched, result]
            self.store.cache(f"control/{key}/{i}", {"plan": plan.model_dump()}, {
                "matched": matched.model_dump(), "control": result.model_dump(), **isolation,
            })
        return evidence

    def validate_concern_responses(self, draft):
        coverage = {row["concern_id"]: row for row in self.supplementary_coverage()}
        responses = {row.concern_id: row for row in draft.concern_responses}
        if len(responses) != len(draft.concern_responses) or set(responses) != set(coverage):
            raise AgentOutputError("Draft must address exactly every supplied supplementary concern once")
        for identifier, response in responses.items():
            actual = coverage[identifier]
            if not set(response.experiment_ids) <= set(actual["observation_ids"]):
                raise AgentOutputError("Concern response cites an experiment not linked to that concern")
            if response.status == "executed" and (actual["status"] != "executed"
                    or set(response.experiment_ids) != set(actual["observation_ids"])):
                raise AgentOutputError("Unsupported, failed or unexecuted supplementary requests cannot be marked executed")
            if response.status == "limited" and (not response.experiment_ids or actual["status"] == "unfulfilled"):
                raise AgentOutputError("A limited empirical response requires linked measured evidence")

    def draft(self, key, best, evidence, *, manuscript_revision=None):
        if self.research_enabled:
            return self.research_draft(key, best, evidence, manuscript_revision=manuscript_revision)
        inputs = {
            "incumbent": best,
            "evidence": [e.model_dump() for e in evidence],
            "sources": self.sources,
            "evidence_package": self.evidence_package(evidence, trials=self.trials),
            "supplementary_actions": self.supplementary_context(),
            "supplementary_coverage": self.supplementary_coverage(),
        }
        if manuscript_revision is not None:
            inputs["manuscript_revision"] = manuscript_revision

        def validate_draft(value):
            self.validate_concern_responses(value)
            if (manuscript_revision is not None
                    and canonical(value.model_dump()) == canonical(manuscript_revision["previous_draft"])):
                raise AgentOutputError("Requested manuscript revision returned an unchanged draft")

        draft = self.ask(
            key,
            "draft",
            inputs,
            Draft,
            validator=validate_draft,
        )
        known = {e.id: e for e in evidence if e.status == "success"}
        for claim in draft.claims:
            e = known.get(claim.experiment_id)
            metrics = observation_metrics(self.config.development, e) if e is not None else {}
            if (
                e is None
                or claim.metric not in metrics
                or not math.isclose(
                    claim.value, metrics[claim.metric], rel_tol=1e-10, abs_tol=1e-10
                )
            ):
                raise IntegrityError("Unsupported manuscript claim")
        self.store.set("live_draft", self.store.put_json(draft, "structured-manuscript"))
        return draft

    def review_draft(self, key, draft, evidence, *, manuscript_revision=None):
        if self.research_enabled:
            return self.research_review(key, draft, evidence, manuscript_revision=manuscript_revision)
        inputs = {
            "draft": draft.model_dump(), "evidence": [e.model_dump() for e in evidence],
            "evidence_package": self.evidence_package(evidence, trials=self.trials),
            "action_contract": self.supplementary_contract(),
            "supplementary_actions": self.supplementary_context(),
            "supplementary_coverage": self.supplementary_coverage(),
        }
        if manuscript_revision is not None:
            inputs["manuscript_revision"] = manuscript_revision
        review = self.ask(key, "review", inputs, LiveReview)
        self.store.set("live_review", self.store.put_json(review, "live-review"))
        return review

    def meta_review(self, key, draft, review, evidence):
        self.transition(key, "meta", "Meta-review the method and manuscript")
        meta = self.ask(key, "meta", {
            "draft": draft.model_dump(), "review": review.model_dump(),
            **({"manuscript": self.store.json(draft.manuscript_ref),
                "scientific_review": self.store.json(review.review_ref)}
               if self.research_enabled and draft.manuscript_ref and review.review_ref else {}),
            **(self.authoritative_writer_context(self.store.json(self.store.get("research_manuscript_bundle")))
               if self.research_enabled and self.store.get("research_manuscript_bundle") else {}),
            "evidence": [e.model_dump() for e in evidence],
            "evidence_package": self.evidence_package(evidence, trials=self.trials),
            "supplementary_actions": self.supplementary_context(),
            "supplementary_coverage": self.supplementary_coverage(),
            "revision_contract": {
                "manuscript_revisions_remaining": self.config.manuscript_revisions - self.manuscript_revisions_used,
                "manuscript_actions": "Revise the current draft, re-review, then meta-review; no implementation or experiments, including when review score is below 8",
                "method_action": "ReviseMethod is separately bounded by meta_refinements and requires executable changes and measured comparison",
            },
        }, LiveMetaReview)
        if self.research_enabled:
            gate = self.store.get("research_review_gate", {})
            if meta.decision == "Accept" and not gate.get("accepted", False):
                meta = LiveMetaReview(decision="ReviseManuscript",
                    reason="Controller veto: scientific review or manuscript checks remain unresolved; "
                           + "; ".join(gate.get("reasons", [])))
        self.store.set("live_meta", self.store.put_json(meta, "live-meta-review"))
        return meta

    def revise_manuscript(self, key, best, evidence, draft, review, meta):
        """Editorial-only route, replayed from chronological local role outputs."""
        revision = {
            "previous_draft": draft.model_dump(),
            "feedback": {"meta": meta.model_dump(), "review": review.model_dump()},
            "instruction": "Revise this manuscript using only the supplied committed evidence; preserve the incumbent and protocol. No new experiment or method change is authorized by this action.",
        }
        selected = canonical(best)
        experiments = self.store.db.execute(
            "SELECT key FROM live_operations WHERE kind='experiment' ORDER BY key"
        ).fetchall()
        record = {
            "status": "pending", "action": "ReviseManuscript",
            "previous_draft_ref": self.store.put_json(draft, "structured-manuscript"),
            "feedback_ref": self.store.put_json(revision["feedback"], "manuscript-feedback"),
            "selected_candidate_ref": self.store.put_json(best["candidate"], "manuscript-incumbent"),
            "selected_source_ref": best["result"].get("source_tree_ref"),
            "scope": "Draft/review routing and source preservation; scientific adequacy is reviewed, not mechanically certified",
        }

        def checkpoint(status, **updates):
            if canonical(best) != selected or experiments != self.store.db.execute(
                "SELECT key FROM live_operations WHERE kind='experiment' ORDER BY key"
            ).fetchall():
                raise IntegrityError("Manuscript revision changed the incumbent or admitted an experiment")
            record.update(status=status, no_new_experiments=True, **updates)
            self.store.set("manuscript-revision/" + key,
                           self.store.put_json(record, "manuscript-revision"))

        checkpoint("pending")
        self.transition(key + "/draft", "draft", "Revise manuscript using existing evidence; preserve the method")
        draft = self.draft(key + "/draft", best, evidence, manuscript_revision=revision)
        checkpoint("revised", revised_draft_ref=self.store.put_json(draft, "structured-manuscript"))
        self.transition(key + "/review", "review", "Review manuscript revision without new experiments")
        review = self.review_draft(key + "/review", draft, evidence, manuscript_revision=revision)
        checkpoint("reviewed", review_ref=self.store.put_json(review, "live-review"))
        meta = self.meta_review(key + "/meta", draft, review, evidence)
        checkpoint("reviewed", meta_ref=self.store.put_json(meta, "live-meta-review"),
                   meta_decision=meta.decision)
        return draft, review, meta

    def downstream(self, prefix, best):
        from ..live.models import Observation

        self.transition(
            prefix + "/ablation",
            "ablation",
            "Evaluate declared controls against matched incumbent runs",
        )
        evidence = []
        for cycle in range(self.config.ablation_refinements + 1):
            try:
                controls = self.controls(f"{prefix}/ablation/{cycle}", best, "ablations")
            except ResearchAdmissionDeferred:
                # No judgment or success is manufactured for unexecuted work.
                # Retain already measured controls when a later cycle defers.
                if not evidence:
                    evidence = [self.baselines["full"], Observation.model_validate(best["result"])]
                break
            evidence = [
                self.baselines["full"],
                Observation.model_validate(best["result"]),
                *controls,
            ]
            critique = self.ask(
                f"{prefix}/ablation/{cycle}/critique",
                "ablation_critic",
                {"incumbent": best, "evidence": [e.model_dump() for e in controls],
                 "evidence_package": self.evidence_package(controls, trials=self.trials)},
                Refinement,
            )
            if critique.decision == "Good" or cycle == self.config.ablation_refinements:
                break
            best, improved = self.refine(
                f"{prefix}/ablation/{cycle}/revision", best,
                {"critique": critique.model_dump(), "evidence": [e.model_dump() for e in controls]}
            )
            if not improved:
                break
        return self.write_review_pipeline(prefix, best, evidence)

    def write_review_pipeline(self, prefix, best, evidence):
        previous_draft = previous_review = None
        meta = None
        for cycle in range(self.config.review_rounds + 1):
            self.transition(
                f"{prefix}/draft/{cycle}", "draft", "Draft structured claims from executed evidence"
            )
            revision = None
            if self.research_enabled and previous_draft is not None:
                revision = {"previous_draft": previous_draft.model_dump(),
                    "feedback": {"review": previous_review.model_dump()},
                    "kind": "peer_rebuttal",
                    "instruction": "Enhance the reviewed manuscript using the actual supplementary results and unresolved concern coverage; preserve unaffected content."}
            draft = self.draft(f"{prefix}/draft/{cycle}", best, evidence, manuscript_revision=revision)
            self.transition(
                f"{prefix}/review/{cycle}",
                "review",
                "Review claims and plan supplementary experiments",
            )
            review = self.review_draft(f"{prefix}/review/{cycle}", draft, evidence)
            if self.research_enabled and review.review_ref:
                typed_review = self.store.json(review.review_ref)
                unresolved = [c for c in typed_review["concerns"] if c["resolution"] != "resolved"]
                if (typed_review["decision"] == "ReviseManuscript" and unresolved
                        and all(c["requested_action"] in {"revise_manuscript", "clarify", "none"}
                                for c in unresolved)):
                    # This is controller routing of the explicit scientific
                    # review, not a fabricated meta-review agent judgment. The
                    # existing editorial loop enforces the campaign-wide limit,
                    # then performs scientific review and actual meta-review.
                    meta = LiveMetaReview(decision="ReviseManuscript",
                        reason="Controller editorial route from scientific review: " + typed_review["summary"])
                    self.store.set("review-routing/" + prefix + "/" + str(cycle), self.store.put_json({
                        "review_ref": review.review_ref, "score": review.score,
                        "route": "manuscript_revision", "decision_origin": "scientific_review",
                        "concern_ids": [c["id"] for c in unresolved],
                        "manuscript_revisions_remaining": self.config.manuscript_revisions - self.manuscript_revisions_used,
                        "new_experiments": False, "method_change": False,
                    }, "editorial-scientific-review-routing"))
                    break
            if review.score >= 8 or cycle == self.config.review_rounds:
                break
            if self.research_enabled and review.review_ref:
                typed_review = self.store.json(review.review_ref)
                if not any(c["resolution"] != "resolved" for c in typed_review["concerns"]):
                    self.store.set("review-routing/" + prefix + "/" + str(cycle), self.store.put_json({
                        "review_ref": review.review_ref, "score": review.score,
                        "route": "meta_review", "reason": "No unresolved typed concerns authorize supplementary experiments; score-only feedback remains a review judgment.",
                        "new_experiments": False,
                    }, "score-only-review-routing"))
                    break
            previous_draft, previous_review = draft, review
            evidence += self.controls(
                f"{prefix}/rebuttal/{cycle}", best, "rebuttal", review.model_dump()
            )
        if meta is None:
            meta = self.meta_review(prefix + "/meta", draft, review, evidence)
        revision_index = 0
        while (meta.decision == "ReviseManuscript"
               and self.manuscript_revisions_used < self.config.manuscript_revisions):
            self.manuscript_revisions_used += 1
            self.store.set("live_manuscript_revisions_used", self.manuscript_revisions_used)
            draft, review, meta = self.revise_manuscript(
                f"{prefix}/manuscript/{revision_index}", best, evidence, draft, review, meta
            )
            revision_index += 1
        self.downstream_evidence = evidence
        return best, meta

    def finish(self, outcome, reason, *, status="complete"):
        state = self.store.get("live_state", {"stage": "created"})
        deferred = [row[0] for row in self.store.db.execute(
            "SELECT value FROM metadata WHERE key LIKE 'research-deferred/%' ORDER BY key")]
        if deferred:
            import json
            state["deferred_research_refs"] = [json.loads(value) for value in deferred]
            state["workflow_coverage"] = "partial_budget_deferred"
            marker = "Optional research blocks were deferred to preserve closing work"
            if marker not in reason:
                reason += ". " + marker + "; recorded deferrals are not completed scientific evidence."
        if status == "complete" and state["stage"] != "complete":
            # Terminal resource/errors can occur from every running stage.
            self.store.event("live/complete", "complete", reason)
            state["stage"] = "complete"
        state.update(status=status, outcome=outcome, reason=reason)
        self.store.set("live_state", state)
        from ..live.reporting import write_report

        write_report(self.store, self.config, self.backend.synthetic)
        return state

    def run(self):
        existing_spec = self.store.get("live_spec_ref")
        if existing_spec and canonical(self.store.json(existing_spec)) != canonical(self.config):
            raise IntegrityError("Frozen campaign schema differs; resume with its original pinned implementation, never rewrite its manifest")
        if self.config.implementation != implementation_identity():
            raise IntegrityError("Implementation changed; use the original pinned runtime for resume or create a fresh campaign")
        engine = self.store.get("engine_version")
        if engine is not None and engine != self.config.engine_version:
            raise IntegrityError("Campaign engine version cannot be migrated in place")
        self.store.set("engine_version", self.config.engine_version)
        ref = self.store.put_json(self.config, "live-specification")
        if self.store.get("live_spec_ref") not in (None, ref):
            raise IntegrityError("Run directory belongs to a different live specification")
        self.store.set("live_spec_ref", ref)
        provenance = {
            "backend": self.backend.name,
            "synthetic": self.backend.synthetic,
            "model": self.config.model,
        }
        if self.store.get("live_backend") not in (None, provenance):
            raise IntegrityError("Agent backend/provenance changed during campaign")
        self.store.set("live_backend", provenance)
        if not self.store.get("live_sources_ref"):
            from .. import backends, process, store, models
            from ..benchmarks import runner as native_runner

            paths = [
                *Path(__file__).parent.glob("*.py"),
                *(Path(__file__).parents[1] / "live").glob("*.py"),
                *(Path(__file__).parents[1] / "integrations").rglob("*.py"),
                *(Path(__file__).parents[1] / "studies").rglob("*.py"),
                *(Path(__file__).parents[1] / "providers").rglob("*.py"),
                *Path(native_runner.__file__).parent.glob("*.py"),
                *[Path(m.__file__) for m in (backends, process, store, models)],
            ]
            snapshots = {
                str(p.relative_to(Path(__file__).parents[1])): self.store.put_file(
                    p, "live-implementation-source"
                )
                for p in paths
            }
            self.store.set(
                "live_sources_ref", self.store.put_json(snapshots, "live-implementation-index")
            )
        if not self.store.get("adapter_sources_ref"):
            sources = getattr(self.adapter, "source_files", lambda: {})()
            refs = {name: self.store.put_file(path, "trusted-adapter-source") for name, path in sources.items()}
            self.store.set("adapter_sources_ref", self.store.put_json(refs, "trusted-adapter-source-index"))
        if self.store.verify():
            raise IntegrityError("Stored evidence was modified")
        if self.config.implementation != implementation_identity():
            raise IntegrityError("Live implementation changed; create a new manifest")
        identity = self.adapter.identity()
        if self.store.get("live_adapter_identity") not in (None, identity):
            raise IntegrityError("Development adapter changed")
        self.store.set("live_adapter_identity", identity)
        if self.research_enabled and self.config.workflow.writer_backend == "paperorchestra":
            from ..integrations.paperorchestra.runtime import verify_upstream, UPSTREAM_REVISION
            writer_identity = {"writer": "paperorchestra", "revision": UPSTREAM_REVISION,
                "files": verify_upstream(Path(self.config.workflow.paperorchestra_root)),
                "compiler_image": self.config.workflow.compiler_image}
            if self.store.get("research_writer_identity") not in (None, writer_identity):
                raise IntegrityError("PaperOrchestra writer changed on resume")
            self.store.set("research_writer_identity", writer_identity)
        saved = self.store.get("live_state", {})
        if saved.get("status") == "complete":
            self.restore_supplementary_records()
            if self.research_enabled:
                self.restore_research_diagnostics()
            return self.finish(saved["outcome"], saved["reason"])
        if saved.get("status") == "blocked" and saved.get("outcome") in {
            "invalid_agent_output", "manuscript_revision_limit", "baseline_alignment_failed"
        }:
            # Exhaustion is terminal for this frozen configuration. Inspecting
            # or resuming it cannot grant another paid correction allowance.
            self.restore_supplementary_records()
            return saved
        # Rebuild chronological role context from committed calls. Loading all
        # later coverage into an earlier resumed prompt would alter its inputs.
        self.trials, self.baselines, self.best = [], {}, None
        self.supplementary_records = {}
        self.manuscript_revisions_used = 0
        self.scientific_reviews = []
        self.research_diagnostics = []
        self.research_deferrals = []
        try:
            self.runner.preflight()
            if self.research_enabled and self.config.workflow.require_pdf:
                from .compiler import IsolatedCompiler
                compiler = IsolatedCompiler(self.runner, self.config.workflow.compiler_image)
                if compiler._image(self.config.workflow.compiler_image) != self.config.workflow.compiler_image:
                    raise ValueError("Required compiler image identity differs from its frozen digest")
            # A source-tree adapter can bind its store-local immutable base during preflight.
            self.sources[0]["text"] = self._task_source_text()
            frozen = self.store.get("live_frozen")
            if frozen:
                return self.finalize(frozen)
            self.transition(
                "limitations", "limitations", "Extract and independently verify limitations"
            )
            limitations = None
            for i in range(self.config.limitation_rounds):
                limitations = self.ask(
                    f"limitations/{i}",
                    "limitations",
                    {
                        "task": self.adapter.context(),
                        "sources": self.sources,
                        "previous": limitations.model_dump() if limitations else None,
                    },
                    LimitationAssessment,
                )
                limitations = self.ask(
                    f"verify/{i}",
                    "verify_limitations",
                    {"task": self.adapter.context(), "assessment": limitations.model_dump()},
                    LimitationAssessment,
                )
                if limitations.relevant:
                    break
            if not limitations.relevant:
                return self.finish(
                    "insufficient_limitations", "No actionable limitations verified within budget"
                )
            count = 1 if self.config.mode == "smoke" else (
                self.config.workflow.seed_ideas if self.research_enabled else self.config.search_rounds + 1)
            if self.research_enabled:
                self.admit_block("seed-generation", model_calls=count * 2)
                ideas = self.research_seeds(limitations, count)
            else:
                seeds = self.ask("seeds", "seeds", {
                    "count": count, "limitations": limitations.model_dump(),
                    "sources": self.sources, "contract": self.adapter.context(),
                }, Seeds)
                ideas = sorted(seeds.ideas, key=lambda x: -x.novelty_score)
            self.transition(
                "baseline",
                "baseline",
                "Reproduce the fixed reference under subset and full protocols",
            )
            base = self.research_baseline() if self.research_enabled else self.adapter.baseline()
            for phase in ("subset", "full"):
                if self.research_enabled:
                    base, self.baselines[phase] = self.research_baseline_result(base, phase)
                else:
                    self.baselines[phase] = self.runner.execute("baseline/" + phase, base, phase)
                if self.baselines[phase].status != "success":
                    return self.finish(
                        "baseline_failed", "Baseline did not produce valid development evidence"
                    )
            if self.research_enabled and not self.research_align_baseline(base):
                return self.finish(
                    "baseline_alignment_failed",
                    "The final reconstructed reference failed subset alignment; original evidence and full reference preserved, search not started",
                    status="blocked",
                )
            self.baseline_candidate = base
            self.transition(
                "search", "search", "Screen, scale and evolve hypotheses using measured evidence"
            )
            successful = []
            search_deferred = False
            rounds = (
                min(2, self.config.search_rounds)
                if self.config.mode == "smoke"
                else self.config.search_rounds
            )
            for i in range(rounds):
                if i == 0:
                    candidates = [(idea, [], None, None) for idea in ideas[:2]]
                else:
                    if self.research_enabled:
                        try:
                            self.admit_block(f"evolve/{i}/planning", model_calls=7,
                                             experiments=2, optional=True)
                        except ResearchAdmissionDeferred:
                            break
                    implementation_parent = deepcopy(self.best) if self.best is not None else {
                        "id": "baseline", "candidate": base.model_dump(),
                        "result": self.baselines["full"].model_dump(),
                    }
                    _, _, _, parent = self.parent_input(implementation_parent)
                    evolved = self.ask(
                        f"evolve/{i}",
                        "evolve",
                        {
                            "attempts": self.trials,
                            "sources": self.sources,
                            "contract": self.adapter.context(),
                            "parent": parent,
                            "evidence_package": self.evidence_package([t["result"] for t in self.trials], trials=self.trials),
                        },
                        MethodChange,
                        validator=lambda change: self.validate_change_plan(change, parent),
                    )
                    if self.research_enabled:
                        novelty_idea, _ = self.research_novelty(f"evolve/{i}/novelty", evolved.idea)
                        evolved = evolved.model_copy(update={"idea": novelty_idea})
                    candidates = [(idea, [], None, None) for idea in ideas[i + 1 : i + 2]] + [
                        (evolved.idea, list(dict.fromkeys([*[t["id"] for t in self.trials], implementation_parent["id"]])), implementation_parent, evolved.change_plan)
                    ]
                for j, (idea, parents, implementation_parent, change_plan) in enumerate(candidates):
                    try:
                        trial = self.evaluate(f"search/{i}/{j}", idea, parents,
                                              implementation_parent=implementation_parent, change_plan=change_plan)
                    except ResearchAdmissionDeferred:
                        search_deferred = True
                        break
                    if trial:
                        successful.append(trial)
                        from ..live.models import Observation

                        if (
                            self.best is None
                            or self.comparison(
                                Observation.model_validate(trial["result"]),
                                Observation.model_validate(self.best["result"]),
                            )["improved"]
                        ):
                            self.set_best(trial)
                    if len(successful) >= self.config.success_target:
                        break
                if search_deferred or len(successful) >= self.config.success_target:
                    break
            if self.best is None:
                best = {
                    "id": "baseline",
                    "candidate": base.model_dump(),
                    "result": self.baselines["full"].model_dump(),
                    "idea": {"title": "Baseline", "hypothesis": "Evaluate the declared reference; no successful research candidate was found", "novelty_score": 0, "novelty_scope": "Reference, no novelty claim", "source_ids": ["task"]},
                }
            else:
                best = self.best
                if self.research_enabled:
                    best = self.research_selection(successful, best)
            if self.research_enabled and self.config.mode == "research" and self.best is None:
                from ..live.models import Observation
                observations = {e.id: e for e in self.baselines.values()}
                observations.update({t["result"]["id"]: Observation.model_validate(t["result"]) for t in self.trials})
                best, meta = self.write_review_pipeline("negative", best, list(observations.values()))
                self.store.set("negative_research_outcome", {"reason": "No candidate passed scientific and numerical selection", "meta": meta.model_dump(), "method_revision_not_started": True})
                if meta.decision == "ReviseManuscript":
                    return self.finish("manuscript_revision_limit", "Negative-outcome manuscript remains unresolved within editorial allowance", status="blocked")
            if self.config.mode == "research" and self.best:
                for cycle in range(self.config.meta_refinements + 1):
                    best, meta = self.downstream(f"method/{cycle}", best)
                    if meta.decision == "ReviseManuscript":
                        return self.finish(
                            "manuscript_revision_limit",
                            "Manuscript revision allowance exhausted; unresolved feedback, latest draft and incumbent preserved",
                            status="blocked",
                        )
                    if meta.decision == "Accept" or cycle == self.config.meta_refinements:
                        break
                    if meta.decision != "ReviseMethod":
                        raise IntegrityError("Unknown meta-review action")
                    best, improved = self.refine(f"meta/{cycle}", best, {
                        "meta": meta.model_dump(),
                        "evidence": [e.model_dump() for e in self.downstream_evidence],
                    })
                    if not improved:
                        break
            self.transition(
                "freeze", "freeze", "Freeze selected method before protected final grading"
            )
            frozen = {
                "best": best,
                "baseline": base.model_dump(),
                "improved": self.best is not None,
            }
            self.store.set("live_frozen", frozen)
            return self.finalize(frozen)
        except AgentOutputError as error:
            self.store.event("live/invalid-agent-output",
                             self.store.get("live_state", {"stage": "created"})["stage"], str(error))
            return self.finish("invalid_agent_output", str(error), status="blocked")
        except ContextProtocolError as error:
            return self.finish("context_protocol_failed", str(error), status="blocked")
        except LiveBudgetExhausted as error:
            return self.finish("budget_exhausted", str(error))
        except (BackendError, AmbiguousCall, ValueError, OSError, PrerequisiteError) as error:
            return self.finish("blocked", str(error), status="blocked")
        except KeyboardInterrupt:
            self.finish(
                "interrupted",
                "Interrupted; committed operations remain reusable",
                status="interrupted",
            )
            raise

    def finalize(self, frozen):
        self.transition(
            "final",
            "final",
            "Grade frozen baseline and selected candidate once; no further agent calls",
        )
        results = {
            "baseline": self.runner.final("baseline", Candidate.model_validate(frozen["baseline"]))
        }
        if frozen["best"]["id"] != "baseline":
            results["selected"] = self.runner.final(
                "selected", Candidate.model_validate(frozen["best"]["candidate"])
            )
        self.store.set("live_final", results)
        if self.research_enabled:
            self.research_final_appendix(results)
        if any(r["status"] != "completed" for r in results.values()):
            return self.finish(
                "final_incomplete",
                "Frozen method preserved; final execution/grading needs attention",
                status="blocked",
            )
        return self.finish(
            "development_improvement" if frozen["improved"] else "no_improvement",
            "Completed protected final grading; development and final scores reported separately",
        )
