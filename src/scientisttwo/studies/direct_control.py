"""An ordinary coding-loop control; no ScientistTwo research roles or final access.

The study coordinator supplies the same frozen LiveSpec, starting source, backend
and ceilings as the research arm. The control can request subset or full trials;
only complete, matched three-seed development results may replace the incumbent.
Both arms must be frozen before a separate coordinator opens final evaluation.
"""

from pathlib import Path
import time
from typing import Literal

from pydantic import Field, model_validator

from ..backends import BackendError
from ..live.adapters import implementation_identity
from ..live.context import present_inputs, validate_presentation
from ..live.evaluation import compare_observations
from ..live.models import Candidate, Observation
from ..live.runtime import AmbiguousCall, Budget, DevelopmentRunner, LiveBudgetExhausted
from ..models import AgentResult, Record
from ..store import IntegrityError, file_digest


PROMPT = (
    "Improve the supplied baseline by an ordinary coding-and-experiment loop. "
    "Return an evaluate action with a complete Candidate proposal conforming to the task's "
    "editable files, command, fixed architecture, training budget and output contract; "
    "or stop with an honest reason. Use actual prior results and failure diagnostics. "
    "You may screen on subset or request full development. Only a valid full-development "
    "result on all three declared seeds can become the final selection. Every overlay is "
    "relative to the frozen starting source, so carry forward desired earlier edits. "
    "Never fabricate predictions, measurements, novelty or causal explanations. Do not "
    "change the evaluator or seek final-test information. The controller executes and scores."
)


class DirectProposal(Record):
    action: Literal["evaluate", "stop"]
    reason: str = Field(min_length=1)
    phase: Literal["subset", "full"] = "full"
    candidate: Candidate | None = None

    @model_validator(mode="after")
    def action_contract(self):
        if (self.action == "evaluate") != (self.candidate is not None):
            raise ValueError("Evaluate requires a candidate; stop must not include one")
        return self


class DirectProposalError(ValueError):
    """A completed model response failed its declared schema; preserve feedback."""


class DirectControlResult(Record):
    status: Literal["frozen", "blocked"]
    reason: str
    improved: bool
    selected_candidate_ref: str | None = None
    selected_observation_ref: str | None = None
    baseline_observation_ref: str | None = None
    history_ref: str
    freeze_ref: str | None = None
    usage: list[dict] = Field(default_factory=list)
    synthetic_agents: bool
    final_evaluation: Literal["not_started"] = "not_started"


class DirectControl:
    """Budgeted single-role control using the production development runner.

    ``runner`` injection is for controlled tests/worker transport. Its budget must
    be this same campaign ledger; the default is the isolated DevelopmentRunner.
    A live backend requires explicit enablement and the adapter's readiness gate.
    """

    def __init__(self, store, config, adapter, backend, runner=None, *, budget=None,
                 allow_live=False):
        if config.provider.kind != "codex":
            raise ValueError("The paired-study control currently supports only the Codex provider")
        if len(config.development.seeds) != 3:
            raise ValueError("This study control requires exactly three matched development seeds")
        if not backend.synthetic and not allow_live:
            raise ValueError("Live control calls require explicit allow_live=True")
        self.store, self.config, self.adapter, self.backend = store, config, adapter, backend
        self.budget = budget or (runner.budget if runner is not None else Budget(store, config))
        if self.budget.store is not store or self.budget.config != config:
            raise ValueError("Control must use its own store and the declared campaign budgets")
        self.runner = runner or DevelopmentRunner(store, config, adapter, self.budget)
        if self.runner.budget is not self.budget:
            raise ValueError("Model calls and experiments must share one budget ledger")

    def _bind(self, max_iterations):
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("A positive explicit iteration ceiling is required")
        if self.store.get("live_spec_ref"):
            raise IntegrityError("Direct control requires a separate run store from the research arm")
        if self.config.implementation != implementation_identity():
            raise IntegrityError("Implementation changed; use the original pinned source")
        identity = {
            "engine": "direct-coding-control-v1", "config": self.config.model_dump(mode="json"),
            "adapter": self.adapter.identity(), "max_iterations": max_iterations,
            "backend": self.backend.name, "synthetic": self.backend.synthetic,
            "control_source_sha256": file_digest(Path(__file__)),
            "protocol": "subset-one-seed/full-three-seeds; development-only freeze",
        }
        if self.store.get("direct_identity") not in (None, identity):
            raise IntegrityError("Direct-control inputs, implementation or provenance changed")
        self.store.set("direct_identity", identity)
        if self.store.verify():
            raise IntegrityError("Direct-control evidence was modified")
        self.store.set("direct_source_ref", self.store.put_file(Path(__file__), "control-source"))
        if self.store.get("direct_sources_ref") is None:
            package = Path(__file__).parents[1]
            sources = {path.relative_to(package).as_posix(): self.store.put_file(
                path, "control-implementation-source") for path in sorted(package.rglob("*.py"))
                if "__pycache__" not in path.parts}
            sources.update({"adapter/" + name: self.store.put_file(path, "trusted-adapter-source")
                            for name, path in getattr(self.adapter, "source_files", lambda: {})().items()})
            self.store.set("direct_sources_ref", self.store.put_json(sources, "control-source-index"))
        return identity

    @staticmethod
    def _tokens(usage):
        return sum(value for key, value in usage.items()
                   if key in {"input_tokens", "output_tokens"}
                   and type(value) is int and value >= 0)

    def _event_tokens(self):
        # Backends can expose observed completion usage even if payload parsing failed.
        return sum(self._tokens(event.get("usage", {}))
                   for event in getattr(self.backend, "events", [])
                   if isinstance(event, dict) and event.get("type") == "turn.completed"
                   and isinstance(event.get("usage"), dict))

    def _ask(self, key, inputs):
        if self.store.get("direct_frozen_ref"):
            raise IntegrityError("No model calls after control selection is frozen")
        inputs, _ = present_inputs(
            "direct_coding", inputs, history_code_chars=self.config.history_code_chars,
            log_tail_chars=self.config.log_tail_chars,
            archive_candidate=lambda value: self.store.put_json(value, "control-history-source"),
        )
        validate_presentation(inputs, max_prompt_chars=self.config.max_prompt_chars,
                              protected_paths=[getattr(self.config.native, field, None)
                                               for field in ("private_dir", "evaluator_dir")])
        request = {"role": "direct_coding", "inputs": inputs,
                   "schema": DirectProposal.model_json_schema(), "model": self.config.model,
                   "prompt_version": "direct-coding-control-v1"}
        self.store.set("direct_request/" + key, self.store.put_json(request, "agent-request"))
        saved = self.store.cached("direct_agent/" + key, request)
        if saved is not None:
            return DirectProposal.model_validate(saved)
        for attempt in range(self.config.model_retries + 1):
            operation = f"model/direct/{key}/{attempt}"
            prior = self.budget.reserve(operation, "model", request)
            if prior is not None:
                if prior["result_ref"] is None:
                    raise AmbiguousCall("Unknown model completion; refusing to repeat " + operation)
                result = self.store.json(prior["result_ref"])
                if prior["status"] == "success":
                    proposal = DirectProposal.model_validate(AgentResult.model_validate(result).payload)
                    self.store.cache("direct_agent/" + key, request, proposal)
                    return proposal
                if prior["status"] == "invalid_output":
                    raise DirectProposalError(result["error"])
                continue
            self.backend.timeout = min(self.config.call_seconds, self.budget.remaining())
            started, result = time.monotonic(), None
            try:
                result = self.backend.invoke("direct_coding", inputs, DirectProposal)
            except (BackendError, OSError) as error:
                self.budget.finish(operation, "failed", {"error": str(error)},
                                   time.monotonic() - started, self._event_tokens())
                self.store.event(operation, "direct_search", str(error))
                continue
            except ValueError as error:
                self.budget.finish(operation, "invalid_output", {"error": str(error)},
                                   time.monotonic() - started, self._event_tokens())
                raise DirectProposalError(str(error)) from error
            finally:
                self.store.set("direct_events/" + operation, self.store.put_json(
                    getattr(self.backend, "events", []), "model-events"))
            if (result.synthetic != self.backend.synthetic or result.model != self.config.model
                    or result.backend != self.backend.name):
                self.budget.finish(operation, "invalid_provenance", result,
                                   time.monotonic() - started, self._tokens(result.usage))
                raise IntegrityError("Control agent/model provenance mismatch")
            try:
                proposal = DirectProposal.model_validate(result.payload)
            except ValueError as error:
                self.budget.finish(operation, "invalid_output",
                                   {"error": str(error), "response": result.model_dump()},
                                   time.monotonic() - started, self._tokens(result.usage))
                raise DirectProposalError(str(error)) from error
            self.budget.finish(operation, "success", result, time.monotonic() - started,
                               self._tokens(result.usage))
            self.store.cache("direct_agent/" + key, request, proposal)
            return proposal
        raise BackendError("Control model retry ceiling exhausted: " + key)

    def _execute(self, key, candidate, phase) -> Observation:
        seeds = self.config.development.seeds[:1] if phase == "subset" else self.config.development.seeds
        result = self.runner.execute("direct/" + key, candidate, phase, seeds=seeds)
        if result.phase != phase:
            raise IntegrityError("Control received a different evaluation phase")
        if result.status == "interrupted":
            raise AmbiguousCall("Unknown experiment completion; refusing to repeat " + key)
        if result.status == "success" and len(result.values) != len(seeds):
            raise IntegrityError("Control result does not contain every declared matched seed")
        return result

    def _compare(self, result, reference):
        compare = getattr(self.adapter, "compare", None)
        return compare(result, reference) if callable(compare) else compare_observations(
            self.config.development, result, reference)

    def _feedback(self, result):
        logs = []
        for ref in result.log_refs:
            receipt = self.store.json(ref)
            logs.append(self.store.read(receipt["log_ref"]).decode(errors="replace"))
        return {"observation": result.model_dump(mode="json"), "execution_logs": logs}

    def _finish(self, identity, reason, history, baseline, best, candidate, *, blocked=False):
        history_ref = self.store.put_json(history, "direct-control-history")
        selected_ref = self.store.put_json(candidate, "direct-selected-candidate") if candidate else None
        observation_ref = self.store.put_json(best, "direct-selected-observation") if best else None
        baseline_ref = self.store.put_json(baseline, "direct-baseline-observation") if baseline else None
        improved = bool(best and baseline and self._compare(best, baseline)["improved"])
        frozen = None
        if not blocked and best is not None:
            if self.store.db.execute("SELECT 1 FROM live_operations WHERE status='running'").fetchone():
                raise AmbiguousCall("Cannot freeze control with unknown operation outcomes")
            frozen = self.store.put_json({
                "engine": identity["engine"], "identity_ref": self.store.put_json(identity),
                "selected_candidate_ref": selected_ref, "selected_observation_ref": observation_ref,
                "baseline_observation_ref": baseline_ref, "history_ref": history_ref,
                "source_tree_ref": best.source_tree_ref, "code_ref": best.code_ref,
                "protocol_hash": best.protocol_hash, "seeds": self.config.development.seeds,
                "selection_basis": "full_development_only", "improved": improved,
                "stop_reason": reason, "final_evaluation": "not_started",
            }, "direct-control-freeze")
            self.store.set("direct_frozen_ref", frozen)
        outcome = DirectControlResult(
            status="frozen" if frozen else "blocked", reason=reason, improved=improved,
            selected_candidate_ref=selected_ref, selected_observation_ref=observation_ref,
            baseline_observation_ref=baseline_ref, history_ref=history_ref, freeze_ref=frozen,
            usage=self.budget.summary(), synthetic_agents=self.backend.synthetic,
        )
        self.store.event("direct/finish", outcome.status, reason)
        self.store.cache("direct/result", identity, outcome)
        self.store.set("direct_result_ref", self.store.put_json(outcome, "direct-control-result"))
        return outcome

    def run(self, *, max_iterations):
        identity = self._bind(max_iterations)
        saved = self.store.cached("direct/result", identity)
        if saved is not None:
            return DirectControlResult.model_validate(saved)
        history, baselines, best, candidate = [], {}, None, None
        try:
            if not self.backend.synthetic:
                gate = getattr(self.adapter, "live_readiness", None)
                if not callable(gate):
                    raise ValueError("Live control requires an explicit adapter readiness gate")
                readiness = gate()
                self.store.set("direct_readiness_ref", self.store.put_json(readiness))
                if not readiness.get("ready"):
                    raise ValueError("Control adapter readiness failed: " + str(readiness.get("errors")))
            self.runner.preflight()
            baseline_candidate = self.adapter.baseline()
            for phase in ("subset", "full"):
                baselines[phase] = self._execute("baseline/" + phase, baseline_candidate, phase)
                if baselines[phase].status != "success":
                    return self._finish(identity, "Baseline failed: " + baselines[phase].reason,
                                        history, baselines.get("full"), None, None, blocked=True)
            best, candidate = baselines["full"], baseline_candidate
            last_candidate = baseline_candidate
            self.store.event("direct/baseline", "direct_search", "Matched baseline measured")
            reason = "Declared iteration ceiling reached"
            for index in range(max_iterations):
                key = f"iteration/{index}"
                inputs = {
                    "instruction": PROMPT, "contract": self.adapter.context(),
                    "starting_candidate": baseline_candidate.model_dump(mode="json"),
                    "incumbent": candidate.model_dump(mode="json"),
                    "last_attempt_candidate": last_candidate.model_dump(mode="json"),
                    "incumbent_observation": best.model_dump(mode="json"),
                    "baselines": {phase: result.model_dump(mode="json")
                                  for phase, result in baselines.items()},
                    "sources": [source.model_dump(mode="json") for source in self.config.sources],
                    "history": history, "iteration": index, "max_iterations": max_iterations,
                    "limits": {name: getattr(self.config, name) for name in (
                        "max_calls", "max_experiments", "campaign_seconds", "call_seconds",
                        "token_stop_threshold", "model_retries")},
                    "resource_limits": self.config.native.resources.model_dump(mode="json"),
                    "evaluation": {"subset_seeds": self.config.development.seeds[:1],
                                   "full_seeds": self.config.development.seeds,
                                   "final_test": "Unavailable to the control"},
                }
                try:
                    proposal = self._ask(key, inputs)
                except DirectProposalError as error:
                    history.append({"id": key, "status": "invalid_proposal", "error": str(error)})
                    self.store.event("direct/" + key, "direct_search", str(error))
                    continue
                if proposal.action == "stop":
                    reason = "Agent stopped: " + proposal.reason
                    history.append({"id": key, "status": "agent_stop", "reason": proposal.reason})
                    break
                last_candidate = proposal.candidate
                result = self._execute(key, proposal.candidate, proposal.phase)
                comparison = self._compare(result, baselines[proposal.phase])
                incumbent_comparison = self._compare(result, best) if proposal.phase == "full" else None
                selected = bool(proposal.phase == "full" and comparison["valid"]
                                and incumbent_comparison["improved"])
                history.append({"id": key, "status": result.status, "reason": proposal.reason,
                                "candidate_ref": self.store.put_json(proposal.candidate),
                                "comparison_to_baseline": comparison,
                                "comparison_to_incumbent": incumbent_comparison,
                                "selected": selected, **self._feedback(result)})
                self.store.event("direct/" + key, "direct_search",
                                 "Incumbent replaced" if selected else "Incumbent preserved")
                if selected:
                    best, candidate = result, proposal.candidate
        except LiveBudgetExhausted as error:
            reason = "Budget exhausted: " + str(error)
        except (AmbiguousCall, BackendError, ValueError) as error:
            return self._finish(identity, str(error), history, baselines.get("full"),
                                best, candidate, blocked=True)
        return self._finish(identity, reason, history, baselines.get("full"), best, candidate)
