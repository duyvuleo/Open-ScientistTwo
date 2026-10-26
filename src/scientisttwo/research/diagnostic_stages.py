"""Controller hooks for exact supplementary diagnostics and artifact accounting."""

import json

from ..live.evaluation import observation_metrics
from ..live.pipeline import EvaluationPipeline
from ..store import IntegrityError
from .diagnostics import DiagnosticInputs, DiagnosticRequest, DiagnosticResult, run_diagnostic
from .diagnostic_runner import ScoringDiagnosticContract, run_scoring_diagnostic


class DiagnosticStages:
    def research_diagnostic_capabilities(self):
        callback = getattr(self.adapter, "diagnostic_capabilities", None)
        offered = callback() if callable(callback) else []
        allowed = {"paired_differences", "paired_interval", "classification", "regression_residuals", "timing_distribution"}
        if not callable(getattr(self.adapter, "diagnostics_contract", None)):
            allowed -= {"classification", "regression_residuals"}
        return ["metric", *sorted(set(offered) & allowed)]

    def restore_research_diagnostics(self):
        """Terminal inspection only; replay builds chronological context as it runs."""
        self.research_diagnostics = []
        for row in self.store.db.execute("SELECT value FROM metadata WHERE key LIKE 'research-diagnostic-record/%' ORDER BY rowid"):
            ref = json.loads(row[0])
            result = DiagnosticResult.model_validate(self.store.json(ref)).model_copy(update={"artifact_ref": ref})
            self.research_diagnostics.append(result.model_dump(mode="json"))

    def _remember_research_diagnostic(self, result):
        if not any(item.get("artifact_ref") == result.artifact_ref for item in self.research_diagnostics):
            self.research_diagnostics.append(result.model_dump(mode="json"))
        self.store.set("research-diagnostic-record/" + result.request.id, result.artifact_ref)

    def research_supplementary_diagnostics(self, key, plan, matched, control=None):
        """Return only measured completed requests plus all immutable result refs."""
        observations = [matched, *([control] if control is not None else [])]
        if any(item.status != "success" for item in observations):
            return [], []
        refs = []
        for observation in observations:
            row = self.store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("development/" + observation.id,)).fetchone()
            if row is None or self.store.json(row[0]) != observation.model_dump(mode="json"):
                raise IntegrityError("Supplementary diagnostic received uncommitted or altered evidence")
            refs.append(row[0])
        seeds = self.store.json(matched.protocol_ref)["seeds"] if matched.protocol_ref else []
        if len(seeds) != plan.seed_count:
            return [], []
        completed, artifact_refs = [], []
        supported = self.research_diagnostic_capabilities()
        for index, diagnostic in enumerate(plan.diagnostics):
            if diagnostic.kind == "metric":
                if all(diagnostic.metric in observation_metrics(self.config.development, item) for item in observations):
                    completed.append(diagnostic)
                continue
            is_pair = diagnostic.kind in {"paired_differences", "paired_interval"}
            lanes = [("pair", refs, matched)] if is_pair else [
                (str(i), [ref], observation) for i, (ref, observation) in enumerate(zip(refs, observations, strict=True))]
            measured = []
            for lane, selected_refs, observation in lanes:
                request = DiagnosticRequest(id=f"{key}/{plan.id}/{index}/{lane}", kind=diagnostic.kind,
                    metric=diagnostic.metric or self.config.development.metric, expected_seeds=seeds,
                    concern_ids=plan.concern_ids, parameters=diagnostic.parameters,
                    evaluation_split=plan.evaluation_split)
                trusted_refs = []
                if diagnostic.kind in {"classification", "regression_residuals"} and diagnostic.kind in supported:
                    partition = "subset" if observation.phase == "subset" else "full"
                    prepared_ref = self.store.get("prepared/" + partition)
                    if prepared_ref is not None:
                        prepared = self.store.json(prepared_ref)
                        declared = self.adapter.diagnostics_contract(diagnostic.kind, observation.phase, prepared=prepared)
                        trusted_refs.append(run_scoring_diagnostic(self.runner, request, selected_refs[0],
                            EvaluationPipeline.model_validate(declared["pipeline"]),
                            ScoringDiagnosticContract.model_validate(declared["contract"])))
                result = run_diagnostic(self.store, request, DiagnosticInputs(observation_refs=selected_refs,
                    trusted_result_refs=trusted_refs, declared_kinds=supported))
                self._remember_research_diagnostic(result)
                artifact_refs.append(result.artifact_ref)
                measured.append(result.status == "success")
            if measured and all(measured):
                completed.append(diagnostic)
        return completed, artifact_refs
