"""Explicit scripted roles for the SAME research engine; never a live fallback."""
from copy import deepcopy
from typing import Callable

from ..artifacts import load_bundle
from ..backends import FixtureBackend
from ..live.models import Candidate, SourceFile
from ..live.source_tree import candidate_snapshot
from ..store import IntegrityError, Store, digest


class ResearchFixtures(FixtureBackend):
    """Adapt a prepared task's fixture recipes to independent scientific roles."""
    def __init__(self, legacy_response, *, store: Store | None = None,
                 materialize: Callable[[Candidate], Candidate] | None = None):
        self.legacy_response = legacy_response
        self.store, self.materialize = store, materialize
        self.seen = []
        super().__init__(self.respond)

    @staticmethod
    def _parent_candidate(inputs):
        parent = inputs["parent"]
        return Candidate(files=[SourceFile(path=v["path"], content=v["content"]) for v in parent["files"]],
            base_snapshot_ref=parent["base_snapshot_ref"], command=parent["command"],
            description="Exact scripted parent")

    def _control(self, inputs):
        candidate = self._parent_candidate(inputs)
        trial = {"candidate": candidate.model_dump(), "idea": inputs["idea"]}
        return self.legacy_response("ablations", {"incumbent": trial})["plans"][0]["candidate"]

    def _target_files(self, candidate, paths):
        """Resolve desired control bytes, including files omitted from an overlay.

        An omitted overlay means the frozen target base, never the incumbent's
        current bytes. Full-file CPU fixtures retain their existing interface.
        """
        target = Candidate.model_validate(candidate.model_dump())
        if target.base_snapshot_ref is None:
            files = {source.path: source.content for source in target.files}
        else:
            if self.store is None:
                raise ValueError("Frozen scripted control target requires a bound Store")
            materialized = self.materialize(target) if self.materialize else target
            materialized = Candidate.model_validate(materialized.model_dump())
            if (materialized.command != candidate.command
                    or materialized.base_snapshot_ref != candidate.base_snapshot_ref):
                raise IntegrityError("Scripted control materialization changed its command or frozen base")
            tree = candidate_snapshot(self.store, materialized)
            bundle = load_bundle(self.store, tree.bundle_ref, verify=True)
            files = {}
            for path in paths:
                artifact = bundle.files.get(path)
                if artifact is None:
                    raise ValueError("Scripted control target is missing declared file: " + path)
                if artifact.size > 250_000:
                    raise ValueError("Scripted control target exceeds the source text limit: " + path)
                try:
                    content = self.store.read(artifact.ref).decode("utf-8")
                except UnicodeError as error:
                    raise ValueError("Scripted control target is not UTF-8 source: " + path) from error
                files[path] = SourceFile(path=path, content=content).content
        missing = set(paths) - set(files)
        if missing:
            raise ValueError("Scripted control target is missing declared file: " + ", ".join(sorted(missing)))
        return files

    def respond(self, role, inputs):
        self.seen.append((role, deepcopy(inputs)))
        if role == "paperorchestra_model":
            from ..integrations.paperorchestra.controller_writer import fixture_model_response
            return fixture_model_response(inputs)
        if role == "paperorchestra_bindings":
            from ..integrations.paperorchestra.manuscript_binding import fixture_bindings
            return fixture_bindings(inputs["latex"], inputs["ledger"]).model_dump(mode="json")
        if role in {"initial_idea", "additional_idea"}:
            seeds = self.legacy_response("seeds", {**inputs, "count": inputs["count"]})["ideas"]
            return seeds[inputs["index"]]
        if role == "novelty_checker":
            sources = inputs["literature"]["sources"]
            return {"score": 0, "overlap": "Scripted methods are established controls.",
                    "distinction": "No novelty claimed.", "confidence": 0,
                    "source_ids": [s["id"] for s in sources],
                    "limitations": ["Scripted assessment is not evidence of scientific novelty"],
                    "status": "unverified"}
        if role == "baseline_plan":
            return {"mode": inputs["required_mode"], "reason": "Declared scripted baseline provenance",
                    "source_ids": ["task"]}
        if role == "baseline_coder":
            return inputs["reference"]
        if role == "baseline_engineer":
            return inputs["candidate"]
        if role == "full_set_coder":
            return {"action": "ReuseSnapshot", "reason": "Trusted runner owns the phase/data scaling"}
        if role == "selection_review":
            return {"selected_id": inputs["eligible_ids"][0],
                    "reason": "Select only among controller-eligible measured results", "limitations": ["Scripted selection"]}
        if role == "method_disposition":
            return {"action": "ChangeMethod", "reason": "Measure the declared prepared fixture revision"}
        if role == "ablation_planner":
            old = self.legacy_response("ablations", inputs)
            return {"plans": [{"id": f"component-{i}", "component": p["purpose"],
                "question": p["purpose"], "changed_files": [p["changed_file"]],
                "invariants": ["Preserve all other source bytes, data, seeds and execution protocol"],
                "known_confounds": ["Scripted design; causal interpretation is not certified"]}
                for i, p in enumerate(old["plans"]) if p["changed_file"] is not None]}
        if role in {"ablation_coder", "rebuttal_coder"}:
            target = Candidate.model_validate(self._control(inputs))
            before = {v["path"]: v["content"] for v in inputs["parent"]["files"]}
            paths = inputs["change_plan"]["changed_files"]
            if target.command != inputs["parent"]["command"]:
                raise ValueError("Scripted control target changes the parent's command")
            if set(paths) - set(before):
                raise ValueError("Scripted parent is missing a declared existing edit target")
            after = self._target_files(target, paths)
            edits = [{"path": p, "before": before[p], "after": after[p],
                      "reason": "Declared fixture component removal"}
                     for p in paths]
            return {"parent_source_ref": inputs["parent"]["parent_source_ref"], "edits": edits,
                    "description": "Scripted exact-parent component test"}
        if role == "control_reviewer":
            return {"decision": "DiagnosticOnly", "reason": "Execute the declared fixture diagnostic",
                    "confounds": ["Scripted design judgment is not empirical causal proof"]}
        if role == "rebuttal_planner":
            plans = self.legacy_response("rebuttal", inputs)
            return {"plans": [{
                **{k: v for k, v in p.items() if k not in {"candidate", "changed_file"}},
                "change_plan": ({"changed_files": [p["changed_file"]], "reason": p["purpose"]}
                                if p["changed_file"] is not None else None),
            } for p in plans["plans"]], "unaddressed_concerns": plans["unaddressed_concerns"]}
        if role in {"initial_drafter", "draft_enhancer"}:
            from .manuscript import fixture_manuscript, ManuscriptRecord
            responses = [{"concern_id": r["concern_id"],
                          "status": {"executed": "executed", "partial": "limited", "unfulfilled": "unaddressed"}[r["status"]],
                          "experiment_ids": r["observation_ids"],
                          "explanation": "Actual recorded execution coverage; scientific resolution remains unverified"}
                         for r in inputs.get("supplementary_coverage", [])]
            if role == "initial_drafter":
                return fixture_manuscript(inputs["ledger"], inputs["sources"],
                                           concern_responses=responses).model_dump()
            manuscript = ManuscriptRecord.model_validate(inputs["previous_manuscript"])
            value = manuscript.model_dump()
            value["limitations"]["text"] += " Editorial feedback has been incorporated without changing experimental evidence."
            value["concern_responses"] = responses
            return value
        if role == "writing_critic":
            return {"decision": "ReviseManuscript" if inputs["pass_index"] == 0 else "Accept",
                    "summary": "Clarify the evidence boundary of the scripted manuscript.",
                    "concerns": [], "confidence": 1}
        if role == "scientific_review":
            from .reviews import fixture_review
            old = self.legacy_response("review", {
                **inputs, "has_rebuttal": bool(inputs.get("supplementary_coverage"))})
            score = old["score"]
            value = fixture_review(score=score, decision="Accept" if score >= 8 else "SupplementaryExperiments").model_dump()
            value["concerns"] = [{
                "id": "concern-" + digest(text.encode())[:20], "category": "reproducibility",
                "severity": "major", "text": text, "requested_action": "supplementary_experiment",
                "evidence_ids": [], "claim_ids": [], "confidence": 1,
                "resolution": "open", "resolution_reason": "",
            } for text in old["concerns"]]
            existing = {c["id"] for c in value["concerns"]}
            for prior in inputs.get("previous_concerns", []):
                if prior["id"] not in existing:
                    value["concerns"].append({**prior,
                        "resolution": "resolved" if score >= 8 else "unresolved",
                        "resolution_reason": "Scripted reviewer disposition; execution evidence does not certify scientific resolution"})
            return value
        return self.legacy_response(role, inputs)
