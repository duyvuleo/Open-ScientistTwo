"""Deterministic, explicitly scripted writing over a verified evidence bundle.

This adapter supplies synthetic agent responses to the official upstream agents.
All experimental numbers are copied from the frozen ledger. Scripted reviewer
scores exercise upstream control flow and are never scientific review evidence.
"""

import json
import math
from pathlib import Path

from .evidence import verify_evidence_bundle
from .runtime import ModelRequest, ModelResponse, UPSTREAM_REVISION


def _tex(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(text))


def _number(value):
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("Fixture cannot render a missing or nonfinite measured value")
    return format(value, ".12g")


class EvidenceFixtureTransport:
    """No model calls, no generated experimental values, no claimed review quality."""

    name = "evidence-grounded-scripted-paperorchestra-v1"
    synthetic = True

    def __init__(self, materials_dir: Path):
        self.materials = Path(materials_dir)
        self.manifest = verify_evidence_bundle(self.materials)
        if self.manifest["upstream_revision"] != UPSTREAM_REVISION:
            raise ValueError("Evidence bundle targets a different PaperOrchestra revision")
        self.ledger = json.loads((self.materials / "evidence_ledger.json").read_text())
        self.citations = json.loads((self.materials / "citation_map.json").read_text())
        all_entries = [(key, item) for key, item in sorted(self.ledger["entries"].items())
                       if item.get("kind") == "measured"]
        if not all_entries:
            raise ValueError("Writing demonstration requires committed measured evidence")
        # A compact example is deliberately not a comprehensive campaign paper.
        self.entries = sorted(all_entries, key=lambda pair: (
            pair[1]["protocol_hash"], pair[1]["metric"], pair[1]["observation_id"]))[:10]
        for _, item in self.entries:
            _number(item["value"])
            for seed in item.get("per_seed", []):
                _number(seed["value"])
        self.requests: list[ModelRequest] = []
        self.refined = False
        self.raw_latex = self._document(refined=False)
        self.refined_latex = self._document(refined=True)

    def _document(self, *, refined):
        lines = [r"\documentclass{article}", r"\usepackage{iclr2025_conference}",
                 r"\usepackage[T1]{fontenc}", r"\usepackage{booktabs,longtable,array}",
                 r"\usepackage{hyperref}", r"\usepackage{url}",
                 r"\hypersetup{hidelinks}", r"\setlength{\emergencystretch}{3em}",
                 r"\begin{document}",
                 r"\lhead{Scripted integration demonstration; not a submission}",
                 r"\begin{center}{\LARGE\bf Scripted writing demonstration}\\[0.5em]",
                 r"{\large ScientistTwo with adapted official PaperOrchestra}\end{center}",
                 r"\begin{abstract}",
                 "This document demonstrates an auditable writing integration using frozen development "
                 "evidence. The official PaperOrchestra outline, section-writing, content-refinement "
                 "and formatting agent methods execute with deterministic scripted responses. "
                 "Experimental values below are copied from the exported evidence ledger. "
                 "The prose and reviewer responses are synthetic workflow fixtures. "
                 "This document establishes neither autonomous writing quality nor scientific acceptance.",
                 r"\end{abstract}", r"\section{Purpose and implementation boundary}",
                 "ScientistTwo provides the research controller and evidence package. "
                 "PaperOrchestra supplies the pinned upstream agent control flow and prompts. "
                 "A frozen literature library replaces upstream retrieval, and a separately isolated "
                 "service compiles and renders the document. These adaptations are explicit. "
                 "The example is not an exact reproduction of the original system or its reported results.",
                 r"\section{Evidence and protocol}",
                 "The writer receives recorded aggregate measurements, matched seed values, source "
                 "identities and citations. It receives no protected final-test results. "
                 "No new training, candidate execution or metric calculation occurs inside this writer. "
                 "The table is a compact selection; the complete exported ledger accompanies the artifact. "
                 "Measurements from different protocols or metrics must not be averaged or treated as "
                 "controlled comparisons. Timing results, if present, retain their own metric labels.",
                 r"\section{Recorded development measurements}",
                 r"\small", r"\begin{longtable}{p{0.27\linewidth}p{0.18\linewidth}p{0.20\linewidth}p{0.23\linewidth}}",
                 r"\toprule Observation & Metric & Aggregate & Seed values \\\midrule\endhead"]
        for key, entry in self.entries:
            seeds = "; ".join(_tex(seed["seed"]) + ": " + _number(seed["value"])
                              for seed in entry.get("per_seed", []))
            # \nolinkurl permits long observation identifiers to wrap naturally.
            observation = entry["observation_id"]
            if any(char in observation for char in "{}\\%#"):
                observation_tex = _tex(observation)
            else:
                observation_tex = r"\nolinkurl{" + observation + "}"
            lines.append(" & ".join([observation_tex, _tex(entry["metric"]),
                                      _number(entry["value"]), seeds]) + r" \\")
        lines += [r"\bottomrule\end{longtable}", r"\normalsize",
                  "Aggregates and per-seed values above are recorded observations, not independently "
                  "recomputed in the writing stage. Each row is bound to a complete evidence identifier "
                  "and protocol identity in the next section. No significance, causality, novelty or "
                  "generalization conclusion follows from this table alone.",
                  r"\section{Evidence identities}"]
        for identifier, entry in self.entries:
            lines += [r"\noindent\textbf{" + _tex(entry["observation_id"]) + r"}\par",
                      r"{\scriptsize\nolinkurl{" + identifier + r"}}\par",
                      r"Phase: " + _tex(entry["phase"]) + r". Protocol: {\scriptsize\nolinkurl{" +
                      entry["protocol_hash"] + r"}}\par\smallskip"]
        lines += [r"\section{Scripted refinement and limitations}",
                  "The reviewer responses and their changing scores are predetermined test inputs. "
                  "They exercise upstream acceptance and refinement branches; they do not measure "
                  "peer-review quality or imply conference acceptance. No model request is needed "
                  "for this scripted example. A live backend requires a separate explicit enable flag "
                  "and bounded model configuration."]
        if refined:
            lines += ["The refinement adds this explicit boundary: successful compilation and artifact "
                      "checks establish execution and provenance only. Scientific validity, literature "
                      "coverage, prose entailment, and autonomous research quality remain unverified. "
                      "Method revisions require new controlled experiments outside this writing pipeline."]
        lines += [r"\section{Reproduction and source attribution}",
                  "The runtime report records the exact upstream source hashes, adaptation list, "
                  "bounded call ledger, and output hashes. The immutable evidence bundle retains the "
                  "measurement records used to populate this document. The PaperOrchestra source is "
                  "provided by Google Research under its upstream Apache license; ScientistTwo "
                  "supplies the integration and service adapters.",
                  r"Pinned PaperOrchestra revision: {\scriptsize\nolinkurl{" + UPSTREAM_REVISION + "}}."]
        if self.citations:
            lines += [r"\section{Frozen source library}",
                      "The following references are included for traceability only. Their presence "
                      "does not verify literature claims or novelty; the source bundle records the "
                      "verification status of each reference.",
                      r"\nocite{" + ",".join(sorted(self.citations)) + "}",
                      r"\bibliographystyle{iclr2025_conference}", r"\bibliography{references}"]
        lines += [r"\end{document}"]
        return "\n".join(lines) + "\n"

    def __call__(self, request: ModelRequest) -> ModelResponse:
        # Detect mutation even when the same transport object is reused.
        verify_evidence_bundle(self.materials)
        self.requests.append(request)
        if request.role == "outline":
            text = json.dumps({"title": "Scripted writing demonstration", "synthetic": True,
                               "sections": ["Purpose", "Evidence", "Results", "Limitations"]})
        elif request.role == "section_writing":
            text = self.raw_latex
        elif request.role == "content_refinement":
            self.refined = True
            text = ('```json\n{"synthetic":true,"change":"Clarify evidence and validation limits; '
                    'no method changes or new experiments."}\n```\n```latex\n'
                    + self.refined_latex + '\n```')
        elif request.role == "peer_review":
            score = 5 if self.refined else 4
            text = json.dumps({"Summary": "Explicitly scripted integration review; no scientific validation.",
                "Strengths": ["Frozen measurement identities retained."],
                "Weaknesses": ["Writing quality and scientific validity are not evaluated by this fixture."],
                "Originality": 1, "Quality": 2, "Clarity": 2, "Significance": 1,
                "Soundness": 2, "Presentation": 2, "Contribution": 1,
                "Questions": ["Clarify the limits of provenance checks."],
                "Limitations": ["All reviewer ratings are predetermined synthetic test inputs."],
                "Ethical Concerns": False, "Overall": score, "Confidence": 1,
                "Decision": "Reject", "synthetic": True})
        elif request.role == "format_review":
            text = json.dumps({"synthetic": True, "figure_and_tables": {}, "other_issues": []})
        elif request.role == "format_revision":
            raise ValueError("Fixture did not request formatting edits; unexpected workflow branch")
        else:
            raise ValueError("Unsupported scripted PaperOrchestra role: " + request.role)
        return ModelResponse(text, observed_tokens=0, model_name=self.name)
