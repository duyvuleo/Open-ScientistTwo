"""Bind the actual PaperOrchestra document to the controller's evidence records.

The canonical record is a lossless, source-addressed review projection, not a
second writer. Its prose consists of exact source slices; numerical/claim spans
are replaced only by explicit ledger references. The unmodified TeX and PDF are
the authoritative publication artifacts. Mechanical identity and value checks
do not establish natural-language entailment or scientific completeness.
"""

from __future__ import annotations

from pathlib import Path
import math
import re
from typing import Literal

from pydantic import Field, model_validator

from ...artifacts import safe_path
from ...models import Record
from ...research.manuscript import (
    SECTIONS, CompilationResult, LedgerClaim, ManuscriptRecord, RenderedManuscript,
    SectionRecord, _bib_key, _number_matches, _source_index,
    validate_manuscript, verify_evidence_ledger,
)
from ...store import IntegrityError, canonical, digest
from .evidence import diagnostic_numeric_values

SectionName = Literal[
    "abstract", "introduction", "related_work", "method", "protocol", "results",
    "ablations", "discussion", "limitations", "reproducibility",
]
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\\?%)?(?![\w.])")
_CITE = re.compile(r"\\(?:no)?cite[a-zA-Z*]*(?:\[[^\]]*\])*\{([^}]+)\}")
_TABLE = re.compile(r"\\begin\{(tabular\*?|longtable|tabularx)\}.*?\\end\{\1\}", re.S)
# Only standard layout arguments are recognized; the match ends before the
# content argument. Arbitrary @{text}, >{macros}, or unrecognized width syntax
# must not become a route for hiding empirical numbers from the value gate.
_DIMENSION = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:em|ex|pt|in|cm|mm|bp|pc|dd|cc|sp|\\(?:line|text|column)width)|\\(?:line|text|column)width)"
_COLUMN_SPEC = r"\s*(?:\|\s*|@\{\s*\}\s*)*(?:[lcr]|[pmb]\{\s*" + _DIMENSION + r"\s*\})(?:\s*\||\s*@\{\s*\})*\s*"
_TABLE_LAYOUT = (
    r"(?<!\\)\\multicolumn\s*\{\s*[1-9]\d*\s*\}\s*\{" + _COLUMN_SPEC + r"\}(?=\s*\{)"
    r"|(?<!\\)\\multirow(?:\s*\[[tcb]\])?\s*\{\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*\}\s*\{\s*(?:\*|=|"
    + _DIMENSION + r")\s*\}(?=\s*\{)"
)
# TeX lengths and identifiers are syntax, not experimental measurements.
_SYNTAX = re.compile(
    r"\\(?:nolinkurl|url|label|ref|eqref|cite\w*|nocite)\{[^{}]*\}"
    r"|(?<!\w)\d*\.\d+(?:em|ex|pt|in|cm|mm|\\(?:line|text|column)width)"
    r"|" + _TABLE_LAYOUT
)


class SourceSpan(Record):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=100000)

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start or self.end - self.start != len(self.quote):
            raise ValueError("Source spans use exact Python character offsets and literal quotes")
        return self


class SourceSectionBinding(SourceSpan):
    section: SectionName


class SourceClaimBinding(SourceSpan):
    claim: LedgerClaim


class SourceNumericBinding(SourceSpan):
    evidence_id: str = Field(min_length=1, max_length=512)
    location: Literal["value", "per_seed_value", "seed", "diagnostic_value", "method_metadata"] = "value"
    seed_index: int | None = Field(default=None, ge=0, le=10000)
    diagnostic_path: list[str | int] | None = Field(default=None, min_length=1, max_length=5)
    metadata_path: list[str | int] | None = Field(default=None, min_length=1, max_length=8)


class PaperOrchestraBindings(Record):
    schema_version: Literal["paperorchestra-controller-bindings-v1"] = "paperorchestra-controller-bindings-v1"
    latex_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    title: str = Field(min_length=1, max_length=1000)
    sections: list[SourceSectionBinding] = Field(min_length=1, max_length=128)
    claims: list[SourceClaimBinding] = Field(default_factory=list, max_length=256)
    numeric_values: list[SourceNumericBinding] = Field(default_factory=list, max_length=2048)
    source_ids: list[str] = Field(default_factory=list, max_length=128)
    table_ids: list[str] = Field(default_factory=list, max_length=128)
    figure_ids: list[str] = Field(default_factory=list, max_length=128)


def _body(latex):
    if (latex.count(r"\begin{document}") != 1 or latex.count(r"\end{document}") != 1
            or len(latex.encode()) > 2_000_000):
        raise ValueError("Bindings require one bounded complete LaTeX document")
    start = latex.index(r"\begin{document}") + len(r"\begin{document}")
    end = latex.index(r"\end{document}")
    if start >= end:
        raise ValueError("LaTeX document closure precedes its body")
    return start, end


def _check_span(span, latex):
    if latex[span.start:span.end] != span.quote:
        raise ValueError("Binding does not match its exact source span")


def _required_numbers(latex):
    """Conservative decimal/percentage and table-cell gate, not a TeX interpreter."""
    syntax = [(m.start(), m.end()) for m in _SYNTAX.finditer(latex)]
    tables = [(m.start(), m.end()) for m in _TABLE.finditer(latex)]
    for match in _NUMBER.finditer(latex):
        if any(start <= match.start() and match.end() <= end for start, end in syntax):
            continue
        if "." in match[0] or "%" in match[0] or any(
            start <= match.start() < end for start, end in tables
        ):
            yield match


def _numeric_value(binding, ledger, diagnostic_values=None):
    entry = ledger.get("entries", {}).get(binding.evidence_id)
    if binding.location == "method_metadata":
        from .writer_context import CONTEXT_VERSION
        if (not entry or entry.get("kind") != "method" or entry.get("metadata_contract") != CONTEXT_VERSION
                or binding.metadata_path is None or binding.seed_index is not None or binding.diagnostic_path is not None):
            raise ValueError("Method metadata requires a typed method entry and exact metadata path")
        matches = [field for field in entry.get("numeric_metadata", [])
                   if field.get("context_path") == binding.metadata_path]
        if (len(matches) != 1 or matches[0].get("scope") != "nonempirical_method_protocol_only"
                or type(matches[0].get("value")) not in (int, float)
                or not math.isfinite(matches[0]["value"])):
            raise ValueError("Method metadata path is absent or not an allowed scalar")
        return matches[0]["value"]
    if binding.metadata_path is not None:
        raise ValueError("Only method-metadata bindings can carry metadata paths")
    if binding.location == "diagnostic_value":
        if (not entry or entry.get("kind") != "diagnostic" or binding.seed_index is not None
                or binding.diagnostic_path is None):
            raise ValueError("Diagnostic numerical bindings require an exact aggregate field path")
        values = (diagnostic_values[binding.evidence_id] if diagnostic_values is not None
                  else diagnostic_numeric_values(entry))
        path = tuple(binding.diagnostic_path)
        if path not in values:
            raise ValueError("Diagnostic numerical binding is not an approved aggregate field")
        return values[path]
    if binding.diagnostic_path is not None:
        raise ValueError("Only diagnostic numerical bindings can carry an aggregate field path")
    if not entry or entry.get("kind") not in {"measured", "derived"}:
        raise ValueError("Numerical binding must reference recorded numerical evidence")
    if binding.location == "value":
        if binding.seed_index is not None:
            raise ValueError("Aggregate binding cannot specify a seed index")
        return entry["value"]
    seeds = entry.get("per_seed", [])
    if binding.seed_index is None or binding.seed_index >= len(seeds):
        raise ValueError("Numerical binding requires an existing recorded seed index")
    return seeds[binding.seed_index]["value" if binding.location == "per_seed_value" else "seed"]


def _numeric_options(entry):
    if entry.get("kind") in {"measured", "derived"}:
        yield "value", None, None, entry["value"]
        for index, seed in enumerate(entry.get("per_seed", [])):
            yield "per_seed_value", index, None, seed["value"]
            yield "seed", index, None, seed["seed"]
    elif entry.get("kind") == "diagnostic":
        for path, number in diagnostic_numeric_values(entry).items():
            yield "diagnostic_value", None, list(path), number
    elif entry.get("kind") == "method" and entry.get("metadata_contract") == "paperorchestra-method-protocol-context-v1":
        for field in entry.get("numeric_metadata", []):
            if field.get("scope") == "nonempirical_method_protocol_only":
                yield "method_metadata", None, field["context_path"], field["value"]


def bind_paperorchestra_manuscript(
    bindings, latex: str, ledger, sources=(), *, synthetic_agents: bool,
    concern_responses=(),
) -> ManuscriptRecord:
    """Validate literal provenance, project source, then run the existing validator.

Every body character is mapped exactly once. Numeric tables cannot evade the
ordinary evidence checks by being left outside the canonical record. Original
source is retained separately for reviewers; placeholders never alter it.
    """
    bindings = PaperOrchestraBindings.model_validate(bindings)
    if digest(latex.encode()) != bindings.latex_sha256:
        raise ValueError("PaperOrchestra binding source hash mismatch")
    if bindings.title not in latex:
        raise ValueError("Canonical title must be a literal source excerpt")
    if ledger.get("includes_final") is not False:
        raise ValueError("Research writing cannot consume protected final evidence")
    start, end = _body(latex)
    cursor = start
    ordered = sorted(bindings.sections, key=lambda item: item.start)
    for section in ordered:
        _check_span(section, latex)
        if section.start != cursor or section.end > end:
            raise ValueError("Section bindings must cover the entire source body without gaps or overlap")
        cursor = section.end
    if cursor != end:
        raise ValueError("Section bindings omit part of the source body")
    replacements = []
    claim_ids = set()
    for bound in bindings.claims:
        _check_span(bound, latex)
        if bound.claim.id in claim_ids or bound.claim.text != bound.quote:
            raise ValueError("Claim identities must be unique and their text must equal the literal source quote")
        claim_ids.add(bound.claim.id)
        if not any(section.start <= bound.start and bound.end <= section.end for section in ordered):
            raise ValueError("Claim must be wholly inside one bound source section")
        replacements.append((bound.start, bound.end, "[Bound claim " + bound.claim.id + "]"))
    covered_numbers = []
    diagnostic_values = {key: diagnostic_numeric_values(entry) for key, entry in ledger.get("entries", {}).items()
                         if entry.get("kind") == "diagnostic"}
    for bound in bindings.numeric_values:
        _check_span(bound, latex)
        if bound.location == "method_metadata":
            supported_claim = any(claim.claim.kind == "method"
                and bound.evidence_id in claim.claim.evidence_ids
                and claim.start <= bound.start and bound.end <= claim.end for claim in bindings.claims)
            supported_section = any(section.section in {"method", "protocol", "reproducibility"}
                and section.start <= bound.start and bound.end <= section.end for section in ordered)
            overlapping_result = any(claim.claim.kind in {"measured", "derived"}
                and claim.start < bound.end and bound.start < claim.end for claim in bindings.claims)
            if not supported_claim or not supported_section or overlapping_result:
                raise ValueError("Method metadata cannot validate an empirical result or an unclassified numeric span")
        if not _NUMBER.fullmatch(bound.quote) or not _number_matches(
            bound.quote.replace(r"\%", "%"), [_numeric_value(bound, ledger, diagnostic_values)]
        ):
            raise ValueError("Rendered numerical value differs from its recorded evidence")
        if (bound.start, bound.end) in covered_numbers:
            raise ValueError("Duplicate numerical source binding")
        covered_numbers.append((bound.start, bound.end))
        if not any(a <= bound.start and bound.end <= b for a, b, _ in replacements):
            replacements.append((bound.start, bound.end, "[Bound value " + bound.evidence_id + "]"))
    numerical_claims = [(b.start, b.end) for b in bindings.claims if b.claim.kind in {"measured", "derived"}]
    for match in _required_numbers(latex):
        if not any(a <= match.start() and match.end() <= b for a, b in covered_numbers + numerical_claims):
            raise ValueError("Unbound rendered numerical statement or table cell: " + match[0])
    # Formatting lengths are projected explicitly so the generic prose validator
    # cannot mistake an upstream table-width directive for a scientific score.
    for match in _SYNTAX.finditer(latex):
        if any("." in value or "%" in value for value in _NUMBER.findall(match[0])):
            if not any(a <= match.start() and match.end() <= b for a, b, _ in replacements):
                replacements.append((match.start(), match.end(), "[Source formatting or identifier]"))
    replacements.sort()
    if any(left[1] > right[0] for left, right in zip(replacements, replacements[1:])):
        raise ValueError("Overlapping source claim/value bindings are ambiguous")
    source_map = _source_index(sources or ledger.get("sources", {}))
    keys = {_bib_key(identifier): identifier for identifier in source_map}
    citations = set()
    for match in _CITE.finditer(latex):
        for key in match[1].split(","):
            if key.strip() not in keys:
                raise ValueError("PaperOrchestra source contains an unknown citation key")
            citations.add(keys[key.strip()])
    if citations - set(bindings.source_ids):
        raise ValueError("Source citations are missing from canonical bibliography bindings")
    sections = {}
    for name in SECTIONS:
        pieces, ids = [], []
        for section in ordered:
            if section.section != name:
                continue
            position = section.start
            text = []
            for left, right, replacement in replacements:
                if section.start <= left and right <= section.end:
                    text += [latex[position:left], replacement]
                    position = right
            text.append(latex[position:section.end])
            pieces.append("".join(text))
            ids.extend(bound.claim.id for bound in bindings.claims
                       if section.start <= bound.start and bound.end <= section.end)
        sections[name] = SectionRecord(
            text="\n".join(pieces) if pieces else "[No source passage mapped to this canonical section.]",
            claim_ids=ids, source_ids=sorted(citations) if pieces else [],
        )
    manuscript = ManuscriptRecord(
        title=bindings.title, **sections, claims=[bound.claim for bound in bindings.claims],
        source_ids=bindings.source_ids, table_ids=bindings.table_ids, figure_ids=bindings.figure_ids,
        synthetic_agents=synthetic_agents, concern_responses=list(concern_responses),
    )
    validation = validate_manuscript(manuscript, ledger, source_map)
    if validation.status != "pass":
        raise ValueError("Invalid source-bound manuscript: " + "; ".join(validation.errors))
    return manuscript


def bind_controller_manuscript(bindings, latex, ledger, sources=(), *, synthetic_agents,
                              concern_responses=()):
    """Controller-only strict-first binding with a recorded narrow normalization.

    The public strict binder stays unchanged. Only its exact overlap error may
    trigger the separately versioned pure normalizer; every other failure and
    every conflicting overlap still fails closed.
    """
    source_map = _source_index(sources or ledger.get("sources", {}))
    try:
        return bind_paperorchestra_manuscript(bindings, latex, ledger, source_map,
            synthetic_agents=synthetic_agents, concern_responses=concern_responses), None
    except ValueError as error:
        if str(error) != "Overlapping source claim/value bindings are ambiguous":
            raise
    from .binding_normalization import normalize_redundant_numeric_claims
    normalized = normalize_redundant_numeric_claims(bindings, latex, ledger, source_map,
        synthetic_agents=synthetic_agents, concern_responses=concern_responses)
    if not normalized["proof"]["changed"]:
        raise IntegrityError("Overlap normalization did not remove a redundant annotation")
    manuscript = bind_paperorchestra_manuscript(normalized["bindings"], latex, ledger, source_map,
        synthetic_agents=synthetic_agents, concern_responses=concern_responses)
    return manuscript, normalized["proof"]


def fixture_bindings(latex: str, ledger) -> PaperOrchestraBindings:
    """Explicit scripted fixture: exact slices and existing values, no new prose."""
    start, end = _body(latex)
    boundaries = [start, *[m.start() for m in re.finditer(r"\\section\*?\{", latex[start:end])]]
    # Matches above are relative to the document body.
    boundaries = [start] + [start + offset for offset in boundaries[1:]] + [end]
    sections = []
    names = [("limitations", "limitations"), ("reproduc", "reproducibility"),
             ("source library", "related_work"), ("protocol", "protocol"),
             ("measurement", "results"), ("identit", "reproducibility"),
             ("purpose", "introduction"), ("method", "method"),
             ("ablation", "ablations"), ("discussion", "discussion")]
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        quote = latex[left:right]
        heading = quote.split("\n", 1)[0].lower()
        section = next((name for word, name in names if word in heading), "abstract" if index == 0 else "discussion")
        sections.append(SourceSectionBinding(start=left, end=right, quote=quote, section=section))
    values = []
    numeric_options = {key: list(_numeric_options(entry)) for key, entry in sorted(ledger["entries"].items())}
    for match in _required_numbers(latex):
        found = None
        for key, options in numeric_options.items():
            for location, index, path, number in options:
                if location == "method_metadata":
                    continue  # Synthetic value matching cannot infer a method claim.
                if _number_matches(match[0].replace(r"\%", "%"), [number]):
                    found = SourceNumericBinding(start=match.start(), end=match.end(), quote=match[0],
                        evidence_id=key, location=location, seed_index=index, diagnostic_path=path)
                    break
            if found:
                break
        if found is None:
            raise ValueError("Scripted source contains a value absent from the evidence ledger")
        values.append(found)
    title = re.search(r"\\title\{([^{}]+)\}", latex)
    title_text = title[1] if title else "Scripted writing demonstration"
    if title_text not in latex:
        raise ValueError("Scripted source must contain an explicit title")
    return PaperOrchestraBindings(latex_sha256=digest(latex.encode()), title=title_text,
        sections=sections, numeric_values=values, source_ids=sorted(ledger.get("sources", {})),
        table_ids=sorted(ledger.get("tables", {})), figure_ids=sorted(ledger.get("figures", {})))


def binding_prompt_context(latex: str, ledger) -> dict:
    """Offer deterministic spans so a model selects evidence instead of counting.

Numeric candidates are value matches only. Choosing the right experiment, seed,
interpretation and attribution remains the binding agent's responsibility; this
helper does not turn the scripted fixture mapping into live ground truth.
    """
    start, end = _body(latex)
    boundaries = [start, *[start + match.start() for match in re.finditer(
        r"\\section\*?\{", latex[start:end])], end]
    sections = [{"span_id": "section-" + str(index), "start": left, "end": right,
                 "quote": latex[left:right]}
                for index, (left, right) in enumerate(zip(boundaries, boundaries[1:]))]
    passages, offset = [], start
    for index, line in enumerate(latex[start:end].splitlines(keepends=True)):
        if line.strip() and len(line) <= 10000:
            passages.append({"span_id": "line-" + str(index), "start": offset,
                             "end": offset + len(line), "quote": line})
        offset += len(line)
    numbers, groups, group_ids = [], {}, {}
    numeric_options = {key: list(_numeric_options(entry)) for key, entry in sorted(ledger.get("entries", {}).items())}
    for index, match in enumerate(_required_numbers(latex)):
        normalized = match[0].replace(r"\%", "%")
        if normalized not in group_ids:
            group_id = "numeric-group-" + str(len(groups))
            group_ids[normalized] = group_id
            candidates = []
            for key, options in numeric_options.items():
                for location, seed_index, path, value in options:
                    if _number_matches(normalized, [value]):
                        candidates.append({"evidence_id": key, "location": location,
                                           "seed_index": seed_index,
                                           **({"metadata_path" if location == "method_metadata" else "diagnostic_path": path}
                                              if path is not None else {})})
            groups[group_id] = {"numeric_text": normalized, "matching_candidates": candidates,
                                "candidates_truncated": False}
        numbers.append({"span_id": "number-" + str(index), "start": match.start(),
            "end": match.end(), "quote": match[0], "candidate_group": group_ids[normalized]})
    return {"schema_version": "paperorchestra-binding-context-v2",
        "latex_sha256": digest(latex.encode()), "offsets": "Python Unicode character offsets; end-exclusive",
        "body_start": start, "body_end": end, "canonical_sections": list(SECTIONS),
        "section_spans": sections, "passage_spans": passages[:1024],
        "passages_truncated": len(passages) > 1024, "required_numeric_spans": numbers,
        "numeric_candidate_groups": groups,
        "instructions": [
            "Copy supplied offsets and literal quotes; do not calculate offsets or rewrite source prose.",
            "Assign each supplied section span a canonical section; preserve all body spans in order.",
            "Bind each required numeric span to its actual experiment and recorded field, not merely a convenient equal value.",
            "Resolve each numeric span's candidate_group in numeric_candidate_groups; copy evidence_id, location and seed_index from the chosen candidate. Experiment labels and values are in the ledger.",
            "For location=diagnostic_value, also copy the exact diagnostic_path. It binds an aggregate confusion count, per-class statistic or matched-seed diagnostic value; individual examples and final labels are never supplied.",
            "For location=method_metadata, copy metadata_path and bind the span inside an explicit kind=method claim citing that evidence ID, within a method/protocol/reproducibility section. Metadata cannot support measured/derived claims or numbers in results/abstract sections. A source constructor literal absent from typed metadata is not a metric substitute; do not bind it merely because another recorded number is equal.",
            "The matching candidates certify numeric equality only; they do not certify correct attribution or scientific support.",
            "Claim text must be exactly its source quote; include only valid evidence IDs and source passage IDs.",
            "An unsupported numeric span requires source correction; do not fabricate evidence or suppress the span.",
        ]}


def _authoritative_receipt(store, report, latex_sha256, pdf_sha256, image_digest):
    selected = report.get("authoritative_compile", {})
    if (report.get("status") != "success" or selected.get("source_tex_sha256") != latex_sha256
            or selected.get("pdf_sha256") != pdf_sha256):
        raise IntegrityError("PaperOrchestra runtime does not bind the selected source and PDF")
    index = selected.get("operation_index")
    operations = report.get("pdf_operations", [])
    if type(index) is not int or not 0 <= index < len(operations):
        raise IntegrityError("Authoritative PaperOrchestra compilation operation is absent")
    operation = operations[index]
    if (operation.get("operation") != "compile"
            or operation.get("source_tex_sha256") != latex_sha256
            or operation.get("sha256") != pdf_sha256):
        raise IntegrityError("Selected PaperOrchestra compile operation has different artifacts")
    for record in report.get("controller_pdf_receipts", []):
        request, outputs = record.get("request", {}), record.get("outputs", {})
        if request.get("operation") != "compile" or "paper.pdf" not in outputs:
            continue
        source_name = request.get("tex_name") or ""
        if not source_name.endswith(".tex"):
            source_name += ".tex"
        if request.get("inputs", {}).get(source_name, {}).get("sha256") != latex_sha256:
            continue
        pdf = store.read(outputs["paper.pdf"])
        if digest(pdf) != pdf_sha256:
            continue
        row = store.db.execute("SELECT * FROM live_operations WHERE key=?", (record.get("operation_key"),)).fetchone()
        if row is None or row["kind"] != "probe" or row["status"] != "success" or not row["result_ref"]:
            raise IntegrityError("PaperOrchestra PDF lacks a successful shared-budget probe")
        saved = store.json(row["result_ref"])
        if canonical(saved) != canonical({k: v for k, v in record.items() if k != "operation_key"}):
            raise IntegrityError("PaperOrchestra PDF receipt differs from its committed operation")
        receipt = record.get("receipt", {})
        process = receipt.get("process", {})
        if (receipt.get("status") != "success" or receipt.get("operation") != "compile"
                or receipt.get("synthetic") or receipt.get("image_digest") != image_digest
                or receipt.get("network_disabled") is not True
                or receipt.get("shell_escape_disabled") is not True
                or "--network=none" not in receipt.get("argv", [])
                or process.get("code") != 0 or process.get("timed_out")
                or process.get("limit_exceeded")):
            raise IntegrityError("PaperOrchestra authoritative PDF violates compiler isolation policy")
        if canonical(request["inputs"]) != canonical(receipt.get("inputs")):
            raise IntegrityError("PaperOrchestra compile input closure differs from its receipt")
        declared = receipt.get("outputs", {}).get("paper.pdf", {})
        if declared != {"sha256": pdf_sha256, "bytes": len(pdf)}:
            raise IntegrityError("PaperOrchestra PDF differs from worker output identity")
        worker_ref, worker_sha = record.get("worker_ref"), record.get("worker_sha256")
        if (not worker_ref or digest(store.read(worker_ref)) != worker_sha
                or receipt.get("worker_sha256") != worker_sha):
            raise IntegrityError("PaperOrchestra PDF worker dependency closure is not sealed")
        return record, row["result_ref"], source_name
    raise IntegrityError("No committed isolated compilation matches the authoritative PaperOrchestra PDF")


def render_bound_manuscript(
    store, manuscript, bindings, ledger, sources=(), *, latex_path, pdf_path,
    runtime_ref, image_digest, project_dir=None, project_files=None, normalization_proof=None,
) -> RenderedManuscript:
    """Seal the actual selected upstream document; do not rewrite or recompile it."""
    latex_path, pdf_path = Path(latex_path), Path(pdf_path)
    for path, limit in ((latex_path, 2_000_000), (pdf_path, 25_000_000)):
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
            raise IntegrityError("PaperOrchestra selected output is absent, nonregular or oversized")
    raw, pdf = latex_path.read_bytes(), pdf_path.read_bytes()
    if not pdf.startswith(b"%PDF-"):
        raise IntegrityError("PaperOrchestra selected PDF has an invalid header")
    latex = raw.decode("utf-8")
    bindings = PaperOrchestraBindings.model_validate(bindings)
    manuscript = ManuscriptRecord.model_validate(manuscript)
    verify_evidence_ledger(store, ledger)
    source_map = _source_index(sources or ledger.get("sources", {}))
    if canonical(source_map) != canonical(ledger.get("sources", {})):
        raise IntegrityError("PaperOrchestra literature differs from its frozen evidence ledger")
    if normalization_proof is None:
        expected = bind_paperorchestra_manuscript(bindings, latex, ledger, source_map,
            synthetic_agents=manuscript.synthetic_agents, concern_responses=manuscript.concern_responses)
    else:
        expected, actual_proof = bind_controller_manuscript(bindings, latex, ledger, source_map,
            synthetic_agents=manuscript.synthetic_agents, concern_responses=manuscript.concern_responses)
        if actual_proof is None or canonical(actual_proof) != canonical(normalization_proof):
            raise IntegrityError("PaperOrchestra normalization proof differs from deterministic replay")
    if canonical(expected) != canonical(manuscript):
        raise IntegrityError("Canonical manuscript is not the exact source-bound review projection")
    report = store.json(runtime_ref)
    receipt, receipt_ref, source_name = _authoritative_receipt(
        store, report, digest(raw), digest(pdf), image_digest)
    if (project_dir is None) == (project_files is None):
        raise ValueError("Supply exactly one frozen PaperOrchestra project source")
    files = {}
    for name, member in receipt["request"]["inputs"].items():
        safe_path(name)
        if project_dir is not None:
            root = Path(project_dir)
            path = root / name
            if (root.is_symlink() or path.is_symlink() or not path.is_file()
                    or not path.resolve().is_relative_to(root.resolve())):
                raise IntegrityError("PaperOrchestra project input is missing or not a regular file")
            payload = path.read_bytes()
        else:
            payload = project_files.get(name)
        if (not isinstance(payload, bytes) or len(payload) != member["bytes"]
                or digest(payload) != member["sha256"]):
            raise IntegrityError("PaperOrchestra project differs from its complete compiled input closure")
        files[name] = payload
    if files[source_name] != raw:
        raise IntegrityError("Selected source differs from actual compiler entrypoint")
    if "main.tex" in files and files["main.tex"] != raw:
        raise IntegrityError("PaperOrchestra compiled project has a conflicting main.tex")
    if "references.bib" not in files:
        raise IntegrityError("PaperOrchestra project is missing its frozen bibliography")
    files["main.tex"] = raw
    manuscript_ref = store.put_json(manuscript, "canonical_manuscript")
    ledger_ref = store.put_json(ledger, "manuscript_evidence_ledger")
    validation = validate_manuscript(manuscript, ledger, source_map)
    validation_ref = store.put_json(validation, "manuscript_validation")
    bindings_ref = store.put_json(bindings, "paperorchestra_controller_bindings")
    normalization_ref = (store.put_json(normalization_proof, "paperorchestra_binding_normalization")
                         if normalization_proof is not None else None)
    pdf_ref = store.put(pdf, "pdf", "paperorchestra_authoritative_pdf")
    files.update({
        "manuscript.md": ("# " + manuscript.title + "\n\n"
            "Authoritative manuscript: the accompanying unchanged PaperOrchestra LaTeX and PDF.\n"
            "This is a literal source view, not an independently generated Markdown manuscript.\n"
            "Mechanical source/value checks passed. Scientific support and claim completeness remain unverified.\n\n"
            "````latex\n" + latex + "\n````\n").encode(),
        "evidence/manuscript.json": canonical(manuscript),
        "evidence/ledger.json": canonical(ledger),
        "evidence/validation.json": canonical(validation),
        "evidence/paperorchestra-bindings.json": canonical(bindings),
    })
    if normalization_ref:
        files["evidence/paperorchestra-binding-normalization.json"] = canonical(normalization_proof)
    files["inputs-manifest.json"] = canonical({
        "schema_version": "manuscript-inputs-v1",
        "files": {name: {"sha256": digest(value), "bytes": len(value)} for name, value in sorted(files.items())},
        "scope": "Authoritative upstream source and frozen controller evidence; scientific entailment remains unverified",
    })
    project_hash = digest(canonical({name: digest(value) for name, value in sorted(files.items())}))
    project = store.root / "manuscripts" / project_hash
    project.mkdir(parents=True, exist_ok=True)
    file_manifest = {}
    for name, value in sorted(files.items()):
        safe_path(name)
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
                raise IntegrityError("Immutable PaperOrchestra publication bundle was modified")
        else:
            with path.open("xb") as stream:
                stream.write(value)
        ref = store.put(value, path.suffix.lstrip(".") or "txt", "manuscript_file")
        file_manifest[name] = {"ref": ref, "sha256": digest(value), "bytes": len(value)}
    target = project / "main.pdf"
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != pdf:
            raise IntegrityError("Authoritative PaperOrchestra PDF was modified")
    else:
        with target.open("xb") as stream:
            stream.write(pdf)
    compiled = CompilationResult(status="success", diagnostics="Preserved exact PaperOrchestra compiled output.",
        pdf_path=str(target), receipt_ref=receipt_ref, image_digest=image_digest,
        network_disabled=True, shell_escape_disabled=True)
    writer = {"type": "paperorchestra", "bindings_ref": bindings_ref, "runtime_ref": runtime_ref,
        "latex_sha256": digest(raw), "source_pdf_ref": pdf_ref, "compilation_receipt_ref": receipt_ref,
        "source_entrypoint": source_name, "image_digest": image_digest}
    if normalization_ref:
        writer["binding_normalization_ref"] = normalization_ref
    manifest = {"schema_version": "manuscript-bundle-v1", "project_sha256": project_hash,
        "files": file_manifest, "template": {"source": "pinned official PaperOrchestra"},
        "manuscript_ref": manuscript_ref, "ledger_ref": ledger_ref, "validation_ref": validation_ref,
        "compilation_attempts": [compiled.model_dump(mode="json")], "pdf_ref": pdf_ref,
        "compilation_status": "success", "writer": writer}
    manifest_ref = store.put_json(manifest, "manuscript_bundle")
    verify_bound_manifest(store, manifest, expected_image_digest=image_digest)
    return RenderedManuscript(manifest_ref=manifest_ref, manuscript_ref=manuscript_ref,
        ledger_ref=ledger_ref, validation_ref=validation_ref, project_path=str(project),
        markdown_ref=file_manifest["manuscript.md"]["ref"], latex_ref=file_manifest["main.tex"]["ref"],
        bibliography_ref=file_manifest["references.bib"]["ref"], figure_refs={},
        compilation=compiled, pdf_ref=pdf_ref)


def verify_bound_manifest(store, manifest, *, expected_image_digest=None):
    """Replay binding and authoritative-PDF checks using only sealed artifacts."""
    writer = manifest.get("writer", {})
    if writer.get("type") != "paperorchestra":
        raise IntegrityError("Expected an authoritative PaperOrchestra manuscript bundle")
    ledger = store.json(manifest["ledger_ref"])
    verify_evidence_ledger(store, ledger)
    manuscript = ManuscriptRecord.model_validate(store.json(manifest["manuscript_ref"]))
    latex = store.read(manifest["files"]["main.tex"]["ref"])
    if digest(latex) != writer["latex_sha256"]:
        raise IntegrityError("Authoritative PaperOrchestra source hash changed")
    raw_bindings = store.json(writer["bindings_ref"])
    normalization_name = "evidence/paperorchestra-binding-normalization.json"
    normalization_ref = writer.get("binding_normalization_ref")
    if normalization_ref is None:
        if normalization_name in manifest["files"]:
            raise IntegrityError("Normalization artifact has no authoritative provenance reference")
        bound = bind_paperorchestra_manuscript(raw_bindings, latex.decode(), ledger,
            synthetic_agents=manuscript.synthetic_agents, concern_responses=manuscript.concern_responses)
    else:
        proof = store.json(normalization_ref)
        bound, actual_proof = bind_controller_manuscript(raw_bindings, latex.decode(), ledger,
            synthetic_agents=manuscript.synthetic_agents, concern_responses=manuscript.concern_responses)
        if actual_proof is None or canonical(actual_proof) != canonical(proof):
            raise IntegrityError("PaperOrchestra normalization proof differs from deterministic replay")
        for name, expected in ((normalization_name, canonical(proof)),
                ("evidence/paperorchestra-bindings.json", canonical(raw_bindings))):
            member = manifest["files"].get(name)
            if (member is None or store.read(member["ref"]) != expected
                    or member["sha256"] != digest(expected) or member["bytes"] != len(expected)):
                raise IntegrityError("PaperOrchestra normalization/raw-binding artifact alias differs")
    if canonical(bound) != canonical(manuscript):
        raise IntegrityError("Canonical review manuscript differs from the authoritative source")
    validation = validate_manuscript(manuscript, ledger)
    if canonical(validation) != canonical(store.json(manifest["validation_ref"])):
        raise IntegrityError("PaperOrchestra evidence validation differs from its sealed proof")
    pdf = store.read(manifest["pdf_ref"])
    if not pdf.startswith(b"%PDF-") or writer["source_pdf_ref"] != manifest["pdf_ref"]:
        raise IntegrityError("Authoritative PaperOrchestra PDF alias differs")
    image = writer["image_digest"]
    if expected_image_digest is not None and image != expected_image_digest:
        raise IntegrityError("PaperOrchestra PDF image differs from the frozen campaign policy")
    receipt, receipt_ref, source_name = _authoritative_receipt(
        store, store.json(writer["runtime_ref"]), digest(latex), digest(pdf), image)
    if receipt_ref != writer["compilation_receipt_ref"] or source_name != writer["source_entrypoint"]:
        raise IntegrityError("PaperOrchestra compile provenance alias differs")
    for name, item in receipt["request"]["inputs"].items():
        member = manifest["files"].get(name)
        if not member or {k: member[k] for k in ("sha256", "bytes")} != item:
            raise IntegrityError("Authoritative PaperOrchestra bundle omits or changes compiled inputs")
        payload = store.read(member["ref"])
        if digest(payload) != item["sha256"] or len(payload) != item["bytes"]:
            raise IntegrityError("Authoritative PaperOrchestra supporting input changed")
    return {"authoritative_source_binding": "pass", "authoritative_pdf": "pass",
        **({"binding_normalization": "pass"} if normalization_ref else {}),
        "shared_compilation_accounting": "pass", "rendered_numeric_values": "pass",
        "scientific_validity": "unverified", "natural_language_claim_completeness": "unverified",
        "citation_entailment": "unverified", "table_label_entailment": "unverified"}


def verify_bound_final_manifest(store, manifest):
    """Ensure protected final reporting preserves the source approved before freeze."""
    development = store.json(manifest["development_bundle_ref"])
    if development.get("writer", {}).get("type") != "paperorchestra":
        return {}
    verify_bound_manifest(store, development)
    before = store.read(development["files"]["main.tex"]["ref"]).decode()
    after = store.read(manifest["files"]["main.tex"]["ref"]).decode()
    expected = before.replace(r"\end{document}", r"\input{final-results.tex}" + "\n" + r"\end{document}")
    with_package = expected.replace(r"\begin{document}", r"\usepackage{longtable}" + "\n" + r"\begin{document}", 1)
    if after not in {expected, with_package}:
        raise IntegrityError("Final appendix changed PaperOrchestra development prose")
    for name, item in development["files"].items():
        if name not in {"main.tex", "manuscript.md", "inputs-manifest.json"}:
            if manifest["files"].get(name) != item:
                raise IntegrityError("Final appendix changed a frozen PaperOrchestra supporting artifact")
    return {"paperorchestra_development_source_preserved": "pass"}
