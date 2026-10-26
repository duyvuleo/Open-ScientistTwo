"""Evidence-bound manuscripts and deterministic, non-executing publication bundles.

Model prose is never interpreted as TeX, Python, HTML, or a numeric table. The
mechanical validator checks identities and recorded values; scientific support
and the completeness of natural-language claims remain explicitly unverified.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
import re
from typing import Callable, Literal

from pydantic import Field, model_validator

from ..artifacts import load_bundle, safe_path
from ..live.models import Candidate, ConcernResponse, Draft, Hypothesis, Observation
from ..live.evidence_context import committed_control_context
from ..live.reporting import tex
from ..models import CriticDecision, Record
from ..store import IntegrityError, canonical, digest, file_digest

SECTIONS = (
    "abstract", "introduction", "related_work", "method", "protocol", "results",
    "ablations", "discussion", "limitations", "reproducibility",
)
_TITLES = {name: name.replace("_", " ").title() for name in SECTIONS}
_ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,191}$"
# This deliberately detects common embedded empirical values, not arbitrary prose
# semantics. Years, section numbers and model names are not automatically scores.
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?(?![\w.])")
_HISTORY_VERSION = "research-history-context-v1"
_HISTORY_LIMIT = 256


class SectionRecord(Record):
    text: str = Field(min_length=1, max_length=50000)
    claim_ids: list[str] = Field(default_factory=list, max_length=256)
    source_ids: list[str] = Field(default_factory=list, max_length=128)


class LedgerClaim(Record):
    id: str = Field(pattern=_ID_PATTERN)
    kind: Literal["measured", "derived", "method", "literature", "interpretation"]
    text: str = Field(min_length=1, max_length=10000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=128)
    source_ids: list[str] = Field(default_factory=list, max_length=128)
    passage_ids: list[str] = Field(default_factory=list, max_length=128)
    value: float | None = None
    attribution_scope: Literal["none", "package_comparison", "isolated_component"] = "none"


class ManuscriptRecord(Record):
    schema_version: Literal["manuscript-v1"] = "manuscript-v1"
    title: str = Field(min_length=1, max_length=1000)
    abstract: SectionRecord
    introduction: SectionRecord
    related_work: SectionRecord
    method: SectionRecord
    protocol: SectionRecord
    results: SectionRecord
    ablations: SectionRecord
    discussion: SectionRecord
    limitations: SectionRecord
    reproducibility: SectionRecord
    claims: list[LedgerClaim] = Field(default_factory=list, max_length=256)
    source_ids: list[str] = Field(default_factory=list, max_length=128)
    table_ids: list[str] = Field(default_factory=list, max_length=128)
    figure_ids: list[str] = Field(default_factory=list, max_length=128)
    synthetic_agents: bool = False
    concern_responses: list[ConcernResponse] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def unique_identifiers(self):
        for values, label in (
            ([claim.id for claim in self.claims], "claim"), (self.source_ids, "source"),
            (self.table_ids, "table"), (self.figure_ids, "figure"),
            ([item.concern_id for item in self.concern_responses], "concern response"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate manuscript {label} identifiers")
        return self


class ManuscriptValidation(Record):
    status: Literal["pass", "fail"]
    errors: list[str]
    checks: dict[str, Literal["pass", "fail", "unverified"]]
    qualifications: list[str] = Field(default_factory=lambda: [
        "Mechanical reference/value checks do not establish scientific truth or causal support.",
        "A citation ID or literal passage match does not establish entailment or novelty.",
        "Natural-language empirical-claim completeness requires independent review.",
    ])


class CompilationPolicy(Record):
    image_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    timeout_seconds: int = Field(default=60, ge=1, le=120)
    memory_mib: int = Field(default=1024, ge=128, le=2048)
    cpus: int = Field(default=1, ge=1, le=2)
    max_attempts: int = Field(default=1, ge=1, le=3)
    network: Literal[False] = False
    shell_escape: Literal[False] = False


class CompilationRequest(Record):
    project_dir: str
    project_manifest_sha256: str
    policy: CompilationPolicy
    attempt: int = Field(ge=1, le=3)
    main_file: Literal["main.tex"] = "main.tex"
    command: list[str] = Field(default_factory=lambda: [
        "pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "main.tex"
    ])
    commands: list[list[str]] = Field(default_factory=lambda: [
        ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ])


class CompilationResult(Record):
    status: Literal["success", "failed", "unavailable"]
    diagnostics: str = Field(max_length=50000)
    pdf_path: str | None = None
    receipt_ref: str | None = None
    image_digest: str | None = None
    network_disabled: bool = False
    shell_escape_disabled: bool = False
    timed_out: bool = False


class RenderedManuscript(Record):
    manifest_ref: str
    manuscript_ref: str
    ledger_ref: str
    validation_ref: str
    project_path: str
    markdown_ref: str
    latex_ref: str
    bibliography_ref: str
    figure_refs: dict[str, str]
    compilation: CompilationResult
    pdf_ref: str | None = None


def _plain(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _source_index(sources):
    values = sources.values() if isinstance(sources, dict) else sources
    result = {}
    for item in values:
        item = _plain(item)
        identifier = item["id"]
        if identifier in result:
            raise ValueError("Duplicate literature source ID")
        item = dict(item)
        # Legacy supplied sources get passages, not a fabricated retrieval record.
        if "passages" not in item:
            text = item.get("text", "")
            item["passages"] = ([{"id": identifier + ":supplied", "text": text,
                                  "sha256": digest(text.encode()), "scope": "supplied_excerpt"}]
                                if text else [])
            item.setdefault("metadata_status", "supplied")
            item.setdefault("content_status", "supplied_excerpt" if text else "metadata_only")
        result[identifier] = item
    return result


def _check_sources(sources):
    for source in sources.values():
        ids = set()
        for passage in source.get("passages", []):
            if passage["id"] in ids:
                raise IntegrityError("Duplicate source passage ID")
            ids.add(passage["id"])
            if digest(passage["text"].encode()) != passage.get("sha256"):
                raise IntegrityError("Source passage hash does not match its text")


def _source_verification(store, sources):
    result = {}
    for identifier, source in sources.items():
        content_identity = "pass" if source.get("passages") else "unverified"
        content_ref = source.get("content_ref")
        if content_ref:
            raw = store.read(content_ref)
            if digest(raw) != source.get("content_sha256"):
                raise IntegrityError("Literature content differs from its recorded hash")
        bound_retrieval = False
        for receipt in source.get("provenance", []):
            if receipt.get("status") != "retrieved":
                continue
            key, body_ref = receipt.get("key"), receipt.get("body_ref")
            if not isinstance(key, str) or not key.startswith("research/literature/request/") or not body_ref:
                continue
            row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", (key,)).fetchone()
            if row is None or canonical(store.json(row[0])) != canonical(receipt):
                raise IntegrityError("Literature retrieval provenance is not bound to its committed request")
            if digest(store.read(body_ref)) != receipt.get("sha256"):
                raise IntegrityError("Literature retrieval body hash mismatch")
            if type(receipt.get("http_status")) is int and 200 <= receipt["http_status"] < 300:
                bound_retrieval = True
        result[identifier] = {
            "metadata_existence": "pass" if source.get("metadata_status") == "verified" and bound_retrieval else "unverified",
            "content_identity": content_identity,
            "support": "unverified",
            "scope": "Metadata retrieval identity and available passage bytes only; no citation entailment",
        }
    return result


def _verify_ref(store, ref):
    """Verify arbitrary artifacts by streaming, without loading checkpoint binaries."""
    row = store.db.execute("SELECT sha,size FROM artifacts WHERE path=?", (ref,)).fetchone()
    path = (store.root / ref).resolve()
    if (row is None or not path.is_relative_to((store.root / "artifacts").resolve())
            or not path.is_file() or path.stat().st_size != row["size"]
            or file_digest(path) != row["sha"]):
        raise IntegrityError("Missing, unregistered or modified manuscript evidence artifact")


def _committed_observation(store, raw):
    observation = Observation.model_validate(_plain(raw))
    row = store.db.execute(
        "SELECT output_ref FROM calls WHERE key=?", ("development/" + observation.id,)
    ).fetchone()
    if row is None:
        raise IntegrityError("Manuscript evidence must be a committed development observation")
    committed = store.json(row[0])
    if canonical(committed) != canonical(observation):
        raise IntegrityError("Manuscript evidence differs from the committed observation")
    if not observation.protocol_ref:
        raise IntegrityError("Manuscript observations require a frozen protocol")
    protocol = store.json(observation.protocol_ref)
    if digest(canonical(protocol)) != observation.protocol_hash:
        raise IntegrityError("Observation protocol hash mismatch")
    if protocol.get("phase") != observation.phase:
        raise IntegrityError("Observation protocol phase mismatch")
    seeds = protocol.get("seeds", [])
    if (not isinstance(seeds, list) or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise IntegrityError("Observation protocol seeds are invalid")
    if observation.status == "success" and len(seeds) != len(observation.values):
        raise IntegrityError("Observation per-seed values differ from its protocol")
    # Hash/registration validation only. Actual metric recomputation remains audit's job.
    for ref in [observation.code_ref, *observation.prediction_refs,
                *observation.log_refs, *observation.evaluation_refs]:
        _verify_ref(store, ref)
    return observation, row[0], protocol


def _committed_history_trial(store, ref):
    """Bind a trial/decision to this store and its committed non-final observation."""
    if not isinstance(ref, str) or not re.fullmatch(r"artifacts/[a-f0-9]{64}\.json", ref):
        raise IntegrityError("Research history requires a content-addressed trial reference")
    _verify_ref(store, ref)
    trial = store.json(ref)
    if not isinstance(trial, dict) or set(trial) != {"id", "idea", "candidate", "result", "decision", "parents"}:
        raise IntegrityError("Research history trial has an unsupported record shape")
    identifier = trial["id"]
    if not isinstance(identifier, str) or not identifier or identifier.startswith("final/"):
        raise IntegrityError("Research history cannot cite final or unnamed trials")
    row = store.db.execute("SELECT input_hash,output_ref FROM calls WHERE key=?",
                           ("live-trial/" + identifier,)).fetchone()
    if (row is None or row["output_ref"] != ref
            or row["input_hash"] != digest(canonical({"key": identifier}))):
        raise IntegrityError("Research history trial is not committed to this campaign")
    observation = Observation.model_validate(trial["result"])
    if observation.phase not in {"subset", "full", "rebuttal"} or observation.id != identifier:
        raise IntegrityError("Research history must bind the same non-final trial observation")
    observation, observation_ref, _ = _committed_observation(store, observation)
    idea = Hypothesis.model_validate(trial["idea"])
    decision = CriticDecision.model_validate(trial["decision"])
    Candidate.model_validate(trial["candidate"])
    if not isinstance(trial["parents"], list) or any(not isinstance(parent, str) for parent in trial["parents"]):
        raise IntegrityError("Research history lineage is malformed")
    if observation.source_tree_ref:
        _verify_ref(store, observation.source_tree_ref)
    identifier = "history:" + digest(canonical([ref, observation_ref]))[:24]
    metrics = observation.metrics or {observation.metric: observation.value}
    entry = {
        "kind": "research_history", "trial_id": trial["id"], "trial_ref": ref,
        "observation_id": observation.id, "observation_ref": observation_ref,
        "phase": observation.phase, "execution_status": observation.status,
        "decision": decision.model_dump(mode="json"), "parents": trial["parents"],
        "title": idea.title, "hypothesis": idea.hypothesis,
        "source_ref": observation.source_tree_ref or observation.code_ref,
        "protocol_ref": observation.protocol_ref, "protocol_hash": observation.protocol_hash,
        "measurement_ids": (["measurement:" + digest(canonical([observation.id, metric]))[:24]
                            for metric in sorted(metrics)] if observation.status == "success" else []),
        "support_scope": "Committed execution status and recorded scientific decision; decision correctness and causal support remain unverified",
    }
    return identifier, entry, observation


def freeze_research_history(store, trials):
    """Freeze only explicitly offered committed trials, never a future store scan."""
    trials = list(trials)
    if len(trials) > _HISTORY_LIMIT:
        raise ValueError("Research history exceeds its bounded record count")
    refs = []
    for offered in trials:
        offered = _plain(offered)
        if not isinstance(offered, dict) or not isinstance(offered.get("id"), str):
            raise IntegrityError("Research history requires an explicitly offered trial")
        row = store.db.execute("SELECT output_ref FROM calls WHERE key=?",
                               ("live-trial/" + offered["id"],)).fetchone()
        if row is None:
            raise IntegrityError("Research history trial has not been committed")
        ref = row[0]
        _committed_history_trial(store, ref)
        if canonical(store.json(ref)) != canonical(offered):
            raise IntegrityError("Offered research history differs from its frozen trial")
        if ref in refs:
            raise IntegrityError("Duplicate research history trial")
        refs.append(ref)
    return refs


def writer_history_contract(ledger):
    """Versioned writer guidance; a history citation is not a numeric measurement."""
    history = ledger.get("research_history")
    if not isinstance(history, dict) or history.get("schema_version") != _HISTORY_VERSION:
        raise ValueError("Versioned committed writer history is required")
    return {
        "schema_version": "research-writer-context-v2",
        "history_context_version": _HISTORY_VERSION,
        "history_ids": list(history["entries"]),
        "rules": [
            "For qualitative search outcomes or critic decisions, use kind=interpretation and cite the relevant history:* evidence_ids from this ledger.",
            "Use each history entry's measurement_ids for numerical claims. Measured/derived claims still require exactly one matching numerical ledger entry; never use a history ID as a numerical measurement.",
            "Execution status and scientific decision are separate facts. A Bad decision after successful execution is not an execution failure, and a decision's reason is not established scientific truth.",
            "Do not use raw trial IDs, unrelated task/literature citations, or unregistered history as substitutes for ledger evidence. Missing or unsupported claims must be omitted or remain explicitly unaddressed.",
            "This frozen snapshot contains non-final research history only. Later trials and final results do not enlarge its evidence scope.",
        ],
    }


def build_evidence_ledger(
    store,
    observations,
    sources=(),
    *,
    diagnostics=(),
    method_facts=None,
    history_refs=None,
    include_final: bool = False,
    method_frozen: bool = False,
    timing_evidence: bool | str = True,
) -> dict:
    """Freeze writer-visible evidence; protected final records are excluded by default.

    Final records require an explicit frozen-method context. The caller must not
    pass that resulting ledger to refinement roles. This helper reads neither
    labels nor checkpoint payloads and never executes candidate source.
    """
    if include_final and not method_frozen:
        raise ValueError("Final evidence requires a frozen method")
    sources = _source_index(sources)
    _check_sources(sources)
    history = None
    if history_refs is not None:
        if (not isinstance(history_refs, (list, tuple)) or len(history_refs) > _HISTORY_LIMIT
                or any(not isinstance(ref, str) for ref in history_refs)
                or len(set(history_refs)) != len(history_refs)):
            raise IntegrityError("Frozen research history references are invalid or duplicated")
        observations = list(observations)
        offered = {_plain(value)["id"]: value for value in observations}
        history = {"schema_version": _HISTORY_VERSION, "trial_refs": list(history_refs), "entries": {}}
        for ref in history_refs:
            identifier, entry, observed = _committed_history_trial(store, ref)
            if observed.id in offered:
                if canonical(offered[observed.id]) != canonical(observed):
                    raise IntegrityError("Writer history conflicts with supplied observation evidence")
            else:
                observations.append(observed)
                offered[observed.id] = observed
            history["entries"][identifier] = {"evidence_id": identifier, **entry}
    entries, records, tables, figures = {}, {}, {}, {}
    for raw in observations:
        if _plain(raw).get("phase") == "final" and not include_final:
            continue
        observation, ref, protocol = _committed_observation(store, raw)
        if observation.id in records:
            raise IntegrityError("Duplicate manuscript observation ID")
        records[observation.id] = {
            "observation_ref": ref, "phase": observation.phase, "status": observation.status,
            "protocol_hash": observation.protocol_hash, "protocol_ref": observation.protocol_ref,
            "code_ref": observation.code_ref, "source_tree_ref": observation.source_tree_ref,
            "seeds": protocol.get("seeds", []), "prediction_refs": observation.prediction_refs,
            "receipt_refs": observation.log_refs, "reason": observation.reason,
        }
        if observation.status != "success":
            continue
        metrics = observation.metrics or {observation.metric: observation.value}
        per_seed = observation.metric_values or {observation.metric: observation.values}
        for metric in sorted(metrics):
            key = "measurement:" + digest(canonical([observation.id, metric]))[:24]
            entries[key] = {
                "kind": "measured", "observation_id": observation.id, "metric": metric,
                "value": metrics[metric], "per_seed": [
                    {"seed": seed, "value": value}
                    for seed, value in zip(protocol["seeds"], per_seed[metric], strict=True)
                ], "protocol_hash": observation.protocol_hash,
                "phase": observation.phase, "observation_ref": ref,
                "source_ref": observation.source_tree_ref or observation.code_ref,
                "numeric_provenance": "committed_trusted_observation; recomputation_is_separate",
            }
            table_id = "table:" + digest(canonical([observation.protocol_hash, metric]))[:20]
            table = tables.setdefault(table_id, {
                "kind": "measurements", "metric": metric, "protocol_hash": observation.protocol_hash,
                "caption": "Recorded aggregate and matched per-seed measurements; no significance claim.",
                "evidence_ids": [],
            })
            table["evidence_ids"].append(key)
        if timing_evidence and observation.timing_refs:
            from .timing_evidence import TIMING_EVIDENCE_VERSION, timing_measurements
            measured = timing_measurements(store, observation, ref, protocol,
                contract=TIMING_EVIDENCE_VERSION if timing_evidence is True else timing_evidence)
            entries.update(measured)
            for key, entry in measured.items():
                table_id = "table:" + digest(canonical([observation.protocol_hash, entry["metric"]]))[:20]
                tables.setdefault(table_id, {
                    "kind": "measurements", "metric": entry["metric"],
                    "protocol_hash": observation.protocol_hash,
                    "caption": "Sealed paired timing: mean of per-seed statistics in seconds, not pooled latency; no significance claim.",
                    "evidence_ids": [],
                })["evidence_ids"].append(key)
    for table_id, table in sorted(tables.items()):
        table["evidence_ids"].sort(key=lambda key: entries[key]["observation_id"])
        figure_id = table_id.replace("table:", "figure:", 1)
        figures[figure_id] = {
            "table_id": table_id, "kind": "per_seed_dot_plot",
            "caption": "Individual seed measurements grouped only within the same protocol and metric.",
        }
    for raw in diagnostics:
        record = dict(_plain(raw))
        ref = record.get("artifact_ref")
        if not ref:
            raise IntegrityError("Writer diagnostics require an immutable artifact reference")
        committed = dict(store.json(ref))
        record.pop("artifact_ref", None)
        committed.pop("artifact_ref", None)
        if canonical(record) != canonical(committed):
            raise IntegrityError("Writer diagnostic differs from its immutable artifact")
        if record.get("status") != "success":
            continue
        key = "diagnostic:" + digest(canonical(record.get("request", record.get("id"))))[:24]
        entries[key] = {"kind": "diagnostic", "artifact_ref": ref,
                        "summary": record.get("summary", {}), "tables": record.get("tables", {}),
                        "arrays": record.get("arrays", {}), "limitations": record.get("limitations", []),
                        "provenance": record.get("provenance", {}),
                        "input_refs": record.get("input_refs", [])}
        for field, value in sorted(record.get("summary", {}).items()):
            if type(value) in (float, int) and math.isfinite(value):
                derived_key = "derived:" + digest(canonical([key, field]))[:24]
                entries[derived_key] = {
                    "kind": "derived", "operation": "diagnostic_summary", "input_ids": [key],
                    "field": field, "value": value,
                    "interpretation": "Trusted diagnostic summary; scientific interpretation unverified",
                }
        for table_name, rows in sorted(record.get("tables", {}).items()):
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise IntegrityError("Diagnostic table requires structured rows")
            if len(rows) > 10000:
                raise IntegrityError("Diagnostic table exceeds manuscript row limit")
            table_id = "table:" + digest(canonical([key, table_name]))[:20]
            tables[table_id] = {
                "kind": "diagnostic", "name": table_name, "diagnostic_ref": ref,
                "columns": sorted({column for row in rows for column in row}), "rows": rows,
                "caption": "Recorded diagnostic output: " + table_name + ". Interpret with its declared limitations.",
            }
    for identifier, raw in sorted((method_facts or {}).items()):
        fact = dict(_plain(raw))
        if not fact.get("evidence_ref") or not fact.get("text"):
            raise ValueError("Method facts require text and immutable evidence_ref")
        store.read(fact["evidence_ref"])
        if fact.get("metadata_contract") == "paperorchestra-method-protocol-context-v1":
            from ..integrations.paperorchestra.writer_context import verify_context_method_fact
            verify_context_method_fact(store, fact, records)
        entries["method:" + identifier] = {"kind": "method", **fact,
            "semantic_support": "unverified"}
    if history is not None:
        entries.update({identifier: {key: value for key, value in entry.items() if key != "evidence_id"}
                        for identifier, entry in history["entries"].items()})
    return {
        "schema_version": "evidence-ledger-v2" if history is not None else "evidence-ledger-v1",
        **({"research_history": history} if history is not None else {}),
        "entries": entries, "observations": records,
        "sources": sources, "source_verification": _source_verification(store, sources),
        "tables": tables, "figures": figures,
        "control_context": committed_control_context(store, [{"id": name} for name in records]),
        "includes_final": include_final,
        **({"timing_evidence_contract": next(entry["numeric_provenance"] for entry in entries.values()
                if entry.get("numeric_provenance", "").startswith("paired-timing-measurements-"))}
           if any(entry.get("numeric_provenance", "").startswith("paired-timing-measurements-") for entry in entries.values()) else {}),
        "claim_scope": "Numerical provenance and source identity only; scientific support unverified",
    }


def derive_comparison(ledger: dict, left: str, right: str, *, operation: str = "difference") -> str:
    """Register a deterministic same-protocol comparison (left minus right)."""
    if operation not in {"difference", "relative_change"}:
        raise ValueError("Unsupported derived comparison")
    a, b = (ledger["entries"][key] for key in (left, right))
    if (a.get("kind") != "measured" or b.get("kind") != "measured"
            or a["metric"] != b["metric"] or a["protocol_hash"] != b["protocol_hash"]
            or [item["seed"] for item in a["per_seed"]] != [item["seed"] for item in b["per_seed"]]):
        raise ValueError("Derived comparisons require matching metrics, protocols and seed order")
    if operation == "relative_change" and b["value"] == 0:
        raise ValueError("Relative change is undefined for a zero reference")
    value = a["value"] - b["value"]
    if operation == "relative_change":
        value /= abs(b["value"])
    key = "derived:" + digest(canonical([left, right, operation]))[:24]
    ledger["entries"][key] = {
        "kind": "derived", "operation": operation, "input_ids": [left, right],
        "value": value, "metric": a["metric"], "protocol_hash": a["protocol_hash"],
        "interpretation": "Descriptive comparison; no statistical significance or causal claim",
    }
    return key


def verify_evidence_ledger(store, ledger):
    """Rebuild evidence from commitments before rendering; derived values are recomputed."""
    version = ledger.get("schema_version")
    if ledger.get("timing_evidence_contract") not in {None, "paired-timing-measurements-v1", "paired-timing-measurements-v2"}:
        raise IntegrityError("Unsupported timing evidence ledger contract")
    if version not in {"evidence-ledger-v1", "evidence-ledger-v2"}:
        raise IntegrityError("Unsupported manuscript evidence ledger version")
    history_refs = None
    if version == "evidence-ledger-v2":
        history = ledger.get("research_history", {})
        if history.get("schema_version") != _HISTORY_VERSION or not isinstance(history.get("trial_refs"), list):
            raise IntegrityError("Versioned manuscript history snapshot is missing")
        history_refs = history["trial_refs"]
    elif "research_history" in ledger:
        raise IntegrityError("Legacy manuscript ledger cannot silently include new history semantics")
    observations = [store.json(item["observation_ref"]) for item in ledger["observations"].values()]
    diagnostics, methods = [], {}
    for key, entry in ledger["entries"].items():
        if entry.get("kind") == "diagnostic":
            record = store.json(entry["artifact_ref"])
            diagnostics.append({**record, "artifact_ref": entry["artifact_ref"]})
        elif entry.get("kind") == "method":
            if not key.startswith("method:"):
                raise IntegrityError("Invalid method evidence identifier")
            methods[key.removeprefix("method:")] = {name: value for name, value in entry.items()
                                                       if name not in {"kind", "semantic_support"}}
    expected = build_evidence_ledger(
        store, observations, ledger.get("sources", {}), diagnostics=diagnostics, method_facts=methods,
        history_refs=history_refs,
        include_final=ledger.get("includes_final", False), method_frozen=ledger.get("includes_final", False),
        timing_evidence=ledger.get("timing_evidence_contract") or False,
    )
    for key, entry in ledger["entries"].items():
        if entry.get("kind") == "derived" and entry.get("operation") != "diagnostic_summary":
            inputs = entry.get("input_ids", [])
            if len(inputs) != 2:
                raise IntegrityError("Derived manuscript comparison has invalid inputs")
            try:
                derived_key = derive_comparison(expected, inputs[0], inputs[1], operation=entry["operation"])
            except (KeyError, ValueError) as exc:
                raise IntegrityError("Invalid derived manuscript comparison") from exc
            if derived_key != key:
                raise IntegrityError("Derived manuscript identifier does not match its inputs")
    if canonical(expected) != canonical(ledger):
        raise IntegrityError("Manuscript evidence ledger differs from committed measurements or derivations")


def _numeric_tokens(text):
    return [match.group(0) for match in _NUMBER.finditer(text)]


def _number_matches(token, values):
    number = float(token.rstrip("%")) / (100 if token.endswith("%") else 1)
    return any(math.isclose(number, value, rel_tol=1e-6, abs_tol=1e-10) for value in values)


def _attribution_errors(claim, ledger):
    """Enforce declared attribution scope, never infer scientific intent from prose."""
    if claim.attribution_scope != "isolated_component":
        return []
    entries = ledger.get("entries", {})
    observation_refs = {value["observation_ref"]: name
                        for name, value in ledger.get("observations", {}).items()}
    observations, visited = set(), set()

    def visit(identifier):
        if identifier in visited:
            return
        visited.add(identifier)
        entry = entries.get(identifier, {})
        if entry.get("observation_id"):
            observations.add(entry["observation_id"])
        for ref in entry.get("input_refs", []):
            if ref in observation_refs:
                observations.add(observation_refs[ref])
        for parent in entry.get("input_ids", []):
            visit(parent)

    for identifier in claim.evidence_ids:
        visit(identifier)
    controls = [item for item in ledger.get("control_context", [])
                if item["control_id"] in observations]
    if not controls or any(item["matched_id"] not in observations for item in controls):
        return [f"Claim {claim.id} declares isolated-component attribution without both arms of a recorded control"]
    if any(item.get("attribution_restriction", {}).get("isolated_component") != "eligible_for_review"
           for item in controls):
        return [f"Claim {claim.id} declares isolated-component attribution from a diagnostic-only, rejected, unreviewed or confounded control"]
    return []


def validate_manuscript(manuscript, ledger, sources=()) -> ManuscriptValidation:
    manuscript = ManuscriptRecord.model_validate(manuscript)
    sources = _source_index(sources or ledger.get("sources", {}))
    errors, citation_errors, id_errors, content_errors, bibliography_errors = [], [], [], [], []
    attribution_errors = []
    try:
        _check_sources(sources)
    except IntegrityError as exc:
        content_errors.append(str(exc))
    entries = ledger.get("entries", {})
    claim_ids = {claim.id for claim in manuscript.claims}
    referenced_claims, referenced_sources = set(), set(manuscript.source_ids)
    for name in SECTIONS:
        section = getattr(manuscript, name)
        referenced_claims.update(section.claim_ids)
        referenced_sources.update(section.source_ids)
        if set(section.claim_ids) - claim_ids:
            errors.append(f"Section {name} references an unknown claim")
        # Empirical numbers must live in structured claims or deterministic tables.
        if name in {"results", "ablations", "discussion", "abstract"}:
            if any("." in token or "%" in token for token in _numeric_tokens(section.text)):
                errors.append(f"Section {name} embeds numeric prose; use a structured evidence claim")
    if claim_ids - referenced_claims:
        errors.append("Manuscript contains claims that no section references")
    for response in manuscript.concern_responses:
        if set(response.experiment_ids) - ledger.get("observations", {}).keys():
            errors.append(f"Concern response {response.concern_id} references unknown experiments")
        if response.status == "executed" and not response.experiment_ids:
            errors.append(f"Concern response {response.concern_id} claims execution without experiments")
    for claim in manuscript.claims:
        attribution_errors.extend(_attribution_errors(claim, ledger))
        unknown = set(claim.evidence_ids) - entries.keys()
        if unknown:
            errors.append(f"Claim {claim.id} references unknown evidence")
        referenced_sources.update(claim.source_ids)
        if claim.kind in {"measured", "derived"}:
            if len(claim.evidence_ids) != 1 or unknown:
                errors.append(f"Claim {claim.id} needs exactly one numerical evidence item")
                continue
            entry = entries[claim.evidence_ids[0]]
            if entry.get("kind") != claim.kind or "value" not in entry:
                errors.append(f"Claim {claim.id} uses the wrong numerical evidence kind")
                continue
            if claim.value is not None and not math.isclose(
                claim.value, entry["value"], rel_tol=1e-10, abs_tol=1e-12
            ):
                errors.append(f"Claim {claim.id} fabricates or changes a recorded value")
            allowed_values = [entry["value"]]
            for token in _numeric_tokens(claim.text):
                if not _number_matches(token, allowed_values):
                    errors.append(f"Claim {claim.id} contains an unsupported numerical statement")
        elif claim.value is not None:
            errors.append(f"Claim {claim.id} attaches a number to a nonnumerical claim")
        if claim.kind == "method":
            if not claim.evidence_ids or any(entries.get(key, {}).get("kind") != "method"
                                             for key in claim.evidence_ids):
                errors.append(f"Method claim {claim.id} requires recorded method evidence")
        if claim.kind == "literature":
            if not claim.source_ids or not claim.passage_ids:
                content_errors.append(f"Literature claim {claim.id} requires sources and available passages")
            available = {item["id"] for source_id in claim.source_ids
                         for item in sources.get(source_id, {}).get("passages", [])}
            if set(claim.passage_ids) - available:
                content_errors.append(f"Literature claim {claim.id} references an unavailable passage")
        elif claim.passage_ids:
            content_errors.append(f"Non-literature claim {claim.id} has unbound passage IDs")
        if claim.kind == "interpretation" and not claim.evidence_ids and not claim.source_ids:
            errors.append(f"Interpretation {claim.id} requires evidence or a source and remains unverified")
    if referenced_sources - sources.keys():
        id_errors.append("Manuscript references an unknown citation ID")
    if referenced_sources - set(manuscript.source_ids):
        bibliography_errors.append("Cited sources must be present in the manuscript bibliography")
    if set(manuscript.table_ids) - ledger.get("tables", {}).keys():
        errors.append("Manuscript references an unknown empirical table")
    if set(manuscript.figure_ids) - ledger.get("figures", {}).keys():
        errors.append("Manuscript references an unknown quantitative figure")
    errors.extend(attribution_errors)
    citation_errors = [*id_errors, *content_errors, *bibliography_errors]
    errors.extend(citation_errors)
    metadata_status = ("pass" if referenced_sources and all(
        ledger.get("source_verification", {}).get(identifier, {}).get("metadata_existence") == "pass"
        for identifier in referenced_sources
    ) else "unverified")
    content_status = ("pass" if referenced_sources and all(
        ledger.get("source_verification", {}).get(identifier, {}).get("content_identity") == "pass"
        for identifier in referenced_sources
    ) else "unverified")
    return ManuscriptValidation(
        status="fail" if errors else "pass", errors=errors,
        checks={"structured_claims": "fail" if errors else "pass",
                "citation_id_resolution": "fail" if id_errors else "pass",
                "citation_existence": "fail" if id_errors else metadata_status,
                "citation_metadata_existence": "fail" if id_errors else metadata_status,
                "cited_content_identity": "fail" if content_errors or id_errors else content_status,
                "bibliography_consistency": "fail" if bibliography_errors else "pass",
                "declared_attribution_scope": "fail" if attribution_errors else "pass",
                "citation_entailment": "unverified", "method_code_consistency": "unverified",
                "scientific_validity": "unverified", "natural_language_claim_completeness": "unverified"},
    )


def _escape_md(value):
    return html.escape(str(value), quote=False).replace("|", "\\|").replace("\n", " ")


def _bib_key(identifier):
    return "source" + digest(identifier.encode())[:20]


def _bibliography(sources, identifiers):
    lines = []
    for identifier in sorted(identifiers):
        source = sources[identifier]
        fields = {"title": source["title"], "author": " and ".join(source.get("authors", [])),
                  "year": source.get("year"), "doi": source.get("doi"), "url": source.get("url"),
                  "note": "Source ID: " + identifier + "; content: " + source.get("content_status", "unverified")}
        lines += ["@misc{" + _bib_key(identifier) + ",", *[
            "  " + name + " = {" + tex(value) + "}," for name, value in fields.items()
            if value not in (None, "")
        ], "}", ""]
    return "\n".join(lines)


def _numeric_text(entry):
    if "value" not in entry:
        return ""
    return format(entry["value"], ".12g")


def _figure(table, ledger):
    """Generate safe SVG and matching pure-TeX vector marks from identical values."""
    rows = [ledger["entries"][key] for key in table["evidence_ids"]]
    points = [item["value"] for row in rows for item in row["per_seed"]]
    lo, hi = min(points), max(points)
    pad = (hi - lo) * .1 or max(abs(lo) * .05, .01)
    lo, hi = lo - pad, hi + pad
    width, height = 760, 70 + 40 * len(rows)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="10" y="20" font-size="14">{html.escape(table["metric"])}</text>']
    latex = [r"\setlength{\unitlength}{0.65pt}", r"\begin{picture}(700," + str(height) + ")"]
    for index, row in enumerate(rows):
        y = 48 + index * 40
        label = row["observation_id"]
        svg.append(f'<text x="10" y="{y+4}" font-size="11">{html.escape(label)}</text>')
        latex.append(r"\put(0," + str(height-y) + r"){\makebox(200,10)[l]{\tiny " + tex(label) + "}}")
        for item in row["per_seed"]:
            x = 300 + (item["value"] - lo) / (hi - lo) * 420
            svg.append(f'<circle cx="{x:.3f}" cy="{y}" r="4" fill="#245ea8"><title>seed {item["seed"]}: {item["value"]:.12g}</title></circle>')
            latex.append(r"\put(" + f"{x-50:.3f},{height-y}" + r"){\circle*{5}}")
    svg += [f'<text x="290" y="{height-5}" font-size="11">{lo:.6g}</text>',
            f'<text x="685" y="{height-5}" font-size="11">{hi:.6g}</text>', '</svg>']
    latex += [r"\put(240,0){\makebox(80,10)[l]{\tiny " + tex(f"{lo:.6g}") + "}}",
              r"\put(630,0){\makebox(70,10)[r]{\tiny " + tex(f"{hi:.6g}") + "}}",
              r"\end{picture}"]
    return "\n".join(svg) + "\n", "\n".join(latex) + "\n"


def _source_payloads(store, ledger):
    files, total = {}, 0
    for observation_id, observation in ledger["observations"].items():
        prefix = "code/" + digest(observation_id.encode())[:20] + "/"
        source_ref = observation.get("source_tree_ref")
        if source_ref:
            source = store.json(source_ref)
            manifest = load_bundle(store, source["bundle_ref"], verify=True)
            if any(spec.role != "source" for spec in manifest.specs):
                raise IntegrityError("Reproduction source bundle contains non-source artifacts")
            if (total + sum(item.size for item in manifest.files.values()) > 50_000_000
                    or len(files) + len(manifest.files) > 10000):
                raise IntegrityError("Manuscript source bundle exceeds its byte/file limits")
            members = {path: store.read(item.ref) for path, item in manifest.files.items()}
        else:
            source = Candidate.model_validate(store.json(observation["code_ref"]))
            if source.base_snapshot_ref:
                raise IntegrityError("Frozen base source requires a materialized source-tree record")
            members = {item.path: item.content.encode() for item in source.files}
        for path, raw in members.items():
            safe_path(path)
            total += len(raw)
            if total > 50_000_000 or len(files) >= 10000:
                raise IntegrityError("Manuscript source bundle exceeds its byte/file limits")
            files[prefix + path] = raw
    return files


def _template_files(template_path):
    if template_path is None:
        return {}, {"status": "generic_article", "iclr_template_used": False}
    root = Path(template_path).resolve()
    if not root.is_dir():
        raise ValueError("An explicit template directory is required")
    allowed = {"iclr2025_conference.sty", "iclr2025_conference.bst", "natbib.sty", "fancyhdr.sty"}
    files = {}
    for name in sorted(allowed):
        source = root / name
        if source.exists():
            if source.is_symlink() or not source.is_file() or source.stat().st_size > 1_000_000:
                raise IntegrityError("Template dependency must be a bounded regular file")
            files[name] = source.read_bytes()
    if not {"iclr2025_conference.sty", "iclr2025_conference.bst"} <= files.keys():
        raise ValueError("Explicit template directory requires iclr2025_conference.sty and iclr2025_conference.bst")
    # Preserve distributed notices verbatim as files, never invent a license.
    for name in ("LICENSE", "LICENSE.txt", "NOTICE", "README.md"):
        source = root / name
        if source.is_file() and not source.is_symlink() and source.stat().st_size <= 1_000_000:
            files["template-notices/" + name] = source.read_bytes()
    provenance = {"status": "copied_explicit_template", "iclr_template_used": True,
                  "source_path": str(root), "files": {name: digest(raw) for name, raw in files.items()},
                  "scope": "Presentation template only; does not imply conference submission or acceptance"}
    return files, provenance


def _compile_project(store, project, project_hash, file_manifest, compiler, compilation_policy):
    compiled = CompilationResult(status="unavailable", diagnostics="No isolated compiler configured; PDF not generated.")
    pdf_ref = None
    attempts = []
    if compiler is not None:
        for attempt in range(1, compilation_policy.max_attempts + 1):
            request = CompilationRequest(project_dir=str(project), project_manifest_sha256=project_hash,
                                         policy=compilation_policy, attempt=attempt)
            compiled = CompilationResult.model_validate(compiler(request))
            for name, member in file_manifest.items():
                if file_digest(project / name) != member["sha256"]:
                    raise IntegrityError("Compiler modified an immutable manuscript input")
            attempts.append(compiled.model_dump(mode="json"))
            if compiled.status == "success":
                if (compiled.image_digest != compilation_policy.image_digest
                        or not compiled.network_disabled or not compiled.shell_escape_disabled
                        or compiled.timed_out or not compiled.receipt_ref or not compiled.pdf_path):
                    raise IntegrityError("Successful compilation lacks matching isolation receipt")
                store.read(compiled.receipt_ref)
                pdf = Path(compiled.pdf_path)
                if (pdf.is_symlink() or not pdf.resolve().is_relative_to(project.resolve())
                        or not pdf.is_file() or pdf.stat().st_size > 50_000_000):
                    raise IntegrityError("Compiler PDF must be a bounded regular file in the project")
                with pdf.open("rb") as stream:
                    if stream.read(5) != b"%PDF-":
                        raise IntegrityError("Compiler output is not a PDF")
                pdf_ref = store.put_file(pdf, kind="manuscript_pdf")
                break
            if compiled.status == "unavailable":
                break
        for name, member in file_manifest.items():
            if file_digest(project / name) != member["sha256"]:
                raise IntegrityError("Compiler modified an immutable manuscript input")
    return compiled, pdf_ref, attempts


def render_manuscript(
    store,
    manuscript,
    ledger,
    sources=(),
    *,
    template_path=None,
    compiler: Callable[[CompilationRequest], CompilationResult] | None = None,
    compilation_policy: CompilationPolicy | None = None,
) -> RenderedManuscript:
    """Render an immutable project. An injected isolated compiler is never implicit.

    Compiler callbacks must enforce the request policy and provide an immutable
    execution receipt. A receipt's presence is not independent sandbox attestation;
    runner enforcement and replay are separate audit responsibilities.
    """
    manuscript = ManuscriptRecord.model_validate(manuscript)
    verify_evidence_ledger(store, ledger)
    sources = _source_index(sources or ledger.get("sources", {}))
    if canonical(sources) != canonical(ledger.get("sources", {})):
        raise IntegrityError("Rendered literature sources differ from the frozen evidence ledger")
    validation = validate_manuscript(manuscript, ledger, sources)
    if validation.status != "pass":
        raise ValueError("Invalid manuscript: " + "; ".join(validation.errors))
    if compiler is not None and compilation_policy is None:
        raise ValueError("Compilation requires an explicitly pinned isolated-runtime policy")
    manuscript_ref = store.put_json(manuscript, "canonical_manuscript")
    ledger_ref = store.put_json(ledger, "manuscript_evidence_ledger")
    validation_ref = store.put_json(validation, "manuscript_validation")
    files, template = _template_files(template_path)
    files.update(_source_payloads(store, ledger))
    claims = {claim.id: claim for claim in manuscript.claims}
    md = ["# " + _escape_md(manuscript.title), "",
          "Synthetic agents: " + str(manuscript.synthetic_agents).lower() + ".",
          "Mechanical evidence checks passed; scientific interpretation and citation entailment remain unverified.", ""]
    latex = [r"\documentclass{article}", r"\usepackage[T1]{fontenc}",
             r"\usepackage{longtable,url,graphicx}"]
    if template["iclr_template_used"]:
        # Keep the vendored template unchanged, but never imply that an internal
        # workflow artifact has been submitted, reviewed or published by ICLR.
        latex += [r"\usepackage{iclr2025_conference}",
                  r"\makeatletter",
                  r"\renewcommand{\@maketitle}{\vbox{\hsize\textwidth",
                  r"{\LARGE\scshape\@title\par}\vskip 0.15in",
                  r"{\bfseries\@author\par}\vskip 0.05in",
                  r"{\small Research workflow draft -- not submitted\par}\vskip 0.3in}}",
                  r"\makeatother",
                  r"\fancyhead{}",
                  r"\lhead{Research workflow draft -- not submitted}"]
    latex += [r"\begin{document}", r"\title{" + tex(manuscript.title) + "}",
              r"\author{ScientistTwo research workflow}", r"\maketitle",
              tex("Synthetic agents: " + str(manuscript.synthetic_agents).lower()) + r"\par",
              tex("Mechanical checks passed; scientific interpretation and citation entailment remain unverified.")]
    for name in SECTIONS:
        section = getattr(manuscript, name)
        md += ["## " + _TITLES[name], "", html.escape(section.text, quote=False), ""]
        latex += [r"\section*{" + tex(_TITLES[name]) + "}", tex(section.text) + r"\par"]
        for claim_id in section.claim_ids:
            claim = claims[claim_id]
            text = claim.text
            if claim.kind in {"measured", "derived"}:
                text += " Recorded value: " + _numeric_text(ledger["entries"][claim.evidence_ids[0]]) + "."
            text += " [" + claim.kind + "; " + claim.id + "; attribution=" + claim.attribution_scope + "]"
            md += ["- " + _escape_md(text)]
            latex += [tex(text) + r"\par"]
        citations = sorted(set(section.source_ids) | {source for key in section.claim_ids
                                                      for source in claims[key].source_ids})
        if citations:
            md += ["Sources: " + ", ".join("[" + _escape_md(source) + "](#" + _bib_key(source) + ")"
                                           for source in citations), ""]
            latex += [r"\cite{" + ",".join(_bib_key(source) for source in citations) + "}"]
    if manuscript.concern_responses:
        md += ["## Concern responses", "", "Execution coverage is distinct from scientific resolution.", ""]
        latex += [r"\section*{Concern responses}", tex("Execution coverage is distinct from scientific resolution.")]
        for response in manuscript.concern_responses:
            text = (response.concern_id + ": " + response.status + "; experiments: "
                    + ", ".join(response.experiment_ids) + ". " + response.explanation)
            md.append("- " + _escape_md(text))
            latex.append(tex(text) + r"\par")
    if ledger.get("control_context"):
        md += ["## Control interpretation limits", "",
               "Recorded design judgments remain distinct from demonstrated causal isolation.", ""]
        latex += [r"\section*{Control interpretation limits}",
                  tex("Recorded design judgments remain distinct from demonstrated causal isolation.")]
        for control in ledger["control_context"]:
            plan, review = control["plan"], control.get("review") or {}
            text = (control["control_id"] + " versus " + control["matched_id"] + ": "
                    + plan.get("question", plan.get("purpose", "Recorded control"))
                    + ". Review: " + review.get("decision", "not recorded") + ". "
                    + review.get("reason", "")
                    + " Invariants: " + "; ".join(plan.get("invariants", []))
                    + ". Known confounds: " + "; ".join(control["known_confounds"])
                    + ". Isolated-component attribution: "
                    + control["attribution_restriction"]["isolated_component"] + ".")
            md += ["- " + _escape_md(text),
                   "  [Control evidence](" + control["control_record_ref"] + ")"]
            latex += [tex(text) + r"\par"]
    md += ["## Empirical tables", ""]
    latex += [r"\section*{Empirical tables}"]
    for table_id in manuscript.table_ids:
        table = ledger["tables"][table_id]
        if table.get("kind") == "diagnostic":
            columns = table["columns"]
            md += ["### " + table_id, "", table["caption"], "",
                   "| " + " | ".join(_escape_md(column) for column in columns) + " |",
                   "| " + " | ".join("---" for _ in columns) + " |"]
            latex += [tex(table_id + ": " + table["caption"]) + r"\par"]
            for row in table["rows"]:
                cells = [json.dumps(row.get(column), sort_keys=True, allow_nan=False) for column in columns]
                md.append("| " + " | ".join(_escape_md(cell) for cell in cells) + " |")
                latex.append(tex("; ".join(column + ": " + cell for column, cell in zip(columns, cells, strict=True))) + r"\par")
            md += ["[Diagnostic evidence](" + table["diagnostic_ref"] + ")", ""]
            continue
        md += ["### " + table_id, "", table["caption"], "", "| Observation | Metric | Aggregate | Seed values | Evidence |",
               "| --- | --- | --- | --- | --- |"]
        latex += [tex(table_id + ": " + table["caption"]) + r"\par",
                  r"\begin{longtable}{p{.25\linewidth}p{.2\linewidth}p{.15\linewidth}p{.3\linewidth}}",
                  r"Observation & Metric & Aggregate & Seed values \\"]
        for key in table["evidence_ids"]:
            entry = ledger["entries"][key]
            seeds = "; ".join(str(item["seed"]) + ": " + format(item["value"], ".12g") for item in entry["per_seed"])
            md.append("| " + " | ".join(_escape_md(value) for value in (
                entry["observation_id"], entry["metric"], _numeric_text(entry), seeds
            )) + " | [observation](" + entry["observation_ref"] + ") |")
            latex.append(" & ".join(tex(value) for value in (entry["observation_id"], entry["metric"], _numeric_text(entry), seeds)) + r" \\")
        latex += [r"\end{longtable}"]
        md += [""]
    figure_files = {}
    for figure_id in manuscript.figure_ids:
        figure = ledger["figures"][figure_id]
        svg, vector = _figure(ledger["tables"][figure["table_id"]], ledger)
        stem = "figures/" + figure_id.replace(":", "-")
        files[stem + ".svg"] = svg.encode()
        files[stem + ".tex"] = vector.encode()
        figure_files[figure_id] = stem + ".svg"
        md += ["![" + figure_id + "](" + stem + ".svg)", figure["caption"], ""]
        latex += [r"\begin{figure}[ht]", r"\centering",
                  r"\resizebox{\linewidth}{!}{\input{" + stem + ".tex}}",
                  r"\caption{" + tex(figure["caption"]) + "}", r"\end{figure}"]
    md += ["## Bibliography", ""]
    for identifier in sorted(manuscript.source_ids):
        source = sources[identifier]
        md += ["<a id=\"" + _bib_key(identifier) + "\"></a>",
               "- " + _escape_md(identifier + ": " + source["title"])
               + " (metadata: " + _escape_md(source.get("metadata_status", "unverified"))
               + "; content: " + _escape_md(source.get("content_status", "unverified")) + ")."]
    bibliography_style = "iclr2025_conference" if template["iclr_template_used"] else "plain"
    latex += [r"\bibliographystyle{" + bibliography_style + "}",
              r"\bibliography{references}", r"\end{document}", ""]
    files["manuscript.md"] = ("\n".join(md) + "\n").encode()
    files["main.tex"] = "\n".join(latex).encode()
    files["references.bib"] = _bibliography(sources, manuscript.source_ids).encode()
    files["evidence/ledger.json"] = canonical(ledger)
    files["evidence/manuscript.json"] = canonical(manuscript)
    files["evidence/validation.json"] = canonical(validation)
    files["evidence/template-provenance.json"] = canonical(template)
    for entry in ledger["entries"].values():
        if entry.get("kind") == "diagnostic":
            files[entry["artifact_ref"]] = store.read(entry["artifact_ref"])
    # Include only declared observation/protocol/receipt records, never a recursive
    # walk of arbitrary JSON paths that could export labels, credentials or models.
    for observation in ledger["observations"].values():
        for ref in [observation["observation_ref"], observation["protocol_ref"], *observation["receipt_refs"]]:
            files[ref] = store.read(ref)
    for control in ledger.get("control_context", []):
        ref = control["control_record_ref"]
        files[ref] = store.read(ref)
    files["REPRODUCE.md"] = (
        "# Reproduction bundle\n\n"
        "Sources are preserved under code/, grouped by observation ID hash. They are data, not executed by this renderer.\n"
        "Use each recorded protocol and execution receipt with the original trusted task adapter, pinned runtime and authorized assets.\n"
        "Prepared datasets, protected labels, credentials and model checkpoint binaries are intentionally external dependencies.\n"
        "The evidence ledger records artifact identities; original prediction artifacts are required for trusted scoring replay.\n"
        "Compile main.tex with the configured pinned offline compiler, shell escape disabled, then bibliography and a second TeX pass.\n"
        "A manuscript bundle alone does not establish executable reproduction, source/citation entailment or scientific validity.\n"
    ).encode()
    files["inputs-manifest.json"] = canonical({
        "schema_version": "manuscript-inputs-v1",
        "files": {name: {"sha256": digest(raw), "bytes": len(raw)} for name, raw in sorted(files.items())},
        "scope": "All rendered inputs except this self-describing manifest; no executable reproduction claim",
    })
    project_hash = digest(canonical({name: digest(raw) for name, raw in sorted(files.items())}))
    project = store.root / "manuscripts" / project_hash
    project.mkdir(parents=True, exist_ok=True)
    file_manifest = {}
    for name, raw in sorted(files.items()):
        safe_path(name)
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.is_file() or target.is_symlink() or digest(target.read_bytes()) != digest(raw):
                raise IntegrityError("Immutable manuscript project was modified")
        else:
            with target.open("xb") as stream:
                stream.write(raw)
        ref = store.put(raw, suffix=target.suffix.lstrip(".") or "txt", kind="manuscript_file")
        file_manifest[name] = {"ref": ref, "sha256": digest(raw), "bytes": len(raw)}
    compiled, pdf_ref, attempts = _compile_project(
        store, project, project_hash, file_manifest, compiler, compilation_policy
    )
    manifest = {"schema_version": "manuscript-bundle-v1", "project_sha256": project_hash,
                "files": file_manifest, "template": template,
                "manuscript_ref": manuscript_ref, "ledger_ref": ledger_ref,
                "validation_ref": validation_ref, "compilation_attempts": attempts,
                "pdf_ref": pdf_ref, "compilation_status": compiled.status}
    manifest_ref = store.put_json(manifest, "manuscript_bundle")
    return RenderedManuscript(
        manifest_ref=manifest_ref, manuscript_ref=manuscript_ref, ledger_ref=ledger_ref,
        validation_ref=validation_ref, project_path=str(project),
        markdown_ref=file_manifest["manuscript.md"]["ref"], latex_ref=file_manifest["main.tex"]["ref"],
        bibliography_ref=file_manifest["references.bib"]["ref"],
        figure_refs={key: file_manifest[path]["ref"] for key, path in figure_files.items()},
        compilation=compiled, pdf_ref=pdf_ref,
    )


def _fixture_claim_entries(ledger, limit=128):
    """Bound fixture citations, not evidence; selection never depends on scores.

    Prefer the ordinary observation measurements to optional timing statistics
    and derived comparisons. Round-robin primary metrics across observations so
    a many-metric observation cannot crowd out the remaining research history.
    The remainder is an evenly spaced deterministic sample of sorted evidence.
    """
    numeric = [(key, entry) for key, entry in sorted(ledger["entries"].items())
               if entry.get("kind") in {"measured", "derived"}]
    if len(numeric) <= limit:
        return numeric
    primary = {}
    for key, entry in numeric:
        if entry["kind"] == "measured" and not entry.get("numeric_provenance", "").startswith("paired-timing-measurements-"):
            primary.setdefault(entry["observation_id"], []).append((key, entry))
    selected = []
    for index in range(max((len(values) for values in primary.values()), default=0)):
        for observation in sorted(primary):
            if index < len(primary[observation]) and len(selected) < limit:
                selected.append(primary[observation][index])
    chosen = {key for key, _ in selected}
    remaining = sorted((item for item in numeric if item[0] not in chosen),
        key=lambda item: (item[1].get("metric", ""), item[1].get("observation_id", ""), item[0]))
    count = min(limit - len(selected), len(remaining))
    selected.extend(remaining[index * len(remaining) // count] for index in range(count))
    return sorted(selected)


def fixture_manuscript(ledger, sources=(), *, title="Scripted research evidence manuscript", concern_responses=()):
    """Complete synthetic writer response; numbers are supplied solely by the ledger."""
    sources = _source_index(sources or ledger.get("sources", {}))
    claims = [LedgerClaim(id="claim-" + str(index), kind=entry["kind"],
                         text="The recorded measurement or deterministic comparison is reported below.",
                         evidence_ids=[key], value=entry["value"])
              for index, (key, entry) in enumerate(_fixture_claim_entries(ledger))]
    sections = {name: SectionRecord(text=(
        "Synthetic writer fixture. This section records the supplied workflow context; "
        "independent scientific interpretation is unverified."
    )) for name in SECTIONS}
    sections["results"].claim_ids = [claim.id for claim in claims]
    sections["related_work"].source_ids = sorted(sources)
    sections["limitations"].text = (
        "Synthetic agents do not demonstrate autonomous research quality. "
        "Numerical evidence does not certify causal attribution or global novelty."
    )
    total = sum(entry.get("kind") in {"measured", "derived"} for entry in ledger["entries"].values())
    if len(claims) < total:
        sections["limitations"].text += (
            f" The fixture cites {len(claims)} of {total} numerical ledger items; "
            f"{total - len(claims)} remain in the unchanged complete evidence ledger. "
            "Citations prioritize primary measurements and a deterministic representative subset; "
            "this selection does not depend on measured outcomes."
        )
    cited = {key for claim in claims for key in claim.evidence_ids}
    table_ids = sorted(sorted(ledger["tables"], key=lambda key: (
        not bool(set(ledger["tables"][key].get("evidence_ids", [])) & cited), key))[:128])
    figure_ids = [key for key, figure in sorted(ledger["figures"].items())
                  if figure["table_id"] in table_ids][:128]
    return ManuscriptRecord(title=title, **sections, claims=claims, source_ids=sorted(sources),
                            table_ids=table_ids, figure_ids=figure_ids,
                            synthetic_agents=True, concern_responses=list(concern_responses))


def legacy_draft_view(manuscript, ledger) -> Draft:
    """Lossy compatibility view only; the canonical manuscript remains authoritative."""
    manuscript = ManuscriptRecord.model_validate(manuscript)
    claims = []
    for claim in manuscript.claims:
        if claim.kind != "measured" or len(claim.evidence_ids) != 1:
            continue
        entry = ledger["entries"][claim.evidence_ids[0]]
        if entry.get("numeric_provenance", "").startswith("paired-timing-measurements-"):
            # The compatibility Draft only supports original Observation metric
            # names. Detailed host timing remains in the canonical manuscript.
            continue
        claims.append({"experiment_id": entry["observation_id"], "metric": entry["metric"],
                       "value": entry["value"]})
    return Draft(title=manuscript.title, claims=claims, limitations=[manuscript.limitations.text],
                 source_ids=manuscript.source_ids, concern_responses=manuscript.concern_responses)


def final_results_appendix(
    store,
    rendered: RenderedManuscript | dict,
    results: dict,
    *,
    compiler: Callable[[CompilationRequest], CompilationResult] | None = None,
    compilation_policy: CompilationPolicy | None = None,
) -> RenderedManuscript:
    """Append protected results deterministically after freeze, without model calls.

    Original manuscripts remain immutable. Final results stay out of the canonical
    writer ledger; this creates a separately identified final publication bundle.
    """
    rendered = RenderedManuscript.model_validate(rendered)
    frozen = store.get("live_frozen")
    if not frozen:
        raise IntegrityError("A final appendix requires a committed frozen method")
    if canonical(results) != canonical(store.get("live_final")):
        raise IntegrityError("Final appendix differs from committed final results")
    if not results or set(results) - {"baseline", "selected"}:
        raise ValueError("Final appendix requires the declared baseline/selected results")
    if compiler is not None and compilation_policy is None:
        raise ValueError("Compilation requires an explicitly pinned isolated-runtime policy")
    frozen_ref = store.put_json(frozen, "frozen_method_for_final_appendix")
    result_refs = {}
    for label, result in results.items():
        from .timing_evidence import verify_final_result_binding
        binding = verify_final_result_binding(store, label, result)
        score = result.get("score")
        if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
            raise IntegrityError("Final result has a nonfinite or invalid numerical score")
        if result.get("status") != "completed" and score is not None:
            raise IntegrityError("Unsuccessful final execution cannot supply a scientific score")
        result_refs[label] = (binding["operation_ref"] if binding["kind"] == "legacy_exact_summary"
                              else store.put_json(result, "paired-final-summary"))
    base = store.json(rendered.manifest_ref)
    if base.get("schema_version") != "manuscript-bundle-v1":
        raise ValueError("Final results can be appended only once to the development manuscript")
    files = {name: store.read(member["ref"]) for name, member in base["files"].items()}
    for name, member in base["files"].items():
        if digest(files[name]) != member["sha256"] or len(files[name]) != member["bytes"]:
            raise IntegrityError("Final appendix source bundle differs from its manifest")
    md = ["## Frozen final evaluation", "",
          "Final measurements were appended after method freeze without model interpretation. "
          "They never enter research refinement. Development and final results remain separate.", "",
          "| Method | Status | Metric | Final score | Evidence |", "| --- | --- | --- | --- | --- |"]
    latex = [r"\appendix", r"\section{Frozen final evaluation}",
             tex("Final measurements were appended after method freeze without model interpretation. Development and final results remain separate."),
             r"\begin{longtable}{p{.2\linewidth}p{.2\linewidth}p{.25\linewidth}p{.2\linewidth}}",
             r"Method & Status & Metric & Final score \\"]
    for label, result in sorted(results.items()):
        cells = [label, result["status"], result.get("metric") or "unavailable",
                 format(result["score"], ".12g") if result.get("score") is not None else "unavailable"]
        md.append("| " + " | ".join(_escape_md(value) for value in cells)
                  + " | [final result](" + result_refs[label] + ") |")
        latex.append(" & ".join(tex(value) for value in cells) + r" \\")
        files[result_refs[label]] = store.read(result_refs[label])
        if result.get("observation_ref"):
            files[result["observation_ref"]] = store.read(result["observation_ref"])
    latex += [r"\end{longtable}", ""]
    md += ["", "Frozen method evidence: [record](" + frozen_ref + ").", ""]
    files[frozen_ref] = store.read(frozen_ref)
    files["final-results.md"] = "\n".join(md).encode()
    files["final-results.tex"] = "\n".join(latex).encode()
    files["manuscript.md"] += b"\n" + files["final-results.md"]
    main = files["main.tex"].decode()
    if main.count(r"\end{document}") != 1:
        raise IntegrityError("Development manuscript has an invalid document closure")
    if base.get("writer", {}).get("type") == "paperorchestra":
        # The authoritative upstream manuscript need not use longtable. Add
        # this dependency only to the separate deterministic final artifact.
        main = main.replace(r"\begin{document}",
                            r"\usepackage{longtable}" + "\n" + r"\begin{document}", 1)
    files["main.tex"] = main.replace(r"\end{document}", r"\input{final-results.tex}" + "\n" + r"\end{document}").encode()
    files.pop("inputs-manifest.json", None)
    files["inputs-manifest.json"] = canonical({
        "schema_version": "manuscript-inputs-v1", "development_bundle_ref": rendered.manifest_ref,
        "frozen_ref": frozen_ref, "final_result_refs": result_refs,
        "files": {name: {"sha256": digest(raw), "bytes": len(raw)} for name, raw in sorted(files.items())},
    })
    project_hash = digest(canonical({name: digest(raw) for name, raw in sorted(files.items())}))
    project = store.root / "manuscripts" / project_hash
    project.mkdir(parents=True, exist_ok=True)
    file_manifest = {}
    for name, raw in sorted(files.items()):
        safe_path(name)
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.is_symlink() or not path.is_file() or digest(path.read_bytes()) != digest(raw):
                raise IntegrityError("Immutable final manuscript bundle was modified")
        else:
            with path.open("xb") as stream:
                stream.write(raw)
        ref = store.put(raw, suffix=path.suffix.lstrip(".") or "txt", kind="final_manuscript_file")
        file_manifest[name] = {"ref": ref, "sha256": digest(raw), "bytes": len(raw)}
    compiled, pdf_ref, attempts = _compile_project(
        store, project, project_hash, file_manifest, compiler, compilation_policy
    )
    manifest_ref = store.put_json({
        "schema_version": "manuscript-final-bundle-v1", "development_bundle_ref": rendered.manifest_ref,
        "project_sha256": project_hash, "files": file_manifest, "frozen_ref": frozen_ref,
        "final_result_refs": result_refs, "model_calls": 0, "compilation_attempts": attempts,
        "compilation_status": compiled.status, "pdf_ref": pdf_ref,
    }, "final_manuscript_bundle")
    return RenderedManuscript(
        manifest_ref=manifest_ref, manuscript_ref=rendered.manuscript_ref, ledger_ref=rendered.ledger_ref,
        validation_ref=rendered.validation_ref, project_path=str(project),
        markdown_ref=file_manifest["manuscript.md"]["ref"], latex_ref=file_manifest["main.tex"]["ref"],
        bibliography_ref=file_manifest["references.bib"]["ref"], figure_refs=rendered.figure_refs,
        compilation=compiled, pdf_ref=pdf_ref,
    )
