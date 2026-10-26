"""Bounded development-evidence exchange with the official PaperOrchestra writer.

An exported bundle contains summaries and artifact identities, never artifact
payloads. Imported TeX stays an external manuscript: reference/value checks do
not establish prose entailment and cannot promote it to a ManuscriptRecord.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Literal

from pydantic import Field

from ...models import Record
from ...research.manuscript import (
    _bib_key, _bibliography, _number_matches, _numeric_tokens,
    build_evidence_ledger, freeze_research_history,
    verify_evidence_ledger,
)
from ...store import IntegrityError, canonical, digest, file_digest

DEFAULT_MAX_BYTES = 2_000_000
MAX_FILES = 12
_NONFINAL_PHASES = {"subset", "full", "rebuttal"}
_BUNDLE_FILES = {
    "idea_sparse.md", "experimental_log.md", "evidence_ledger.json",
    "literature/metadata.json", "citation_map.json", "references.bib", "figures/info.json",
}
_CITE = re.compile(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\])*\{([^}]+)\}")


def _aggregate_matches(actual, expected):
    if type(expected) is float:
        return (type(actual) in {int, float} and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(_aggregate_matches(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_aggregate_matches(left, right) for left, right in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def diagnostic_numeric_values(entry):
    """Allowlisted aggregate-only diagnostic numbers; never per-example arrays.

    Seed-level paired values are experimental aggregates. Classification arrays
    contain only confusion counts and the class vocabulary, with no row IDs.
    Unknown fields or diagnostic families fail closed rather than being dropped.
    """
    def reject():
        raise IntegrityError("Diagnostic requires the verified aggregate-only export contract")

    if (entry.get("kind") != "diagnostic"
            or entry.get("provenance", {}).get("split") != "development"
            or entry.get("provenance", {}).get("implementation") != "scientisttwo.research.diagnostics"):
        reject()
    provenance = entry["provenance"]
    if (set(entry) != {"kind", "artifact_ref", "summary", "tables", "arrays", "limitations", "provenance", "input_refs"}
            or set(provenance) != {"implementation", "version", "implementation_sha256", "numpy_version", "input_hash", "split"}
            or provenance["version"] != 1
            or not all(isinstance(provenance[key], str) and re.fullmatch(r"[a-f0-9]{64}", provenance[key])
                       for key in ("implementation_sha256", "input_hash"))
            or not isinstance(provenance["numpy_version"], str)
            or not re.fullmatch(r"[A-Za-z0-9_.+\-]{1,64}", provenance["numpy_version"])
            or not isinstance(entry["input_refs"], list)
            or any(not isinstance(ref, str) or not re.fullmatch(r"artifacts/[a-f0-9]{64}\.json", ref)
                   for ref in entry["input_refs"])):
        reject()
    tables, arrays, summary = (entry.get(name, {}) for name in ("tables", "arrays", "summary"))
    if not all(isinstance(value, dict) for value in (tables, arrays, summary)):
        reject()
    if "paired_differences" in tables:
        from ...research.diagnostics import paired_differences, paired_interval

        rows = tables["paired_differences"]
        if (not isinstance(rows, list) or not 1 <= len(rows) <= 32
                or any(not isinstance(row, dict) or set(row) != {"seed", "baseline", "candidate", "difference"}
                       for row in rows)):
            reject()
        function = paired_interval if "confidence" in summary else paired_differences
        try:
            expected = function([row["baseline"] for row in rows], [row["candidate"] for row in rows],
                                [row["seed"] for row in rows],
                                **({"confidence": summary["confidence"]} if function is paired_interval else {}))
        except (ValueError, TypeError):
            reject()
        if any(canonical(entry.get(name, {})) != canonical(expected[name])
               for name in ("tables", "arrays", "summary", "limitations")):
            reject()
    else:
        if entry["limitations"] != ["Aggregate development-label diagnostics; no individual labels exported"]:
            reject()
        prefixes = sorted(name.removesuffix("confusion_matrix") for name in arrays
                          if name.endswith("confusion_matrix"))
        if (not prefixes or len(prefixes) > 32 or any(
                prefix and not re.fullmatch(r"seed_[0-9]+/", prefix) for prefix in prefixes)):
            reject()
        if set(arrays) != {prefix + name for prefix in prefixes for name in ("confusion_matrix", "class_ids")}:
            reject()
        expected_tables = {prefix + "per_class" for prefix in prefixes}
        if prefixes != [""]:
            expected_tables.add("per_seed")
        if set(tables) != expected_tables:
            reject()
        per_seed = []
        for prefix in prefixes:
            matrix, classes = arrays[prefix + "confusion_matrix"], arrays[prefix + "class_ids"]
            if (not isinstance(classes, list) or not 1 <= len(classes) <= 64
                    or any(type(value) is not int for value in classes) or len(set(classes)) != len(classes)
                    or not isinstance(matrix, list) or len(matrix) != len(classes)
                    or any(not isinstance(row, list) or len(row) != len(classes)
                           or any(type(value) is not int or value < 0 for value in row) for row in matrix)):
                reject()
            count = sum(map(sum, matrix))
            if not 0 < count <= 10_000_000:
                reject()
            rows = []
            for index, class_id in enumerate(classes):
                tp = matrix[index][index]
                support, predicted = sum(matrix[index]), sum(row[index] for row in matrix)
                precision, recall = tp / predicted if predicted else 0., tp / support if support else 0.
                rows.append({"class_id": class_id, "support": support, "predicted_count": predicted,
                             "precision": precision, "recall": recall,
                             "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.})
            if not _aggregate_matches(tables[prefix + "per_class"], rows):
                reject()
            values = {"example_count": count, "accuracy": sum(matrix[i][i] for i in range(len(classes))) / count,
                      "macro_f1": sum(row["f1"] for row in rows) / len(rows),
                      "confusion_axes": "rows=true, columns=predicted", "zero_division": 0}
            if prefix:
                per_seed.append({"seed": int(prefix[5:-1]), **values})
            elif not _aggregate_matches(summary, values):
                reject()
        if per_seed:
            # The trusted scorer preserves declared seed order, not string sorting.
            order = summary.get("seeds")
            if (set(summary) != {"seeds"} or not isinstance(order, list)
                    or len(order) != len(set(order)) or set(order) != {row["seed"] for row in per_seed}):
                reject()
            expected = [next(row for row in per_seed if row["seed"] == seed) for seed in order]
            if not _aggregate_matches(tables["per_seed"], expected):
                reject()
    values = {}

    def visit(value, path):
        if type(value) in {int, float}:
            if not math.isfinite(value):
                reject()
            values[tuple(path)] = value
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, [*path, key])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, [*path, index])

    for name in ("tables", "arrays", "summary"):
        visit(entry[name], [name])
    return values


def _verify_diagnostic_exports(store, ledger):
    for entry in ledger.get("entries", {}).values():
        if entry.get("kind") != "diagnostic":
            continue
        diagnostic_numeric_values(entry)
        record = store.json(entry["artifact_ref"])
        request = record.get("request", {})
        paired = "paired_differences" in entry["tables"]
        kind = ("paired_interval" if "confidence" in entry["summary"] else "paired_differences") if paired else "classification"
        seeds = ([row["seed"] for row in entry["tables"]["paired_differences"]]
                 if paired else entry["summary"].get("seeds", request.get("expected_seeds")))
        if (record.get("status") != "success" or request.get("evaluation_split") != "development"
                or request.get("kind") != kind or request.get("expected_seeds") != seeds
                or not isinstance(seeds, list) or not seeds
                or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(seeds) != len(set(seeds))):
            raise IntegrityError("Only successful declared development aggregate diagnostics are exportable")
        row = store.db.execute("SELECT output_ref FROM calls WHERE key=?",
                               ("research-diagnostic/" + request["id"],)).fetchone()
        if row is None or row[0] != entry["artifact_ref"]:
            raise IntegrityError("Aggregate diagnostic lacks its committed controller operation")


class ExternalClaim(Record):
    id: str = Field(min_length=1, max_length=192)
    quote: str = Field(min_length=1, max_length=10000)
    kind: Literal["measured", "derived", "method", "literature", "interpretation"]
    evidence_ids: list[str] = Field(default_factory=list, max_length=128)
    source_ids: list[str] = Field(default_factory=list, max_length=128)
    value: float | None = None


class ExternalEvidenceBindings(Record):
    schema_version: Literal["paperorchestra-evidence-bindings-v1"] = "paperorchestra-evidence-bindings-v1"
    latex_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    claims: list[ExternalClaim] = Field(default_factory=list, max_length=256)


class ExternalManuscript(Record):
    schema_version: Literal["paperorchestra-external-manuscript-v1"] = "paperorchestra-external-manuscript-v1"
    artifact_kind: Literal["external_latex"] = "external_latex"
    latex_ref: str
    latex_sha256: str
    bundle_manifest_ref: str
    bundle_manifest_sha256: str
    evidence_ledger_ref: str
    bindings_ref: str | None = None
    validation_ref: str
    validation: dict
    controller_accepted: Literal[False] = False


def _nonfinal(ledger):
    if ledger.get("includes_final") is not False:
        raise IntegrityError("PaperOrchestra exports require explicitly non-final evidence")
    for observation in ledger.get("observations", {}).values():
        if observation.get("phase") not in _NONFINAL_PHASES:
            raise IntegrityError("Protected final evidence cannot be exported to a writer")
    for entry in ledger.get("entries", {}).values():
        if "phase" in entry and entry["phase"] not in _NONFINAL_PHASES:
            raise IntegrityError("Protected final history cannot be exported to a writer")
        if entry.get("kind") == "diagnostic":
            diagnostic_numeric_values(entry)
    for table in ledger.get("tables", {}).values():
        if table.get("kind") == "measurements":
            continue
        matches = [entry for entry in ledger.get("entries", {}).values()
                   if entry.get("kind") == "diagnostic" and entry.get("artifact_ref") == table.get("diagnostic_ref")]
        if (table.get("kind") != "diagnostic" or len(matches) != 1
                or matches[0]["tables"].get(table.get("name")) != table.get("rows")):
            raise IntegrityError("Only verified aggregate measurement/diagnostic tables are exportable")


def build_campaign_export_ledger(store, observation_ids=None) -> dict:
    """Rebuild a NEW ledger from an explicit snapshot of committed development calls.

    A default snapshot selects current non-final commitments only. The returned
    ledger is new evidence packaging, not a retrospective change to old drafts.
    Neither the database nor old ledger artifacts are changed by this helper.
    """
    rows = list(store.db.execute(
        "SELECT key,output_ref FROM calls WHERE key LIKE 'development/%' ORDER BY key"
    ))
    available = {row["key"].removeprefix("development/"): row["output_ref"] for row in rows}
    if observation_ids is not None:
        selected = list(observation_ids)
        if len(selected) != len(set(selected)) or set(selected) - available.keys():
            raise IntegrityError("Export observation IDs must be unique committed development IDs")
    else:
        selected = [key for key, ref in available.items()
                    if store.json(ref).get("phase") in _NONFINAL_PHASES]
    if not selected or len(selected) > 256:
        raise ValueError("Export requires between one and 256 committed development observations")
    observations = [store.json(available[key]) for key in selected]
    if any(value.get("phase") not in _NONFINAL_PHASES for value in observations):
        raise IntegrityError("Protected final observations cannot be selected for export")
    # Read only the source snapshot of an existing development ledger; never
    # import the old manuscript, final report, run configuration, or backend logs.
    sources = {}
    for row in store.db.execute("SELECT value FROM metadata WHERE key LIKE 'manuscript-ledger/%' ORDER BY key"):
        prior = store.json(json.loads(row["value"]))
        if prior.get("includes_final") is False:
            sources = prior.get("sources", {})
    trials = []
    for identifier in selected:
        row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("live-trial/" + identifier,)).fetchone()
        if row:
            trials.append(store.json(row["output_ref"]))
    return build_evidence_ledger(store, observations, sources,
                                 history_refs=freeze_research_history(store, trials))


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def _experimental_log(ledger):
    lines = ["# Frozen development experiments", "",
        "All numerical values below are committed observations. They are not model-written scores.",
        "Comparisons are valid only within identical metric, protocol hash, and matched seeds.",
        "Final-test evidence is excluded. Source identity checks do not recompute scores.",
        "Scientific interpretation, novelty, and causal attribution remain unverified.", ""]
    for identifier, entry in sorted(ledger["entries"].items()):
        lines += ["## " + identifier, "", "```json", json.dumps(entry, sort_keys=True, indent=2), "```", ""]
    return "\n".join(lines).encode()


def export_evidence_bundle(store, destination: Path, *, ledger: dict, title: str,
                           hypothesis: str, upstream_revision: str,
                           max_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    """Export only a verified frozen development ledger; do not copy linked files."""
    if not re.fullmatch(r"[a-f0-9]{40}", upstream_revision):
        raise ValueError("PaperOrchestra upstream_revision must be a pinned commit SHA")
    if not 1 <= max_bytes <= DEFAULT_MAX_BYTES:
        raise ValueError("Invalid evidence bundle byte limit")
    if not title.strip() or len(title) > 1000 or not hypothesis.strip() or len(hypothesis) > 10000:
        raise ValueError("A bounded title and hypothesis are required")
    _nonfinal(ledger)
    verify_evidence_ledger(store, ledger)
    _verify_diagnostic_exports(store, ledger)
    source_ids = sorted(ledger.get("sources", {}))
    sources = ledger.get("sources", {})
    citation_map = {}
    metadata = {}
    for identifier in source_ids:
        source = sources[identifier]
        metadata[identifier] = {key: source[key] for key in (
            "id", "title", "authors", "year", "doi", "arxiv_id", "url", "metadata_status", "content_status"
        ) if key in source}
        metadata[identifier]["verification"] = ledger.get("source_verification", {}).get(identifier, {})
        citation_map[_bib_key(identifier)] = {
            "source_id": identifier, "title": source.get("title", identifier),
            "authors": source.get("authors", []), "year": source.get("year") or "",
            "abstract": "No independently verified abstract supplied; do not invent literature claims.",
        }
    source_name = store.root.name
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,192}", source_name):
        source_name = "campaign-" + digest(canonical(sorted(ledger["observations"])))[:20]
    backend = store.get("live_backend", {})
    provenance = {
        "source_run_id": source_name,
        "source_ledger_sha256": digest(canonical(ledger)),
        "observation_refs": {key: value["observation_ref"] for key, value in ledger["observations"].items()},
        "protocol_hashes": sorted({value["protocol_hash"] for value in ledger["observations"].values()}),
        "synthetic_research_agents": backend.get("synthetic", "unverified"),
        "packaging": "New export-time ledger from committed development evidence; historical records are unchanged.",
    }
    idea = ("# " + title + "\n\n" + hypothesis + "\n\n"
        "This is a workflow reproduction example, not a novelty or significance claim.\n"
        "Use only the accompanying frozen development evidence and existing citation IDs.\n"
        "Execution failures and recorded scientific rejection are different outcomes.\n"
        "No final-test evidence, datasets, labels, checkpoint binaries, credentials, or run configuration are included.\n"
        "Source literature may be supplied or unverified; preserve its recorded verification scope.\n")
    payloads = {
        "idea_sparse.md": idea.encode(), "experimental_log.md": _experimental_log(ledger),
        "evidence_ledger.json": _json_bytes(ledger),
        "literature/metadata.json": _json_bytes(metadata), "citation_map.json": _json_bytes(citation_map),
        "references.bib": _bibliography(sources, source_ids).encode(), "figures/info.json": b"[]\n",
    }
    manifest = {
        "schema_version": "paperorchestra-evidence-bundle-v1", "upstream_revision": upstream_revision,
        "provenance": provenance,
        "files": {name: {"sha256": digest(raw), "bytes": len(raw)} for name, raw in sorted(payloads.items())},
        "limits": {"max_bytes": max_bytes, "max_files": MAX_FILES},
        "verification_scope": "Committed artifact identity and recorded values; metric recomputation and prose entailment are separate.",
        "includes_final": False,
    }
    payloads["manifest.json"] = _json_bytes(manifest)
    if sum(map(len, payloads.values())) > max_bytes:
        raise ValueError("Evidence bundle exceeds its byte limit")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Evidence export requires a new directory; existing bundles are immutable")
    destination.mkdir(parents=True)
    for name, raw in payloads.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    verify_evidence_bundle(destination, store=store)
    return manifest


def verify_evidence_bundle(path: Path, store=None) -> dict:
    """Verify exact bounded file identities, and optionally source commitments."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise IntegrityError("Evidence bundle must be an ordinary directory")
    paths = list(path.rglob("*"))
    if len(paths) > MAX_FILES + 2 or any(p.is_symlink() for p in paths):
        raise IntegrityError("Evidence bundle contains too many paths or a symlink")
    files = {p.relative_to(path).as_posix(): p for p in paths if p.is_file()}
    directories = {p.relative_to(path).as_posix() for p in paths if p.is_dir()}
    if set(files) != _BUNDLE_FILES | {"manifest.json"}:
        raise IntegrityError("Evidence bundle has missing, extra, or partial files")
    if directories != {"literature", "figures"}:
        raise IntegrityError("Evidence bundle contains unexpected directories")
    if sum(p.stat().st_size for p in files.values()) > DEFAULT_MAX_BYTES:
        raise IntegrityError("Evidence bundle exceeds the hard byte limit")
    manifest = json.loads(files["manifest.json"].read_bytes())
    if (manifest.get("schema_version") != "paperorchestra-evidence-bundle-v1"
            or manifest.get("includes_final") is not False or set(manifest.get("files", {})) != _BUNDLE_FILES
            or not re.fullmatch(r"[a-f0-9]{40}", str(manifest.get("upstream_revision", "")))
            or manifest.get("limits", {}).get("max_files") != MAX_FILES):
        raise IntegrityError("Unsupported or final evidence bundle manifest")
    limit = manifest.get("limits", {}).get("max_bytes", 0)
    if type(limit) is not int or not 1 <= limit <= DEFAULT_MAX_BYTES:
        raise IntegrityError("Invalid evidence bundle limit")
    if sum(p.stat().st_size for p in files.values()) > limit:
        raise IntegrityError("Evidence bundle exceeds its declared byte limit")
    for name, expected in manifest["files"].items():
        if files[name].stat().st_size != expected.get("bytes") or file_digest(files[name]) != expected.get("sha256"):
            raise IntegrityError("Modified frozen evidence file: " + name)
    ledger = json.loads(files["evidence_ledger.json"].read_bytes())
    _nonfinal(ledger)
    if digest(canonical(ledger)) != manifest["provenance"]["source_ledger_sha256"]:
        raise IntegrityError("Exported evidence does not match its source ledger identity")
    if store is not None:
        verify_evidence_ledger(store, ledger)
    return manifest


def _validate_external(tex: str, latex_sha256: str, ledger: dict, citation_map: dict, bindings):
    errors = []
    cited = {key.strip() for match in _CITE.findall(tex) for key in match.split(",")}
    unknown = cited - citation_map.keys()
    if unknown:
        errors.append("Unknown citation keys: " + ", ".join(sorted(unknown)))
    checks = {"citation_id_resolution": "fail" if unknown else "pass",
              "structured_evidence_bindings": "unverified", "prose_entailment": "unverified",
              "natural_language_claim_completeness": "unverified", "scientific_validity": "unverified",
              "method_code_consistency": "unverified", "citation_entailment": "unverified"}
    parsed = None
    if bindings is not None:
        parsed = ExternalEvidenceBindings.model_validate(bindings)
        if parsed.latex_sha256 != latex_sha256:
            errors.append("Evidence bindings target a different LaTeX artifact")
        if len({claim.id for claim in parsed.claims}) != len(parsed.claims):
            errors.append("Duplicate external claim IDs")
        entries, sources = ledger["entries"], ledger.get("sources", {})
        for claim in parsed.claims:
            if claim.quote not in tex:
                errors.append("Claim quote is absent from LaTeX: " + claim.id)
            if set(claim.evidence_ids) - entries.keys() or set(claim.source_ids) - sources.keys():
                errors.append("Unknown evidence or source binding: " + claim.id)
                continue
            if claim.kind in {"measured", "derived"}:
                entry = entries.get(claim.evidence_ids[0], {}) if len(claim.evidence_ids) == 1 else {}
                if (entry.get("kind") != claim.kind or claim.value is None or "value" not in entry
                        or not math.isclose(claim.value, entry["value"], rel_tol=1e-10, abs_tol=1e-12)):
                    errors.append("Numerical claim differs from its recorded evidence: " + claim.id)
                elif any(not _number_matches(token, [entry["value"]])
                         for token in _numeric_tokens(claim.quote.replace(r"\%", "%"))):
                    errors.append("Numerical claim quote contains an unsupported value: " + claim.id)
            elif claim.value is not None:
                errors.append("Nonnumerical claim declares a numerical value: " + claim.id)
            if claim.kind == "method" and (not claim.evidence_ids or any(
                    entries[key].get("kind") != "method" for key in claim.evidence_ids)):
                errors.append("Method claim requires method evidence: " + claim.id)
            if claim.kind == "literature" and not claim.source_ids:
                errors.append("Literature claim requires a source binding: " + claim.id)
            if claim.kind == "interpretation" and not claim.evidence_ids and not claim.source_ids:
                errors.append("Interpretation requires evidence and remains unverified: " + claim.id)
        checks["structured_evidence_bindings"] = "fail" if errors else ("pass" if parsed.claims else "unverified")
    return {"status": "fail" if errors else "unverified", "errors": errors, "checks": checks,
            "scope": "External LaTeX preserved only. No acceptance, execution, empirical-claim completeness, or prose entailment is established."}, parsed


def import_external_manuscript(store, bundle_dir: Path, latex_path: Path, *,
                               evidence_bindings: dict | None = None) -> ExternalManuscript:
    """Preserve external TeX and its mechanical report; never compile or promote it."""
    manifest = verify_evidence_bundle(bundle_dir)
    latex_path = Path(latex_path)
    if latex_path.is_symlink() or not latex_path.is_file() or latex_path.stat().st_size > DEFAULT_MAX_BYTES:
        raise IntegrityError("External LaTeX must be a bounded ordinary file")
    raw = latex_path.read_bytes()
    tex = raw.decode("utf-8")
    ledger = json.loads((Path(bundle_dir) / "evidence_ledger.json").read_bytes())
    citations = json.loads((Path(bundle_dir) / "citation_map.json").read_bytes())
    report, parsed = _validate_external(tex, digest(raw), ledger, citations, evidence_bindings)
    report["checks"]["source_commitments"] = "unverified"
    return ExternalManuscript(
        latex_ref=store.put(raw, suffix="tex", kind="paperorchestra_external_manuscript"),
        latex_sha256=digest(raw), bundle_manifest_ref=store.put_json(manifest, "paperorchestra_bundle_manifest"),
        bundle_manifest_sha256=digest(canonical(manifest)),
        evidence_ledger_ref=store.put_json(ledger, "paperorchestra_frozen_evidence"),
        bindings_ref=store.put_json(parsed, "paperorchestra_external_bindings") if parsed else None,
        validation_ref=store.put_json(report, "paperorchestra_external_validation"), validation=report,
    )
