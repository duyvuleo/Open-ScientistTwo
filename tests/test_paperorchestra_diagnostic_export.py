"""Synthetic diagnostic handoff fixtures; no scientific or live-quality claim."""

from copy import deepcopy
import json

import pytest

from scientisttwo.integrations.paperorchestra.evidence import (
    diagnostic_numeric_values, export_evidence_bundle, verify_evidence_bundle,
)
from scientisttwo.integrations.paperorchestra.manuscript_binding import (
    bind_paperorchestra_manuscript, binding_prompt_context, fixture_bindings,
)
from scientisttwo.live.models import Candidate, Observation, SourceFile
from scientisttwo.research.diagnostics import (
    DiagnosticInputs, DiagnosticRequest, classification_diagnostics, run_diagnostic,
)
from scientisttwo.research.manuscript import build_evidence_ledger
from scientisttwo.store import IntegrityError, Store, canonical, digest


def observation(store, name, values):
    protocol = {"phase": "full", "seeds": list(range(len(values))), "adapter": "synthetic"}
    source = Candidate(files=[SourceFile(path="train.py", content="# synthetic source")],
                       command=["python", "train.py"], description="Not executed")
    value = Observation(id=name, phase="full", status="success", metric="accuracy",
                        value=sum(values) / len(values), values=values,
                        protocol_ref=store.put_json(protocol), protocol_hash=digest(canonical(protocol)),
                        code_ref=store.put_json(source),
                        prediction_refs=[store.put_json({"PRIVATE_ROW_LABELS": [0, 1]})])
    store.cache("development/" + name, {}, value)
    return value, store.put_json(value)


def classification(store):
    observed, observed_ref = observation(store, "baseline", [0.61])
    request = DiagnosticRequest(id="classification", kind="classification", expected_seeds=[0])
    raw = classification_diagnostics([0, 0, 0, 1], [0, 0, 1, 1])
    output = {"tables": {"seed_0/per_class": raw["tables"]["per_class"],
                          "per_seed": [{"seed": 0, **raw["summary"]}]},
              "arrays": {"seed_0/" + name: value for name, value in raw["arrays"].items()},
              "summary": {"seeds": [0]}, "limitations": raw["limitations"]}
    sealed = {"request": request.model_dump(), "observation_refs": [observed_ref],
              "output": output, "result_refs": [], "receipt_refs": []}
    store.cache("trusted-diagnostic/classification", {"synthetic": True}, sealed)
    sealed_ref = store.db.execute("SELECT output_ref FROM calls WHERE key='trusted-diagnostic/classification'").fetchone()[0]
    result = run_diagnostic(store, request, DiagnosticInputs(observation_refs=[observed_ref],
        trusted_result_refs=[sealed_ref], declared_kinds=["classification"]))
    return build_evidence_ledger(store, [observed], diagnostics=[result])


def export(store, destination, ledger):
    return export_evidence_bundle(store, destination, ledger=ledger, title="Synthetic aggregates",
                                  hypothesis="No scientific claim", upstream_revision="a" * 40)


def diagnostic_entry(ledger):
    return next((key, value) for key, value in ledger["entries"].items() if value["kind"] == "diagnostic")


def test_classification_aggregate_export_excludes_individual_payloads(tmp_path):
    store = Store(tmp_path / "source")
    ledger = classification(store)
    manifest = export(store, tmp_path / "export", ledger)
    assert verify_evidence_bundle(tmp_path / "export", store) == manifest
    content = "\n".join(path.read_text() for path in (tmp_path / "export").rglob("*") if path.is_file())
    assert "PRIVATE_ROW_LABELS" not in content
    assert "confusion_matrix" in content and "per_class" in content
    assert diagnostic_numeric_values(diagnostic_entry(ledger)[1])[
        ("tables", "seed_0/per_class", 1, "precision")] == 0.5


@pytest.mark.parametrize("kind", ["paired_differences", "paired_interval"])
def test_paired_seed_aggregate_export_uses_committed_development_only(tmp_path, kind):
    store = Store(tmp_path / "source")
    left, left_ref = observation(store, "baseline", [.5, .6, .7])
    right, right_ref = observation(store, "candidate", [.6, .65, .72])
    result = run_diagnostic(store, DiagnosticRequest(id=kind, kind=kind, metric="accuracy",
        expected_seeds=[0, 1, 2]), DiagnosticInputs(observation_refs=[left_ref, right_ref], declared_kinds=[kind]))
    ledger = build_evidence_ledger(store, [left, right], diagnostics=[result])
    assert result.status == "success"
    export(store, tmp_path / "export", ledger)
    assert verify_evidence_bundle(tmp_path / "export", store)


@pytest.mark.parametrize("change", ["labels", "predictions", "residuals", "row_ids", "provenance", "count", "final"])
def test_raw_arrays_extra_columns_and_inconsistent_aggregates_rejected(tmp_path, change):
    store = Store(tmp_path / "source")
    ledger = classification(store)
    entry = diagnostic_entry(ledger)[1]
    if change in {"labels", "predictions", "residuals"}:
        entry["arrays"][change] = [0, 1, 0, 1]
    elif change == "row_ids":
        entry["tables"]["seed_0/per_class"][0]["row_id"] = "private-example"
    elif change == "provenance":
        entry["provenance"]["raw_labels"] = [0, 1]
    elif change == "count":
        entry["arrays"]["seed_0/confusion_matrix"][0][0] += 1
    else:
        entry["provenance"]["split"] = "final"
    with pytest.raises(IntegrityError, match="aggregate-only"):
        export(store, tmp_path / "rejected", ledger)
    assert not (tmp_path / "rejected").exists()


def test_uncommitted_diagnostic_is_not_exportable(tmp_path):
    store = Store(tmp_path / "source")
    ledger = classification(store)
    store.db.execute("DELETE FROM calls WHERE key='research-diagnostic/classification'")
    store.db.commit()
    with pytest.raises(IntegrityError, match="committed controller"):
        export(store, tmp_path / "rejected", ledger)


LATEX = r"""\documentclass{article}
\title{Synthetic aggregates}
\begin{document}
\section{Recorded measurements}
\begin{tabular}{lr}
Support & 3 \\
Precision & 0.5 \\
\end{tabular}
\section{Limitations}
Synthetic fixture; interpretation remains unverified.
\end{document}
"""


def test_diagnostic_table_numbers_bind_to_exact_field_paths(tmp_path):
    store = Store(tmp_path / "source")
    ledger = classification(store)
    bindings = fixture_bindings(LATEX, ledger)
    assert all(bound.location == "diagnostic_value" for bound in bindings.numeric_values)
    assert all(bound.diagnostic_path for bound in bindings.numeric_values)
    manuscript = bind_paperorchestra_manuscript(bindings, LATEX, ledger, synthetic_agents=True)
    assert "[Bound value diagnostic:" in manuscript.results.text
    context = binding_prompt_context(LATEX, ledger)
    groups = context["numeric_candidate_groups"]
    assert all(any(candidate.get("diagnostic_path") for candidate in group["matching_candidates"])
               for group in groups.values())
    # Number equality cannot justify pointing at a different cell or a raw-data path.
    bad = deepcopy(bindings)
    bad.numeric_values[0].diagnostic_path = ["arrays", "individual_predictions", 0]
    with pytest.raises(ValueError, match="approved aggregate"):
        bind_paperorchestra_manuscript(bad, LATEX, ledger, synthetic_agents=True)


def test_mutating_rendered_diagnostic_value_still_fails_binding(tmp_path):
    store = Store(tmp_path / "source")
    ledger = classification(store)
    bindings = fixture_bindings(LATEX, ledger)
    altered = LATEX.replace("0.5", "0.9")
    bindings.latex_sha256 = digest(altered.encode())
    for section in bindings.sections:
        section.quote = altered[section.start:section.end]
    for numeric in bindings.numeric_values:
        numeric.quote = altered[numeric.start:numeric.end]
    with pytest.raises(ValueError, match="differs from its recorded evidence"):
        bind_paperorchestra_manuscript(bindings, altered, ledger, synthetic_agents=True)
    assert json.loads(json.dumps(ledger))["includes_final"] is False
