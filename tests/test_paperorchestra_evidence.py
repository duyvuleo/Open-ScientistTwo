"""Explicitly synthetic provenance tests, never experimental result fixtures."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from scientisttwo.integrations.paperorchestra.evidence import (
    build_campaign_export_ledger, export_evidence_bundle, import_external_manuscript,
    verify_evidence_bundle,
)
from scientisttwo.live.models import Candidate, Hypothesis, Observation, SourceFile
from scientisttwo.models import CriticDecision
from scientisttwo.research.manuscript import build_evidence_ledger, freeze_research_history
from scientisttwo.store import IntegrityError, Store, canonical, digest

UPSTREAM = "a" * 40


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "synthetic-campaign")
    yield value
    value.close()


def committed(store, name="baseline", phase="full", value=.6):
    candidate = Candidate(files=[SourceFile(path="train.py", content="print('synthetic; never run')\n")],
                          command=["python", "train.py"], description="Synthetic unit-test artifact")
    protocol = {"phase": phase, "seeds": [0], "adapter": "synthetic_test"}
    result = Observation(id=name, phase=phase, status="success", code_ref=store.put_json(candidate),
                         protocol_ref=store.put_json(protocol), protocol_hash=digest(canonical(protocol)),
                         metric="accuracy", value=value, values=[value],
                         prediction_refs=[store.put_json({"do_not_export_payload": ["synthetic label"]})],
                         log_refs=[store.put_json({"do_not_export_log": "synthetic machine details"})])
    store.cache("development/" + name, {"synthetic": True}, result)
    return result


def exported(store, tmp_path, *, sources=()):
    result = committed(store)
    ledger = build_evidence_ledger(store, [result], sources, history_refs=[])
    destination = tmp_path / "materials"
    manifest = export_evidence_bundle(store, destination, ledger=ledger,
        title="Synthetic test", hypothesis="No empirical claim", upstream_revision=UPSTREAM)
    return destination, ledger, manifest


def test_export_is_bounded_deterministic_and_payload_free(store, tmp_path):
    directory, ledger, manifest = exported(store, tmp_path)
    assert verify_evidence_bundle(directory, store) == manifest
    assert manifest["provenance"]["source_ledger_sha256"] == digest(canonical(ledger))
    joined = "\n".join(path.read_text() for path in directory.rglob("*") if path.is_file())
    assert "do_not_export_payload" not in joined and "do_not_export_log" not in joined
    assert str(store.root) not in joined
    assert "synthetic-campaign" in joined
    assert json.loads((directory / "figures/info.json").read_text()) == []
    assert len(list(directory.rglob("*.json"))) == 5
    with pytest.raises(FileExistsError):
        export_evidence_bundle(store, directory, ledger=ledger, title="Test", hypothesis="Test",
                               upstream_revision=UPSTREAM)


@pytest.mark.parametrize("change", ["score", "protocol", "prediction", "uncommitted", "final"])
def test_export_refuses_tampered_or_final_source_evidence(store, tmp_path, change):
    result = committed(store)
    if change == "final":
        final = committed(store, "final-heldout", phase="final")
        ledger = build_evidence_ledger(store, [final], include_final=True, method_frozen=True)
    else:
        ledger = build_evidence_ledger(store, [result])
    if change == "score":
        next(iter(ledger["entries"].values()))["value"] = .99
    elif change == "protocol":
        (store.root / result.protocol_ref).write_text("{}")
    elif change == "prediction":
        (store.root / result.prediction_refs[0]).write_text("{}")
    elif change == "uncommitted":
        store.db.execute("DELETE FROM calls WHERE key='development/baseline'")
    with pytest.raises(IntegrityError):
        export_evidence_bundle(store, tmp_path / "rejected", ledger=ledger, title="Test",
                               hypothesis="Test", upstream_revision=UPSTREAM)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("change", ["missing", "extra", "empty_directory", "edited", "symlink", "oversized"])
def test_verifier_rejects_partial_altered_and_unbounded_exports(store, tmp_path, change):
    directory, _, _ = exported(store, tmp_path)
    target = directory / "experimental_log.md"
    if change == "missing":
        target.unlink()
    elif change == "extra":
        (directory / "checkpoint.bin").write_bytes(b"not permitted")
    elif change == "empty_directory":
        (directory / "unregistered").mkdir()
    elif change == "edited":
        target.write_text("fabricated result")
    elif change == "symlink":
        target.unlink()
        target.symlink_to(store.root / "campaign.sqlite")
    else:
        target.write_bytes(b"x" * 2_000_001)
    with pytest.raises(IntegrityError):
        verify_evidence_bundle(directory)


def test_export_keeps_history_and_source_ids(store, tmp_path):
    result = committed(store, "search/0/subset/0", phase="subset")
    trial = {"id": result.id, "idea": Hypothesis(title="Synthetic idea", hypothesis="No scientific claim",
        novelty_score=0, novelty_scope="Synthetic", source_ids=["paper"]).model_dump(),
        "candidate": store.json(result.code_ref), "result": result.model_dump(),
        "decision": CriticDecision(decision="Bad", reason="Synthetic rejection after success").model_dump(),
        "parents": []}
    store.cache("live-trial/" + result.id, {"key": result.id}, trial)
    source = {"id": "paper", "title": "Synthetic reference", "authors": ["Fixture"], "year": 2026}
    ledger = build_evidence_ledger(store, [], [source], history_refs=freeze_research_history(store, [trial]))
    destination = tmp_path / "materials"
    export_evidence_bundle(store, destination, ledger=ledger, title="Test", hypothesis="Test",
                           upstream_revision=UPSTREAM)
    history = ledger["research_history"]["entries"]
    assert len(history) == 1
    assert next(iter(history)) in (destination / "experimental_log.md").read_text()
    mapping = json.loads((destination / "citation_map.json").read_text())
    assert next(iter(mapping.values()))["source_id"] == "paper"
    assert next(iter(mapping.values()))["authors"] == ["Fixture"]
    assert next(iter(mapping)) in (destination / "references.bib").read_text()


def test_campaign_selection_excludes_final_and_can_run_readonly(store, tmp_path):
    dev = committed(store)
    committed(store, "heldout", phase="final", value=.987654)
    db = sqlite3.connect(f"file:{store.root / 'campaign.sqlite'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    readonly = Store.__new__(Store)
    readonly.root, readonly.db = store.root, db
    try:
        ledger = build_campaign_export_ledger(readonly)
        assert set(ledger["observations"]) == {dev.id}
        assert "0.987654" not in json.dumps(ledger)
        with pytest.raises(IntegrityError, match="final"):
            build_campaign_export_ledger(readonly, ["heldout"])
        export_evidence_bundle(readonly, tmp_path / "readonly-export", ledger=ledger,
            title="Test", hypothesis="Test", upstream_revision=UPSTREAM)
    finally:
        db.close()


def test_empty_unknown_and_duplicate_observation_selections_fail(store):
    committed(store)
    for ids in (["absent"], ["baseline", "baseline"]):
        with pytest.raises(IntegrityError):
            build_campaign_export_ledger(store, ids)
    with pytest.raises(ValueError):
        build_campaign_export_ledger(store, [])


def test_raw_diagnostic_payload_requires_separate_contract(store, tmp_path):
    result = committed(store)
    diagnostic = {"status": "success", "request": {"id": "synthetic"}, "summary": {"samples": 2},
                  "arrays": {"labels": [0, 1]}}
    ref = store.put_json(diagnostic)
    ledger = build_evidence_ledger(store, [result], diagnostics=[{**diagnostic, "artifact_ref": ref}])
    with pytest.raises(IntegrityError, match="aggregate-only"):
        export_evidence_bundle(store, tmp_path / "blocked", ledger=ledger,
            title="Test", hypothesis="Test", upstream_revision=UPSTREAM)


def test_external_manuscript_stays_unverified_artifact_without_conversion(store, tmp_path):
    directory, ledger, _ = exported(store, tmp_path)
    latex = tmp_path / "main.tex"
    latex.write_text(r"\documentclass{article}\begin{document}Synthetic result: 0.6.\end{document}")
    imported = import_external_manuscript(store, directory, latex)
    assert imported.artifact_kind == "external_latex" and imported.controller_accepted is False
    assert imported.validation["status"] == "unverified"
    assert imported.validation["checks"]["prose_entailment"] == "unverified"
    assert imported.validation["checks"]["source_commitments"] == "unverified"
    assert store.read(imported.latex_ref) == latex.read_bytes()
    assert store.json(imported.evidence_ledger_ref) == ledger
    assert "manuscript_ref" not in imported.model_dump()


@pytest.mark.parametrize("change", [None, "value", "quote", "id", "latex", "duplicate", "quote_value"])
def test_import_validates_structured_claim_bindings_and_preserves_rejections(store, tmp_path, change):
    directory, ledger, _ = exported(store, tmp_path)
    latex = tmp_path / "main.tex"
    latex.write_text("Synthetic result: 0.6. Unsupported: 0.9.")
    claim = {"id": "c1", "quote": "Synthetic result: 0.6", "kind": "measured",
             "evidence_ids": [next(iter(ledger["entries"]))], "value": .6}
    bindings = {"latex_sha256": digest(latex.read_bytes()), "claims": [claim]}
    if change == "value":
        claim["value"] = .9
    elif change == "quote":
        claim["quote"] = "Unwritten sentence"
    elif change == "id":
        claim["evidence_ids"] = ["measurement:invented"]
    elif change == "latex":
        bindings["latex_sha256"] = "0" * 64
    elif change == "duplicate":
        bindings["claims"].append(deepcopy(claim))
    elif change == "quote_value":
        claim["quote"] = "Unsupported: 0.9"
    result = import_external_manuscript(store, directory, latex, evidence_bindings=bindings)
    assert result.validation["status"] == ("fail" if change else "unverified")
    assert result.validation["checks"]["structured_evidence_bindings"] == ("fail" if change else "pass")
    assert result.validation["checks"]["natural_language_claim_completeness"] == "unverified"
    assert result.controller_accepted is False


def test_unknown_external_citations_are_flagged(store, tmp_path):
    directory, _, _ = exported(store, tmp_path)
    latex = tmp_path / "main.tex"
    latex.write_text(r"A claim \citep[see]{unknown}.")
    result = import_external_manuscript(store, directory, latex)
    assert result.validation["status"] == "fail"
    assert result.validation["checks"]["citation_id_resolution"] == "fail"


def test_export_budget_and_upstream_revision_must_be_explicit(store, tmp_path):
    ledger = build_evidence_ledger(store, [committed(store)])
    for revision, budget in (("main", 10000), (UPSTREAM, 10), (UPSTREAM, 2_000_001)):
        with pytest.raises(ValueError):
            export_evidence_bundle(store, tmp_path / "rejected", ledger=ledger, title="Test",
                hypothesis="Test", upstream_revision=revision, max_bytes=budget)
    assert not (tmp_path / "rejected").exists()


def test_public_cpu_example_is_frozen_nonfinal_and_omits_machine_paths():
    # This verifies the distributed record, not independent metric recomputation.
    root = Path(__file__).parents[1] / "examples/paperorchestra/materials"
    manifest = verify_evidence_bundle(root)
    ledger = json.loads((root / "evidence_ledger.json").read_text())
    assert manifest["provenance"]["synthetic_research_agents"] is True
    assert ledger["includes_final"] is False
    assert len(ledger["observations"]) == 11
    assert sum(value["kind"] == "measured" for value in ledger["entries"].values()) == 22
    assert len(ledger["research_history"]["entries"]) == 8
    payload = b"\n".join(path.read_bytes() for path in root.rglob("*") if path.is_file())
    assert b"/Users/" not in payload and b"/home/" not in payload
