"""Mechanical audit of versioned research evidence; never a scientific verdict.

Historical and incompatible campaigns are inspected read-only and never upgraded.
New derivations are replayed in a temporary SQLite/artifact overlay. This module
does not construct an agent backend or perform literature requests or compilation.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile

from ..artifacts import ArtifactFile, _checked_path, safe_path, verified_file
from ..live.adapters import implementation_identity, load_development
from ..live.evaluation import compare_observations
from ..live.models import Candidate, LiveSpec, Observation
from ..live.parent_audit import audit_parent_edits
from ..live.parent_edits import apply_parent_patch, parent_context
from ..live.source_tree import candidate_snapshot
from ..models import AgentResult, Profile, TaskSpec
from ..store import IntegrityError, Store, canonical, digest, file_digest
from .contracts import FullSetHandoff, SelectionReview
from .literature import (
    LiteratureConfig,
    NoveltyValidation,
    SourceRecord,
    retrieve_literature,
    validate_novelty,
)
from .manuscript import ManuscriptRecord, validate_manuscript, verify_evidence_ledger


class _ReplayStore(Store):
    """Copy ledger metadata; read immutable original payloads and write only scratch."""

    def __init__(self, root, original):
        super().__init__(root)
        original.db.backup(self.db)
        self.original = original

    def read(self, name):
        if (self.root / name).is_file():
            return super().read(name)
        return self.original.read(name)

    def materialize_artifact(self, name):
        """Copy only a requested payload into private scratch, never link originals.

        Metadata-only overlays cannot satisfy streaming consumers with read()
        fallback alone. Verify both the frozen registration and original bytes,
        stream the copy, then verify it again before publishing the local file.
        Unrequested checkpoint/model binaries remain solely in the source store.
        """
        row = self.db.execute("SELECT sha,size FROM artifacts WHERE path=?", (name,)).fetchone()
        if row is None or not name.startswith("artifacts/"):
            raise IntegrityError("Unregistered replay artifact")
        target = _checked_path(self.root, name)
        if target.exists():
            return
        artifact = ArtifactFile(ref=name, sha256=row["sha"], size=row["size"])
        source = verified_file(self.original, artifact)
        target.parent.mkdir(parents=True, exist_ok=True)
        _checked_path(self.root, name)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix="replay-", suffix=".tmp", dir=target.parent,
                                              delete=False) as outgoing:
                temporary = Path(outgoing.name)
                hasher, count = hashlib.sha256(), 0
                with source.open("rb") as incoming:
                    for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                        count += len(chunk)
                        if count > artifact.size:
                            raise IntegrityError("Original replay artifact grew during copying")
                        hasher.update(chunk)
                        outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if count != artifact.size or hasher.hexdigest() != artifact.sha256:
                raise IntegrityError("Original replay artifact changed during copying")
            if target.exists() or target.is_symlink():
                raise IntegrityError("Replay artifact target appeared during copying")
            temporary.replace(target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _metadata(store, prefix):
    for row in store.db.execute("SELECT key,value FROM metadata ORDER BY key"):
        if row["key"].startswith(prefix):
            yield row["key"].removeprefix(prefix), json.loads(row["value"])


def _agent(store, key):
    accepted = store.get("accepted-agent-output/" + key)
    if accepted:
        proof = store.json(accepted)
        if proof.get("status") != "accepted" or proof.get("invocation_key") != key:
            raise IntegrityError("Scientific response has no accepted contract binding")
        key = proof["response_key"]
    ref = store.get("request/" + key)
    if ref is None:
        raise IntegrityError("Scientific response is missing its immutable request")
    request = store.json(ref)
    row = store.db.execute("SELECT * FROM calls WHERE key=?", ("live-agent/" + key,)).fetchone()
    if row is None or row["input_hash"] != digest(canonical(request)):
        raise IntegrityError("Scientific response is not bound to its request")
    response = AgentResult.model_validate(store.json(row["output_ref"]))
    operations = [
        r
        for r in store.db.execute("SELECT * FROM live_operations WHERE kind='model'")
        if r["key"].startswith("model/" + key + "/") and r["status"] == "success"
    ]
    if not any(
        r["result_ref"] == row["output_ref"] and r["request_hash"] == row["input_hash"]
        for r in operations
    ):
        raise IntegrityError("Scientific response has no matching completed model operation")
    return request, response.payload


def _observation(store, identifier):
    row = store.db.execute(
        "SELECT output_ref FROM calls WHERE key=?", ("development/" + identifier,)
    ).fetchone()
    if row is None:
        raise IntegrityError("Missing committed development observation")
    return Observation.model_validate(store.json(row[0]))


def _without_retrieval_times(value):
    if isinstance(value, list):
        return [_without_retrieval_times(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _without_retrieval_times(item)
            for key, item in value.items()
            if key not in {"created_at", "retrieved_at"}
        }
    return value


def _literature(store, replay):
    checks = []
    for row in store.db.execute(
        "SELECT * FROM calls WHERE key LIKE 'research/literature/bundle/%'"
    ):
        bundle = store.json(row["output_ref"])
        if bundle.get("schema_version") != "research-literature-v1":
            raise IntegrityError("Unknown literature bundle schema")
        provenance = bundle["provenance"]
        if provenance["operation_key"] != row["key"]:
            raise IntegrityError("Literature operation key changed")
        local_ref = store.get("research_local_sources")
        if local_ref is None:
            raise IntegrityError("Literature bundle is missing its frozen supplied-source inputs")
        config = LiteratureConfig.model_validate(provenance["config"])
        local_sources = store.json(local_ref)
        inputs = {
            "schema_version": bundle["schema_version"],
            "provider_version": provenance["provider_version"],
            "config": config.model_dump(mode="json"),
            "query": bundle["query"],
            "local_sources": local_sources,
        }
        if row["input_hash"] != digest(canonical(inputs)) or row[
            "key"
        ] != "research/literature/bundle/" + digest(canonical(inputs)):
            raise IntegrityError(
                "Literature search bundle is not bound to its declared query and inputs"
            )
        for source in bundle["sources"]:
            record = SourceRecord.model_validate(source)
            if (
                record.content_ref
                and digest(store.read(record.content_ref)) != record.content_sha256
            ):
                raise IntegrityError("Retrieved source content hash changed")
            for passage in record.passages:
                if digest(passage.text.encode()) != passage.sha256:
                    raise IntegrityError("Source passage hash changed")
        for receipt in provenance["requests"]:
            if (
                receipt.get("body_ref")
                and digest(store.read(receipt["body_ref"])) != receipt["sha256"]
            ):
                raise IntegrityError("Raw literature response hash changed")
            saved = store.db.execute(
                "SELECT output_ref FROM calls WHERE key=?", (receipt["key"],)
            ).fetchone()
            if receipt["status"] == "unknown_completion":
                if store.get(receipt["key"]) is None:
                    raise IntegrityError("Unknown literature completion has no durable admission")
            elif saved is None:
                raise IntegrityError("Literature response has no committed request operation")
            else:
                raw = store.json(saved[0])
                if any(
                    raw.get(name) != receipt.get(name)
                    for name in ("url", "status", "body_ref", "sha256", "http_status")
                ):
                    raise IntegrityError(
                        "Literature bundle differs from committed transport receipt"
                    )
        # Remove only the convenience bundle in scratch; raw HTTP responses remain
        # cached. Any attempt to perform fresh external I/O is a hard audit failure.
        with replay.db:
            replay.db.execute("DELETE FROM calls WHERE key=?", (row["key"],))

        def no_network(*args, **kwargs):
            raise IntegrityError("Literature replay requires an unrecorded external request")

        reproduced = retrieve_literature(
            replay, config, bundle["query"], local_sources, allow_network=True, transport=no_network
        )
        if canonical(_without_retrieval_times(reproduced)) != canonical(
            _without_retrieval_times(bundle)
        ):
            raise IntegrityError(
                "Literature metadata or passages differ from offline raw-response replay"
            )
        checks.append(
            {
                "bundle_ref": row["output_ref"],
                "status": "pass",
                "retrieval_status": bundle["status"],
                "source_count": len(bundle["sources"]),
                "metadata_and_passages": "offline_replay_pass",
            }
        )
    for key, ref in _metadata(store, "novelty/"):
        recorded = NoveltyValidation.model_validate(store.json(ref))
        request, payload = _agent(store, key)
        if request["role"] != "novelty_checker":
            raise IntegrityError("Novelty judgment came from the wrong role")
        bundle = request["inputs"]["literature"]
        row = store.db.execute(
            "SELECT output_ref FROM calls WHERE key=?", (bundle["provenance"]["operation_key"],)
        ).fetchone()
        if row is None or canonical(store.json(row[0])) != canonical(bundle):
            raise IntegrityError("Novelty checker received an unfrozen literature bundle")
        rebuilt = validate_novelty(payload, bundle)
        if rebuilt != recorded:
            raise IntegrityError(
                "Novelty evidence validation changed the raw response or effective status"
            )
        if "novelty_score" in request["inputs"]["idea"]:
            raise IntegrityError("Independent novelty checker received the generator's score")
        checks.append(
            {
                "novelty": key,
                "status": "pass",
                "effective_status": rebuilt.effective_status,
                "scientific_novelty": "unverified",
                "usable_references": rebuilt.usable_reference_count,
            }
        )
    return checks


def _handoffs(store, replay, adapter):
    checks = []
    materialize = getattr(adapter, "materialize", None)
    paths = getattr(adapter, "editable_paths", None)
    paths = paths() if callable(paths) else adapter.context().get("editable_files")
    options = {"materialize": materialize, "editable_paths": paths}
    for key, ref in _metadata(store, "full-handoff/"):
        saved = store.json(ref)
        request, payload = _agent(store, key)
        handoff = FullSetHandoff.model_validate(payload)
        if canonical(saved.get("handoff", saved)) != canonical(handoff):
            raise IntegrityError("Full-stage handoff differs from its model response")
        inputs = request["inputs"]
        screened = Observation.model_validate(inputs["subset_result"])
        if (
            canonical(_observation(store, screened.id)) != canonical(screened)
            or screened.status != "success"
        ):
            raise IntegrityError(
                "Full-stage handoff was not based on a successful committed subset result"
            )
        parent_ref = inputs["parent"]["parent_source_ref"]
        tree = store.json(parent_ref)
        parent = Candidate.model_validate(store.json(tree["proposal_ref"]))
        context = parent_context(replay, parent, parent_ref, **options)
        context["trial_id"] = inputs["parent"]["trial_id"]
        if canonical(context) != canonical(inputs["parent"]):
            raise IntegrityError("Full-stage coder did not receive the exact screened source")
        if (
            screened.source_tree_ref
            and store.json(screened.source_tree_ref)["bundle_ref"] != tree["bundle_ref"]
        ):
            raise IntegrityError("Full-stage parent differs from screened executable source")
        candidate = parent
        if handoff.action != "ReuseSnapshot":
            if handoff.action == "ScalingPatch" and not set(
                handoff.change_plan.changed_files
            ) <= set(inputs["scaling_paths"]):
                raise IntegrityError("Full-stage scaling patch exceeds declared scaling files")
            candidate, proof = apply_parent_patch(
                replay,
                parent,
                parent_ref,
                handoff.patch,
                declared_paths=handoff.change_plan.changed_files,
                **options,
            )
            if canonical(proof) != canonical(saved["source_proof"]):
                raise IntegrityError(
                    "Full-stage patch differs from independently replayed source proof"
                )
        adapter.validate(candidate)
        expected = candidate_snapshot(replay, materialize(candidate) if materialize else candidate)
        prefix = key.removesuffix("/full-handoff")
        first = store.db.execute(
            "SELECT output_ref FROM calls WHERE key=?", ("development/" + prefix + "/full/0",)
        ).fetchone()
        if first:
            observed = Observation.model_validate(store.json(first[0]))
            if (
                not observed.source_tree_ref
                or store.json(observed.source_tree_ref)["bundle_ref"] != expected.bundle_ref
            ):
                raise IntegrityError(
                    "Full-development execution differs from accepted handoff source"
                )
        if handoff.action == "ScientificChange":
            recheck = _observation(store, key + "/rescreen")
            if (
                not recheck.source_tree_ref
                or store.json(recheck.source_tree_ref)["bundle_ref"] != expected.bundle_ref
            ):
                raise IntegrityError("Scientific handoff rescreen used a different source")
            if first:
                trial = store.db.execute(
                    "SELECT output_ref FROM calls WHERE key=?", ("live-trial/" + recheck.id,)
                ).fetchone()
                if (
                    trial is None
                    or store.json(trial[0])["decision"]["decision"] != "Good"
                    or recheck.status != "success"
                ):
                    raise IntegrityError(
                        "Scientific handoff advanced without accepted renewed screening"
                    )
        checks.append(
            {
                "handoff": key,
                "status": "pass",
                "action": handoff.action,
                "full_execution": "recorded" if first else "not_exercised",
                "mechanism_equivalence": "unverified",
            }
        )
    return checks


def _selection(store, config, adapter):
    ref = store.get("selection_review")
    if ref is None:
        return []
    selection = SelectionReview.model_validate(store.json(ref))
    request, payload = _agent(store, "selection/review")
    if selection != SelectionReview.model_validate(payload):
        raise IntegrityError("Selection differs from its raw reviewer response")
    candidates = request["inputs"]["candidates"]
    eligible = request["inputs"]["eligible_ids"]
    if selection.selected_id not in eligible:
        raise IntegrityError("Selector chose a candidate outside the controller's eligible set")
    found = [trial for trial in candidates if trial["id"] == selection.selected_id]
    if len(found) != 1:
        raise IntegrityError("Selected trial is absent or ambiguous")
    chosen = found[0]
    compare = getattr(adapter, "compare", None)
    compare = (
        compare
        if callable(compare)
        else lambda a, b: compare_observations(config.development, a, b)
    )
    chosen_observation = _observation(store, chosen["result"]["id"])
    if (
        canonical(chosen_observation) != canonical(chosen["result"])
        or chosen["decision"]["decision"] != "Good"
    ):
        raise IntegrityError("Selection references altered or rejected evidence")
    for trial in candidates:
        observed = _observation(store, trial["result"]["id"])
        if canonical(observed) != canonical(trial["result"]) or observed.phase != "full":
            raise IntegrityError("Selection candidate is not committed full-development evidence")
        comparison = compare(observed, chosen_observation)
        if comparison["valid"] and comparison.get("delta", 0) > config.development.epsilon:
            raise IntegrityError("Selection chose a numerically inferior candidate")
    return [
        {
            "selection_ref": ref,
            "status": "pass",
            "selected_id": selection.selected_id,
            "scientific_selection_judgment": "unverified",
        }
    ]


def _manuscripts(store, replay, config):
    checks, seen = [], set()
    for row in store.db.execute(
        "SELECT path FROM artifacts WHERE kind IN ('manuscript_bundle','final_manuscript_bundle')"
    ):
        manifest = store.json(row["path"])
        final = manifest.get("schema_version") == "manuscript-final-bundle-v1"
        if manifest.get("schema_version") not in {
            "manuscript-bundle-v1",
            "manuscript-final-bundle-v1",
        }:
            raise IntegrityError("Unknown manuscript bundle version")
        files = manifest["files"]
        paperorchestra = not final and manifest.get("writer", {}).get("type") == "paperorchestra"
        if (
            digest(canonical({name: item["sha256"] for name, item in sorted(files.items())}))
            != manifest["project_sha256"]
        ):
            raise IntegrityError("Rendered project hash differs from its declared inputs")
        project = store.root / "manuscripts" / manifest["project_sha256"]
        if project.is_symlink():
            raise IntegrityError("Manuscript project cannot be a symbolic link")
        for name, item in files.items():
            safe_path(name)
            raw = store.read(item["ref"])
            if len(raw) != item["bytes"] or digest(raw) != item["sha256"]:
                raise IntegrityError("Rendered manuscript file hash or length changed")
            path = project / name
            if project.exists() and (
                path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(project.resolve())
                or path.stat().st_size != item["bytes"]
                or file_digest(path) != item["sha256"]
            ):
                raise IntegrityError("Materialized manuscript project differs from sealed inputs")
        check = {
            "bundle_ref": row["path"],
            "status": "pass",
            "final_appendix": final,
            "pdf": "unverified",
            "scientific_validity": "unverified",
        }
        if final:
            if manifest["model_calls"] != 0 or canonical(
                store.json(manifest["frozen_ref"])
            ) != canonical(store.get("live_frozen")):
                raise IntegrityError("Final appendix is not bound to the frozen method")
            development = store.json(manifest["development_bundle_ref"])
            if development.get("schema_version") != "manuscript-bundle-v1":
                raise IntegrityError("Final appendix does not preserve its development manuscript")
            if development.get("writer", {}).get("type") == "paperorchestra":
                from ..integrations.paperorchestra.manuscript_binding import verify_bound_final_manifest
                check["authoritative_writer"] = verify_bound_final_manifest(store, manifest)
            for label, ref in manifest["final_result_refs"].items():
                from .timing_evidence import verify_final_result_binding
                result = store.json(ref)
                if canonical(result) != canonical(store.get("live_final", {}).get(label)):
                    raise IntegrityError("Final manuscript summary differs from committed final results")
                verify_final_result_binding(store, label, result, summary_ref=ref)
        else:
            ledger = store.json(manifest["ledger_ref"])
            if ledger.get("includes_final"):
                raise IntegrityError("Research manuscript ledger exposes final-test evidence")
            # This verifier is read-only and streams original artifact paths.
            verify_evidence_ledger(store, ledger)
            if paperorchestra:
                from ..integrations.paperorchestra.manuscript_binding import verify_bound_manifest
                check["authoritative_writer"] = verify_bound_manifest(
                    store, manifest, expected_image_digest=config.workflow.compiler_image)
            manuscript = ManuscriptRecord.model_validate(store.json(manifest["manuscript_ref"]))
            validation = validate_manuscript(manuscript, ledger, ledger["sources"])
            if validation.status != "pass" or canonical(validation) != canonical(
                store.json(manifest["validation_ref"])
            ):
                raise IntegrityError(
                    "Manuscript claim/citation validation differs from its recorded proof"
                )
            if manuscript.synthetic_agents != store.get("live_backend", {}).get("synthetic"):
                raise IntegrityError(
                    "Manuscript synthetic-agent declaration differs from campaign backend"
                )
            seen.add(manifest["manuscript_ref"])
            check["claim_checks"] = validation.checks
        attempts = manifest.get("compilation_attempts", [])
        if paperorchestra and manifest["compilation_status"] == "success":
            # The writer helper verifies the shared probe, exact upstream source,
            # supporting files, isolated compiler receipt and resulting PDF.
            check["pdf"] = "pass"
        elif manifest["compilation_status"] == "success":
            if not attempts or not manifest.get("pdf_ref"):
                raise IntegrityError("Successful manuscript compilation lacks a PDF or receipt")
            last = attempts[-1]
            if (
                last["status"] != "success"
                or not last["network_disabled"]
                or not last["shell_escape_disabled"]
                or last["timed_out"]
                or last["image_digest"] != config.workflow.compiler_image
            ):
                raise IntegrityError("Compilation violates the frozen isolation policy")
            pdf = store.read(manifest["pdf_ref"])
            if not pdf.startswith(b"%PDF-") or len(pdf) > 50_000_000:
                raise IntegrityError("Compiled PDF is invalid or oversized")
            operation_key = f"artifact/compile/{manifest['project_sha256']}/{len(attempts)}"
            operation = store.db.execute(
                "SELECT * FROM live_operations WHERE key=?", (operation_key,)
            ).fetchone()
            if (
                operation is None
                or operation["kind"] != "probe"
                or operation["status"] != "success"
            ):
                raise IntegrityError("Compilation has no bounded probe completion")
            completed = store.json(operation["result_ref"])
            if completed["pdf_ref"] != manifest["pdf_ref"] or completed["result"] != last:
                raise IntegrityError("Compiled manuscript differs from its operation receipt")
            if digest(store.read(completed["worker_ref"])) != completed["worker_sha256"]:
                raise IntegrityError("Compiler dependency closure changed")
            receipt = store.json(last["receipt_ref"])
            if (
                receipt.get("returncode") != 0
                or receipt.get("timed_out")
                or receipt.get("limit_exceeded")
                or receipt.get("image_id") != config.workflow.compiler_image
                or "--network=none" not in receipt.get("argv", [])
            ):
                raise IntegrityError("Successful PDF references a failed container receipt")
            job = store.json(receipt["job_ref"])
            from .compiler_worker import COMMANDS

            if (
                job["project_manifest_sha256"] != manifest["project_sha256"]
                or job.get("commands") != COMMANDS
                or job.get("worker_sha256") != completed["worker_sha256"]
            ):
                raise IntegrityError("Compiler executed a different manuscript input set")
            report = store.json(completed["report_ref"])
            if (
                report.get("status") != "success"
                or report.get("pdf_sha256") != digest(pdf)
                or report.get("pdf_bytes") != len(pdf)
                or report.get("project_manifest_sha256") != manifest["project_sha256"]
            ):
                raise IntegrityError("PDF differs from the compiler worker's verified output")
            check["pdf"] = "pass"
        elif config.workflow.require_pdf:
            check["required_pdf_gate"] = "unverified"
        checks.append(check)
    for key, ref in _metadata(store, "manuscript/"):
        if ref not in seen:
            raise IntegrityError("Historical manuscript has no validated immutable rendered bundle")
    for row in store.db.execute(
        "SELECT path FROM artifacts WHERE kind IN ('rendered-manuscript','final-rendered-manuscript')"
    ):
        wrapper = store.json(row["path"])
        manifest = store.json(wrapper["manifest_ref"])
        if manifest["schema_version"] == "manuscript-bundle-v1":
            for field in ("manuscript_ref", "ledger_ref", "validation_ref"):
                if manifest[field] != wrapper[field]:
                    raise IntegrityError("Rendered manuscript wrapper differs from its bundle")
        for field, name in (
            ("markdown_ref", "manuscript.md"),
            ("latex_ref", "main.tex"),
            ("bibliography_ref", "references.bib"),
        ):
            if wrapper[field] != manifest["files"][name]["ref"]:
                raise IntegrityError("Rendered artifact alias differs from its manifest")
        if wrapper["pdf_ref"] != manifest["pdf_ref"]:
            raise IntegrityError("Rendered PDF alias differs from its manifest")
    return checks


def _diagnostics(store, replay, adapter):
    from .diagnostics import DiagnosticInputs, DiagnosticResult, run_diagnostic

    checks = []
    callback = getattr(adapter, "diagnostic_capabilities", None)
    allowed = {
        "paired_differences",
        "paired_interval",
        "classification",
        "regression_residuals",
        "timing_distribution",
    }
    if not callable(getattr(adapter, "diagnostics_contract", None)):
        allowed -= {"classification", "regression_residuals"}
    declared = ["metric", *sorted(set(callback() if callable(callback) else []) & allowed)]
    for key, ref in _metadata(store, "research-diagnostic-record/"):
        result = DiagnosticResult.model_validate(store.json(ref))
        if result.request.id != key:
            raise IntegrityError("Diagnostic identity differs from its registration")
        row = store.db.execute(
            "SELECT * FROM calls WHERE key=?", ("research-diagnostic/" + key,)
        ).fetchone()
        if row is None or row["output_ref"] != ref:
            raise IntegrityError("Diagnostic is not a committed controller operation")
        observations, trusted = [], []
        for input_ref in result.input_refs:
            value = store.json(input_ref)
            if "phase" in value and "metric" in value:
                observations.append(input_ref)
            else:
                trusted.append(input_ref)
        inputs = DiagnosticInputs(
            observation_refs=observations, trusted_result_refs=trusted, declared_kinds=declared
        )
        reproduced = run_diagnostic(replay, result.request, inputs).model_dump(mode="json")
        reproduced["artifact_ref"] = None
        if canonical(reproduced) != canonical(result):
            raise IntegrityError("Diagnostic replay differs from its frozen output")
        checks.append(
            {
                "diagnostic": key,
                "status": "pass",
                "execution_status": result.status,
                "scientific_concern_resolution": "unverified",
            }
        )
    return checks


def _read_frozen_spec(output):
    database = output / "campaign.sqlite"
    if not database.is_file():
        raise ValueError("Research campaign database not found")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute("SELECT value FROM metadata WHERE key='live_spec_ref'").fetchone()
        if row is None:
            return None
        ref = json.loads(row[0])
        registered = db.execute("SELECT sha FROM artifacts WHERE path=?", (ref,)).fetchone()
        path = (output / ref).resolve()
        if (
            registered is None
            or not path.is_relative_to(output / "artifacts")
            or not path.is_file()
        ):
            raise IntegrityError("Unregistered research specification")
        raw = path.read_bytes()
        if digest(raw) != registered[0]:
            raise IntegrityError("Frozen research specification was modified")
        return LiveSpec.model_validate_json(raw)


def audit_research(output, *, reexecute=False, include_final=False):
    """Audit current research-v2 campaigns without live calls or implicit upgrades."""
    output = Path(output).resolve()
    try:
        config = _read_frozen_spec(output)
    except (OSError, ValueError, IntegrityError, sqlite3.Error) as error:
        return {"status": "fail", "errors": [str(error)], "live_model_calls": 0}
    if (
        config is None
        or config.engine_version != "research-v2"
        or config.implementation != implementation_identity()
    ):
        return {
            "status": "unverified",
            "requires_pinned_runtime": True,
            "reason": "Historical or incompatible campaigns require their original frozen implementation for execution and audit",
            "live_model_calls": 0,
            "legacy_evidence_modified": False,
        }
    store = Store(output)
    checks, errors = {}, []
    try:
        errors.extend(store.verify())
        if errors:
            return {"status": "fail", "errors": errors, "live_model_calls": 0}
        pilot = config.development.adapter == "bundled-pilot"
        if pilot:
            from .pilot_bridge import create_pilot_campaign, audit_pilot_campaign

            task = TaskSpec.model_validate(store.json(store.get("task_ref")))
            profile = Profile.model_validate(store.json(store.get("profile_ref")))
            reconstructed, adapter, runner = create_pilot_campaign(
                store, task, profile, **store.get("runtime")
            )
            if reconstructed != config:
                raise IntegrityError("Pilot factory differs from frozen research specification")
            base = audit_pilot_campaign(
                store, adapter, runner, reexecute=reexecute, include_final=include_final
            )
            errors.extend(base["errors"])
        else:
            from ..live.audit import audit_live

            base = audit_live(output, reexecute_evaluators=reexecute)
            errors.extend(base.get("errors", []))
            adapter = load_development(config.native, config.development)
            if store.get("authorized_source_snapshot"):
                adapter.bind_source(store.get("authorized_source_snapshot"))
        with tempfile.TemporaryDirectory(prefix="scientisttwo-research-audit-") as temporary:
            replay = _ReplayStore(Path(temporary), store)
            try:
                for name, callback in (
                    ("literature_and_novelty", lambda: _literature(store, replay)),
                    ("full_stage_handoffs", lambda: _handoffs(store, replay, adapter)),
                    ("selection", lambda: _selection(store, config, adapter)),
                    ("parent_edits", lambda: audit_parent_edits(store, adapter)),
                    ("manuscripts", lambda: _manuscripts(store, replay, config)),
                    ("diagnostics", lambda: _diagnostics(store, replay, adapter)),
                ):
                    try:
                        records = callback()
                        checks[name] = {
                            "status": "pass" if records else "not_exercised",
                            "records": records,
                        }
                    except (OSError, ValueError, KeyError, TypeError, IntegrityError) as error:
                        checks[name] = {"status": "fail", "error": str(error)}
                        errors.append(name + ": " + str(error))
            finally:
                replay.close()
        final_row = store.db.execute(
            "SELECT min(rowid) FROM live_operations WHERE key LIKE 'final/%'"
        ).fetchone()[0]
        if (
            final_row is not None
            and store.db.execute(
                "SELECT 1 FROM live_operations WHERE kind='model' AND rowid>?", (final_row,)
            ).fetchone()
        ):
            errors.append("Model invocation followed protected final evaluation")
        return {
            "status": ("fail" if errors else "unverified"
                       if pilot and reexecute and base.get("reexecution") != "pass"
                       else "pass"),
            "checks": checks,
            "execution_audit": base,
            "errors": errors,
            "live_model_calls": 0,
            "scope": "Mechanical evidence, exact source and ledger consistency only; absent branches remain not_exercised",
            "scientific_validity": "unverified",
            "novelty_certification": "unverified",
            "all_13_task_readiness": "unverified",
        }
    except (OSError, ValueError, KeyError, TypeError, IntegrityError) as error:
        return {
            "status": "fail",
            "checks": checks,
            "errors": [*errors, str(error)],
            "live_model_calls": 0,
        }
    finally:
        store.close()
