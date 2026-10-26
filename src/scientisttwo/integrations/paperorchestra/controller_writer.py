"""PaperOrchestra writing inside the research controller's accounting boundary.

The upstream agents remain the writers. The controller supplies their model and
isolated PDF services, then indexes the exact selected source for scientific
review. Replaying a partial attempt restores committed services rather than
repeating model calls or container operations with an unknown completion.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import shutil
import time

from pydantic import Field, create_model

from ...backends import CodexBackend
from ...live.output_contracts import AgentOutputError
from ...live.runtime import AmbiguousCall
from ...models import Record
from ...research.manuscript import ManuscriptRecord, RenderedManuscript
from ...store import IntegrityError, canonical, digest, file_digest
from .evidence import export_evidence_bundle, verify_evidence_bundle
from .pdf_tools import ContainerPdfTools
from .writer_context import build_writer_context, writing_scope_instruction
from .runtime import (
    ModelResponse, RuntimeConfig, SOURCE_HASHES, UPSTREAM_REVISION,
    run_paperorchestra, verify_upstream,
)


class PaperOrchestraResponse(Record):
    output: str = Field(min_length=1, max_length=250_000)


def response_size_policy(max_characters):
    """Describe an opt-in text budget without claiming a provider token limit."""
    if max_characters is None:
        return None
    if type(max_characters) is not int or not 1 <= max_characters <= 250_000:
        raise ValueError("PaperOrchestra output-character budget must be an integer in 1..250000")
    return {"schema_version": "paperorchestra-output-character-budget-v1",
            "max_output_characters": max_characters, "field": "output",
            "counting": "Unicode code points in decoded output text, including JSON/LaTeX fences and escaping inside that text; outer transport JSON escaping is separate",
            "scope": "Complete concise upstream artifact; optional response-size adapter deviation",
            "provider_token_cap": False, "truncate_response": False, "automatic_retry": False,
            "preserve": ["required fields and closing syntax", "evidence and citation identifiers",
                         "numerical measurements", "required protocol facts", "uncertainty and limitations"]}


@lru_cache(maxsize=64)
def response_schema(max_characters):
    if response_size_policy(max_characters) is None:
        return PaperOrchestraResponse
    return create_model("BoundedPaperOrchestraResponse", __base__=Record,
        output=(str, Field(min_length=1, max_length=max_characters)))


def bounded_response_instruction(policy):
    return (
        "Controller output-character budget takes precedence over upstream verbosity and length targets: "
        f"the complete output text must contain at most {policy['max_output_characters']} Unicode characters. "
        "Count the entire decoded output, including JSON/LaTeX fences and all escaping inside the requested text. "
        "Produce a concise but COMPLETE short-note artifact in the requested format, not an excerpt or summary of missing content. "
        "Keep every required field, closing delimiter, evidence/citation identifier, numerical measurement, necessary protocol fact, "
        "and material uncertainty. Shorten prose, repeated explanation and optional commentary first. "
        "Do not fabricate evidence, omit required content to conceal a problem, or leave an unfinished JSON/LaTeX fragment. "
        "If these constraints cannot both be satisfied, state the inability without claiming completion; validation must fail closed. "
        "This is a character-budget mitigation, not a guaranteed model token cap."
    )


def authoritative_evidence_inventory(ledger):
    """Count supplied typed records, never infer measurements from an outline."""
    scope = "Counts of supplied ledger entries, not experiment runs, datasets, statistical replications or proposed outline items"
    if not isinstance(ledger, dict) or not ledger.get("entries"):
        return {"schema_version": "paperorchestra-evidence-inventory-v1", "status": "unavailable",
                "ledger_sha256": digest(canonical(ledger)), "counts_by_kind": None,
                "total_entries": None, "measured_entry_count": None, "measured_entry_ids": None,
                "scope": scope, "limitation": "An empty or absent ledger inventory does not establish zero measurements"}
    entries = ledger["entries"]
    if (not isinstance(entries, dict) or any(not isinstance(key, str) or not key
            or not isinstance(value, dict) or not isinstance(value.get("kind"), str)
            or not value["kind"] for key, value in entries.items())):
        raise IntegrityError("Supplied ledger entries lack valid typed evidence identities")
    measured = sorted(key for key, value in entries.items() if value["kind"] == "measured")
    return {"schema_version": "paperorchestra-evidence-inventory-v1", "status": "available",
            "ledger_sha256": digest(canonical(ledger)),
            "counts_by_kind": dict(sorted(Counter(value["kind"] for value in entries.values()).items())),
            "total_entries": len(entries), "measured_entry_count": len(measured),
            "measured_entry_ids": measured, "includes_final": ledger.get("includes_final"), "scope": scope}


EVIDENCE_AUTHORITY_INSTRUCTION = (
    "Controller evidence authority takes precedence over generated outline plans and mistaken counts in prose. "
    "Use the supplied typed evidence inventory and frozen experimental log for measured facts. "
    "An outline is a writing plan, not experimental evidence: correct its counts or requested tables when they conflict with available records. "
    "Only kind=measured entries count as measured ledger records; research_history, method, derived and diagnostic records retain their distinct types. "
    "Never invent, duplicate, relabel or pad measurements to satisfy an outline. Preserve all valid required numerical facts, evidence IDs, "
    "citation IDs, protocol details and limitations. If the inventory is unavailable, leave its counts unverified rather than assert zero."
)


@dataclass(frozen=True)
class ControllerWriterResult:
    manuscript: ManuscriptRecord
    rendered: RenderedManuscript
    evidence_ref: str


def fixture_model_response(inputs):
    """Scripted upstream-role responses; never enabled for a live backend."""
    from .fixtures import EvidenceFixtureTransport
    fixture = object.__new__(EvidenceFixtureTransport)
    fixture.ledger = inputs["scripted_ledger"]
    fixture.citations = inputs.get("scripted_citations", {})
    fixture.entries = sorted(
        ((key, item) for key, item in fixture.ledger["entries"].items()
         if item.get("kind") == "measured"),
        key=lambda pair: (pair[1]["protocol_hash"], pair[1]["metric"],
                          pair[1]["observation_id"]),
    )[:10]
    role = inputs["upstream_role"]
    if role == "outline":
        text = json.dumps({"title": "Scripted writing demonstration", "synthetic": True,
                           "sections": ["Purpose", "Evidence", "Results", "Limitations"]})
    elif role == "section_writing":
        text = fixture._document(refined=False)
    elif role == "content_refinement":
        previous = inputs.get("scripted_previous_latex")
        if previous:
            # An explicit changed passage keeps editorial test revisions observable
            # without inventing new scores or modifying the scientific method.
            note = ("Editorial feedback has been incorporated using the existing evidence. "
                    "No new experimental work or scientific resolution is implied.\n")
            text = previous.replace(r"\end{document}", note + r"\end{document}")
        else:
            text = fixture._document(refined=True)
        text = ('```json\n{"synthetic":true,"change":"Clarify evidence limits only"}\n```'
                '\n```latex\n' + text + '\n```')
    elif role == "peer_review":
        score = 5 if inputs.get("refinement_completed") else 4
        text = json.dumps({"Summary": "Explicitly scripted writing review.",
            "Strengths": ["Recorded evidence retained."], "Weaknesses": [],
            "Originality": 1, "Quality": 2, "Clarity": 2, "Significance": 1,
            "Soundness": 2, "Presentation": 2, "Contribution": 1,
            "Questions": [], "Limitations": ["Synthetic review; quality unverified."],
            "Ethical Concerns": False, "Overall": score, "Confidence": 1,
            "Decision": "Reject", "synthetic": True})
    elif role == "format_review":
        text = json.dumps({"synthetic": True, "figure_and_tables": {}, "other_issues": []})
    else:
        raise ValueError("Unexpected scripted PaperOrchestra role: " + role)
    return {"output": text}


def _tree(root, *, extensions=None, maximum=100_000_000):
    """Bounded regular-file manifest, independent of temporary workspace paths."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError("PaperOrchestra tree must be a regular directory")
    result, total = {}, 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise IntegrityError("PaperOrchestra artifacts cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise IntegrityError("PaperOrchestra artifacts cannot contain special files")
        if extensions is not None and path.suffix.lower() not in extensions:
            continue
        total += path.stat().st_size
        if total > maximum or len(result) >= 2500:
            raise IntegrityError("PaperOrchestra artifact quota exceeded")
        result[path.relative_to(root).as_posix()] = {
            "sha256": file_digest(path), "bytes": path.stat().st_size,
        }
    return result


class BudgetedPdfTools:
    """Durable PDF probes sharing campaign and optional suite accounting."""

    def __init__(self, controller, key, delegate, identity):
        self.controller, self.key, self.delegate = controller, key, delegate
        self.synthetic, self.isolation = delegate.synthetic, delegate.isolation
        self.identity, self.index, self.receipts, self.error = identity, 0, [], None

    def _call(self, kind, source, *, tex_name=None, output=None, timeout_seconds=None):
        store, budget = self.controller.store, self.controller.budget
        if output is not None and (Path(output).is_symlink() or
                any(parent.is_symlink() for parent in Path(output).absolute().parents)):
            raise IntegrityError("PDF output path cannot traverse a symlink")
        self.index += 1
        key = f"paperorchestra/{self.key}/pdf/{self.index:04d}"
        if kind == "compile":
            inputs = _tree(source, extensions={".tex", ".bib", ".bst", ".sty", ".cls",
                ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg", ".def", ".clo", ".cfg"})
        else:
            path = Path(source)
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 100_000_000:
                raise IntegrityError("Invalid bounded PDF input")
            inputs = {"paper.pdf": {"sha256": file_digest(path), "bytes": path.stat().st_size}}
        request = {"operation": kind, "inputs": inputs, "tex_name": tex_name,
                   "identity": self.identity}
        prior = budget.reserve(key, "probe", request)
        if prior:
            if not prior["result_ref"]:
                raise AmbiguousCall("PDF operation outcome is unknown; refusing duplicate execution: " + key)
            saved = store.json(prior["result_ref"])
            if prior["status"] != "success":
                raise AgentOutputError("Prior PaperOrchestra PDF operation failed: " + saved["error"])
            for ref in saved["outputs"].values():
                store.read(ref)
        else:
            started = time.monotonic()
            try:
                timeout = min(timeout_seconds or 120, budget.remaining())
                if kind == "compile":
                    self.delegate.compile(source, output, tex_name, timeout)
                    outputs = {"paper.pdf": store.put_file(Path(output), "paperorchestra_pdf")}
                elif kind == "text":
                    text = self.delegate.text(source, timeout_seconds=timeout)
                    if not isinstance(text, str) or len(text.encode()) > 500_000:
                        raise IntegrityError("Extracted PDF text exceeds its contract")
                    outputs = {"paper.txt": store.put(text.encode(), "txt", "paperorchestra_pdf_text")}
                else:
                    paths = self.delegate.render(source, output, timeout_seconds=timeout)
                    if not paths or len(paths) > 100:
                        raise IntegrityError("PDF rendering exceeded its page contract")
                    outputs = {}
                    for path in paths:
                        path = Path(path)
                        if (path.is_symlink() or not path.is_file() or
                                path.parent.resolve() != Path(output).resolve() or
                                path.stat().st_size > 25_000_000):
                            raise IntegrityError("PDF rendering returned an invalid page")
                        outputs[path.name] = store.put_file(path, "paperorchestra_pdf_page")
                receipts = getattr(self.delegate, "receipts", [])
                worker = Path(__file__).with_name("pdf_worker.py")
                saved = {"request": request, "outputs": outputs,
                         "receipt": receipts[-1] if receipts else {"synthetic": self.synthetic},
                         "worker_ref": store.put_file(worker, "paperorchestra-pdf-worker"),
                         "worker_sha256": file_digest(worker)}
                budget.finish(key, "success", saved, time.monotonic() - started)
            except Exception as error:
                budget.finish(key, "failed", {"request": request, "error": str(error)},
                              time.monotonic() - started)
                raise
        self.receipts.append({"operation_key": key, **saved})
        if kind == "text":
            return store.read(saved["outputs"]["paper.txt"]).decode()
        if kind == "compile":
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_bytes(store.read(saved["outputs"]["paper.pdf"]))
            return None
        Path(output).mkdir(parents=True, exist_ok=True)
        restored = []
        for name, ref in saved["outputs"].items():
            if Path(name).name != name:
                raise IntegrityError("Invalid cached PDF page name")
            path = Path(output) / name
            path.write_bytes(store.read(ref))
            restored.append(path)
        return restored

    def compile(self, work_dir, pdf_path, tex_name, timeout_seconds=120):
        return self._guarded("compile", work_dir, tex_name=tex_name, output=pdf_path,
                             timeout_seconds=timeout_seconds)

    def text(self, pdf_path, *, timeout_seconds=None):
        return self._guarded("text", pdf_path, timeout_seconds=timeout_seconds)

    def render(self, pdf_path, output_dir, *, timeout_seconds=None):
        return self._guarded("render", pdf_path, output=output_dir, timeout_seconds=timeout_seconds)

    def _guarded(self, *args, **kwargs):
        try:
            return self._call(*args, **kwargs)
        except BaseException as error:
            self.error = error
            raise


@contextmanager
def _image_attachments(backend, images):
    """Attach real image bytes on the existing backend, retaining its accounting."""
    if not images or backend.synthetic:
        yield
        return
    from ...providers.backend import APIBackend
    if isinstance(backend, APIBackend):
        with backend.attach_images(images):
            yield
        return
    if not isinstance(backend, CodexBackend):
        raise AgentOutputError("Configured writer backend cannot attach PaperOrchestra page images")
    original = backend.executor

    def execute(argv, timeout, **kwargs):
        directory = Path(argv[argv.index("--cd") + 1])
        argv = list(argv)
        for index, item in enumerate(images):
            source = item["path"]
            suffix = ".png" if item["mime_type"] == "image/png" else ".jpg"
            target = directory / f"paperorchestra-page-{index}{suffix}"
            shutil.copyfile(source, target)
            if file_digest(target) != item["sha256"]:
                raise IntegrityError("Page image changed while staging its backend attachment")
            argv[-1:-1] = ["--image", str(target)]
        return original(argv, timeout, **kwargs)

    backend.executor = execute
    try:
        yield
    finally:
        backend.executor = original


class ControllerTransport:
    name = "controller-accounted-paperorchestra"

    def __init__(self, controller, key, ledger, pdf, *, previous_latex=None):
        self.controller, self.key, self.ledger, self.pdf = controller, key, ledger, pdf
        self.synthetic = controller.backend.synthetic
        self.index, self.refined, self.error = 0, False, None
        self.previous_latex = previous_latex
        self.writer_context = build_writer_context(controller.store, ledger)
        self.writing_scope = getattr(controller.config.workflow, "paperorchestra_writing_scope", "upstream")
        self.scope_instruction = writing_scope_instruction(self.writing_scope)
        self.response_policy = response_size_policy(getattr(
            controller.config.workflow, "paperorchestra_max_response_chars", None))
        self.output_schema = response_schema(self.response_policy["max_output_characters"]
                                             if self.response_policy else None)
        if self.response_policy is not None and controller.config.model_retries != 0:
            raise ValueError("Bounded PaperOrchestra responses require model_retries=0; malformed or oversized responses cannot be retried automatically")

    def __call__(self, request):
        self.index += 1
        key = f"{self.key}/upstream/{self.index:04d}"
        store = self.controller.store
        try:
            if request.model_name != self.controller.config.model:
                raise IntegrityError("PaperOrchestra requested a different controller model")
            media, images, total_bytes = [], [], 0
            if len(request.media) > 16:
                raise IntegrityError("Too many writer media attachments")
            for part in request.media:
                path = Path(part.path)
                if (path.is_symlink() or not path.is_file() or path.stat().st_size > 25_000_000
                        or file_digest(path) != part.sha256):
                    raise IntegrityError("PaperOrchestra media differs from its frozen identity")
                total_bytes += path.stat().st_size
                if total_bytes > 40_000_000:
                    raise IntegrityError("PaperOrchestra media exceeds its aggregate byte quota")
                ref = store.put_file(path, "paperorchestra_model_media")
                record = {"sha256": part.sha256, "mime_type": part.mime_type, "artifact_ref": ref}
                if part.mime_type == "application/pdf":
                    record["extracted_text"] = self.pdf.text(path, timeout_seconds=request.timeout_seconds)
                elif part.mime_type in {"image/png", "image/jpeg"}:
                    header = path.read_bytes()[:8]
                    if not (header.startswith(b"\x89PNG\r\n\x1a\n") if part.mime_type == "image/png"
                            else header.startswith(b"\xff\xd8\xff")):
                        raise IntegrityError("PaperOrchestra image MIME does not match its bytes")
                    images.append({**record, "path": store.root / ref})
                else:
                    raise IntegrityError("Unsupported PaperOrchestra model media")
                media.append(record)
            inputs = {"upstream_role": request.role, "prompt": request.prompt,
                      "response_kind": request.response_kind, "media": media,
                      "image_delivery": "synthetic_test" if self.synthetic else "actual_cli_attachments",
                      "response_instruction": "Return the COMPLETE upstream response verbatim in output, "
                      "including requested JSON/LaTeX fences. Preserve evidence and citation identities. "
                      "Never invent empirical results or executed supplementary work."}
            from ...research.controller import PROMPTS
            inventory = authoritative_evidence_inventory(self.ledger)
            inputs.update(evidence_inventory=inventory,
                          instruction=EVIDENCE_AUTHORITY_INSTRUCTION + " " + PROMPTS["paperorchestra_model"],
                          response_instruction=EVIDENCE_AUTHORITY_INSTRUCTION + " " + inputs["response_instruction"])
            inputs["method_protocol_context"] = self.writer_context
            if self.scope_instruction:
                inputs.update(writing_scope=self.writing_scope,
                    instruction=self.scope_instruction + " " + inputs["instruction"],
                    response_instruction=self.scope_instruction + " " + inputs["response_instruction"])
            context_proof = {"context_ref": store.put_json(self.writer_context, "paperorchestra-method-protocol-context"),
                "context_sha256": digest(canonical(self.writer_context)), "writing_scope": self.writing_scope,
                "scope_instruction_sha256": digest((self.scope_instruction or "").encode()),
                "upstream_prompt_sha256": digest(request.prompt.encode()), "includes_final": False}
            context_key = "paperorchestra-writer-context/" + key
            context_ref = store.put_json(context_proof, "paperorchestra-writer-context-proof")
            if store.get(context_key) not in (None, context_ref):
                raise IntegrityError("PaperOrchestra writer context changed for an existing invocation")
            store.set(context_key, context_ref)
            if self.response_policy is not None:
                instruction = bounded_response_instruction(self.response_policy)
                # ResearchController places this explicit role instruction ahead
                # of the quoted upstream prompt. The upstream source is unchanged.
                inputs.update(instruction=instruction + " " + inputs["instruction"],
                              response_size_policy=self.response_policy,
                              response_instruction=instruction + " " + inputs["response_instruction"])
                proof = {"policy": self.response_policy, "upstream_role": request.role,
                         "response_kind": request.response_kind,
                         "upstream_prompt_sha256": digest(request.prompt.encode()),
                         "output_schema_sha256": digest(canonical(self.output_schema.model_json_schema())),
                         "controller_instruction_sha256": digest(inputs["instruction"].encode()),
                         "adaptation": "Explicit bounded complete-response request; upstream sources and evidence remain unchanged"}
                proof_key = "paperorchestra-response-policy/" + key
                proof_ref = store.put_json(proof, "paperorchestra-response-size-policy")
                if store.get(proof_key) not in (None, proof_ref):
                    raise IntegrityError("PaperOrchestra response-size policy changed for an existing invocation")
                store.set(proof_key, proof_ref)
            inventory_proof = {"inventory": inventory, "upstream_role": request.role,
                "upstream_prompt_sha256": digest(request.prompt.encode()),
                "controller_instruction_sha256": digest(inputs["instruction"].encode()),
                "source_priority": "typed_ledger_and_frozen_log_over_generated_outline",
                "no_measurements_generated": True}
            inventory_key = "paperorchestra-evidence-inventory/" + key
            inventory_ref = store.put_json(inventory_proof, "paperorchestra-evidence-inventory")
            if store.get(inventory_key) not in (None, inventory_ref):
                raise IntegrityError("PaperOrchestra evidence inventory changed for an existing invocation")
            store.set(inventory_key, inventory_ref)
            if self.synthetic:
                from ...research.manuscript import _bib_key
                inputs.update(scripted_ledger=self.ledger,
                    scripted_citations={_bib_key(k): {} for k in self.ledger.get("sources", {})},
                    refinement_completed=self.refined,
                    scripted_previous_latex=self.previous_latex)
            with _image_attachments(self.controller.backend, images):
                result = self.controller.ask(key, "paperorchestra_model", inputs, self.output_schema)
            row = store.db.execute("SELECT output_ref FROM calls WHERE key=?",
                                   ("live-agent/" + key,)).fetchone()
            if row is None:
                raise IntegrityError("Controller writer response has no committed call")
            response = store.json(row[0])
            usage = response.get("usage", {})
            if not self.synthetic and any(type(usage.get(k)) is not int or usage[k] < 0
                                         for k in ("input_tokens", "output_tokens")):
                raise IntegrityError("Live writer response lacks valid observed token accounting")
            tokens = sum(usage.get(k, 0) for k in ("input_tokens", "output_tokens"))
            if request.role == "content_refinement":
                self.refined = True
            return ModelResponse(result.output, tokens, self.controller.config.model)
        except BaseException as error:
            self.error = error
            raise


def _seal_attempt(store, output):
    files = _tree(output)
    for name, item in files.items():
        item["ref"] = store.put_file(output / name, "paperorchestra_attempt_artifact")
    return store.put_json({"schema_version": "paperorchestra-controller-attempt-v1", "files": files},
                          "paperorchestra_controller_attempt")



def _run_official(controller, key, inputs, ledger, identity, pdf_tools, previous_latex, request, feedback):
    store, config = controller.store, controller.config
    workflow, image = config.workflow, config.workflow.compiler_image
    root = Path(workflow.paperorchestra_root)
    saved = store.cached("paperorchestra/runtime/" + key, request)
    if saved is not None:
        output = Path(saved["output_directory"])
        if not output.resolve().is_relative_to((store.root / "paperorchestra").resolve()):
            raise IntegrityError("Cached writer output path escaped the campaign")
        report = store.json(saved["runtime_ref"])
        attempt = store.json(report["attempt_ref"])
        for name, member in attempt["files"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise IntegrityError("Invalid cached writer artifact path")
            target = output / name
            raw = store.read(member["ref"])
            if digest(raw) != member["sha256"] or len(raw) != member["bytes"]:
                raise IntegrityError("Cached writer output differs from its sealed attempt")
            if target.exists():
                if target.is_symlink() or not target.is_file() or file_digest(target) != member["sha256"]:
                    raise IntegrityError("Preserved writer attempt has been modified")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
        return (output / "final_paper.tex", output / "final_paper.pdf", output,
                saved["runtime_ref"])
    base = store.root / "paperorchestra" / digest(key.encode())[:24]
    base.mkdir(parents=True, exist_ok=True)
    materials = base / "materials"
    if not materials.exists():
        idea = inputs.get("incumbent", {}).get("idea", {})
        export_evidence_bundle(store, materials, ledger=ledger,
            title=idea.get("title") or "Evidence-grounded research manuscript",
            hypothesis=idea.get("hypothesis") or "Report the recorded development experiments and their limits.",
            upstream_revision=UPSTREAM_REVISION)
    verify_evidence_bundle(materials, store=store)
    if json.loads((materials / "evidence_ledger.json").read_text()) != ledger:
        raise IntegrityError("Frozen writing material differs from the current ledger")
    number = 1
    while (base / f"attempt-{number:04d}").exists():
        number += 1
    output = base / f"attempt-{number:04d}"
    if pdf_tools is None:
        executor = getattr(controller.runner, "raw_executor", None) or controller.runner.executor
        pdf_tools = ContainerPdfTools(image, runtime=config.native.runtime,
            context=config.native.context, executor=executor, budget_seconds=controller.budget.remaining(), staging_root=base,
            swap_policy=workflow.paperorchestra_swap_policy)
    pdf = BudgetedPdfTools(controller, key, pdf_tools, identity)
    transport = ControllerTransport(controller, key, ledger, pdf, previous_latex=previous_latex)
    runtime_config = RuntimeConfig(model_name=config.model,
        max_reflections=workflow.paperorchestra_reflections,
        review_ensemble=workflow.paperorchestra_ensemble,
        max_model_calls=workflow.paperorchestra_max_calls,
        max_tokens=min(config.token_stop_threshold or 2_000_000, 2_000_000),
        timeout_seconds=min(controller.budget.remaining(), 14_400),
        allow_live=not controller.backend.synthetic, allow_synthetic_pdf=pdf.synthetic)
    options = {}
    if previous_latex is not None:
        initial = base / "previous-source.tex"
        if initial.exists() and initial.read_text() != previous_latex:
            raise IntegrityError("Prior manuscript source changed during editorial revision")
        initial.write_text(previous_latex)
        options = {"initial_tex": initial, "revision_feedback": canonical(feedback or {}).decode(),
                   "expected_initial_tex_sha256": digest(previous_latex.encode())}
    try:
        result = run_paperorchestra(root, materials, output, transport, pdf, runtime_config,
            expected_manifest_sha256=file_digest(materials / "manifest.json"), **options)
    except BaseException:
        if output.exists():
            ref = _seal_attempt(store, output)
            store.set("paperorchestra-interrupted/" + key + f"/{number:04d}", ref)
        raise
    attempt_ref = _seal_attempt(store, output)
    report = json.loads(result.report_path.read_text())
    report.update(controller_identity=identity, controller_pdf_receipts=pdf.receipts,
                  attempt_ref=attempt_ref)
    runtime_ref = store.put_json(report, "paperorchestra_controller_runtime")
    store.set("paperorchestra-attempt/" + key + f"/{number:04d}", runtime_ref)
    if result.status != "success":
        if transport.error is not None:
            raise transport.error
        if pdf.error is not None:
            raise pdf.error
        raise AgentOutputError("PaperOrchestra writing failed: " + str(result.error))
    latex = result.final_tex.read_text()
    if previous_latex is not None and (latex == previous_latex or
            report.get("revision", {}).get("status") != "accepted"):
        raise AgentOutputError("PaperOrchestra did not accept the requested manuscript revision; prior source retained")
    store.cache("paperorchestra/runtime/" + key, request,
                {"output_directory": str(output), "runtime_ref": runtime_ref})
    return result.final_tex, result.final_pdf, output, runtime_ref

def write_controller_manuscript(controller, key, inputs, ledger, *, previous_manuscript=None,
                                feedback=None, pdf_tools=None):
    """Execute the official writer and return its evidence-bound controller view."""
    from .manuscript_binding import (
        PaperOrchestraBindings, bind_controller_manuscript, render_bound_manuscript,
        verify_bound_manifest, binding_prompt_context,
    )
    store, config = controller.store, controller.config
    workflow = config.workflow
    root = Path(workflow.paperorchestra_root)
    verify_upstream(root)
    image = workflow.compiler_image
    if not image:
        raise ValueError("PaperOrchestra controller writer requires a pinned compiler image")
    if pdf_tools is not None and pdf_tools.synthetic and not controller.backend.synthetic:
        raise ValueError("Live writing cannot use synthetic PDF operations")
    identity = {"schema_version": "paperorchestra-controller-writer-v1",
                "upstream_revision": UPSTREAM_REVISION, "upstream_sources": SOURCE_HASHES,
                "integration_sources": {path.name: file_digest(path)
                    for path in sorted(Path(__file__).parent.glob("*.py"))},
                "image_digest": image, "pdf_worker_sha256": file_digest(Path(__file__).with_name("pdf_worker.py")),
                "controller_writer_sha256": file_digest(Path(__file__)),
                "runtime_sha256": file_digest(Path(__file__).with_name("runtime.py")),
                "binding_sha256": file_digest(Path(__file__).with_name("manuscript_binding.py")),
                "model": config.model, "backend": controller.backend.name,
                "synthetic_agents": controller.backend.synthetic,
                "synthetic_pdf": bool(pdf_tools is not None and pdf_tools.synthetic),
                "runtime": config.native.runtime, "context": config.native.context,
                "swap_policy": workflow.paperorchestra_swap_policy,
                "max_reflections": workflow.paperorchestra_reflections,
                "review_ensemble": workflow.paperorchestra_ensemble,
                "max_model_calls": workflow.paperorchestra_max_calls}
    identity["evidence_inventory"] = authoritative_evidence_inventory(ledger)
    identity["method_protocol_context_sha256"] = digest(canonical(build_writer_context(store, ledger)))
    identity["writing_scope"] = getattr(workflow, "paperorchestra_writing_scope", "upstream")
    size_policy = response_size_policy(getattr(workflow, "paperorchestra_max_response_chars", None))
    if size_policy is not None:
        identity["response_size_policy"] = size_policy
    previous_latex, previous_ref = None, None
    if previous_manuscript is not None:
        previous_ref = store.put_json(previous_manuscript, "research-manuscript")
        previous_bundle = store.get("paperorchestra-rendered/" + previous_ref)
        if not previous_bundle:
            raise IntegrityError("PaperOrchestra revision requires its exact previous authoritative source")
        previous_rendered = RenderedManuscript.model_validate(store.json(previous_bundle))
        previous_latex = store.read(previous_rendered.latex_ref).decode()
    request = {"identity": identity, "inputs": inputs, "ledger": ledger,
               "previous_manuscript_ref": previous_ref,
               "previous_latex_sha256": digest(previous_latex.encode()) if previous_latex else None,
               "feedback": feedback}
    cached = store.cached("paperorchestra/controller/" + key, request)
    if cached is not None:
        rendered = RenderedManuscript.model_validate(cached["rendered"])
        for field in ("manifest_ref", "manuscript_ref", "ledger_ref", "validation_ref",
                      "markdown_ref", "latex_ref", "bibliography_ref", "pdf_ref"):
            if ref := getattr(rendered, field):
                store.read(ref)
        for member in store.json(rendered.manifest_ref)["files"].values():
            store.read(member["ref"])
        verify_bound_manifest(store, store.json(rendered.manifest_ref), expected_image_digest=image)
        store.read(cached["evidence_ref"])
        return ControllerWriterResult(ManuscriptRecord.model_validate(cached["manuscript"]),
                                      rendered, cached["evidence_ref"])
    # Freeze before any external work. Different source or configuration cannot
    # adopt a partly completed pipeline merely because its stage key matches.
    store.cache("paperorchestra/admission/" + key, request, {"identity": identity})
    final_tex, final_pdf, output, runtime_ref = _run_official(
        controller, key, inputs, ledger, identity, pdf_tools, previous_latex, request, feedback)
    latex = final_tex.read_text()
    binding_inputs = {"latex": latex, "latex_sha256": digest(latex.encode()), "ledger": ledger,
                      "method_protocol_context": build_writer_context(store, ledger),
                      "source_binding_context": binding_prompt_context(latex, ledger),
                      "sources": inputs.get("sources", []), "synthetic_agents": controller.backend.synthetic,
                      "supplementary_coverage": inputs.get("supplementary_coverage", []),
                      "authoritative_runtime_ref": runtime_ref}
    responses = [{"concern_id": item["concern_id"],
        "status": {"executed": "executed", "partial": "limited", "unfulfilled": "unaddressed"}[item["status"]],
        "experiment_ids": item["observation_ids"],
        "explanation": "Recorded supplementary coverage; scientific resolution remains unverified."}
        for item in inputs.get("supplementary_coverage", [])]

    def bind(value):
        return bind_controller_manuscript(value, latex, ledger,
            inputs.get("sources", []), synthetic_agents=controller.backend.synthetic,
            concern_responses=responses)

    def validate(value):
        try:
            bind(value)
        except (ValueError, IntegrityError) as error:
            raise AgentOutputError(str(error)) from error

    bindings = controller.ask(key + "/bindings", "paperorchestra_bindings", binding_inputs,
                              PaperOrchestraBindings, validator=validate)
    manuscript, normalization_proof = bind(bindings)
    rendered = render_bound_manuscript(store, manuscript, bindings, ledger,
        inputs.get("sources", []), latex_path=final_tex, pdf_path=final_pdf,
        runtime_ref=runtime_ref, image_digest=image,
        project_dir=output / "content_refinement_workdir", normalization_proof=normalization_proof)
    saved = {"manuscript": manuscript.model_dump(mode="json"),
             "rendered": rendered.model_dump(mode="json"), "evidence_ref": runtime_ref}
    store.cache("paperorchestra/controller/" + key, request, saved)
    store.set("paperorchestra-rendered/" + rendered.manuscript_ref,
              store.put_json(rendered, "rendered-manuscript"))
    return ControllerWriterResult(manuscript, rendered, runtime_ref)
