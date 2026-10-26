"""Run pinned official PaperOrchestra agents with explicit, bounded service adapters.

Only reviewed upstream Python is executed here. Generated LaTeX is passed to an
isolated PDF service; the runtime never invokes a shell, compiler, model SDK, or
network client. Frozen literature replaces the upstream retrieval stage and is
always disclosed. This is an adapted integration, not the unmodified upstream CLI.
"""

from __future__ import annotations

import builtins
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from types import ModuleType, SimpleNamespace
from typing import Protocol


UPSTREAM_REVISION = "ca1b3fa01c2970fc7cda32d16245db38d57b3f56"
UPSTREAM_URL = "https://github.com/google-research/paper-orchestra"
# Complete executed upstream module closure and copied template closure. A new
# revision requires a deliberate review and change to this lock.
SOURCE_HASHES = {
    "LICENSE": "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
    "utils/prompt_utils.py": "72b19f8adda3b870bc797c30e1260331ce5e429c346e39b05a0ad932b858df9e",
    "autoraters/agent_review.py": "7d1cf9326d68a5fe5a2744d1c284f3333e9781db99600ed3606cf01c7e00062b",
    "methods/agents/content_refinement_agent.py": "c6dc3e5d36d85adaae931b8036327756c630cccb8f1dddc9d9282d1eb4c1a038",
    "methods/agents/outline_agent.py": "bc1724362938e5ed2ff96f509c0d0a6169c55461b0ab860bb9cd52a4a0d7e169",
    "methods/agents/section_writing_agent.py": "747e150ee450a7ad6a9d50b6f4a31e20e643609d619e1183de7d28e83a6f015e",
    "methods/prompts/content_refinement_agent.py": "539c0584d8e85505a17495f9a646c1551201e5a1c95264464516b1fcc5ff6a59",
    "methods/prompts/outline_agent.py": "fb9d318dc432208cff41ee1fc340b920ca392e9db4da5dbc1d78a7493ebb2c5b",
    "methods/prompts/section_writing_agent.py": "e222611a6f57708fe8a1b8fbee73ddcc5318eac774cd46f03478118345d3a383",
    "templates/iclr2025/fancyhdr.sty": "b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea",
    "templates/iclr2025/guidelines.md": "853f9f8b9a8b57d5e0eca44c1abc77dba352440d4d4aee240431421eb3ae4eeb",
    "templates/iclr2025/iclr2025_conference.bst": "2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5",
    "templates/iclr2025/iclr2025_conference.sty": "9e280a9ff99be84ad190f5cafe047a1a32c2eeb9ca14232ddceae42d8489c697",
    "templates/iclr2025/math_commands.tex": "90473c4d0542070db244cea73ef962d6cddc5b2a746757e6a40ddf5fdfb90ba9",
    "templates/iclr2025/natbib.sty": "88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d",
    "templates/iclr2025/references.bib": "cdd86e7d4c31854dcf2145871657c944588a6d44c3b72e160ff4baa8df1a52fb",
    "templates/iclr2025/template.tex": "bb1ea888e8791267ecca2b05867d302d121924b3f088cbd23f4dd98663df5692",
}


def compatible_template(source: bytes):
    """Add the package required by the pinned section prompt to a staged copy.

    Return bytes and provenance; never write the upstream checkout. Comments are
    masked only while locating active preamble commands, preserving every source
    byte except the single package insertion. Reapplying is a no-op.
    """
    text = source.decode("utf-8")
    active = re.sub(r"(?<!\\)%[^\r\n]*", lambda match: " " * len(match.group()), text)
    starts = list(re.finditer(r"\\begin\s*\{\s*document\s*\}", active))
    if len(starts) != 1:
        raise ValueError("Compatibility template requires exactly one active begin-document marker")
    preamble = active[:starts[0].start()]
    packages = re.findall(r"\\(?:usepackage|RequirePackage)\s*(?:\[[^\]]*\]\s*)?\{([^}]*)\}", preamble)
    existing = sum(package.strip() == "booktabs" for group in packages for package in group.split(","))
    if existing > 1:
        raise ValueError("Compatibility template already declares booktabs more than once")
    addition = "\\usepackage{booktabs}" + ("\r\n" if "\r\n" in text else "\n")
    staged = (text[:starts[0].start()] + addition + text[starts[0].start():]).encode() if not existing else source
    proof = {"schema_version": "paperorchestra-template-compatibility-v1",
             "source_sha256": _sha(source), "staged_sha256": _sha(staged),
             "added_packages": ["booktabs"] if not existing else [],
             "change": "single_booktabs_package_insertion" if not existing else "already_compatible",
             "reason": "Pinned section prompt requires booktabs table rules while preserving the supplied preamble",
             "section_prompt_sha256": SOURCE_HASHES["methods/prompts/section_writing_agent.py"],
             "upstream_checkout_modified": False}
    return staged, proof


@dataclass(frozen=True)
class RuntimeConfig:
    model_name: str = "scripted"
    max_reflections: int = 1
    review_ensemble: int = 3
    max_model_calls: int = 24
    max_tokens: int = 200_000
    timeout_seconds: float = 600
    allow_live: bool = False
    allow_synthetic_pdf: bool = False

    def __post_init__(self):
        if (not self.model_name or not 0 <= self.max_reflections <= 3
                or not 1 <= self.review_ensemble <= 3
                or not 1 <= self.max_model_calls <= 120
                or not 1 <= self.max_tokens <= 2_000_000
                or not 1 <= self.timeout_seconds <= 14_400):
            raise ValueError("Invalid bounded PaperOrchestra runtime configuration")


@dataclass(frozen=True)
class MediaPart:
    path: Path
    mime_type: str
    sha256: str


@dataclass(frozen=True)
class ModelRequest:
    role: str
    prompt: str
    model_name: str
    response_kind: str
    media: tuple[MediaPart, ...] = ()
    timeout_seconds: float = 120


@dataclass(frozen=True)
class ModelResponse:
    raw_text: str
    observed_tokens: int = 0
    model_name: str = "scripted"


class ModelTransport(Protocol):
    synthetic: bool
    name: str

    def __call__(self, request: ModelRequest) -> ModelResponse: ...


class PdfTools(Protocol):
    synthetic: bool
    isolation: str

    def compile(self, work_dir: Path, pdf_path: Path, tex_name: str,
                timeout_seconds: float) -> None: ...

    def text(self, pdf_path: Path, *, timeout_seconds: float | None = None) -> str: ...

    def render(self, pdf_path: Path, output_dir: Path,
               *, timeout_seconds: float | None = None) -> list[Path]: ...


@dataclass(frozen=True)
class RuntimeResult:
    status: str
    final_tex: Path | None
    final_pdf: Path | None
    report_path: Path
    error: str | None = None


class ScriptedTransport:
    """Explicit test/replay responses; never computes or fabricates experiments."""

    name = "scripted-paperorchestra"
    synthetic = True

    def __init__(self, responses: list[ModelResponse | str]):
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    def __call__(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise ValueError("Scripted PaperOrchestra response trace exhausted")
        result = self.responses.pop(0)
        return result if isinstance(result, ModelResponse) else ModelResponse(result)


class _Abort(BaseException):
    """Cross upstream broad Exception handlers without permitting silent fallback."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def bundled_upstream_root() -> Path:
    """Return the packaged source closure; accessing it performs no execution."""
    return Path(__file__).resolve().parents[2] / "_vendor" / "paperorchestra"


def resolve_upstream_root(root: Path | str | None = None) -> Path:
    """Resolve and verify bundled source or an explicitly supplied override."""
    selected = Path(root).expanduser().resolve() if root is not None else bundled_upstream_root()
    verify_upstream(selected)
    return selected


def verify_upstream(root: Path | None = None) -> dict[str, str]:
    """Verify bytes against a lock; HEAD labels alone do not establish identity."""
    root = Path(root).resolve() if root is not None else bundled_upstream_root()
    for name, expected in SOURCE_HASHES.items():
        path = root / name
        if (path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(root)
                or _sha(path.read_bytes()) != expected):
            raise ValueError(f"Pinned PaperOrchestra source mismatch: {name}")
    return dict(SOURCE_HASHES)


def _snapshot_materials(source: Path, target: Path) -> dict[str, str]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("PaperOrchestra materials must be a regular directory")
    files, size = {}, 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Symlinks are forbidden in PaperOrchestra materials")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Non-regular PaperOrchestra material")
        size += path.stat().st_size
        if size > 32_000_000 or len(files) >= 256:
            raise ValueError("PaperOrchestra materials exceed bounded input contract")
        name = path.relative_to(source).as_posix()
        data = path.read_bytes()
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        files[name] = _sha(data)
    required = {"idea_sparse.md", "experimental_log.md", "citation_map.json", "references.bib"}
    if not required <= files.keys():
        raise ValueError("PaperOrchestra materials missing: " + ", ".join(sorted(required - files.keys())))
    citations = json.loads((target / "citation_map.json").read_text())
    if not isinstance(citations, dict) or any(
        not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_:.+-]+", key)
        or not isinstance(item, dict) or not isinstance(item.get("title"), str)
        or not isinstance(item.get("authors", []), list)
        or any(not isinstance(author, str) for author in item.get("authors", []))
        for key, item in citations.items()
    ):
        raise ValueError("Frozen citation map does not match upstream bibkey metadata contract")
    figures = target / "figures" / "info.json"
    if not figures.exists():
        figures.parent.mkdir(parents=True, exist_ok=True)
        _write_json(figures, [])
    return files


def _parse_json(text: str):
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    value = json.loads(match[1] if match else text,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(value, (dict, list)):
        raise ValueError("Expected a JSON object or array")
    return value


def _parse_latex(text: str):
    match = re.search(r"```latex\s*(.*?)```", text, re.DOTALL)
    result = (match[1] if match else text).strip()
    if not result or r"\begin{document}" not in result or r"\end{document}" not in result:
        raise ValueError("Expected a complete LaTeX document")
    return result


def _parse_refinement(text: str):
    if "```json" not in text or "```latex" not in text:
        raise ValueError("Content refinement requires JSON worklog and LaTeX blocks")
    return json.dumps({"json": _parse_json(text), "latex": _parse_latex(text)}, allow_nan=False)


@dataclass
class _Part:
    text: str | None = None
    data: bytes | None = None
    mime_type: str | None = None

    @classmethod
    def from_text(cls, *, text):
        return cls(text=text)

    @classmethod
    def from_bytes(cls, *, data, mime_type):
        return cls(data=data, mime_type=mime_type)


@dataclass
class _Runtime:
    root: Path
    output: Path
    transport: ModelTransport
    pdf: PdfTools
    config: RuntimeConfig
    started: float = field(default_factory=time.monotonic)
    phase: str = "outline"
    observed_tokens: int = 0
    calls: list[dict] = field(default_factory=list)
    pdf_operations: list[dict] = field(default_factory=list)
    modules: dict[str, ModuleType] = field(default_factory=dict)
    revision_feedback: str | None = None

    def remaining(self):
        remaining = self.config.timeout_seconds - (time.monotonic() - self.started)
        if remaining <= 0:
            raise _Abort("PaperOrchestra wall-clock budget exhausted")
        return remaining

    def owned_path(self, value):
        path = Path(value)
        if path.is_symlink() or not path.resolve().is_relative_to(self.output.resolve()):
            raise _Abort("PaperOrchestra attempted to access a path outside its owned workspace")
        return path

    def media(self, path, mime_type):
        path = self.owned_path(path)
        if not path.is_file() or path.stat().st_size > 25_000_000:
            raise _Abort("Missing or oversized PaperOrchestra media")
        return MediaPart(path, mime_type, _sha(path.read_bytes()))

    def call(self, *, prompt="", contents=(), images=(), model_name=None,
             generation_configs=None, result_parsing_func=None, **kwargs):
        self.remaining()
        if len(self.calls) >= self.config.max_model_calls:
            raise _Abort("PaperOrchestra model-call budget exhausted")
        if self.observed_tokens >= self.config.max_tokens:
            raise _Abort("PaperOrchestra observed-token budget exhausted")
        system = (generation_configs or {}).get("system_instruction", "")
        texts, media = [system, prompt], []
        for part in contents:
            if part.text is not None:
                texts.append(part.text)
            elif part.data is not None:
                if part.mime_type != "application/pdf" or len(part.data) > 25_000_000:
                    raise _Abort("Unsupported or oversized upstream binary context")
                path = self.output / "media" / (_sha(part.data) + ".pdf")
                path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    path.write_bytes(part.data)
                media.append(self.media(path, part.mime_type))
        for path in images:
            mime = "image/jpeg" if Path(path).suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            media.append(self.media(path, mime))
        parser = result_parsing_func or _parse_json
        kind = "latex" if parser is _parse_latex else (
            "json_latex" if parser is _parse_refinement else "json")
        if getattr(parser, "__name__", "") == "<lambda>":
            kind = "latex"
        role = self.phase
        if "Analyze the provided page images" in prompt:
            role = "format_review"
        elif self.phase == "content_refinement" and kind == "latex":
            role = "format_revision"
        if self.revision_feedback and role in {"content_refinement", "peer_review"}:
            texts.append(
                "--- FROZEN SCIENTISTTWO MANUSCRIPT REVISION REQUEST ---\n"
                + self.revision_feedback
                + "\n--- END MANUSCRIPT REVISION REQUEST ---\n"
                "Address these requests using the supplied evidence only. This is a manuscript "
                "revision: do not invent experiments, change the method, or alter recorded results. "
                "Explicitly identify requests that the evidence cannot support in the worklog."
            )
        request = ModelRequest(role, "\n\n".join(text for text in texts if text),
                               model_name or self.config.model_name, kind, tuple(media),
                               self.remaining())
        if len(request.prompt.encode()) > 4_000_000:
            raise _Abort("PaperOrchestra prompt exceeds the bounded input limit")
        record = {"index": len(self.calls) + 1, "role": role, "status": "started",
                  "model_name": request.model_name, "response_kind": kind,
                  "synthetic": self.transport.synthetic, "prompt": request.prompt,
                  "media": [{"path": str(x.path.relative_to(self.output)),
                             "mime_type": x.mime_type, "sha256": x.sha256} for x in media]}
        self.calls.append(record)
        log = self.output / "model_calls" / f"{record['index']:03d}.json"
        log.parent.mkdir(exist_ok=True)
        _write_json(log, record)
        try:
            response = self.transport(request)
            if (not isinstance(response, ModelResponse) or not isinstance(response.raw_text, str)
                    or not isinstance(response.observed_tokens, int) or response.observed_tokens < 0
                    or len(response.raw_text.encode()) > 4_000_000):
                raise ValueError("Invalid model response contract")
            self.observed_tokens += response.observed_tokens
            record.update(observed_tokens=response.observed_tokens,
                          response_model_name=response.model_name, raw_response=response.raw_text)
            self.remaining()
            parsed = parser(response.raw_text)
            if parsed is None:
                raise ValueError("Upstream model output did not parse")
            record["status"] = "completed"
            _write_json(log, record)
            if self.observed_tokens > self.config.max_tokens:
                raise _Abort("PaperOrchestra observed-token threshold crossed; no further calls allowed")
            return {"raw_response": response.raw_text, "parsed_response": parsed}
        except _Abort as error:
            record.update(status="stopped", error=str(error))
            _write_json(log, record)
            raise
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            _write_json(log, record)
            raise _Abort("PaperOrchestra model call failed: " + record["error"]) from error

    def compile(self, work_dir, output_pdf, tex_name):
        work, target = self.owned_path(work_dir), self.owned_path(output_pdf)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", tex_name):
            raise _Abort("Unsafe LaTeX entrypoint")
        if not (work / (tex_name + ".tex")).is_file():
            raise _Abort("LaTeX entrypoint is missing")
        source_tex_sha256 = _sha((work / (tex_name + ".tex")).read_bytes())
        try:
            self.pdf.compile(work, target, tex_name, self.remaining())
            if (not target.is_file() or target.stat().st_size > 25_000_000
                    or target.read_bytes()[:5] != b"%PDF-"):
                raise ValueError("Isolated PDF service did not produce a bounded PDF")
            if _sha((work / (tex_name + ".tex")).read_bytes()) != source_tex_sha256:
                raise ValueError("PDF compilation changed its authoritative LaTeX source")
            self.pdf_operations.append({"operation": "compile", "tex_name": tex_name,
                                        "source_tex_sha256": source_tex_sha256,
                                        "pdf": str(target.relative_to(self.output)),
                                        "sha256": _sha(target.read_bytes())})
            self.remaining()
        except Exception as error:
            raise _Abort(f"Isolated PaperOrchestra compilation failed: {error}") from error

    def text(self, pdf_path, **kwargs):
        try:
            self.remaining()
            result = self.pdf.text(self.owned_path(pdf_path), timeout_seconds=self.remaining())
            if not isinstance(result, str) or not result.strip() or len(result.encode()) > 4_000_000:
                raise ValueError("Invalid extracted PDF text")
            self.remaining()
            return result
        except Exception as error:
            raise _Abort(f"Isolated PDF extraction failed: {error}") from error

    def render(self, pdf_path, output_dir, **kwargs):
        try:
            self.remaining()
            paths = self.pdf.render(self.owned_path(pdf_path), self.owned_path(output_dir),
                                    timeout_seconds=self.remaining())
            if not isinstance(paths, list) or not 1 <= len(paths) <= 40:
                raise ValueError("PDF renderer must return one to forty page images")
            result = [str(self.media(path, "image/png").path) for path in paths]
            self.remaining()
            return result
        except Exception as error:
            raise _Abort(f"Isolated PDF rendering failed: {error}") from error

    def load(self, name):
        if name in self.modules:
            return self.modules[name]
        relative = name.replace(".", "/") + ".py"
        if relative not in SOURCE_HASHES:
            raise _Abort("Unreviewed upstream import: " + name)
        source = (self.root / relative).read_bytes()
        if _sha(source) != SOURCE_HASHES[relative]:
            raise _Abort("Upstream source changed during execution: " + relative)
        module = ModuleType(name)
        module.__file__ = str(self.root / relative)
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": self.import_module}
        self.modules[name] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        return module

    def import_module(self, name, globals=None, locals=None, fromlist=(), level=0):
        if name in self.modules:
            return self.modules[name]
        if name.replace(".", "/") + ".py" in SOURCE_HASHES:
            return self.load(name)
        if name not in {"os", "traceback", "json", "shutil", "typing", "numpy"}:
            raise _Abort("Unreviewed upstream dependency: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    def setup_modules(self):
        def module(name, **attributes):
            value = ModuleType(name)
            value.__dict__.update(attributes)
            self.modules[name] = value
            return value
        module("google.genai", types=SimpleNamespace(Part=_Part))
        module("utils.common_utils", load_md_file=lambda path: self.owned_path(path).read_text())
        module("utils.gemini_utils", call_gemini_with_contents=self.call,
               call_gemini_with_images=self.call, call_gemini_with_text_prompt=self.call,
               parse_gemini_latex_results=_parse_latex,
               parse_gemini_json_latex_response=_parse_refinement)
        module("utils.llm_backend_utils", call_llm_with_images=self.call)
        module("utils.pdf_utils", compile_latex=self.compile, load_paper=self.text,
               pdf_to_grid_images=self.render)
        review = self.load("autoraters.agent_review")
        original = review.perform_review_agentreview

        def bounded_review(*args, **kwargs):
            before, self.phase = self.phase, "peer_review"
            try:
                kwargs["num_review_ensemble"] = self.config.review_ensemble
                return original(*args, **kwargs)
            finally:
                self.phase = before
        review.perform_review_agentreview = bounded_review


def run_paperorchestra(upstream_root: Path | None, materials_dir: Path, output_dir: Path,
                      transport: ModelTransport, pdf_tools: PdfTools,
                      config: RuntimeConfig | None = None, *,
                      expected_manifest_sha256: str | None = None,
                      initial_tex: Path | None = None,
                      revision_feedback: str | None = None,
                      expected_initial_tex_sha256: str | None = None) -> RuntimeResult:
    """Execute official outline, writing, review/refinement and formatting agents.

    The output directory must not exist: unknown/partial attempts cannot be
    silently rerun. Every attempt retains its own report and model-call ledger.
    PDF test doubles require a separate explicit flag and remain labeled synthetic.
    Editorial revisions start from the exact supplied TeX and use the official
    content-refinement agent directly. Durable controller transport/PDF adapters
    can replay committed calls into a fresh attempt without repeating external work.
    """
    config = config or RuntimeConfig()
    if (expected_manifest_sha256 is not None
            and not re.fullmatch(r"[a-f0-9]{64}", expected_manifest_sha256)):
        raise ValueError("Expected evidence manifest identity must be a SHA256 digest")
    initial_bytes = None
    if initial_tex is None:
        if revision_feedback is not None or expected_initial_tex_sha256 is not None:
            raise ValueError("Manuscript revision requires an authoritative initial_tex")
    else:
        if (not isinstance(revision_feedback, str) or not revision_feedback.strip()
                or len(revision_feedback.encode()) > 1_000_000):
            raise ValueError("Manuscript revision requires bounded, nonempty revision_feedback")
        if config.max_reflections == 0:
            raise ValueError("Manuscript revision requires at least one content reflection")
        if (expected_initial_tex_sha256 is not None
                and not re.fullmatch(r"[a-f0-9]{64}", expected_initial_tex_sha256)):
            raise ValueError("Expected initial TeX identity must be a SHA256 digest")
        initial = Path(initial_tex)
        if initial.is_symlink() or not initial.is_file() or initial.stat().st_size > 4_000_000:
            raise ValueError("Initial manuscript must be a bounded regular TeX file")
        initial_bytes = initial.read_bytes()
        _parse_latex(initial_bytes.decode())
        if (expected_initial_tex_sha256 is not None
                and _sha(initial_bytes) != expected_initial_tex_sha256):
            raise ValueError("Initial manuscript differs from the authorized TeX identity")
    root = Path(upstream_root).resolve() if upstream_root is not None else bundled_upstream_root()
    output = Path(output_dir).absolute()
    if not isinstance(getattr(transport, "synthetic", None), bool):
        raise ValueError("Transport must declare whether responses are synthetic")
    if not transport.synthetic and not config.allow_live:
        raise ValueError("Live PaperOrchestra calls disabled; no model request was made")
    if pdf_tools.synthetic and not config.allow_synthetic_pdf:
        raise ValueError("Synthetic PDF tools are forbidden outside explicit tests")
    if not pdf_tools.synthetic and pdf_tools.isolation != "container":
        raise ValueError("PaperOrchestra requires isolated container PDF tools")
    if output.exists() or output.is_symlink():
        raise ValueError("PaperOrchestra output already exists; preserve it and select a new attempt")
    output = output.resolve()
    if output.resolve().is_relative_to(root):
        raise ValueError("PaperOrchestra output cannot modify its upstream checkout")
    if output.is_relative_to(Path(materials_dir).resolve()):
        raise ValueError("PaperOrchestra output cannot be inside its input materials")
    verify_upstream(root)
    output.mkdir(parents=True)
    runtime = _Runtime(root, output, transport, pdf_tools, config,
                       revision_feedback=revision_feedback)
    report_path = output / "runtime-report.json"
    report = {"schema_version": "paperorchestra-runtime-v1", "status": "running",
              "upstream": {"url": UPSTREAM_URL, "revision": UPSTREAM_REVISION,
                           "files": dict(SOURCE_HASHES)},
              "runtime_sha256": _sha(Path(__file__).read_bytes()),
              "config": asdict(config), "transport": transport.name,
              "synthetic_agents": transport.synthetic, "synthetic_pdf": pdf_tools.synthetic,
              "adaptations": ["Frozen evidence-backed literature replaces HybridLiteratureAgent retrieval.",
                              "Model calls use an explicit bounded transport instead of upstream SDKs.",
                              "PDF compilation, extraction and rendering use an isolated adapter.",
                              "Official agent methods run once; upstream whole-paper retry loop is bypassed.",
                              "Strict JSON/LaTeX parsing rejects malformed responses without hidden retries.",
                              f"Content reflections={config.max_reflections}; review ensemble={config.review_ensemble}."],
              "stages": [], "error": None}
    if initial_bytes is not None:
        (output / "initial_manuscript.tex").write_bytes(initial_bytes)
        (output / "revision_feedback.txt").write_text(revision_feedback)
        report["revision"] = {
            "status": "pending", "initial_tex": "initial_manuscript.tex",
            "initial_tex_sha256": _sha(initial_bytes),
            "feedback": "revision_feedback.txt",
            "feedback_sha256": _sha(revision_feedback.encode()),
            "mode": "official_content_refinement_from_authoritative_parent",
        }
        report["adaptations"].append(
            "Editorial requests start from the exact parent LaTeX, bypass outline/section "
            "generation, and include frozen controller feedback in refinement/review prompts.")
    _write_json(report_path, report)
    final_tex = final_pdf = None
    try:
        materials = output / "raw_materials"
        report["input_files"] = _snapshot_materials(Path(materials_dir), materials)
        if expected_manifest_sha256 is not None:
            from .evidence import verify_evidence_bundle
            manifest_file = materials / "manifest.json"
            if (not manifest_file.is_file()
                    or _sha(manifest_file.read_bytes()) != expected_manifest_sha256):
                raise _Abort("Frozen evidence manifest differs from the authorized input identity")
            manifest = verify_evidence_bundle(materials)
            if manifest["upstream_revision"] != UPSTREAM_REVISION:
                raise _Abort("Evidence bundle targets a different PaperOrchestra revision")
            report["verified_evidence_manifest_sha256"] = expected_manifest_sha256
        writeup = output / "latex_writeup"
        writeup.mkdir()
        notices = output / "upstream-notices"
        notices.mkdir()
        shutil.copyfile(root / "LICENSE", notices / "LICENSE")
        (notices / "README.txt").write_text(
            "PaperOrchestra source and template attribution: Google Research.\n"
            + UPSTREAM_URL + "\nRevision: " + UPSTREAM_REVISION + "\n"
            "The upstream checkout is unchanged. ScientistTwo supplies the service adapters; "
            "runtime-report.json records the deviations from the upstream CLI.\n")
        for name in SOURCE_HASHES:
            if name.startswith("templates/iclr2025/"):
                shutil.copyfile(root / name, writeup / Path(name).name)
        staged_template, template_proof = compatible_template((writeup / "template.tex").read_bytes())
        (writeup / "template.tex").write_bytes(staged_template)
        template_proof.update(source_path="templates/iclr2025/template.tex",
                              staged_path="latex_writeup/template.tex")
        _write_json(output / "template-compatibility.json", template_proof)
        report["template_compatibility"] = template_proof
        report["adaptations"].append(
            "The staged template includes one booktabs package declaration to satisfy the pinned section prompt; upstream template bytes remain unchanged.")
        shutil.copyfile(materials / "references.bib", writeup / "references.bib")
        shutil.copytree(materials / "figures", writeup / "figures")
        runtime.setup_modules()
        outline = output / "outline.json"
        if initial_bytes is None:
            runtime.load("methods.agents.outline_agent").OutlineAgent(
                model_name=config.model_name).run(
                    idea_file=str(materials / "idea_sparse.md"),
                    experimental_log_file=str(materials / "experimental_log.md"),
                    latex_template_file=str(writeup / "template.tex"),
                    conference_guidelines_file=str(writeup / "guidelines.md"),
                    output_filepath=str(outline))
            report["stages"].append("official_outline_agent")
        literature = output / "literature_agent_output"
        literature.mkdir()
        for name in ["citation_map.json", "references.bib"]:
            shutil.copyfile(materials / name, literature / name)
        if initial_bytes is None:
            shutil.copyfile(outline, literature / "outline_v1.json")
        shutil.copyfile(writeup / "template.tex", literature / "updated_template.tex")
        report["stages"].append("adapted_frozen_literature")
        raw_tex = writeup / "raw_draft_paper.tex"
        if initial_bytes is None:
            runtime.phase = "section_writing"
            runtime.load("methods.agents.section_writing_agent").SectionWritingAgent(
                outline_path=str(outline), template_path=str(writeup / "template.tex"),
                idea_path=str(materials / "idea_sparse.md"),
                experimental_log_path=str(materials / "experimental_log.md"),
                citation_map_path=str(literature / "citation_map.json"),
                figures_info_path=str(writeup / "figures" / "info.json"),
                guidelines_path=str(writeup / "guidelines.md"), model_name=config.model_name,
            ).run(output_path=str(raw_tex))
            report["stages"].append("official_section_writing_agent")
        else:
            raw_tex.write_bytes(initial_bytes)
            report["stages"].append("authoritative_parent_manuscript")
        if not raw_tex.is_file():
            raise _Abort("Official section writer did not produce a draft")
        refinement = output / "content_refinement_workdir"
        shutil.copytree(writeup, refinement)
        runtime.phase = "content_refinement"
        candidate = runtime.load("methods.agents.content_refinement_agent").ContentRefinementAgent(
            experimental_log_path=str(materials / "experimental_log.md"),
            citation_map_path=str(literature / "citation_map.json"),
            guidelines_path=str(writeup / "guidelines.md"), model_name=config.model_name,
            max_reflections=config.max_reflections, work_dir=str(refinement),
        ).run(texfile_path=str(raw_tex))
        candidate = runtime.owned_path(candidate)
        if not candidate.is_file() or candidate.read_bytes()[:5] != b"%PDF-":
            raise _Abort("Official content refinement did not produce its final PDF")
        report["stages"].extend(["official_content_refinement_agent", "official_formatting_loop"])
        verify_upstream(root)
        for name, expected in report["input_files"].items():
            if _sha((materials / name).read_bytes()) != expected:
                raise _Abort("Frozen PaperOrchestra inputs changed during writing")
        final_tex, final_pdf = output / "final_paper.tex", output / "final_paper.pdf"
        shutil.copyfile(refinement / "final_refined_paper.tex", final_tex)
        shutil.copyfile(candidate, final_pdf)
        tex_sha, pdf_sha = _sha(final_tex.read_bytes()), _sha(final_pdf.read_bytes())
        compiled = [(index, operation) for index, operation in enumerate(runtime.pdf_operations)
                    if operation["operation"] == "compile"
                    and operation["source_tex_sha256"] == tex_sha
                    and operation["sha256"] == pdf_sha]
        if not compiled:
            raise _Abort("Final PDF has no compilation bound to its authoritative LaTeX source")
        operation_index, operation = compiled[-1]
        report["authoritative_compile"] = {
            "source_tex_sha256": tex_sha, "pdf_sha256": pdf_sha,
            "operation_index": operation_index, "pdf": operation["pdf"],
        }
        if initial_bytes is not None:
            if ((output / "initial_manuscript.tex").read_bytes() != initial_bytes
                    or (output / "revision_feedback.txt").read_text() != revision_feedback):
                raise _Abort("Frozen parent manuscript or revision feedback changed during writing")
            worklog = refinement / "content_refinement_worklog.json"
            decisions = json.loads(worklog.read_text()) if worklog.is_file() else {}
            accepted = any(item.get("outcome", "").startswith("ACCEPTED_")
                           for item in decisions.values())
            changed = final_tex.read_bytes() != initial_bytes
            report["revision"].update(
                status="accepted" if accepted and changed else (
                    "unchanged" if accepted else "rejected"),
                source_changed=changed, decisions=decisions,
            )
        runtime.remaining()
        report["status"] = "success"
        report["outputs"] = {p.name: _sha(p.read_bytes()) for p in [final_tex, final_pdf]}
    except (_Abort, Exception) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if "revision" in report:
            report["revision"]["status"] = "failed"
        final_tex = final_pdf = None
    finally:
        report.update(model_calls=len(runtime.calls), observed_tokens=runtime.observed_tokens,
                      elapsed_seconds=time.monotonic() - runtime.started,
                      pdf_operations=runtime.pdf_operations,
                      completed_calls=sum(item["status"] == "completed" for item in runtime.calls))
        _write_json(report_path, report)
    return RuntimeResult(report["status"], final_tex, final_pdf, report_path, report["error"])
