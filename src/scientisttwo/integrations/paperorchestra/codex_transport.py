"""Response-only Codex transport for official PaperOrchestra agent prompts.

PDF inputs are explicitly converted to text by the isolated PDF adapter. Images
are actual CLI attachments, never path strings masquerading as visual evidence.
"""

from pathlib import Path
import shutil
import time

from pydantic import Field

from ...backends import BackendError, CodexBackend
from ...models import Record
from ...process import run_process
from ...store import file_digest


class _RawResponse(Record):
    output: str = Field(min_length=1, max_length=250_000)


class CodexTransport:
    name = "codex-paperorchestra"
    synthetic = False

    def __init__(self, *, model, allow_live=False, pdf_tools=None, executable="codex",
                 call_timeout=180, executor=run_process):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("PaperOrchestra requires an explicit Codex model")
        if not 1 <= call_timeout <= 600:
            raise ValueError("Codex call timeout must be between 1 and 600 seconds")
        self.model, self.allow_live = model, allow_live
        self.pdf_tools, self.executable = pdf_tools, executable
        self.call_timeout, self.executor = call_timeout, executor
        self.last_usage = {}

    def __call__(self, request):
        from .runtime import ModelResponse

        if not self.allow_live:
            raise BackendError("Live PaperOrchestra Codex calls are disabled")
        if request.model_name != self.model:
            raise ValueError("PaperOrchestra request model differs from configured transport")
        deadline = time.monotonic() + min(self.call_timeout, request.timeout_seconds)

        def remaining():
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise BackendError("PaperOrchestra model request deadline exhausted")
            return seconds
        images, pdf_text, total = [], [], 0
        if len(request.media) > 16:
            raise ValueError("Too many PaperOrchestra media attachments")
        for item in request.media:
            path = Path(item.path)
            if path.is_symlink() or not path.is_file():
                raise ValueError("PaperOrchestra attachment must be a regular file")
            size = path.stat().st_size
            total += size
            if not 0 < size <= 20_000_000 or total > 40_000_000:
                raise ValueError("PaperOrchestra attachment quota exceeded")
            if file_digest(path) != item.sha256:
                raise ValueError("PaperOrchestra attachment changed after admission")
            if item.mime_type == "application/pdf":
                if self.pdf_tools is None or self.pdf_tools.synthetic:
                    raise ValueError("Live PDF inputs require an isolated real text extractor")
                extracted = self.pdf_tools.text(path, timeout_seconds=remaining())
                if not extracted.strip() or len(extracted.encode()) > 500_000:
                    raise ValueError("PDF text is empty or oversized")
                pdf_text.append({"sha256": item.sha256, "extracted_text": extracted})
            elif item.mime_type in {"image/png", "image/jpeg"}:
                prefix = path.read_bytes()[:8]
                if not (prefix.startswith(b"\x89PNG\r\n\x1a\n") if item.mime_type == "image/png"
                        else prefix.startswith(b"\xff\xd8\xff")):
                    raise ValueError("Media MIME does not match its file header")
                images.append(item)
            else:
                raise ValueError("Unsupported PaperOrchestra media type")

        def execute(argv, timeout, **kwargs):
            # CodexBackend creates a fresh empty invocation directory. Attach
            # bounded verified copies so the CLI cannot race a mutable source.
            directory = Path(argv[argv.index("--cd") + 1])
            arguments = list(argv)
            for index, item in enumerate(images):
                suffix = ".png" if item.mime_type == "image/png" else ".jpg"
                target = directory / f"paper-page-{index}{suffix}"
                shutil.copyfile(item.path, target)
                if file_digest(target) != item.sha256:
                    raise ValueError("Image changed during attachment staging")
                arguments[-1:-1] = ["--image", str(target)]
            return self.executor(arguments, timeout, **kwargs)

        backend = CodexBackend(
            Path.cwd(), self.model, executable=self.executable, allow_live=True,
            timeout=remaining(), executor=execute,
        )
        response = backend.invoke("paperorchestra/" + request.role, {
            "task": request.prompt,
            "response_instruction": "Execute the supplied PaperOrchestra writing/review task. "
                "Put the complete requested response verbatim in output; preserve JSON/LaTeX "
                "fences when requested. Do not summarize it. Use only supplied experimental "
                "evidence and literature; never invent data, citations or executed work.",
            "requested_response_kind": request.response_kind,
            "pdf_text_adaptation": pdf_text,
            "attached_image_count": len(images),
        }, _RawResponse)
        self.last_usage = response.usage
        tokens = 0
        for key in ("input_tokens", "output_tokens"):
            value = response.usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BackendError("Codex usage missing or invalid; cannot enforce writer budget")
            tokens += value
        return ModelResponse(raw_text=response.payload["output"], observed_tokens=tokens,
                             model_name=response.model)
