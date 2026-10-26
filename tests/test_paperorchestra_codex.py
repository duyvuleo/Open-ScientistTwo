from pathlib import Path
from types import SimpleNamespace

import pytest

from scientisttwo.backends import BackendError
from scientisttwo.integrations.paperorchestra import codex_transport as transport_module
from scientisttwo.integrations.paperorchestra.codex_transport import CodexTransport
from scientisttwo.models import AgentResult
from scientisttwo.process import ProcessResult
from scientisttwo.store import file_digest


def request(media=(), model="gpt-6-astra"):
    return SimpleNamespace(role="outline", prompt="Make an outline from supplied facts",
                           model_name=model, response_kind="json", media=media,
                           timeout_seconds=90)


def test_live_disabled_before_backend_construction(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled transport constructed backend")
    monkeypatch.setattr(transport_module, "CodexBackend", forbidden)
    with pytest.raises(BackendError, match="disabled"):
        CodexTransport(model="gpt-6-astra")(request())


def test_codex_images_are_real_attachments_and_usage_is_counted(tmp_path, monkeypatch):
    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fixture-image")
    media = SimpleNamespace(path=image, mime_type="image/png", sha256=file_digest(image))
    invocations = []

    def executor(argv, timeout, **kwargs):
        copied = Path(argv[argv.index("--image") + 1])
        assert copied != image and copied.read_bytes() == image.read_bytes()
        assert argv[-1] == "-"
        invocations.append(argv)
        return ProcessResult(0, "", "", 0)

    class FakeBackend:
        def __init__(self, _, model, **kwargs):
            self.execute = kwargs["executor"]
            assert 0 < kwargs["timeout"] <= 90 and kwargs["allow_live"]

        def invoke(self, role, inputs, schema):
            assert role == "paperorchestra/outline"
            assert inputs["attached_image_count"] == 1
            self.execute(["codex", "exec", "--cd", str(tmp_path), "-"], 90)
            return AgentResult(payload={"output": '{"title":"fixture"}'}, backend="codex",
                               model="gpt-6-astra", synthetic=False,
                               usage={"input_tokens": 100, "cached_input_tokens": 50,
                                      "output_tokens": 20})

    monkeypatch.setattr(transport_module, "CodexBackend", FakeBackend)
    result = CodexTransport(model="gpt-6-astra", allow_live=True, executor=executor)(request((media,)))
    assert result.observed_tokens == 120
    assert len(invocations) == 1


def test_modified_media_and_model_mismatch_rejected(tmp_path):
    path = tmp_path / "page.png"
    path.write_bytes(b"modified")
    media = SimpleNamespace(path=path, mime_type="image/png", sha256="0" * 64)
    transport = CodexTransport(model="gpt-6-astra", allow_live=True)
    with pytest.raises(ValueError, match="changed"):
        transport(request((media,)))
    with pytest.raises(ValueError, match="differs"):
        transport(request(model="other-model"))


def test_pdf_without_isolated_extractor_fails(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-fixture")
    media = SimpleNamespace(path=path, mime_type="application/pdf", sha256=file_digest(path))
    with pytest.raises(ValueError, match="extractor"):
        CodexTransport(model="gpt-6-astra", allow_live=True)(request((media,)))
