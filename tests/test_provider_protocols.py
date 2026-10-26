"""Synthetic provider contracts and local-only HTTP; never contacts a model service."""

import base64
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import subprocess
import sys
import threading
from urllib.error import URLError

from pydantic import ValidationError
import pytest

from scientisttwo.backends import StructuredOutputCodec
from scientisttwo.providers.config import ModelProvider, assert_credential_free, readiness
from scientisttwo.providers.protocols import (
    build_request, normalize_response, normalize_usage, provider_schema, validate_request,
)
from scientisttwo.providers.worker import execute_request
from test_structured_map_codec import ROLE_SCHEMAS


SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
          "required": ["answer"], "additionalProperties": False}
FAKE_KEY = "synthetic-credential-for-protocol-tests"


def request(kind="openai_compatible", **provider):
    return {"version": "provider-http-v1", "model": "org/custom-model:revision",
            "provider": {"kind": kind, "max_output_tokens": 256,
                         **({"base_url": "http://127.0.0.1:12345/v1"} if kind == "openai_compatible" else {}),
                         **provider},
            "prompt": "Return a synthetic JSON artifact", "wire_schema": deepcopy(SCHEMA),
            "images": [], "timeout_seconds": 2.0}


def response(kind="openai_compatible", text='{"answer":"synthetic"}'):
    if kind == "openai":
        return {"id": "resp_synthetic", "model": "resolved-model", "status": "completed",
                "output": [{"type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": text}]}],
                "usage": {"input_tokens": 10, "output_tokens": 5,
                          "input_tokens_details": {"cached_tokens": 4},
                          "output_tokens_details": {"reasoning_tokens": 2}, "total_tokens": 15}}
    if kind == "anthropic":
        return {"id": "msg_synthetic", "model": "resolved-claude", "type": "message", "role": "assistant",
                "stop_reason": "end_turn", "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 10, "output_tokens": 5,
                          "cache_creation_input_tokens": 20, "cache_read_input_tokens": 30}}
    return {"id": "chatcmpl_synthetic", "model": "custom-model", "choices": [
        {"finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class MemoryResponse(io.BytesIO):
    def __init__(self, body, status=200):
        super().__init__(body)
        self.status = status


class MemoryOpener:
    def __init__(self, payload, status=200):
        self.payload, self.status, self.requests = payload, status, []

    def open(self, req, timeout):
        self.requests.append((req, timeout))
        body = self.payload if isinstance(self.payload, bytes) else json.dumps(self.payload).encode()
        return MemoryResponse(body, self.status)


def test_defaults_and_readiness_never_construct_a_client(monkeypatch):
    assert ModelProvider().kind == "codex"
    p = ModelProvider(kind="openai", max_output_tokens=100)
    assert p.base_url == "https://api.openai.com/v1" and p.api_key_env == "OPENAI_API_KEY"
    assert readiness(p, {})["ready"] is False
    result = readiness(p, {"OPENAI_API_KEY": FAKE_KEY})
    assert result["ready"] and FAKE_KEY not in json.dumps(result)
    assert ModelProvider(kind="openai_compatible", base_url="http://localhost:8000/v1", max_output_tokens=100).api_key_env is None
    assert readiness(ModelProvider())["live_model_calls"] == 0


def test_selected_credential_guard_allows_configuration_without_environment(monkeypatch):
    monkeypatch.delenv("SYNTHETIC_MODEL_TEST_KEY", raising=False)
    provider = ModelProvider(kind="openai_compatible", base_url="https://example.com/v1",
                             max_output_tokens=100, api_key_env="SYNTHETIC_MODEL_TEST_KEY")
    assert assert_credential_free(provider, {"model": "org/model"}, {}) is None
    assert assert_credential_free(provider, {"model": "org/model"},
                                  {"SYNTHETIC_MODEL_TEST_KEY": FAKE_KEY}) is None


@pytest.mark.parametrize("value", [
    {"model": FAKE_KEY}, {FAKE_KEY: "mapping-key"}, [{"inputs": FAKE_KEY}],
    {"nested": json.dumps({"value": FAKE_KEY})},
    {"nested": json.dumps(json.dumps({"value": FAKE_KEY}))},
    {"nested": json.dumps({"value": "".join(f"\\u{ord(char):04x}" for char in FAKE_KEY)})},
])
def test_selected_credential_guard_rejects_literal_and_escaped_inputs(value, monkeypatch):
    monkeypatch.delenv("SYNTHETIC_MODEL_TEST_KEY", raising=False)
    provider = ModelProvider(kind="openai_compatible", base_url="https://example.com/v1",
                             max_output_tokens=100, api_key_env="SYNTHETIC_MODEL_TEST_KEY")
    with pytest.raises(ValueError, match="Selected credential") as error:
        assert_credential_free(provider, value, {"SYNTHETIC_MODEL_TEST_KEY": FAKE_KEY})
    assert FAKE_KEY not in str(error.value)


@pytest.mark.parametrize("value", ["", "synthetic\ninvalid", "synthetic nonascii-\u00e9"])
def test_selected_credential_guard_rejects_present_invalid_environment(value, monkeypatch):
    monkeypatch.setenv("SYNTHETIC_MODEL_TEST_KEY", value)
    with pytest.raises(ValidationError, match="environment variable is missing or invalid") as error:
        ModelProvider(kind="openai_compatible", base_url="https://example.com/v1",
                      max_output_tokens=100, api_key_env="SYNTHETIC_MODEL_TEST_KEY")
    if value:
        assert value not in str(error.value)


def test_provider_never_persists_selected_credential_in_endpoint(monkeypatch):
    monkeypatch.setenv("SYNTHETIC_MODEL_TEST_KEY", FAKE_KEY)
    with pytest.raises(ValidationError, match="Selected credential") as error:
        ModelProvider(kind="openai_compatible", base_url="https://example.com/" + FAKE_KEY,
                      max_output_tokens=100, api_key_env="SYNTHETIC_MODEL_TEST_KEY")
    assert FAKE_KEY not in str(error.value)


def test_guard_reads_only_the_selected_credential(monkeypatch):
    monkeypatch.delenv("SYNTHETIC_MODEL_TEST_KEY", raising=False)
    provider = ModelProvider(kind="openai_compatible", base_url="https://example.com/v1",
                             max_output_tokens=100, api_key_env="SYNTHETIC_MODEL_TEST_KEY")
    assert assert_credential_free(provider, {"ordinary_text": FAKE_KEY},
                                  {"UNSELECTED_TEST_KEY": FAKE_KEY}) is None


@pytest.mark.parametrize("fields", [
    {"kind": "openai"}, {"kind": "openai", "max_output_tokens": True},
    {"kind": "openai_compatible", "max_output_tokens": 100},
    {"kind": "codex", "base_url": "https://example.com"},
    {"kind": "anthropic", "max_output_tokens": 100, "structured_output": "json_object"},
    {"kind": "anthropic", "max_output_tokens": 100, "temperature": .5},
    {"kind": "anthropic", "max_output_tokens": 100, "api_version": "2023-06-01", "temperature": 1.5},
    {"kind": "openai", "max_output_tokens": 100, "api_key": FAKE_KEY},
    {"kind": "openai", "max_output_tokens": 100, "api_key_env": "invalid-name"},
    {"kind": "openai", "max_output_tokens": 100, "api_version": "2023-06-01"},
])
def test_invalid_configs_fail_without_echoing_values(fields):
    with pytest.raises(ValidationError) as error:
        ModelProvider.model_validate(fields)
    assert FAKE_KEY not in str(error.value)


@pytest.mark.parametrize("url", [
    "https://user:password@example.com/v1", "https://example.com/v1?api_key=" + FAKE_KEY,
    "https://example.com/v1#credential", "file:///tmp/model", "http://worker.internal/v1",
    "https://example.com:0/v1", "https://example.com/v1\nheader", "https://example.com\\other",
    "http://localhost.attacker.example/v1", "https://example.com:invalid/v1",
])
def test_endpoint_validation(url):
    with pytest.raises(ValidationError) as error:
        ModelProvider(kind="openai_compatible", base_url=url, max_output_tokens=100)
    assert FAKE_KEY not in str(error.value)


def test_explicit_lan_http_and_ipv6_loopback():
    p = ModelProvider(kind="openai_compatible", base_url="http://worker.internal:8000/v1/",
                      max_output_tokens=100, allow_insecure_http=True)
    assert p.base_url == "http://worker.internal:8000/v1"
    assert ModelProvider(kind="openai_compatible", base_url="http://[::1]:8000/v1", max_output_tokens=100)


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
def test_nonofficial_origin_never_inherits_commercial_credential(kind):
    with pytest.raises(ValidationError, match="explicit api_key_env"):
        ModelProvider(kind=kind, base_url="https://proxy.example/v1", max_output_tokens=100)
    explicit = ModelProvider(kind=kind, base_url="https://proxy.example/v1", max_output_tokens=100,
                             api_key_env="PROXY_MODEL_KEY")
    assert explicit.api_key_env == "PROXY_MODEL_KEY"
    official = "api.openai.com" if kind == "openai" else "api.anthropic.com"
    assert ModelProvider(kind=kind, base_url=f"https://{official}:443/v1", max_output_tokens=100)


@pytest.mark.parametrize("kind,endpoint,token_field", [
    ("openai", "/responses", "max_output_tokens"),
    ("anthropic", "/messages", "max_tokens"),
    ("openai_compatible", "/chat/completions", "max_tokens"),
])
def test_requests_are_stateless_response_only_and_keep_configured_model(kind, endpoint, token_field):
    url, body, headers = build_request(request(kind))
    assert url.endswith(endpoint) and body["model"] == "org/custom-model:revision"
    assert body[token_field] == 256 and body["stream"] is False
    assert not {"tools", "functions", "api_key", "Authorization", "x-api-key"} & body.keys()
    assert not {"Authorization", "x-api-key"} & headers.keys()
    if kind == "openai":
        assert body["store"] is False and body["text"]["format"]["strict"] is True
    elif kind == "anthropic":
        assert headers["anthropic-version"] == "2026-10-21"
        assert body["output_config"]["format"]["schema"] == SCHEMA
    else:
        assert body["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize("kind", ["openai", "anthropic", "openai_compatible"])
def test_explicit_prompt_mode_never_silently_requests_provider_schema(kind):
    _, body, _ = build_request(request(kind, structured_output="prompt"))
    assert not {"response_format", "output_config", "text"} & body.keys()
    assert "Required JSON schema" in json.dumps(body)


@pytest.mark.parametrize("kind", ["openai", "openai_compatible"])
def test_json_object_mode_is_explicit(kind):
    _, body, _ = build_request(request(kind, structured_output="json_object"))
    assert (body["text"]["format"] if kind == "openai" else body["response_format"]) == {"type": "json_object"}


def test_anthropic_projects_only_schema_keywords_and_preserves_original():
    schema = {"type": "object", "properties": {"minimum": {"type": "integer", "minimum": 10},
        "xs": {"type": "array", "minItems": 3, "maxItems": 5, "items": {"type": "string", "pattern": "^x$"}}},
        "required": ["minimum", "xs"], "additionalProperties": False}
    original = deepcopy(schema)
    projected = provider_schema(ModelProvider(kind="anthropic", max_output_tokens=100), schema)
    assert schema == original and "minimum" in projected["properties"]
    assert "minimum" not in projected["properties"]["minimum"]
    assert '"minimum": 10' in projected["properties"]["minimum"]["description"]
    assert "minItems" not in projected["properties"]["xs"]
    assert "pattern" not in projected["properties"]["xs"]["items"]


@pytest.mark.parametrize("role,model", [(role, model) for role, models in ROLE_SCHEMAS.items() for model in models])
def test_every_current_research_schema_can_be_projected_for_anthropic(role, model):
    codec = StructuredOutputCodec(model.model_json_schema())
    original = deepcopy(codec.wire_schema)
    projected = provider_schema(ModelProvider(kind="anthropic", max_output_tokens=100), original)
    assert projected["type"] == "object" and original == codec.wire_schema


def test_anthropic_recursive_schema_fails_before_network():
    q = request("anthropic")
    q["wire_schema"] = {"type": "object", "properties": {"child": {"$ref": "#"}}, "additionalProperties": False}
    with pytest.raises(ValueError, match="recursive"):
        build_request(q)


@pytest.mark.parametrize("kind", ["openai", "anthropic", "openai_compatible"])
def test_image_formats_are_real_base64_blocks(kind):
    q = request(kind, vision=True)
    q["images"] = [{"mime_type": "image/png", "data_base64": base64.b64encode(b"\x89PNG\r\n\x1a\nsynthetic").decode()}]
    _, body, _ = build_request(q)
    text = json.dumps(body)
    assert q["images"][0]["data_base64"] in text
    assert ("input_image" if kind == "openai" else "media_type" if kind == "anthropic" else "image_url") in text
    q["provider"]["vision"] = False
    with pytest.raises(ValueError, match="vision"):
        build_request(q)


@pytest.mark.parametrize("image", [
    {"mime_type": "image/png", "data_base64": "not base64"},
    {"mime_type": "image/jpeg", "data_base64": base64.b64encode(b"not JPEG").decode()},
    {"mime_type": "application/pdf", "data_base64": ""},
])
def test_invalid_media_fails_before_dispatch(image):
    q = request(vision=True)
    q["images"] = [image]
    with pytest.raises(ValueError):
        validate_request(q)


@pytest.mark.parametrize("kind", ["openai", "anthropic", "openai_compatible"])
def test_complete_usage_is_not_double_counted(kind):
    result = normalize_response(kind, response(kind))
    assert result["status"] == "completed" and result["usage_known"]
    assert result["usage"]["input_tokens"] == (60 if kind == "anthropic" else 10)
    assert result["usage"]["output_tokens"] == 5
    if kind == "openai":
        assert result["usage"]["reasoning_tokens"] == 2
    assert result["reported_model"] is not None
    assert result["output_text"] == '{"answer":"synthetic"}'


@pytest.mark.parametrize("bad", [None, {}, {"prompt_tokens": 2},
    {"prompt_tokens": True, "completion_tokens": 3}, {"prompt_tokens": -1, "completion_tokens": 3},
    {"prompt_tokens": 1, "completion_tokens": "3"},
    {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 99}])
def test_missing_or_malformed_usage_is_unknown(bad):
    p = response()
    p["usage"] = bad
    result = normalize_response("openai_compatible", p)
    assert result["usage"] is None and result["usage_known"] is False


def test_missing_optional_cache_counters_preserves_reported_legacy_usage():
    assert normalize_usage("anthropic", {"input_tokens": 2, "output_tokens": 3}) == {"input_tokens": 2, "output_tokens": 3}
    assert normalize_usage("anthropic", {"input_tokens": 2, "output_tokens": 3, "cache_read_input_tokens": None}) is None
    assert normalize_usage("openai", {"input_tokens": 2, "output_tokens": 3,
                                      "input_tokens_details": {"cached_tokens": 4}}) is None


@pytest.mark.parametrize("kind", ["openai", "anthropic", "openai_compatible"])
def test_refusal_and_truncation_keep_known_usage(kind):
    p = response(kind)
    if kind == "openai":
        p["output"][0]["content"] = [{"type": "refusal", "refusal": "Synthetic refusal"}]
    elif kind == "anthropic":
        p["stop_details"] = {"type": "refusal", "explanation": "Synthetic refusal"}
    else:
        p["choices"][0]["message"]["refusal"] = "Synthetic refusal"
    r = normalize_response(kind, p)
    assert r["status"] == "refusal" and r["usage_known"] and r["output_text"] is None
    p = response(kind)
    if kind == "openai":
        p.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    elif kind == "anthropic":
        p["stop_reason"] = "max_tokens"
    else:
        p["choices"][0]["finish_reason"] = "length"
    r = normalize_response(kind, p)
    assert r["status"] == "truncated" and r["usage_known"]


@pytest.mark.parametrize("kind", ["openai", "anthropic", "openai_compatible"])
def test_unrequested_tools_are_rejected(kind):
    p = response(kind)
    if kind == "openai":
        p["output"].append({"type": "function_call", "arguments": "{}"})
    elif kind == "anthropic":
        p["content"].append({"type": "tool_use", "name": "shell", "input": {}})
    else:
        p["choices"][0]["message"]["tool_calls"] = [{"function": {"name": "shell"}}]
    r = normalize_response(kind, p)
    assert r["status"] == "invalid_response" and r["usage_known"] and r["output_text"] is None


def test_anthropic_server_tool_usage_is_not_accepted_as_plain_output():
    p = response("anthropic")
    p["usage"]["server_tool_use"] = {"web_search_requests": 1}
    assert normalize_response("anthropic", p)["error_code"] == "unexpected_output_activity"


@pytest.mark.parametrize("status", [301, 307, 400, 401, 403, 429, 500])
def test_http_error_body_never_becomes_public_diagnostic(status):
    op = MemoryOpener({"error": {"message": "Untrusted server text " + FAKE_KEY}}, status)
    result = execute_request(request(), opener=op, environ={})
    assert result["status"] == "http_error" and result["usage_known"] is False
    assert FAKE_KEY not in json.dumps(result) and "Untrusted" not in json.dumps(result)


def test_worker_reads_selected_key_only_and_keeps_it_out_of_body_and_result():
    q = request(api_key_env="SCIENTISTTWO_FAKE_KEY")
    op = MemoryOpener(response())
    result = execute_request(q, opener=op, environ={"SCIENTISTTWO_FAKE_KEY": FAKE_KEY, "UNRELATED_KEY": "never-used"})
    wire, timeout = op.requests[0]
    assert wire.get_header("Authorization") == "Bearer " + FAKE_KEY
    assert FAKE_KEY.encode() not in wire.data and FAKE_KEY not in json.dumps(result)
    assert result["status"] == "completed" and result["request_sent"] is True and timeout == 2
    assert result["response_sha256"]


def test_missing_key_is_known_not_sent_and_does_not_open_connection():
    op = MemoryOpener(response())
    result = execute_request(request(api_key_env="SCIENTISTTWO_FAKE_KEY"), opener=op, environ={})
    assert not op.requests and result["status"] == "not_sent" and result["request_sent"] is False
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_credential_echo_is_suppressed_even_with_json_escaping():
    raw = json.dumps(response(text=json.dumps({"answer": FAKE_KEY}))).replace("synthetic-credential", "synthetic-cred\\u0065ntial").encode()
    result = execute_request(request(api_key_env="SCIENTISTTWO_FAKE_KEY"),
        opener=MemoryOpener(raw), environ={"SCIENTISTTWO_FAKE_KEY": FAKE_KEY})
    assert result["error_code"] == "credential_echo_detected" and result["output_text"] is None
    assert result["usage_known"] and FAKE_KEY not in json.dumps(result)


def test_credential_echo_inside_encoded_output_json_is_suppressed():
    encoded_key = "".join("\\u" + format(ord(char), "04x") for char in FAKE_KEY)
    inner = '{"answer":"' + encoded_key + '"}'
    payload = response(text=inner)
    assert FAKE_KEY not in json.dumps(payload)
    result = execute_request(request(api_key_env="SCIENTISTTWO_FAKE_KEY"),
        opener=MemoryOpener(payload), environ={"SCIENTISTTWO_FAKE_KEY": FAKE_KEY})
    assert result["error_code"] == "credential_echo_detected" and result["output_text"] is None
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_credential_in_body_is_rejected_before_request():
    q = request(api_key_env="SCIENTISTTWO_FAKE_KEY")
    q["prompt"] = FAKE_KEY
    op = MemoryOpener(response())
    r = execute_request(q, opener=op, environ={"SCIENTISTTWO_FAKE_KEY": FAKE_KEY})
    assert r["status"] == "not_sent" and r["error_code"] == "credential_in_request_body" and not op.requests


def test_transport_failure_remains_unknown_and_message_is_not_echoed():
    class Failing:
        def open(self, req, timeout):
            raise URLError(FAKE_KEY)
    r = execute_request(request(), opener=Failing(), environ={})
    assert r["status"] == "transport_error" and r["usage"] is None and r["request_sent"]
    assert FAKE_KEY not in json.dumps(r)


@pytest.mark.parametrize("raw", [b"not JSON", b'{"usage":{},"usage":{}}', b'NaN'])
def test_bad_response_json_never_becomes_zero_usage(raw):
    r = execute_request(request(), opener=MemoryOpener(raw), environ={})
    assert r["status"] == "invalid_response" and r["usage"] is None


def test_response_size_limit_blocks_unbounded_body(monkeypatch):
    import scientisttwo.providers.worker as worker
    monkeypatch.setattr(worker, "RESPONSE_LIMIT", 100)
    r = execute_request(request(), opener=MemoryOpener(b"x" * 101), environ={})
    assert r["error_code"] == "response_read_failed" and r["usage_known"] is False


@contextmanager
def local_server(*, redirect=False):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append((self.path, self.headers.get("Authorization"), body))
            if redirect:
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/redirect-target")
                self.end_headers()
                return
            data = json.dumps(response()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args):
            pass
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("Loopback socket access unavailable in this execution sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_worker_process_uses_only_local_fake_http_and_ignores_proxy():
    with local_server() as (server, calls):
        q = request(base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key_env="SCIENTISTTWO_FAKE_KEY")
        env = {"PATH": os.environ.get("PATH", ""), "SCIENTISTTWO_FAKE_KEY": FAKE_KEY,
               "HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1"}
        p = subprocess.run([sys.executable, "-I", "-m", "scientisttwo.providers.worker"],
            input=json.dumps(q), capture_output=True, text=True, env=env, timeout=10)
    assert p.returncode == 0 and not p.stderr
    result = json.loads(p.stdout)
    assert result["status"] == "completed" and len(calls) == 1
    assert calls[0][0] == "/v1/chat/completions" and calls[0][1] == "Bearer " + FAKE_KEY
    assert FAKE_KEY not in p.stdout


def test_redirect_never_sends_a_second_request():
    with local_server(redirect=True) as (server, calls):
        q = request(base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key_env="SCIENTISTTWO_FAKE_KEY")
        result = execute_request(q, environ={"SCIENTISTTWO_FAKE_KEY": FAKE_KEY})
    assert len(calls) == 1 and result["error_code"] == "http_redirect_rejected"
    assert result["usage"] is None and FAKE_KEY not in json.dumps(result)


def test_worker_bad_input_exits_without_echoing_input_or_secret():
    value = json.dumps({"api_key": FAKE_KEY})
    p = subprocess.run([sys.executable, "-I", "-m", "scientisttwo.providers.worker"],
        input=value, capture_output=True, text=True, timeout=10)
    assert p.returncode == 0 and not p.stderr and FAKE_KEY not in p.stdout
    assert json.loads(p.stdout)["request_sent"] is False
