"""Real local HTTP/worker subprocess; responses are synthetic, never LLM inference."""
from contextlib import closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from types import SimpleNamespace
import time

import pytest

from scientisttwo.models import Record
from scientisttwo.live.adapters import implementation_identity
from scientisttwo.providers.backend import APIBackend, APIBackendError
from scientisttwo.providers.config import ModelProvider
from scientisttwo.store import Store


class Answer(Record):
    answer: str


@contextmanager
def fake_server(delay=0):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
            if delay:
                time.sleep(delay)
            data = json.dumps({"id": "synthetic-completion", "model": "resolved-test-model",
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant",
                    "content": '{"answer":"fake HTTP response"}'}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("Loopback socket binding is unavailable in this execution sandbox")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def configuration(url, *, seconds=5, auth=None):
    return SimpleNamespace(provider=ModelProvider(kind="openai_compatible", base_url=url,
        api_key_env=auth, max_output_tokens=128), model="test/custom-model", max_calls=3,
        token_stop_threshold=100, call_seconds=seconds, model_retries=0,
        engine_version="research-v2", implementation=implementation_identity())


def test_real_worker_roundtrip_with_fake_provider_and_environment_only_auth(tmp_path, monkeypatch):
    secret = "fixture" + "-local-http-credential"
    monkeypatch.setenv("SCIENTISTTWO_HTTP_TEST_KEY", secret)
    with fake_server() as (url, requests), closing(Store(tmp_path)) as store:
        store.set("live_deadline", time.time() + 30)
        backend = APIBackend(store, configuration(url, auth="SCIENTISTTWO_HTTP_TEST_KEY"), allow_live=True)
        result = backend.invoke("critic", {"source": "explicitly synthetic test"}, Answer)
        assert result.payload == {"answer": "fake HTTP response"}
        assert result.usage["input_tokens"] == 7 and result.usage["output_tokens"] == 4
        assert len(requests) == 1 and requests[0]["path"] == "/v1/chat/completions"
        assert requests[0]["auth"] == "Bearer " + secret
        assert requests[0]["body"]["model"] == "test/custom-model"
        assert backend.accounting()["usage_complete"]
        assert backend.accounting()["known_observed_tokens"] == 11
        assert all(secret.encode() not in path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())


def test_killed_http_worker_retains_unknown_usage_and_does_not_retry(tmp_path):
    with fake_server(delay=4) as (url, requests), closing(Store(tmp_path)) as store:
        store.set("live_deadline", time.time() + 30)
        backend = APIBackend(store, configuration(url, seconds=1), allow_live=True)
        started = time.monotonic()
        with pytest.raises(APIBackendError) as error:
            backend.invoke("critic", {}, Answer)
        assert time.monotonic() - started < 3
        assert not error.value.usage_known and not backend.accounting()["usage_complete"]
        count = len(requests)
        with pytest.raises(APIBackendError, match="prior_usage_or_completion_unknown"):
            backend.invoke("critic", {}, Answer)
        assert len(requests) == count and backend.accounting()["claimed_calls"] == 1
