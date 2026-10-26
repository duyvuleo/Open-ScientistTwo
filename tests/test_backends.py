import hashlib
import json
import sys
import time

import pytest

from scientisttwo.backends import BackendError, CodexBackend
from scientisttwo.models import CriticDecision
from scientisttwo.process import ProcessResult, clean_environment, run_process


def test_codex_disabled_before_any_process(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("A live process must not be launched")

    backend = CodexBackend(tmp_path / "not-created", "explicit-model", executor=forbidden)
    with pytest.raises(BackendError, match="disabled"):
        backend.invoke("critic", {}, CriticDecision)
    assert not backend.workspace.exists()


def fake_cli(
    tmp_path, response, event='{"type":"turn.completed","usage":{"input_tokens":12}}', code=0
):
    path = tmp_path / "fake-codex"
    path.write_text(f"""#!{sys.executable}
import sys, pathlib
argv = sys.argv
assert '--json' in argv and '--ephemeral' in argv
assert argv[argv.index('--sandbox') + 1] == 'read-only'
sys.stdin.read()
pathlib.Path(argv[argv.index('--output-last-message') + 1]).write_text({json.dumps(response)!r})
print({event!r})
sys.exit({code})
""")
    path.chmod(0o755)
    return str(path)


def test_fake_codex_cli_structured_output(tmp_path):
    cli = fake_cli(tmp_path, {"decision": "Good", "reason": "Fixture"})
    backend = CodexBackend(tmp_path / "work", "fake-model", executable=cli, allow_live=True)
    result = backend.invoke("critic", {}, CriticDecision)
    assert result.payload["decision"] == "Good" and result.usage["input_tokens"] == 12


@pytest.mark.parametrize(
    "response,event,code",
    [
        ({"decision": "maybe"}, '{"type":"turn.completed"}', 0),
        ({"decision": "Good", "reason": "fixture"}, "not-json", 0),
        ({"decision": "Good", "reason": "fixture"}, '{"type":"turn.failed"}', 0),
        ({"decision": "Good", "reason": "fixture"}, '{"type":"turn.completed"}', 1),
    ],
)
def test_invalid_codex_outputs_rejected(tmp_path, response, event, code):
    cli = fake_cli(tmp_path, response, event, code)
    backend = CodexBackend(tmp_path / "work", "fake-model", executable=cli, allow_live=True)
    with pytest.raises(BackendError):
        backend.invoke("critic", {}, CriticDecision)


def test_process_timeout_cancels_descendants(tmp_path):
    marker = tmp_path / "child-survived"
    script = tmp_path / "parent.py"
    child = f"import time; from pathlib import Path; time.sleep(0.8); Path({str(marker)!r}).write_text('bad')"
    script.write_text(
        f"import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c',{child!r}])\ntime.sleep(10)\n"
    )
    result = run_process([sys.executable, str(script)], timeout=0.2)
    assert result.timed_out
    time.sleep(0.9)
    assert not marker.exists()


def test_environment_does_not_forward_arbitrary_credentials(monkeypatch):
    monkeypatch.setenv("SCIENTISTTWO_TEST_SENTINEL", "dummy-test-value")
    assert "SCIENTISTTWO_TEST_SENTINEL" not in clean_environment()


@pytest.mark.parametrize(
    "event",
    [
        '{"type":"thread.started"}',
        '{"type":"turn.completed"}\n{"type":"turn.completed"}',
        '{"type":"item.completed","item":{"type":"command_execution"}}\n{"type":"turn.completed"}',
    ],
)
def test_codex_rejects_incomplete_duplicate_and_tool_turns(tmp_path, event):
    cli = fake_cli(tmp_path, {"decision": "Good", "reason": "fixture"}, event)
    backend = CodexBackend(tmp_path / "work", "fake-model", executable=cli, allow_live=True)
    with pytest.raises(BackendError):
        backend.invoke("critic", {}, CriticDecision)


@pytest.mark.parametrize(
    "message,category",
    [
        ("Authentication failed", "auth"),
        ("HTTP 403 Forbidden", "auth"),
        ("The model is not supported", "model_unavailable"),
        ("Model fixture-model is not supported with this account", "model_unavailable"),
        ("The requested model fixture-model does not exist", "model_unavailable"),
        ("You do not have access to this model", "model_unavailable"),
        ("Invalid JSON schema for response", "schema"),
        ("Usage limit reached", "rate_or_usage_limit"),
        ("Error sending request", "network"),
        ("Unexpected argument --example", "config"),
        ("Sandbox initialization failed", "tools_or_sandbox"),
        ("Unclassified service failure", "unknown"),
    ],
)
def test_codex_exit_diagnostics_never_expose_raw_output(tmp_path, message, category):
    secret = "TEST_ONLY_SECRET_NEVER_RETAIN"
    stdout = json.dumps({"type": "error", "message": secret, "code": secret})
    stderr = f"{message}: Authorization: Bearer {secret}; /private/user/context.txt"

    def executor(*args, **kwargs):
        return ProcessResult(1, stdout, stderr, 0)

    backend = CodexBackend(tmp_path, "fake-model", allow_live=True, executor=executor)
    with pytest.raises(BackendError) as caught:
        backend.invoke("critic", {}, CriticDecision)
    error = str(caught.value)
    diagnostic = json.loads(error.split("; response discarded; ", 1)[1])
    assert diagnostic == {
        "diagnostic_categories": [category],
        "event_error_codes": [],
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
    }
    assert secret not in error and message not in error and "/private/user" not in error
    assert backend.events == []


@pytest.mark.parametrize("exit_code,event_type", [(1, "error"), (0, "turn.failed")])
def test_codex_failure_event_codes_are_allowlisted(tmp_path, exit_code, event_type):
    secret = "Bearer fixture-credential-do-not-retain"
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": secret}),
            json.dumps(
                {
                    "type": event_type,
                    "error": {
                        "code": "invalid_json_schema",
                        "message": "Authentication failed " + secret,
                        "details": {"code": secret},
                    },
                }
            ),
        ]
    )

    def executor(*args, **kwargs):
        return ProcessResult(exit_code, stdout, secret, 0)

    backend = CodexBackend(tmp_path, "fake-model", allow_live=True, executor=executor)
    with pytest.raises(BackendError) as caught:
        backend.invoke("critic", {}, CriticDecision)
    error = str(caught.value)
    diagnostic = json.loads(error[error.index("{") :])
    assert diagnostic["diagnostic_categories"] == ["schema"]
    assert diagnostic["event_error_codes"] == ["invalid_json_schema"]
    assert secret not in error and "Authentication failed" not in error
    assert backend.events == []


def test_codex_accepts_informational_todo_list(tmp_path):
    event = "\n".join(
        [
            json.dumps({"type": "item.completed", "item": {"type": "todo_list", "items": []}}),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 3}}),
        ]
    )
    cli = fake_cli(tmp_path, {"decision": "Good", "reason": "Fixture"}, event)
    backend = CodexBackend(tmp_path, "fake-model", executable=cli, allow_live=True)
    assert backend.invoke("critic", {}, CriticDecision).usage == {"input_tokens": 3}


@pytest.mark.parametrize(
    "item_type", ["command_execution", "mcp_tool_call", "web_search", "file_change"]
)
def test_codex_still_rejects_execution_items_without_contents(tmp_path, item_type):
    secret = "sk-fixture-tool-argument"
    event = json.dumps({"type": "item.completed", "item": {"type": item_type, "args": secret}})
    cli = fake_cli(tmp_path, {"decision": "Good", "reason": "Fixture"}, event)
    backend = CodexBackend(tmp_path, "fake-model", executable=cli, allow_live=True)
    with pytest.raises(BackendError) as caught:
        backend.invoke("critic", {}, CriticDecision)
    assert "item_type=" + item_type in str(caught.value)
    assert secret not in str(caught.value)
    assert backend.events == []


@pytest.mark.parametrize(
    "item_type", ["Bearer fixture-secret", "sk_fixture_secret", "x" * 65, {"token": "secret"}]
)
def test_codex_hashes_unsafe_unknown_item_types(tmp_path, item_type):
    event = json.dumps({"type": "item.completed", "item": {"type": item_type}})
    cli = fake_cli(tmp_path, {"decision": "Good", "reason": "Fixture"}, event)
    backend = CodexBackend(tmp_path, "fake-model", executable=cli, allow_live=True)
    with pytest.raises(BackendError) as caught:
        backend.invoke("critic", {}, CriticDecision)
    assert "item_type_sha256=" in str(caught.value)
    assert str(item_type) not in str(caught.value)
    assert backend.events == []


@pytest.mark.parametrize("completed,valid_output", [(True, True), (False, True), (True, False)])
def test_codex_nonfatal_error_item_is_sanitized_and_still_requires_completion(
    tmp_path, completed, valid_output
):
    secret = "TEST_ONLY_WARNING_NEVER_RETAIN"
    event = json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "error", "message": "Network is unreachable " + secret},
        }
    )
    if completed:
        event += "\n" + json.dumps({"type": "turn.completed"})
    response = (
        {"decision": "Good", "reason": "Fixture"} if valid_output else {"decision": "unknown"}
    )
    cli = fake_cli(tmp_path, response, event)
    backend = CodexBackend(tmp_path, "fake-model", executable=cli, allow_live=True)
    if completed and valid_output:
        assert backend.invoke("critic", {}, CriticDecision).payload["decision"] == "Good"
    else:
        with pytest.raises(BackendError):
            backend.invoke("critic", {}, CriticDecision)
    assert secret not in json.dumps(backend.events)
    diagnostic = backend.events[0]["item"]["diagnostic"]
    assert diagnostic["diagnostic_categories"] == ["network"]


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
@pytest.mark.parametrize("ending", ["\n", "\r\n"])
def test_backend_accepts_literal_unicode_json_strings_with_lf_or_crlf(tmp_path, separator, ending):
    text = "before" + separator + "after"
    events = ending.join([
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": text}}, ensure_ascii=False),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 1}}),
    ])
    cli = fake_cli(tmp_path, {"decision": "Good", "reason": text}, events)
    backend = CodexBackend(tmp_path, "fake-model", executable=cli, allow_live=True)
    assert backend.invoke("critic", {}, CriticDecision).payload["reason"] == text


@pytest.mark.parametrize("event", [
    '\n{"type":"turn.completed"}',
    '{"type":"turn.completed"}\n',  # fake CLI supplies one more LF: reject the blank record.
    '{"type":"turn.completed"}\r{"type":"thread.started"}',
    '{"type":[]}',
    '{"type":"turn.started","type":"turn.completed"}',
])
def test_backend_keeps_physical_blank_malformed_and_ambiguous_events_invalid(tmp_path, event):
    cli = fake_cli(tmp_path, {"decision": "Good", "reason": "fixture"}, event)
    with pytest.raises(BackendError):
        CodexBackend(tmp_path, "fake-model", executable=cli, allow_live=True).invoke("critic", {}, CriticDecision)


def test_diagnostic_categories_ignore_scientific_prose_but_keep_actual_errors():
    from scientisttwo.backends import failure_diagnostic
    prose = "invalid value; unsupported model; authentication failed"
    for item_type in ("agent_message", "reasoning"):
        stdout = json.dumps({"type": "item.completed", "item": {"type": item_type, "text": prose}})
        assert json.loads(failure_diagnostic(stdout, ""))["diagnostic_categories"] == ["unknown"]
    error = json.dumps({"type": "item.completed", "item": {"type": "error", "message": "invalid value"}})
    assert json.loads(failure_diagnostic(error, ""))["diagnostic_categories"] == ["config"]
    assert json.loads(failure_diagnostic("", "invalid value"))["diagnostic_categories"] == ["config"]
