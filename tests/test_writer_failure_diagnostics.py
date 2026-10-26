"""Synthetic failure traces; no model calls or credential-bearing diagnostics."""
import json

import pytest

from scientisttwo.backends import failure_diagnostic
from scientisttwo.process import ProcessResult
from scientisttwo.studies import media_relay as relay
import test_study_media_relay as relay_tests
from test_study_media_relay import policy, request

fake_cli = relay_tests.fake_cli


def test_output_length_marker_is_an_error_hint_not_scientific_prose():
    secret = "sensitive-fixture-do-not-store"
    error = json.dumps({"type": "error", "message": "invalid value for max_output_tokens " + secret})
    parsed = json.loads(failure_diagnostic(error, ""))
    assert parsed["diagnostic_categories"] == ["config", "output_length"]
    assert secret not in json.dumps(parsed)
    prose = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": error}})
    assert json.loads(failure_diagnostic(prose, ""))["diagnostic_categories"] == ["unknown"]


@pytest.mark.parametrize("text,status,code", [
    ('{"output":"partial-private-fixture', 'invalid_json', 'unterminated_string'),
    ('{"output":"private-fixture"}', 'valid_json', None),
    ('{"x":1,"x":2}', 'invalid_json', 'nonfinite_or_ambiguous_json'),
    ('{"x":NaN}', 'invalid_json', 'nonfinite_or_ambiguous_json'),
])
def test_structured_syntax_diagnostic_is_bounded_and_contains_no_text(text, status, code):
    event = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": text}})
    result = relay.structured_message_diagnostic("\n".join([event] * 7))
    assert result["count"] == 7 and len(result["records"]) == 4 and result["omitted"] == 3
    assert result["records"][0]["status"] == status
    assert result["records"][0].get("error_code") == code
    assert "private-fixture" not in json.dumps(result)


def test_six_event_writer_failure_stays_unknown_and_cannot_retry(tmp_path, fake_cli):
    cli, counter = fake_cli
    secret = "sensitive-fixture-do-not-store"
    events = [
        {"type": "thread.started"}, {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": '{"output":"' + secret}},
        {"type": "item.completed", "item": {"type": "error", "message": "invalid value " + secret}},
        {"type": "error", "message": "max_output_tokens " + secret},
        {"type": "turn.failed", "error": {"message": "max_output_tokens " + secret}},
    ]
    p = policy()
    invoked = []

    def execute(*args, **kwargs):
        invoked.append(True)
        return ProcessResult(1, "\n".join(json.dumps(event) for event in events), "", .1)

    service = relay.LocalService(tmp_path / "ledger", p, executable=cli, executor=execute, synthetic=True)
    original = request(p)
    response = service.serve(original)
    receipt = relay.read_json(service.claims / original["request_id"] / "dispatch.json", 16384)
    assert receipt["usage_status"] == "unknown" and receipt["observed_tokens"] is None
    assert receipt["cli_diagnostics"]["jsonl"]["parsed_count"] == 6
    assert receipt["cli_diagnostics"]["agent_messages"]["records"][0]["error_code"] == "unterminated_string"
    assert json.loads(response["process"]["stderr"])["diagnostic_categories"] == ["config", "output_length"]
    assert secret not in json.dumps(receipt) + json.dumps(response)
    assert service.serve(original) == response
    with pytest.raises(relay.UnknownDispatch):
        service.serve(request(p, request_id="b" * 32))
    assert len(invoked) == 1 and not counter.exists()
