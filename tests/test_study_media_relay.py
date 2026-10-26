"""Operational relay tests use an executable fake CLI; no model calls."""
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from scientisttwo.models import Record
import pytest

from scientisttwo.studies import media_relay as relay


class Answer(Record):
    answer: str


def policy(**kwargs):
    value = dict(run_id="synthetic-study-arm", expires_at=time.time() + 120,
                 model="gpt-6-astra", allowed_models=("gpt-6-astra",), source_sha256="1" * 64,
                 relay_sha256=relay.sha(Path(relay.__file__).read_bytes()))
    value.update(kwargs)
    return relay.RelayPolicy(**value)


def request(p, **kwargs):
    now = time.time()
    value = dict(schema_version=2, request_id="a" * 32, policy_sha256=p.identity,
                 run_id=p.run_id, model=p.model, role="critic", created_at=now,
                 deadline=min(now + p.call_seconds, p.expires_at), timeout=p.call_seconds,
                 prompt=json.dumps({"role": "critic", "prompt_version": "scientisttwo-v1",
                                    "instructions": relay.INSTRUCTIONS, "inputs": {"evidence": "real"}}),
                 source_sha256=p.source_sha256, media=[],
                 schema={"type": "object", "properties": {"answer": {"type": "string"}},
                         "required": ["answer"], "additionalProperties": False})
    value.update(kwargs)
    return value


@pytest.fixture
def fake_cli(tmp_path):
    cli = tmp_path / "fake-codex"
    counter = tmp_path / "invocations.jsonl"
    cli.write_text("#!" + sys.executable + "\n" + "\n".join([
        "import json,sys,pathlib,hashlib",
        f"counter=pathlib.Path({str(counter)!r})",
        "with counter.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')",
        "assert sys.argv[1] == 'exec'",
        "assert sys.argv[sys.argv.index('--model')+1] == 'gpt-6-astra'",
        "assert sys.argv[sys.argv.index('--sandbox')+1] == 'read-only'",
        "prompt=json.load(sys.stdin)",
        "schema=json.loads(pathlib.Path(sys.argv[sys.argv.index('--output-schema')+1]).read_text())",
        "assert schema['type']=='object'",
        "images=[pathlib.Path(sys.argv[i+1]) for i,arg in enumerate(sys.argv[:-1]) if arg=='--image']",
        "assert all(path.parent==pathlib.Path(sys.argv[sys.argv.index('--cd')+1]) for path in images)",
        "assert all(path.read_bytes().startswith((b'\\x89PNG\\r\\n\\x1a\\n',b'\\xff\\xd8\\xff')) for path in images)",
        "pathlib.Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps({'answer':'actual fake CLI'}))",
        "print(json.dumps({'type':'thread.started','thread_id':'fixture-only'}))",
        "print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'actual fake CLI'}}))",
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'cached_input_tokens':3,'output_tokens':2}}))",
    ]) + "\n")
    cli.chmod(0o700)
    return cli, counter


def test_real_backend_roundtrip_rebuilds_local_paths_and_parses_actual_events(tmp_path, fake_cli):
    from scientisttwo.backends import CodexBackend
    cli, counter = fake_cli
    p = policy()
    queue = tmp_path / "remote-private-queue"
    executor = relay.QueueExecutor(queue, p, poll_seconds=0.01)
    service = relay.LocalService(tmp_path / "local-private-ledger", p, executable=cli, synthetic=True)
    result, errors = [], []

    def remote():
        try:
            result.append(CodexBackend(tmp_path / "agent", p.model, allow_live=True,
                                       timeout=30, executor=executor).invoke("critic", {"evidence": "real"}, Answer))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=remote)
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline and not list(queue.glob("*.request.json")):
        time.sleep(0.01)
    queued = next(queue.glob("*.request.json"))
    original = relay.read_json(queued, relay.REQUEST_LIMIT)
    response = service.serve(original)
    relay.publish_new(queue / (original["request_id"] + ".response.json"), response)
    thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    assert result[0].model == "gpt-6-astra" and result[0].synthetic is False
    assert result[0].payload == {"answer": "actual fake CLI"}
    assert result[0].usage == {"input_tokens": 10, "cached_input_tokens": 3, "output_tokens": 2}
    argv = json.loads(counter.read_text().splitlines()[0])
    assert "scientisttwo-local-codex-" in argv[argv.index("--cd") + 1]
    assert "--strict-config" in argv and "--ignore-user-config" in argv
    assert len(list(service.claims.glob("*/claim.json"))) == 1


def test_completed_replay_is_exact_and_does_not_call_fake_cli_twice(tmp_path, fake_cli):
    cli, counter = fake_cli
    p = policy()
    service = relay.LocalService(tmp_path / "state", p, executable=cli, synthetic=True)
    item = request(p)
    first = service.serve(item)
    assert service.serve(item) == first
    assert len(counter.read_text().splitlines()) == 1
    receipt = relay.read_json(service.claims / item["request_id"] / "dispatch.json", 16384)
    assert receipt["observed_tokens"] == 12
    with pytest.raises(relay.RelayError, match="different bytes"):
        service.serve(dict(item, created_at=item["created_at"] - 1))


def test_unknown_dispatch_never_retries_or_allows_later_paid_call(tmp_path, fake_cli):
    cli, _ = fake_cli
    p = policy()
    calls = []

    def crash(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("simulated client died after possible dispatch")

    service = relay.LocalService(tmp_path / "state", p, executable=cli, executor=crash, synthetic=True)
    item = request(p)
    with pytest.raises(RuntimeError, match="simulated"):
        service.serve(item)
    with pytest.raises(relay.UnknownDispatch):
        service.serve(item)
    with pytest.raises(relay.UnknownDispatch):
        service.serve(request(p, request_id="b" * 32))
    assert len(calls) == 1


@pytest.mark.parametrize("limits", [{"max_calls": 1}, {"token_stop_threshold": 12}])
def test_failed_attempts_count_and_observed_token_threshold_stops_next_call(tmp_path, fake_cli, limits):
    cli, counter = fake_cli
    p = policy(**limits)
    service = relay.LocalService(tmp_path / "state", p, executable=cli, synthetic=True)
    service.serve(request(p))
    with pytest.raises(relay.RelayError, match="threshold"):
        service.serve(request(p, request_id="b" * 32))
    assert len(counter.read_text().splitlines()) == 1


@pytest.mark.parametrize("change", [
    {"model": "gpt-6.1-sol"}, {"run_id": "different"}, {"policy_sha256": "0" * 64},
    {"timeout": 601}, {"deadline": float("inf")}, {"request_id": "../escape"},
    {"source_sha256": "2" * 64},
    {"schema": {"$ref": "https://attacker.invalid/schema"}},
    {"role": "shell"},
])
def test_rejects_unbound_or_unbounded_request_before_any_dispatch(tmp_path, fake_cli, change):
    cli, counter = fake_cli
    p = policy()
    service = relay.LocalService(tmp_path / "state", p, executable=cli, synthetic=True)
    with pytest.raises((relay.RelayError, ValueError)):
        service.serve(request(p, **change))
    assert not counter.exists() and not list(service.claims.iterdir())


def test_remote_executor_rejects_command_injection_and_any_capability_flag_change(tmp_path):
    p = policy()
    executor = relay.QueueExecutor(tmp_path / "queue", p)
    directory = tmp_path / "remote-codex-temp"
    directory.mkdir(mode=0o700)
    argv = relay.cli_argv("codex", directory, p.model)
    argv += ["--dangerously-bypass-approvals-and-sandbox"]
    with pytest.raises(relay.RelayError, match="argv differs"):
        executor(argv, 10, stdin="{}")
    assert not list((tmp_path / "queue").iterdir())


def test_queue_deadline_marks_cancellation_and_no_success(tmp_path):
    from scientisttwo.backends import BackendError, CodexBackend
    p = policy()
    executor = relay.QueueExecutor(tmp_path / "queue", p, poll_seconds=0.01)
    with pytest.raises(BackendError, match="timed out"):
        CodexBackend(tmp_path / "agent", p.model, allow_live=True, timeout=0.05,
                     executor=executor).invoke("critic", {}, Answer)
    assert len(list((tmp_path / "queue").glob("*.cancel.json"))) == 1


def test_actual_process_timeout_cancels_fake_cli_and_counts_reserved_attempt(tmp_path):
    cli = tmp_path / "slow-fake-cli"
    cli.write_text("#!" + sys.executable + "\nimport time\ntime.sleep(30)\n")
    cli.chmod(0o700)
    p = policy(call_seconds=2.2)
    service = relay.LocalService(tmp_path / "state", p, executable=cli, synthetic=True)
    response = service.serve(request(p))
    assert response["process"]["timed_out"] and response["process"]["code"] != 0
    assert response["response"] is None
    assert len(list(service.claims.glob("*/dispatch.json"))) == 1


def test_policy_and_transport_files_reject_replacement_and_symlinks(tmp_path, fake_cli):
    cli, _ = fake_cli
    p = policy()
    state = tmp_path / "state"
    relay.LocalService(state, p, executable=cli, synthetic=True)
    with pytest.raises(relay.RelayError, match="policy changed"):
        relay.LocalService(state, policy(), executable=cli, synthetic=True)
    path = state / "immutable.json"
    relay.publish_new(path, {"a": 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        relay.publish_new(path, {"a": 2})
    assert path.read_bytes() == before
    link = state / "link.json"
    link.symlink_to(path)
    with pytest.raises(OSError):
        relay.read_json(link, 100)


def test_response_hash_and_request_binding_are_checked(tmp_path):
    p = policy()
    item = request(p)
    good = relay.response_envelope(item, relay._result(0), '{"answer":"ok"}')
    relay.validate_response(good, item)
    with pytest.raises(relay.RelayError, match="size/hash"):
        relay.validate_response(dict(good, response='{"answer":"altered"}'), item)
    with pytest.raises(relay.RelayError, match="pending request"):
        relay.validate_response(dict(good, request_id="b" * 32), item)


def test_policy_uses_explicit_model_allowlist_and_source_binding():
    p = policy()
    assert asdict(p)["model"] in p.allowed_models
    assert asdict(p)["source_sha256"] == "1" * 64
    assert policy(model="gpt-6.1-sol", allowed_models=("gpt-6.1-sol",)).model == "gpt-6.1-sol"
    for change in ({"model": "gpt-6.1-sol"}, {"max_calls": 1001}, {"call_seconds": 601},
                   {"token_stop_threshold": 10000001}, {"source_sha256": "mutable"}):
        with pytest.raises(relay.RelayError):
            policy(**change)


def test_cli_diagnostics_are_redacted_and_failed_usage_still_counts(tmp_path, fake_cli):
    from scientisttwo.process import ProcessResult
    cli, _ = fake_cli
    p = policy(token_stop_threshold=12)
    sensitive = "sensitive-account-diagnostic-fixture"
    def failure(*args, **kwargs):
        return ProcessResult(1, json.dumps({"type": "turn.completed", "usage":
                             {"input_tokens": 10, "output_tokens": 2}}), sensitive, 0.1)
    service = relay.LocalService(tmp_path / "state", p, executable=cli, executor=failure, synthetic=True)
    response = service.serve(request(p))
    assert sensitive not in json.dumps(response) and response["process"]["stdout"] == ""
    with pytest.raises(relay.RelayError, match="threshold"):
        service.serve(request(p, request_id="b" * 32))


def test_fatal_or_tool_events_fail_closed_without_retaining_account_error_text():
    for event in ({"type": "error", "message": "sensitive-diagnostic"},
                  {"type": "item.completed", "item": {"type": "command_execution", "command": "whoami"}}):
        safe, error = relay.safe_events(json.dumps(event))
        assert safe == "" and error
    safe, error = relay.safe_events(json.dumps({"type": "item.completed", "item":
                                   {"type": "error", "message": "sensitive-diagnostic"}}))
    assert error is None and "sensitive-diagnostic" not in safe and '"redacted":true' in safe


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6SUQAAAAASUVORK5CYII=")


def media(raw=PNG, *, index=0, mime="image/png"):
    return {"filename": f"media-{index:02d}" + (".png" if mime == "image/png" else ".jpg"),
            "mime_type": mime, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "data_base64": base64.b64encode(raw).decode()}


def writer_request(p, **kwargs):
    attachments = kwargs.get("media", [media()])
    prompt = json.dumps({"role": "paperorchestra_model", "prompt_version": "scientisttwo-v1",
                         "instructions": relay.INSTRUCTIONS, "inputs": {"task": "Inspect this page",
                         "media": [{"sha256": item.get("sha256"), "mime_type": item.get("mime_type")}
                                   for item in attachments]}})
    value = dict(role="paperorchestra_model", prompt=prompt, media=[media()])
    value.update(kwargs)
    return request(p, **value)


def test_image_bytes_roundtrip_through_real_backend_and_fresh_local_cli_paths(tmp_path, fake_cli):
    from scientisttwo.backends import CodexBackend
    cli, counter = fake_cli
    p = policy()
    queue = relay.QueueExecutor(tmp_path / "queue", p, poll_seconds=.01)
    service = relay.LocalService(tmp_path / "state", p, executable=cli, synthetic=True)
    results, errors = [], []

    def with_page(argv, timeout, **kwargs):
        directory = Path(argv[argv.index("--cd") + 1])
        path = directory / "paperorchestra-page-0.png"
        path.write_bytes(PNG)
        args = list(argv)
        args[-1:-1] = ["--image", str(path)]
        return queue(args, timeout, **kwargs)

    def remote():
        try:
            results.append(CodexBackend(tmp_path, p.model, allow_live=True, timeout=30,
                                        executor=with_page).invoke("paperorchestra_model", {
                                            "media": [{"sha256": media()["sha256"], "mime_type": "image/png"}]
                                        }, Answer))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=remote)
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline and not list(queue.queue.glob("*.request.json")):
        time.sleep(.01)
    item = relay.read_json(next(queue.queue.glob("*.request.json")), relay.REQUEST_LIMIT)
    assert item["media"] == [media()]
    response = service.serve(item)
    relay.publish_new(queue.queue / (item["request_id"] + ".response.json"), response)
    thread.join(timeout=5)
    assert not thread.is_alive() and not errors and results[0].payload["answer"] == "actual fake CLI"
    argv = json.loads(counter.read_text().splitlines()[0])
    assert Path(argv[argv.index("--image") + 1]).name == "media-00.png"
    assert "--disable" in argv and "view_image" in argv and "web_search=\"disabled\"" in argv
    receipt = relay.read_json(service.claims / item["request_id"] / "dispatch.json", 16384)
    assert receipt["media"] == [{k: media()[k] for k in ("filename", "mime_type", "sha256", "bytes")}]
    assert receipt["synthetic"] and receipt["usage_status"] == "known"


@pytest.mark.parametrize("change", [
    {"filename": "../escape.png"}, {"mime_type": "application/pdf"},
    {"data_base64": "not base64"}, {"bytes": 0}, {"bytes": True},
    {"sha256": "0" * 64}, {"extra": "untrusted"},
])
def test_bad_media_is_rejected_before_claim_or_dispatch(tmp_path, fake_cli, change):
    cli, counter = fake_cli
    p = policy()
    service = relay.LocalService(tmp_path / "state", p, executable=cli, synthetic=True)
    with pytest.raises(relay.RelayError):
        service.serve(writer_request(p, media=[{**media(), **change}]))
    assert not counter.exists() and not list(service.claims.iterdir())


def test_media_bounds_mime_and_role_are_checked(tmp_path, monkeypatch):
    p = policy()
    with pytest.raises(relay.RelayError, match="Only the PaperOrchestra"):
        relay.validate_request(request(p, media=[media()]), p)
    with pytest.raises(relay.RelayError, match="Too many"):
        relay.validate_request(writer_request(p, media=[media(index=i) for i in range(17)]), p)
    with pytest.raises(relay.RelayError, match="header"):
        relay.validate_request(writer_request(p, media=[media(b"fake")]), p)
    monkeypatch.setattr(relay, "MEDIA_TOTAL_BYTES", len(PNG))
    with pytest.raises(relay.RelayError, match="aggregate"):
        relay.validate_request(writer_request(p, media=[media(), media(index=1)]), p)
    relay.validate_request(writer_request(p, media=[media(b"\xff\xd8\xffjpeg", mime="image/jpeg")]), p)


def test_source_pin_and_live_opt_in_are_required(tmp_path, fake_cli):
    cli, counter = fake_cli
    p = policy()
    with pytest.raises(relay.RelayError, match="allow_live"):
        relay.LocalService(tmp_path / "state", p, executable=cli)
    with pytest.raises(relay.RelayError, match="frozen source"):
        relay.QueueExecutor(tmp_path / "queue", policy(relay_sha256="0" * 64))
    assert not counter.exists()


@pytest.mark.parametrize("stdout", ["", '{"type":"turn.failed"}',
    '{"type":"turn.completed","usage":{"input_tokens":1}}',
    '{"type":"turn.completed","usage":{"input_tokens":true,"output_tokens":1}}',
    '\n'.join([json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2}})] * 2),
])
def test_unknown_usage_never_becomes_a_zero_cost_retry(tmp_path, fake_cli, stdout):
    from scientisttwo.process import ProcessResult
    cli, counter = fake_cli
    p = policy()
    service = relay.LocalService(tmp_path / "state", p, executable=cli, synthetic=True,
                                  executor=lambda *args, **kwargs: ProcessResult(1, stdout, "private diagnostic", .1))
    item = request(p)
    first = service.serve(item)
    assert first["process"]["code"] != 0 and "private diagnostic" not in json.dumps(first)
    assert first == service.serve(item), "Known transport result replays without redispatch"
    with pytest.raises(relay.UnknownDispatch, match="usage is unknown"):
        service.serve(request(p, request_id="b" * 32))
    assert not counter.exists()


def test_all_current_research_roles_and_direct_coding_are_supported():
    from scientisttwo.research.controller import PROMPTS
    assert relay.allowed_roles() == frozenset(PROMPTS) | {"direct_coding"}
    for role in relay.allowed_roles():
        prompt = json.dumps({"role": role, "prompt_version": "scientisttwo-v1",
                             "instructions": relay.INSTRUCTIONS, "inputs": {}})
        assert relay.validate_prompt_schema(prompt, {"type": "object"}) == role


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_unicode_inside_json_string_survives_real_local_service(tmp_path, separator, line_ending):
    """A real fake process emits literal Unicode, not an escaped ASCII fixture."""
    cli = tmp_path / "unicode-fake-codex"
    cli.write_text("#!" + sys.executable + "\n" + f'''
import json,sys,pathlib
sys.stdin.read()
text = "Known invalid value in scientific evidence " + {separator!r} + " still valid JSON"
pathlib.Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps({{'answer':text}}, ensure_ascii=False))
events = [{{'type':'thread.started'}}, {{'type':'item.completed','item':{{'type':'agent_message','text':text}}}},
          {{'type':'turn.completed','usage':{{'input_tokens':123,'cached_input_tokens':100,'output_tokens':45}}}}]
sys.stdout.buffer.write(({line_ending!r}.join(json.dumps(x,ensure_ascii=False) for x in events)+{line_ending!r}).encode())
''')
    cli.chmod(0o700)
    p = policy()
    service = relay.LocalService(tmp_path / "state", p, executable=cli, synthetic=True)
    item = request(p)
    response = service.serve(item)
    assert response["process"]["code"] == 0 and separator in json.loads(response["response"])["answer"]
    assert relay.usage_tokens(response["process"]["stdout"]) == 168
    assert json.loads(response["process"]["stderr"])["diagnostic_categories"] == ["unknown"]
    receipt = relay.read_json(service.claims / item["request_id"] / "dispatch.json", 16384)
    assert receipt["observed_tokens"] == 168 and receipt["usage_status"] == "known"
    diagnostic = receipt["cli_diagnostics"]
    assert diagnostic["original_process"]["code"] == 0
    assert diagnostic["jsonl"]["completed_count"] == 1
    assert diagnostic["jsonl"]["event_counts"] == {"thread.started": 1, "item.completed": 1, "turn.completed": 1}
    assert diagnostic["jsonl"]["usage_error"] is None
    assert "scientific evidence" not in json.dumps(diagnostic)
    assert service.serve(item) == response


_COMPLETED = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}})


@pytest.mark.parametrize("stdout,reason", [
    ("", "missing_completed_turn"),
    ("\n" + _COMPLETED, "blank_event_line"),
    (_COMPLETED + "\n\n", "blank_event_line"),
    (_COMPLETED + "\n \t\r\n", "blank_event_line"),
    (_COMPLETED + '\n{"type":"thread.started"}\n\n{}', "blank_event_line"),
    (_COMPLETED + '\r{"type":"thread.started"}', "malformed_json"),
    (_COMPLETED + "\nnot-json", "malformed_json"),
    (_COMPLETED + "\n" + _COMPLETED, "duplicate_completed_turn"),
    ('{"type":"turn.completed","usage":null}', "invalid_usage_object"),
    ('{"type":"turn.completed","usage":{"input_tokens":true,"output_tokens":2}}', "invalid_usage_count"),
    ('{"type":"turn.completed","usage":{"input_tokens":-1,"output_tokens":2}}', "invalid_usage_count"),
    ('{"type":"turn.completed","usage":{"input_tokens":1}}', "invalid_usage_count"),
    ('{"type":"turn.completed","usage":{"input_tokens":NaN,"output_tokens":2}}', "malformed_json"),
    ('{"type":"turn.started","type":"turn.completed"}', "malformed_json"),
    ('{"type":[]}', "invalid_event_type"),
    ('[]', "non_object_event"),
])
def test_local_failure_receipt_preserves_original_status_and_specific_parse_reason(tmp_path, fake_cli, stdout, reason):
    from scientisttwo.process import ProcessResult
    cli, counter = fake_cli
    p = policy()
    sensitive = "private-account-fixture"
    service = relay.LocalService(tmp_path / "state", p, executable=cli, synthetic=True,
        executor=lambda *args, **kwargs: ProcessResult(17, stdout, sensitive, .2,
            stdout_bytes=len(stdout.encode()), stderr_bytes=len(sensitive)))
    item = request(p)
    response = service.serve(item)
    receipt = relay.read_json(service.claims / item["request_id"] / "dispatch.json", 16384)
    assert response["process"]["code"] == 125
    assert receipt["observed_tokens"] is None and receipt["usage_status"] == "unknown"
    diagnostic = receipt["cli_diagnostics"]
    assert diagnostic["original_process"]["code"] == 17
    assert diagnostic["jsonl"]["usage_error"]["code"] == reason
    assert sensitive not in json.dumps(receipt) + json.dumps(response)
    assert service.serve(item) == response
    with pytest.raises(relay.UnknownDispatch, match="usage is unknown"):
        service.serve(request(p, request_id="b" * 32))
    assert not counter.exists()


def test_unknown_event_strings_and_original_limit_text_do_not_leak_into_diagnostics(tmp_path, fake_cli):
    from scientisttwo.process import ProcessResult
    cli, _ = fake_cli
    p = policy()
    secret = "private-untrusted-string"
    stdout = json.dumps({"type": secret, "item": {"type": secret}}) + "\n" + _COMPLETED
    service = relay.LocalService(tmp_path / "state", p, executable=cli, synthetic=True,
        executor=lambda *args, **kwargs: ProcessResult(9, stdout, secret, .1, timed_out=True,
            stdout_truncated=True, stderr_truncated=True, limit_exceeded=secret))
    item = request(p)
    service.serve(item)
    receipt = relay.read_json(service.claims / item["request_id"] / "dispatch.json", 16384)
    diagnostic = receipt["cli_diagnostics"]
    assert secret not in json.dumps(diagnostic)
    assert diagnostic["original_process"] == {
        "code": 9, "timed_out": True, "limit_exceeded": True,
        "limit_exceeded_sha256": hashlib.sha256(secret.encode()).hexdigest(),
        "stdout_truncated": True, "stderr_truncated": True, "stdout_bytes": 0, "stderr_bytes": 0}
    assert diagnostic["jsonl"]["event_counts"] == {"unknown": 1, "turn.completed": 1}
    assert diagnostic["jsonl"]["item_counts"] == {"unknown": 1}


@pytest.mark.parametrize("stdout", ['{"type":[]}', '{"type":{}}',
    '{"type":"item.completed","item":{"type":[]}}'])
def test_safe_event_parser_rejects_untyped_events_without_crashing(stdout):
    safe, error = relay.safe_events(stdout)
    assert safe == "" and error


def test_frozen_policy_json_roundtrip_requires_every_field():
    p = policy()
    assert relay.RelayPolicy.from_mapping(json.loads(relay.canonical(asdict(p)))) == p
    missing = asdict(p)
    missing.pop("token_stop_threshold")
    with pytest.raises(relay.RelayError, match="incomplete"):
        relay.RelayPolicy.from_mapping(missing)


def test_media_matches_frozen_controller_prompt_identity():
    p = policy()
    value = writer_request(p)
    value["media"] = [media(PNG + b"modified")]
    with pytest.raises(relay.RelayError, match="frozen media bindings"):
        relay.validate_request(value, p)
