"""A local subprocess simulates the node step; no SSH, Slurm or model service."""

import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import pytest

from scientisttwo.node_queue_transport import NodeQueueConfig, NodeQueueTransport
from scientisttwo.process import ProcessResult, run_process
from scientisttwo.studies import media_relay as relay


PROGRAM = r'''
import hashlib,json,os,sys,uuid
from pathlib import Path
operation,root=sys.argv[1:]
root=Path(root)
if operation=='next':
 for path in sorted(root.glob('*.request.json')):
  if not path.with_name(path.name.replace('.request.','.response.')).exists():
   sys.stdout.buffer.write(path.read_bytes()); break
elif operation=='complete':
 raw=sys.stdin.buffer.read(); value=json.loads(raw); key=value['request_id']
 request=json.loads((root/(key+'.request.json')).read_bytes())
 canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 assert value['request_sha256']==hashlib.sha256(canonical(request)).hexdigest()
 out=root/(key+'.response.json')
 if out.exists(): assert json.loads(out.read_bytes())==value
 else:
  tmp=root/('.tmp-'+uuid.uuid4().hex)
  with tmp.open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
  os.link(tmp,out); tmp.unlink()
 print(json.dumps({'status':'stored','request_id':key}))
else: raise ValueError('Unsupported operation')
'''


def config(**changes):
    values = dict(host="fixture@example.test", queue="/home/fixture/study/queue", job_id="3920",
        node="fixture-node03", python="/usr/bin/python3", srun="/usr/bin/srun", cpus=1,
        memory_mib=1024, step_seconds=60, connect_timeout_seconds=10,
        exchange_timeout_seconds=15, deadline=time.time() + 120, return_reserve_seconds=20,
        remote_program_sha256=hashlib.sha256(PROGRAM.encode()).hexdigest())
    values.update(changes)
    return NodeQueueConfig(**values)


def test_command_pins_same_node_cpu_only_and_quotes_program_and_queue():
    options = config(queue="/home/fixture/a 'quoted' $queue")
    transport = NodeQueueTransport(options, PROGRAM)
    for operation in ("next", "complete"):
        argv = transport.command(operation)
        assert argv[:-1] == ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                            "-o", "ConnectTimeout=10", options.host]
        remote = shlex.split(argv[-1])
        assert remote == ["/usr/bin/srun", "--jobid=3920", "--overlap", "--exact",
            "--nodelist=fixture-node03", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
            "--mem=1024M", "--gres=none", "--time=00:01:00", "--", "/usr/bin/python3",
            "-I", "-c", PROGRAM, operation, options.queue]


@pytest.mark.parametrize("changes", [
    {"cpus": 2}, {"memory_mib": 2048}, {"step_seconds": 61}, {"connect_timeout_seconds": 11},
    {"exchange_timeout_seconds": 46}, {"deadline": float("inf")}, {"return_reserve_seconds": 0},
    {"return_reserve_seconds": 16},
    {"host": "-oProxyCommand=bad"}, {"node": "node;command"}, {"job_id": "0;command"},
    {"queue": "/tmp/queue"}, {"queue": "/home/user/../queue"}, {"python": "python3"},
])
def test_allocation_paths_and_time_limits_are_explicit_and_bounded(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_program_hash_must_match_and_operation_cannot_inject_commands():
    with pytest.raises(ValueError, match="pinned hash"):
        NodeQueueTransport(config(), PROGRAM + "\n")
    transport = NodeQueueTransport(config(), PROGRAM)
    with pytest.raises(ValueError):
        transport.command("complete; touch /tmp/forbidden")


@pytest.mark.parametrize("publication_failure", [False, True])
def test_real_local_cli_and_atomic_node_publication_preserve_service_ledger(tmp_path, publication_failure):
    queue = tmp_path / "simulated-node-queue"
    queue.mkdir()
    now = time.time()
    policy = relay.RelayPolicy(run_id="node-transport-fixture", expires_at=now + 120,
        model="gpt-6-astra", allowed_models=("gpt-6-astra",), source_sha256="1" * 64,
        relay_sha256=relay.sha(Path(relay.__file__).read_bytes()), call_seconds=60)
    request = {"schema_version": 2, "request_id": "a" * 32, "policy_sha256": policy.identity,
        "run_id": policy.run_id, "model": policy.model, "role": "critic", "created_at": now,
        "deadline": now + 60, "timeout": 60, "source_sha256": policy.source_sha256, "media": [],
        "prompt": json.dumps({"role": "critic", "prompt_version": "scientisttwo-v1",
            "instructions": relay.INSTRUCTIONS, "inputs": {"evidence": "Synthetic transport fixture"}}),
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}},
                   "required": ["answer"], "additionalProperties": False}}
    (queue / (request["request_id"] + ".request.json")).write_bytes(relay.canonical(request))
    executable = tmp_path / "fake-model-cli"
    counter = tmp_path / "model-count"
    executable.write_text("#!" + sys.executable + "\n" + f'''
import json,pathlib,sys
sys.stdin.read()
with pathlib.Path({str(counter)!r}).open('a') as stream:stream.write('called\\n')
pathlib.Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps({{'answer':'fixture'}}))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':7,'output_tokens':5}}}}))
''')
    executable.chmod(0o700)
    model_timeouts = []

    def model_executor(argv, timeout, **kwargs):
        model_timeouts.append(timeout)
        return run_process(argv, timeout, **kwargs)

    service = relay.LocalService(tmp_path / "unchanged-service", policy, executable=executable,
                                 executor=model_executor, synthetic=True)
    wire_calls, receipts = [], []
    fail_once = publication_failure

    def simulated_node(argv, timeout, **kwargs):
        nonlocal fail_once
        remote = shlex.split(argv[-1])
        assert remote[0] == "/usr/bin/srun" and "--gres=none" in remote
        assert "--nodelist=fixture-node03" in remote
        wire_calls.append(remote[-2])
        if remote[-2] == "complete" and fail_once:
            fail_once = False
            return ProcessResult(255, "", "fixture connection ended before acknowledgement", .1)
        # Only the SSH/Slurm boundary is simulated. Execute the exact pinned
        # queue program and perform the actual file publication in a subprocess.
        return run_process([sys.executable, "-I", "-c", remote[-3], remote[-2], str(queue)], timeout, **kwargs)

    transport = NodeQueueTransport(config(), PROGRAM, executor=simulated_node, receipt_sink=receipts.append)
    if publication_failure:
        with pytest.raises(RuntimeError, match="transfer failed"):
            transport.serve_once(service)
        assert counter.read_text() == "called\n"
    assert transport.serve_once(service)
    assert wire_calls == ["next", "complete"] * (2 if publication_failure else 1)
    assert 0 < model_timeouts[0] <= 40
    assert service.executor is model_executor
    response = json.loads((queue / (request["request_id"] + ".response.json")).read_bytes())
    relay.validate_response(response, request)
    assert response["process"]["code"] == 0
    assert len(list(service.claims.iterdir())) == 1
    assert counter.read_text() == "called\n"
    assert transport.serve_once(service) is False
    # Republishing identical completion remains idempotent, without a new call.
    assert transport.exchange("complete", relay.canonical(response)) == {
        "status": "stored", "request_id": request["request_id"]}
    assert counter.read_text() == "called\n"
    assert all(item["node"] == "fixture-node03" for item in receipts)
    assert "Synthetic transport fixture" not in json.dumps(receipts)


@pytest.mark.parametrize("result", [
    ProcessResult(255, "private stdout", "private account text", .1),
    ProcessResult(0, "private stdout", "private account text", .1, timed_out=True),
    ProcessResult(0, "private stdout", "private account text", .1, stdout_truncated=True),
])
def test_transfer_failure_diagnostics_do_not_include_raw_text(result):
    receipts = []
    transport = NodeQueueTransport(config(), PROGRAM,
        executor=lambda *args, **kwargs: result, receipt_sink=receipts.append)
    with pytest.raises(RuntimeError) as caught:
        transport.exchange("next")
    assert "private" not in str(caught.value) + json.dumps(receipts)
    assert receipts[0]["stderr_sha256"] == hashlib.sha256(result.stderr.encode()).hexdigest()


def test_expired_transport_does_not_call_executor():
    def forbidden(*args, **kwargs):
        pytest.fail("Expired transport must not execute a process")
    transport = NodeQueueTransport(config(deadline=99), PROGRAM, executor=forbidden, clock=lambda: 100)
    with pytest.raises(TimeoutError):
        transport.exchange("next")


def test_insufficient_return_time_does_not_invoke_model_and_restores_executor():
    request = {"deadline": 110, "request_id": "a" * 32}
    calls = []

    def original(*args, **kwargs):
        calls.append(True)

    class Service:
        executor = staticmethod(original)

        def serve(self, item):
            return self.executor([], 300)

    service = Service()
    transport = NodeQueueTransport(config(deadline=200), PROGRAM, clock=lambda: 100,
        executor=lambda *args, **kwargs: ProcessResult(0, json.dumps(request), "", 0))
    with pytest.raises(TimeoutError, match="response publication"):
        transport.serve_once(service)
    assert not calls and service.executor is original


def test_exchange_never_exceeds_transport_deadline_or_output_limit():
    called = []

    def executor(argv, timeout, **kwargs):
        called.append((timeout, kwargs))
        return ProcessResult(0, "", "", 0)

    transport = NodeQueueTransport(config(deadline=105), PROGRAM, executor=executor, clock=lambda: 100)
    assert transport.exchange("next") is None
    assert called == [(5, {"stdin": None, "max_output_bytes": relay.REQUEST_LIMIT})]
    with pytest.raises(ValueError):
        transport.exchange("next", b"unexpected")
    with pytest.raises(ValueError):
        transport.exchange("complete", b"x" * (relay.RESPONSE_LIMIT + 1))
