"""Trusted subprocess fixtures prove capture, cancellation, and output limits.

These scripts are test infrastructure. Generated research candidates continue to
execute only through container runners; no model endpoint is contacted here.
"""

import subprocess
import sys
import time
import tracemalloc
from types import SimpleNamespace

import pytest

from scientisttwo.live.timing_runtime import ContainerTimingTransport
from scientisttwo.process import OutputLimits, ProcessResult, run_process


def command(source):
    return [sys.executable, "-u", "-c", source]


def test_legacy_process_result_stdin_and_unicode_are_preserved():
    assert ProcessResult(0, "out", "err", 0).limit_exceeded is None
    text = "héllo κόσμε\n" * 5000
    result = run_process(
        command(
            "import sys; sys.stdout.write(sys.stdin.read()); sys.stderr.write('diagnostic\\n')"
        ),
        5,
        stdin=text,
    )
    assert result.code == 0 and result.stdout == text and result.stderr == "diagnostic\n"
    assert result.stdout_bytes == len(text.encode())
    assert not result.stdout_truncated and not result.stderr_truncated


def test_stdout_and_stderr_flood_keep_bounded_memory_and_head_tail():
    size = 24 * 1024**2
    script = f"""import os
for descriptor, name in ((1, b'OUT'), (2, b'ERR')):
    os.write(descriptor, b'START-' + name + b'\\n')
    for _ in range({size // 65536}):
        os.write(descriptor, b'x' * 65536)
    os.write(descriptor, b'\\nEND-' + name + b'\\n')
"""
    tracemalloc.start()
    try:
        result = run_process(command(script), 15, max_output_bytes=65536)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.code == 0 and not result.timed_out and result.limit_exceeded is None
    assert result.stdout_truncated and result.stderr_truncated
    for output, name, count in (
        (result.stdout, "OUT", result.stdout_bytes),
        (result.stderr, "ERR", result.stderr_bytes),
    ):
        assert output.startswith("START-" + name + "\n") and output.endswith("\nEND-" + name + "\n")
        assert len(output.encode()) == 65536
        assert count == size + len("START-OUT\n\nEND-OUT\n")
    assert peak < 4 * 1024**2, "Capture must not retain the 48 MiB child output in controller RAM"


@pytest.mark.parametrize("cap", [0, 1, 17, 1024])
def test_small_output_caps_retain_exact_counts(cap):
    result = run_process(command("import os; os.write(1,b'a'*5000)"), 5, max_output_bytes=cap)
    assert result.stdout_bytes == 5000 and len(result.stdout) == cap
    assert result.stdout_truncated and not result.stderr_truncated


def test_byte_quota_stops_process_tree_and_preserves_partial_artifact(tmp_path):
    output, marker = tmp_path / "output", tmp_path / "escaped-child"
    output.mkdir()
    child = f"import time; from pathlib import Path; time.sleep(0.9); Path({str(marker)!r}).write_text('escaped')"
    source = f"""import subprocess,sys,time
from pathlib import Path
subprocess.Popen([sys.executable,'-c',{child!r}])
Path({str(output / "partial.bin")!r}).write_bytes(b'x' * 2000000)
time.sleep(60)
"""
    result = run_process(
        command(source),
        5,
        output_limits=OutputLimits((output,), max_bytes=100000, max_files=10),
        poll_interval=0.05,
    )
    assert result.limit_exceeded and "byte quota" in result.limit_exceeded
    assert not result.timed_out and result.code != 0 and result.seconds < 3
    assert (output / "partial.bin").is_file()
    time.sleep(1)
    assert not marker.exists()


def test_fast_completed_process_still_checks_file_quota(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    script = f"from pathlib import Path\np=Path({str(root)!r})\nfor i in range(4): (p/str(i)).write_text('')"
    result = run_process(
        command(script), 5, output_limits=OutputLimits((root,), max_bytes=100, max_files=2)
    )
    assert result.limit_exceeded and "file quota" in result.limit_exceeded
    assert not result.timed_out and len(list(root.iterdir())) == 4


def test_timeout_cancels_a_parent_that_stopped_reading_stdin(tmp_path):
    marker = tmp_path / "descendant"
    child = f"import time; from pathlib import Path; time.sleep(0.7); Path({str(marker)!r}).write_text('alive')"
    source = f"import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c',{child!r}])\ntime.sleep(60)"
    result = run_process(command(source), 0.15, stdin="input" * 500000)
    assert result.timed_out and result.limit_exceeded is None and result.seconds < 2
    time.sleep(0.8)
    assert not marker.exists()


def test_quota_roots_do_not_follow_symlinks_or_double_count(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    (root / "nested").mkdir(parents=True)
    outside.mkdir()
    (root / "nested/data.bin").write_bytes(b"12345")
    (outside / "large.bin").write_bytes(b"x" * 100000)
    (root / "linked").symlink_to(outside, target_is_directory=True)
    limit = OutputLimits((root, root / "nested", root), max_bytes=5, max_files=2)
    assert limit.check() is None
    assert "not a regular directory" in OutputLimits((root / "linked",), 100000, 10).check()


def test_custom_monitor_failures_are_fail_closed_without_raw_messages():
    calls = []

    def monitor():
        calls.append(1)
        if len(calls) > 1:
            raise OSError("untrusted diagnostic should not be persisted")
        return None

    result = run_process(command("import time; time.sleep(60)"), 5, monitor=monitor)
    assert result.limit_exceeded == "Output monitor failed: OSError"
    assert not result.timed_out and result.code != 0


def test_preexisting_over_quota_outputs_do_not_start_process(tmp_path):
    output, marker = tmp_path / "output", tmp_path / "started"
    output.mkdir()
    (output / "existing").write_bytes(b"12345")
    result = run_process(
        command(f"from pathlib import Path; Path({str(marker)!r}).touch()"),
        5,
        output_limits=OutputLimits((output,), max_bytes=1, max_files=10),
    )
    assert result.limit_exceeded and not marker.exists()
    assert (output / "existing").read_bytes() == b"12345"


def timing_fixture(tmp_path, monkeypatch, script, *, bytes_limit=1000000, log_limit=65536):
    original_popen = subprocess.Popen
    cleaned = []

    def launch(argv, **kwargs):
        assert argv[0] == "docker"
        return original_popen(command(script), **kwargs)

    monkeypatch.setattr("scientisttwo.live.timing_runtime.subprocess.Popen", launch)
    monkeypatch.setattr(
        "scientisttwo.live.timing_runtime.run_process", lambda argv, timeout: cleaned.append(argv)
    )
    source, public = tmp_path / "source", tmp_path / "public"
    source.mkdir()
    public.mkdir()
    resources = SimpleNamespace(
        cpus=1,
        memory_mb=512,
        gpus=None,
        output_max_bytes=bytes_limit,
        output_max_files=10,
        log_max_bytes=log_limit,
    )
    transport = ContainerTimingTransport(
        prefix=["docker"],
        image="synthetic-container",
        source=source,
        public=public,
        control=tmp_path / "control",
        outputs=tmp_path / "output",
        resources=resources,
        handler_source="# fixed test fixture only",
        parameters={},
    )
    return transport, cleaned


def test_timing_worker_stderr_is_bounded_and_records_truncation(tmp_path, monkeypatch):
    script = """import os,json,sys
os.write(2,b'START\\n')
for _ in range(128): os.write(2,b'x'*65536)
os.write(2,b'\\nEND\\n')
print(json.dumps({'type':'ready','protocol':1,'device':'cpu'}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps({'request_id':r['request_id'],'status':'ok','result':{}}),flush=True)
"""
    transport, cleaned = timing_fixture(tmp_path, monkeypatch, script)
    assert transport.request({}, timeout=1) == {}
    transport.close()
    log = transport.log_path.read_text()
    assert len(log.encode()) == 65536 and log.startswith("START\n") and log.endswith("\nEND\n")
    receipt = transport.receipt()
    assert receipt["stderr_truncated"] and receipt["stderr_bytes"] > 8_000_000
    assert cleaned and all(argv[-1].startswith("scientisttwo-timing-") for argv in cleaned)


def test_timing_worker_output_quota_interrupts_blocked_rpc(tmp_path, monkeypatch):
    output = tmp_path / "output" / "oversized.bin"
    script = f"""import json,sys,time
from pathlib import Path
print(json.dumps({{'type':'ready','protocol':1,'device':'cpu'}}),flush=True)
sys.stdin.readline()
Path({str(output)!r}).write_bytes(b'x'*2000000)
time.sleep(60)
"""
    transport, cleaned = timing_fixture(tmp_path, monkeypatch, script, bytes_limit=1000)
    started = time.monotonic()
    with pytest.raises(RuntimeError):
        transport.request({}, timeout=10)
    assert time.monotonic() - started < 3
    assert transport.limit_exceeded and "byte quota" in transport.limit_exceeded
    assert output.is_file() and transport.closed
    assert cleaned and all("--force" in argv for argv in cleaned)
