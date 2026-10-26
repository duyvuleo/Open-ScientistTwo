"""Synthetic RPC traces test timing contracts, never benchmark performance claims."""

import itertools
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from scientisttwo.live.timing import JsonLineTransport, run_paired_timing
from scientisttwo.live.timing_runtime import ContainerTimingTransport
from scientisttwo.live.timing_vectorization import expected_outputs, vectorization_oracle
from scientisttwo.store import Store


class SyntheticTransport:
    def __init__(self, arm, calls):
        self.arm, self.calls = arm, calls

    def request(self, payload, *, timeout):
        assert timeout > 0
        self.calls.append((self.arm, payload))
        return {"answer": payload["value"], "model_claimed_seconds": -1234}


def test_pair_order_warmup_exclusion_and_parent_only_timing():
    calls, checked = [], []
    transports = {arm: SyntheticTransport(arm, calls) for arm in ("baseline", "candidate")}

    def oracle(arm, payload, response):
        assert response["answer"] == payload["value"]
        checked.append(arm)
        return {"valid": True, "fixture": "synthetic"}

    clock = itertools.count(0, 1_000_000).__next__
    report = run_paired_timing(transports, [{"value": 1}], oracle, clock_ns=clock)
    assert len(report.samples) == len(checked) == 30
    assert len(report.baseline.seconds) == len(report.candidate.seconds) == 10
    assert report.baseline.median_seconds == report.candidate.median_seconds == 0.001
    assert report.baseline.iqr_seconds == 0
    assert [arm for arm, _ in calls[:4]] == ["baseline", "candidate", "candidate", "baseline"]
    assert [s.arm for s in report.samples if s.phase == "measurement"][:4] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert report.gpu_synchronization == "not_applicable"


def test_multiple_workloads_are_summed_per_matched_pair():
    transports = {arm: SyntheticTransport(arm, []) for arm in ("baseline", "candidate")}
    report = run_paired_timing(
        transports,
        [{"value": 1}, {"value": 2}],
        lambda *args: {"valid": True},
        clock_ns=itertools.count(0, 1_000_000).__next__,
    )
    assert report.baseline.seconds == [0.002] * 10
    assert report.payload_count == 2 and len(report.samples) == 60


def test_warmup_correctness_and_gpu_synchronization_are_required():
    transports = {arm: SyntheticTransport(arm, []) for arm in ("baseline", "candidate")}
    with pytest.raises(ValueError, match="correctness"):
        run_paired_timing(transports, [{"value": 1}], lambda *args: {"valid": False})
    with pytest.raises(ValueError, match="synchronization"):
        run_paired_timing(transports, [{"value": 1}], lambda *args: {"valid": True}, device="cuda")


def rpc_process(source):
    return subprocess.Popen(
        [sys.executable, "-u", "-c", source],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def test_real_fixture_subprocess_rpc_and_parent_timing():
    process = rpc_process("""import json, sys
print(json.dumps({"type":"ready","protocol":1,"device":"cpu"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"request_id":request["request_id"],"status":"ok","result":{"answer":request["payload"]["value"]}}), flush=True)
""")
    transport = JsonLineTransport(process)
    try:
        assert transport.request({"value": 7}, timeout=1) == {"answer": 7}
        assert transport.request({"value": 11}, timeout=1) == {"answer": 11}
        assert transport.startup_seconds >= 0
    finally:
        transport.close()
    assert process.poll() is not None


@pytest.mark.parametrize(
    "response",
    [
        '{"request_id":999,"status":"ok","result":{}}',
        '{"request_id":1,"status":"failed"}',
        "x" * 200,
    ],
)
def test_rpc_rejects_failed_mismatched_and_oversized_responses(response):
    process = rpc_process(
        'import sys\nprint(\'{"type":"ready","protocol":1}\',flush=True)\n'
        f"sys.stdin.readline()\nprint({response!r},flush=True)\n"
    )
    transport = JsonLineTransport(process, max_message_bytes=128)
    try:
        with pytest.raises(ValueError):
            transport.request({}, timeout=1)
    finally:
        transport.close()


def test_rpc_timeout_is_bounded_and_process_can_be_cancelled():
    process = rpc_process(
        'import time\nprint(\'{"type":"ready","protocol":1}\',flush=True)\ntime.sleep(60)'
    )
    transport = JsonLineTransport(process)
    try:
        with pytest.raises(TimeoutError):
            transport.request({}, timeout=0.05)
    finally:
        transport.close()
    assert process.poll() is not None


def test_independent_vector_oracle_seals_raw_outputs_and_rejects_reuse(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    np.savez(
        public / "inputs.npz",
        x0=np.arange(9).reshape(1, 3, 3, 1),
        k0=np.ones((2, 2, 1, 1)),
        b0=np.zeros((1, 1, 1, 1)),
    )
    expected = expected_outputs(public, [{"stride": 1, "padding": 0, "activation": None}])
    np.testing.assert_array_equal(expected["0"], np.array([8, 12, 20, 24]).reshape(1, 2, 2, 1))
    outputs = {arm: tmp_path / arm for arm in ("baseline", "candidate")}
    for path in outputs.values():
        path.mkdir()
        np.savez(path / "predictions-1.npz", **expected)
    store = Store(tmp_path / "store")
    try:
        oracle = vectorization_oracle(store, outputs, expected)
        evidence = oracle(
            "baseline", {"workload": "all_cases"}, {"predictions": "predictions-1.npz"}
        )
        assert evidence["valid"] is True and evidence["max_abs_error"] == 0
        assert store.read(evidence["prediction_ref"])
        with pytest.raises(ValueError, match="fresh"):
            oracle("baseline", {"workload": "all_cases"}, {"predictions": "predictions-1.npz"})
        np.savez(outputs["candidate"] / "wrong.npz", **{"0": expected["0"] + 1})
        with pytest.raises(ValueError, match="correctness"):
            oracle("candidate", {"workload": "all_cases"}, {"predictions": "wrong.npz"})
    finally:
        store.close()


def test_container_transport_has_no_host_fallback(tmp_path):
    with pytest.raises(ValueError, match="no host fallback"):
        ContainerTimingTransport(
            prefix=[sys.executable],
            image="ignored",
            source=tmp_path,
            public=tmp_path,
            control=tmp_path / "control",
            outputs=tmp_path / "out",
            resources=SimpleNamespace(),
            handler_source="",
            parameters={},
        )


@pytest.mark.parametrize("runtime,device", [("docker", "cpu"), ("podman", "cuda")])
def test_container_argv_isolates_source_inputs_and_worker(tmp_path, monkeypatch, runtime, device):
    source, public = tmp_path / "source", tmp_path / "public"
    source.mkdir()
    public.mkdir()
    launched, cleaned = [], []

    class FakeProcess:
        def __init__(self, argv, **kwargs):
            launched.append((argv, kwargs))
            self.stdin, self.stdout, self.stderr = (tempfile.TemporaryFile() for _ in range(3))

        def poll(self):
            return 0

    class FakeRPC:
        ready = {"device": device}
        startup_seconds = 0.1

        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr("scientisttwo.live.timing_runtime.subprocess.Popen", FakeProcess)
    monkeypatch.setattr("scientisttwo.live.timing_runtime.JsonLineTransport", FakeRPC)
    monkeypatch.setattr(
        "scientisttwo.live.timing_runtime.run_process", lambda argv, *args: cleaned.append(argv)
    )
    transport = ContainerTimingTransport(
        prefix=[runtime],
        image="sha256:fixture",
        source=source,
        public=public,
        control=tmp_path / "control",
        outputs=tmp_path / "out",
        resources=SimpleNamespace(memory_mb=1024, cpus=1, seconds=900,
                                  gpus="all" if device == "cuda" else None),
        handler_source="",
        parameters={},
        device=device,
        execution_timeout=60,
    )
    transport.close()
    argv, options = launched[0]
    assert "--network=none" in argv and "--read-only" in argv and "--cap-drop=ALL" in argv
    assert f"{source}:/workspace:ro" in argv and f"{public}:/data/public:ro" in argv
    assert options["start_new_session"] is True
    assert argv[argv.index("--label") + 1] == "scientisttwo.timeout-seconds=60.0"
    assert "--force" in cleaned[0]
    assert transport.receipt()["closed"] is True
    if runtime == "podman":
        assert "nvidia.com/gpu=all" in argv
