import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from scientisttwo import runtime_contracts as contracts
from scientisttwo.benchmarks.models import NativeResources
from scientisttwo.benchmarks.runner import NativeRunner
from scientisttwo.live import timing_vectorization
from scientisttwo.live.timing_runtime import ContainerTimingTransport
from scientisttwo.process import ProcessResult
from scientisttwo.store import Store, digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def clean_execution_environment(monkeypatch):
    monkeypatch.delenv(contracts.EXECUTION_CONFIG_ENV, raising=False)
    monkeypatch.delenv(contracts.EXECUTION_CONFIG_SHA_ENV, raising=False)


@pytest.fixture
def policy(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    receipts = root / "receipts"
    receipts.mkdir(mode=0o700)
    wrapper, podman, control = (root / name for name in ("shim", "podman", "negative.json"))
    wrapper.write_text("shim")
    podman.write_text("runtime")
    control.write_text('{"verified":true}')
    config = {
        "schema_version": 1, "job_id": "42", "node": "worker", "uid": os.getuid(),
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc", "gpu_gres": "gpu:a100:8",
        "max_cpus": 8, "max_memory_mb": 65536, "controller_memory_mb": 2048,
        "max_seconds": 3600, "expires_at": 1,  # Historical evidence remains inspectable.
        "receipt_dir": str(receipts), "podman": str(podman), "podman_sha256": sha(podman),
        "negative_control": str(control), "negative_control_sha256": sha(control),
        "guard_sha256": sha(Path(contracts.__file__).with_name("slurm_execution.py")),
        "future_policy_field": {"preserved_in_full_config_hash": True},
    }
    path = root / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv(contracts.EXECUTION_CONFIG_ENV, str(path))
    monkeypatch.setenv(contracts.EXECUTION_CONFIG_SHA_ENV, sha(path))
    monkeypatch.setattr(contracts.shutil, "which", lambda name: str(wrapper))
    return path, config, wrapper


def test_ordinary_runtime_identity_unchanged():
    spec = SimpleNamespace(resources=NativeResources(), runtime="docker", assets=[])
    assert contracts.execution_identity("docker") is None
    assert contracts.runtime_identity(spec, {"execution": "sha256:" + "a" * 64}) == {
        "images": {"execution": "sha256:" + "a" * 64},
        "resources": spec.resources.model_dump(mode="json"), "runtime": "docker",
        "assets": [], "hardware": {"status": "unverified"},
    }


@pytest.mark.parametrize("key,value", [
    (contracts.EXECUTION_CONFIG_ENV, "/tmp/config"),
    (contracts.EXECUTION_CONFIG_SHA_ENV, "a" * 64),
])
def test_half_declared_policy_fails_closed(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="declared together"):
        contracts.execution_identity("podman")


def test_identity_binds_full_config_wrapper_and_dependencies(policy):
    path, config, wrapper = policy
    identity = contracts.execution_identity("podman")
    assert identity["config_sha256"] == sha(path)
    assert identity["runtime_wrapper"]["sha256"] == sha(wrapper)
    assert identity["dependencies"]["guard"]["sha256"] == config["guard_sha256"]
    assert identity["allocation"]["controller_memory_mb"] == 2048
    assert "future_policy_field" not in identity
    spec = SimpleNamespace(resources=NativeResources(), runtime="podman", assets=[])
    assert contracts.runtime_identity(spec, {})["execution_enforcement"] == identity
    with pytest.raises(ValueError, match="requires the Podman"):
        contracts.execution_identity("docker")


@pytest.mark.parametrize("target", ["config", "podman", "negative_control"])
def test_changed_policy_bytes_rejected(policy, target):
    path, config, _ = policy
    changed = path if target == "config" else Path(config[target])
    changed.write_text(changed.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        contracts.execution_identity("podman")


def write_receipt(policy, identity, *, receipt_name=None, **changes):
    _, config, _ = policy
    name = receipt_name or "scientisttwo-native-" + "a" * 18
    value = {
        "schema_version": 1, "container_name": name,
        "config_sha256": identity["config_sha256"], "guard_sha256": config["guard_sha256"],
        "job_id": config["job_id"], "node": config["node"], "uid": config["uid"],
        "step_id": "5", "started_at": 110.0, "completed_at": 120.0,
        "admission_id": "c" * 32, "status": "completed",
        "guard": {"status": "verified", "gpu_uuid": None},
        "description": {"gpu_uuid": None}, "cleanup": {"confirmed_absent": True},
        "parser_errors": [], **changes,
    }
    if isinstance(value.get("guard"), dict):
        value["guard"]["description_sha256"] = hashlib.sha256(
            json.dumps(value["description"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    path = Path(config["receipt_dir"]) / (name + ".json")
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return name, path


def test_receipt_copied_to_immutable_campaign_artifact(policy, tmp_path):
    identity = contracts.execution_identity("podman")
    name, path = write_receipt(policy, identity)
    raw = path.read_bytes()
    store = Store(tmp_path / "campaign")
    try:
        sealed = contracts.seal_execution_receipt(store, name, identity, started_at=100, wait_seconds=0)
        path.write_text("{}")
        assert sealed["sha256"] == hashlib.sha256(raw).hexdigest()
        assert store.read(sealed["evidence_ref"]) == raw
        assert sealed["step_id"] == "5"
        assert store.verify() == []
    finally:
        store.close()


@pytest.mark.parametrize("changes,reason", [
    ({"started_at": 90}, "unavailable"),
    ({"completed_at": None}, "unavailable"),
    ({"container_name": "another"}, "does not match"),
    ({"config_sha256": "f" * 64}, "does not match"),
    ({"guard_sha256": "f" * 64}, "does not match"),
    ({"job_id": "99"}, "does not match"),
    ({"admission_id": None}, "does not match"),
    ({"cleanup": {"confirmed_absent": False}}, "cleanup failed"),
    ({"guard": None}, "cleanup failed"),
    ({"guard": {"status": "verified", "gpu_uuid": "other"}}, "GPU differs"),
])
def test_receipt_must_be_fresh_bound_and_enforced(policy, tmp_path, changes, reason):
    identity = contracts.execution_identity("podman")
    name, _ = write_receipt(policy, identity, **changes)
    store = Store(tmp_path / "campaign")
    try:
        with pytest.raises(ValueError, match=reason):
            contracts.seal_execution_receipt(store, name, identity, started_at=100, wait_seconds=0)
    finally:
        store.close()


def test_receipt_gpu_matches_stage_and_authorized_uuid(policy, tmp_path):
    identity = contracts.execution_identity("podman")
    gpu = identity["allocation"]["gpu_uuid"]
    name, _ = write_receipt(policy, identity, guard={"status": "verified", "gpu_uuid": gpu},
                            description={"gpu_uuid": gpu})
    store = Store(tmp_path / "campaign")
    try:
        sealed = contracts.seal_execution_receipt(
            store, name, identity, started_at=100, expected_gpu_uuid=gpu, wait_seconds=0)
        assert sealed["gpu_uuid"] == gpu
    finally:
        store.close()


def test_host_timeout_rejected_after_sealing_even_with_verified_guard_and_cleanup(policy, tmp_path):
    identity = contracts.execution_identity("podman")
    name, path = write_receipt(policy, identity, timed_out=True, returncode=124)
    raw = path.read_bytes()
    store = Store(tmp_path / "campaign")
    try:
        with pytest.raises(ValueError, match="trusted host timeout"):
            contracts.seal_execution_receipt(store, name, identity, started_at=100, wait_seconds=0)
        rows = list(store.db.execute("SELECT path FROM artifacts WHERE kind='execution-enforcement'"))
        assert len(rows) == 1 and store.read(rows[0]["path"]) == raw
        assert store.verify() == []
    finally:
        store.close()


def test_normal_forced_timing_cleanup_remains_accepted(policy, tmp_path):
    identity = contracts.execution_identity("podman")
    name, _ = write_receipt(policy, identity, timed_out=False, returncode=137)
    store = Store(tmp_path / "campaign")
    try:
        proof = contracts.seal_execution_receipt(store, name, identity, started_at=100, wait_seconds=0)
        assert proof["status"] == "verified"
        assert proof["returncode"] == 137 and proof["timed_out"] is False
    finally:
        store.close()


@pytest.mark.parametrize("outer_returncode", [0, 124])
def test_native_stage_cannot_accept_a_trusted_host_timeout(policy, tmp_path, monkeypatch, outer_returncode):
    """Synthetic event disagreement also fails closed instead of trusting stdout."""
    policy[1]["expires_at"] = time.time() + 3600
    policy[0].write_text(json.dumps(policy[1]))
    monkeypatch.setenv(contracts.EXECUTION_CONFIG_SHA_ENV, sha(policy[0]))
    identity = contracts.execution_identity("podman")
    runner = NativeRunner.__new__(NativeRunner)
    runner.spec = SimpleNamespace(runtime="podman", resources=NativeResources())
    runner.prefix = ["podman"]
    runner.store = Store(tmp_path / "native")
    def execute(argv, _timeout, **_kwargs):
        name = argv[argv.index("--name") + 1]
        started = time.time()
        write_receipt(policy, identity, receipt_name=name, started_at=started,
                      completed_at=started + 0.001, timed_out=True, returncode=124)
        return ProcessResult(outer_returncode, '{"synthetic_valid_output":true}', "", 0.01)
    runner.executor = execute
    try:
        result, ref = runner._launch("execute", "sha256:" + "a" * 64,
                                     {"command": ["python", "train.py"]}, [], 1)
        receipt = runner.store.json(ref)
        assert result.code == (outer_returncode or 125)
        assert receipt["returncode"] != 0 and receipt["execution_enforcement"] is None
        assert "trusted host timeout" in receipt["execution_enforcement_error"]
        proof_refs = list(runner.store.db.execute(
            "SELECT path FROM artifacts WHERE kind='execution-enforcement'"))
        assert len(proof_refs) == 1
        assert runner.store.json(proof_refs[0]["path"])["timed_out"] is True
        assert runner.store.verify() == []
    finally:
        runner.close()


def test_effective_timeout_respects_slurm_rounding_and_short_caller_budget():
    identity = {"allocation": {"expires_at": 1000}}
    assert contracts.effective_execution_timeout(900, identity, now=156) == 765
    assert contracts.effective_execution_timeout(60, identity, now=156) == 60
    assert contracts.effective_execution_timeout(900) == 900
    with pytest.raises(ValueError, match="Insufficient time"):
        contracts.effective_execution_timeout(900, identity, now=930)


def test_native_launch_caps_long_timeout_to_approved_window(policy, tmp_path, monkeypatch):
    policy[1]["expires_at"] = time.time() + 150
    policy[0].write_text(json.dumps(policy[1]))
    monkeypatch.setenv(contracts.EXECUTION_CONFIG_SHA_ENV, sha(policy[0]))
    identity = contracts.execution_identity("podman")
    runner = NativeRunner.__new__(NativeRunner)
    runner.spec = SimpleNamespace(runtime="podman", resources=NativeResources(seconds=900))
    runner.prefix = ["podman"]
    runner.store = Store(tmp_path / "native")
    calls = []
    def execute(argv, timeout, **_kwargs):
        calls.append((argv, timeout))
        name = argv[argv.index("--name") + 1]
        started = time.time()
        write_receipt(policy, identity, receipt_name=name, started_at=started,
                      completed_at=started + 0.001, timed_out=False, returncode=0)
        return ProcessResult(0, "", "", 0.01)
    runner.executor = execute
    try:
        result, _ = runner._launch("execute", "sha256:" + "a" * 64,
                                   {"command": ["python", "train.py"]}, [], 900)
        assert result.code == 0
        argv, timeout = calls[0]
        assert timeout == 105
        assert argv[argv.index("--label") + 1] == "scientisttwo.timeout-seconds=105.0"
    finally:
        runner.close()


@pytest.mark.parametrize("host_timeout", [False, True])
def test_valid_final_timing_response_then_host_timeout_cannot_complete_pair(
        policy, tmp_path, monkeypatch, host_timeout):
    """Synthetic RPC responses exercise real paired-control and receipt sealing."""
    identity = contracts.execution_identity("podman")
    transports = []
    monkeypatch.setattr(timing_vectorization, "verify_tree", lambda *_: None)
    monkeypatch.setattr(timing_vectorization, "expected_outputs", lambda *_: {})
    monkeypatch.setattr(timing_vectorization, "vectorization_oracle",
                        lambda *_: lambda *_args: {"valid": True, "synthetic_test": True})
    class SyntheticTransport(ContainerTimingTransport):
        def __init__(self, **options):
            self.arm = options["control"].parent.name
            self.name = "scientisttwo-timing-" + digest(self.arm.encode())[:18]
            self.enforcement = identity
            self.enforcement_started = 100
            self.expected_gpu_uuid = None
            self.closed = False
            self.argv = ["synthetic-test-only"]
            self.started = time.monotonic()
            self.transport = SimpleNamespace(startup_seconds=0.01)
            self.process = SimpleNamespace(poll=lambda: 137)
            self.stderr_bytes, self.stderr_truncated, self.limit_exceeded = 0, False, None
            self.responses = 0
            options["control"].mkdir(parents=True)
            self.log_path = options["control"] / "worker.log"
            self.log_path.write_text("synthetic RPC workload log")
            transports.append(self)

        def request(self, payload, *, timeout):
            assert payload == {"workload": "all_cases"} and timeout > 0
            self.responses += 1
            return {"predictions": "synthetic-valid-response-" + str(self.responses)}

        def close(self):
            assert self.responses == 15  # Five warmups plus ten measured requests.
            self.closed = True
            timed_out = self.arm == "candidate" and host_timeout
            write_receipt(policy, identity, receipt_name=self.name, timed_out=timed_out,
                          returncode=124 if timed_out else 137)

    monkeypatch.setattr(timing_vectorization, "ContainerTimingTransport", SyntheticTransport)
    spec = SimpleNamespace(runtime="podman", context=None,
                           resources=SimpleNamespace(seconds=60, gpus=None))
    prepared = {"public": str(tmp_path / "public"), "files": {}, "cases": []}
    output = tmp_path / "pair"
    store = Store(tmp_path / "paired-campaign")
    try:
        def run():
            return timing_vectorization.run_vectorization_pair(
                store, spec, "sha256:" + "a" * 64, tmp_path / "baseline",
                tmp_path / "candidate", prepared, output, timeout=60)
        if host_timeout:
            with pytest.raises(RuntimeError, match="enforcement or cleanup failed"):
                run()
        else:
            report, receipts = run()
            assert len(report.samples) == 30 and set(receipts) == {"baseline", "candidate"}
        assert all(transport.responses == 15 and transport.closed for transport in transports)
        attempt = store.json(store.get("paired_timing/" + digest(str(output).encode())))
        assert attempt["status"] == ("failed" if host_timeout else "completed")
        assert set(attempt["receipts"]) == {"baseline", "candidate"}
        if host_timeout:
            assert "trusted host timeout" in attempt["retention_errors"]["candidate/receipt"]
        else:
            assert attempt["retention_errors"] == {}
        raw_proofs = [store.json(row["path"]) for row in store.db.execute(
            "SELECT path FROM artifacts WHERE kind='execution-enforcement'")]
        assert len(raw_proofs) == 2
        assert sum(row.get("timed_out") is True for row in raw_proofs) == int(host_timeout)
        assert store.verify() == []
    finally:
        store.close()


def test_wrapper_drift_during_stage_rejected(policy, tmp_path):
    identity = contracts.execution_identity("podman")
    name, _ = write_receipt(policy, identity)
    policy[2].write_text("changed wrapper")
    store = Store(tmp_path / "campaign")
    try:
        with pytest.raises(ValueError, match="changed while"):
            contracts.seal_execution_receipt(store, name, identity, started_at=100, wait_seconds=0)
    finally:
        store.close()


def nvidia_policy(policy, monkeypatch):
    path, config, _ = policy
    for name in ("nvidia_container_runtime", "nvidia_cdi_hook", "nvidia_ctk"):
        target = path.parent / name
        target.write_text("binary:" + name)
        config[name] = str(target)
        config[name + "_sha256"] = sha(target)
    cdi = path.parent / "gpu.json"
    cdi.write_text(json.dumps({
        "containerEdits": {"hooks": [{"path": config["nvidia_cdi_hook"]}]},
        "devices": [{"name": config["gpu_uuid"], "containerEdits": {
            "hooks": [{"path": config["nvidia_cdi_hook"]}]}}],
    }))
    runtime = path.parent / "nvidia.toml"
    runtime.write_text('[nvidia-ctk]\npath = ' + json.dumps(config["nvidia_ctk"]) + "\n")
    for name, target in (("cdi_spec", cdi), ("nvidia_config", runtime)):
        config[name] = str(target)
        config[name + "_sha256"] = sha(target)
    path.write_text(json.dumps(config))
    monkeypatch.setenv(contracts.EXECUTION_CONFIG_SHA_ENV, sha(path))
    return config


def test_nvidia_closure_pins_actual_hook_and_ctk_binaries(policy, monkeypatch):
    config = nvidia_policy(policy, monkeypatch)
    identity = contracts.execution_identity("podman")
    for name in ("nvidia_ctk", "nvidia_cdi_hook"):
        assert identity["dependencies"][name]["sha256"] == config[name + "_sha256"]
    Path(config["nvidia_cdi_hook"]).write_text("changed")
    with pytest.raises(ValueError, match="dependency changed: nvidia_cdi_hook"):
        contracts.execution_identity("podman")


@pytest.mark.parametrize("missing", ["nvidia_ctk", "nvidia_cdi_hook"])
def test_nvidia_closure_rejects_unpinned_indirect_binaries(policy, monkeypatch, missing):
    config = nvidia_policy(policy, monkeypatch)
    del config[missing]
    del config[missing + "_sha256"]
    policy[0].write_text(json.dumps(config))
    monkeypatch.setenv(contracts.EXECUTION_CONFIG_SHA_ENV, sha(policy[0]))
    with pytest.raises(ValueError, match="explicitly hash-pinned"):
        contracts.execution_identity("podman")
