"""Synthetic contract traces; real node negative controls remain a release gate."""

import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from scientisttwo import slurm_execution as execution


GPU = "GPU-4f81c4b7-3011-b470-5205-7afcfb47f569"
IMAGE = "sha256:" + "a" * 64
NAME = "scientisttwo-native-" + "b" * 18


@pytest.fixture
def config(tmp_path):
    for filename in ("podman", "srun", "python"):
        (tmp_path / filename).write_text("synthetic executable fixture\n")
    receipt = tmp_path / "receipts"
    receipt.mkdir(mode=0o700)
    validation = tmp_path / "negative.json"
    validation.write_text(json.dumps({
        "node": "synthetic-node", "uid": 1234,
        "podman_sha256": execution.sha256(tmp_path / "podman"),
        "cpu_enforced": True, "memory_enforced": True,
        "nproc_enforced": True, "timeout_cleanup_verified": True,
        "synthetic_test_trace": True,
    }))
    data = {
        "schema_version": 1, "job_id": "3920", "node": "synthetic-node", "uid": 1234,
        "podman": str(tmp_path / "podman"),
        "podman_sha256": execution.sha256(tmp_path / "podman"),
        "guard_sha256": execution.sha256(execution.__file__),
        "srun": str(tmp_path / "srun"), "python": str(tmp_path / "python"),
        "receipt_dir": str(receipt),
        "max_cpus": 8, "max_memory_mb": 65536, "max_seconds": 3600,
        "controller_memory_mb": 512,
        "expires_at": 9999999999, "gpu_uuid": GPU, "gpu_gres": "gpu:a100:8",
        "negative_control": str(validation),
        "negative_control_sha256": execution.sha256(validation),
    }
    file = tmp_path / "config.json"
    file.write_text(json.dumps(data))
    return execution.load_config(file, execution.sha256(file), check_identity=False)


def command(*extra):
    return [
        "run", "--rm", "--pull=never", "--name", NAME, "--network=none",
        "--read-only", "--cap-drop=ALL", "--security-opt", "no-new-privileges",
        "--memory=1024m", "--cpus=2", "--pids-limit=64",
        "--user", "65534:65534", "--shm-size=64m",
        "--tmpfs", "/tmp:rw,nosuid,size=64m", "-w", "/tmp",
        "--label", "scientisttwo.timeout-seconds=45",
        *extra, IMAGE, "python", "/bridge/bridge.py",
    ]


def test_plan_replaces_ignored_limits_with_slurm_and_kernel_nproc(config):
    plan = execution.parse_run(command(), config)
    argv = plan["container_argv"]
    assert not any(item.startswith(("--cpus=", "--memory=", "--pids-limit=")) for item in argv)
    assert "--ulimit=nproc=64:64" in argv
    assert "--userns=keep-id:uid=65534,gid=65534" in argv
    assert argv[-2:] == ["python", "/bridge/bridge.py"]
    launch = execution.slurm_argv(plan, config)
    for option in (
        "--jobid=3920", "--overlap", "--exact", "--cpus-per-task=2",
        "--mem=1024M", "--gres=none", "--export=NONE", "--time=1",
    ):
        assert option in launch
    assert "--" in launch


def test_gpu_requires_single_full_uuid_and_step_requests_explicit_authorized_gres(config):
    plan = execution.parse_run(command("--device", "nvidia.com/gpu=" + GPU), config)
    assert plan["description"]["gpu_uuid"] == GPU
    assert "--gres=gpu:a100:8" in execution.slurm_argv(plan, config)
    assert "CUDA_VISIBLE_DEVICES=" + GPU in plan["container_argv"]
    for device in ("all", "4", GPU + ",other"):
        with pytest.raises(ValueError, match="single authorized"):
            execution.parse_run(command("--device", "nvidia.com/gpu=" + device), config)


def cpu_config(config):
    data = {key: value for key, value in config.items() if not key.startswith("_")}
    data.update(gpu_uuid=None, gpu_gres="none", max_cpus=2, max_memory_mb=8192,
                controller_memory_mb=512)
    source = Path(config["_path"])
    source.write_text(json.dumps(data))
    return execution.load_config(source, execution.sha256(source), check_identity=False)


def test_cpu_only_policy_requires_no_gpu_and_launches_no_gpu_step(config):
    configured = cpu_config(config)
    plan = execution.parse_run(command(), configured)
    assert configured["gpu_uuid"] is None and configured["gpu_gres"] == "none"
    assert plan["description"]["gpu_uuid"] is None
    assert "--gres=none" in execution.slurm_argv(plan, configured)
    assert "CUDA_VISIBLE_DEVICES=" in plan["container_argv"]
    assert "NVIDIA_VISIBLE_DEVICES=void" in plan["container_argv"]
    assert "--device" not in plan["container_argv"]


@pytest.mark.parametrize("device", [GPU, "all", "0"])
def test_cpu_only_policy_rejects_every_gpu_request(config, device):
    with pytest.raises(ValueError, match="single authorized GPU"):
        execution.parse_run(command("--device", "nvidia.com/gpu=" + device), cpu_config(config))


@pytest.mark.parametrize("gpu_uuid,gres", [
    (None, "gpu:a100:8"), (None, "all"),
    (GPU, "none"), (GPU, "gpu:a100:0"), (GPU, "gpu:a100:-1"), (GPU, "all"),
])
def test_cpu_and_gpu_policy_declarations_cannot_be_mixed(config, gpu_uuid, gres):
    data = {key: value for key, value in config.items() if not key.startswith("_")}
    data.update(gpu_uuid=gpu_uuid, gpu_gres=gres)
    source = Path(config["_path"])
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        execution.load_config(source, execution.sha256(source), check_identity=False)


def test_cpu_pair_fits_two_cpu_eight_gib_envelope_with_controller_reserve(config):
    configured = cpu_config(config)
    plans = []
    for suffix in ("b", "c", "d"):
        argv = command()
        argv[argv.index("--cpus=2")] = "--cpus=1"
        argv[argv.index("--memory=1024m")] = "--memory=2048m"
        argv[argv.index("--name") + 1] = "scientisttwo-timing-" + suffix * 18
        plans.append(execution.parse_run(argv, configured))
    execution.acquire_admission(plans[0], configured)
    execution.acquire_admission(plans[1], configured)
    with execution.admission_ledger(configured) as ledger:
        assert sum(row["cpus"] for row in ledger["active"].values()) == 2
        assert sum(row["memory_mb"] for row in ledger["active"].values()) == 4096
        assert all(row["gpu_uuid"] is None for row in ledger["active"].values())
    with pytest.raises(ValueError, match="CPU admission budget"):
        execution.acquire_admission(plans[2], configured)


@pytest.mark.parametrize("option,value", [
    ("--cpus=2", "--cpus=0.5"),
    ("--cpus=2", "--cpus=9"),
    ("--memory=1024m", "--memory=65537m"),
    ("--memory=1024m", "--memory=1g"),
    ("--pids-limit=64", "--pids-limit=0"),
    ("--network=none", "--network=host"),
    ("--read-only", "--privileged"),
    ("--pull=never", "--pull=always"),
    ("--cap-drop=ALL", "--cap-drop=none"),
])
def test_invalid_and_over_budget_runs_fail_closed(config, option, value):
    argv = command()
    argv[argv.index(option)] = value
    with pytest.raises(ValueError):
        execution.parse_run(argv, config)


def test_missing_timeout_ambiguous_options_and_guard_mount_fail_closed(config):
    for args in (
        command("--cpus", "2"),
        command("-v", "/tmp/file:/scientisttwo-guard.py:rw"),
        command("-e", "CUDA_VISIBLE_DEVICES=0"),
        command("--security-opt", "label=disable"),
    ):
        with pytest.raises(ValueError):
            execution.parse_run(args, config)
    args = command()
    label = args.index("--label")
    del args[label:label + 2]
    with pytest.raises(ValueError, match="--label"):
        execution.parse_run(args, config)


def test_expiring_window_does_not_silently_reduce_experiment_budget(config):
    with pytest.raises(ValueError, match="Insufficient time"):
        execution.parse_run(command(), config, now=config["expires_at"] - 60)


def test_receipts_are_outside_candidate_writable_mounts(config):
    with pytest.raises(ValueError, match="receipts writable"):
        execution.parse_run(command("-v", config["receipt_dir"] + ":/outputs:rw"), config)


def test_task_local_nvidia_runtime_is_used_only_for_gpu(config, tmp_path):
    config["nvidia_container_runtime"] = str(tmp_path / "nvidia-container-runtime")
    config["nvidia_config"] = str(tmp_path / "nvidia.toml")
    plan = execution.parse_run(command("--device", "nvidia.com/gpu=" + GPU), config)
    argv = plan["container_argv"]
    assert argv[:4] == [config["podman"], "--runtime", config["nvidia_container_runtime"], "run"]
    assert "--device" not in argv
    assert "NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=" + GPU in argv
    cpu = execution.parse_run(command(), config)["container_argv"]
    assert cpu[:2] == [config["podman"], "run"]


def test_nvidia_helper_binary_changes_fail_before_launch(config, tmp_path):
    data = {key: value for key, value in config.items() if not key.startswith("_")}
    for name in ("nvidia_container_runtime", "nvidia_config", "cdi_spec", "nvidia_cdi_hook", "nvidia_ctk"):
        path = tmp_path / name
        path.write_text("synthetic frozen helper\n")
        data[name], data[name + "_sha256"] = str(path), execution.sha256(path)
    source = Path(config["_path"])
    source.write_text(json.dumps(data))
    execution.load_config(source, execution.sha256(source), check_identity=False)
    Path(data["nvidia_cdi_hook"]).write_text("modified helper\n")
    with pytest.raises(ValueError, match="dependency changed: nvidia_cdi_hook"):
        execution.load_config(source, execution.sha256(source), check_identity=False)


def test_partial_nvidia_helper_pin_is_rejected(config, tmp_path):
    data = {key: value for key, value in config.items() if not key.startswith("_")}
    data["nvidia_ctk"] = str(tmp_path / "tool")
    source = Path(config["_path"])
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="helper paths and hashes"):
        execution.load_config(source, execution.sha256(source), check_identity=False)


def test_runtime_and_negative_controls_are_frozen(config):
    Path(config["podman"]).write_text("changed\n")
    with pytest.raises(ValueError, match="runtime wrapper changed"):
        execution.load_config(config["_path"], config["_sha256"], check_identity=False)


def test_missing_negative_controls_never_enable_adapter(config):
    path = Path(config["negative_control"])
    evidence = json.loads(path.read_text())
    evidence["nproc_enforced"] = False
    path.write_text(json.dumps(evidence))
    data = {k: v for k, v in config.items() if not k.startswith("_")}
    data["negative_control_sha256"] = execution.sha256(path)
    source = Path(config["_path"])
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not all passed"):
        execution.load_config(source, execution.sha256(source), check_identity=False)


def test_config_cannot_be_replaced_without_new_bound_hash(config):
    with pytest.raises(ValueError, match="identity mismatch"):
        execution.load_config(config["_path"], "0" * 64, check_identity=False)


def test_shared_step_limit_is_not_smaller_experiment_limit():
    observed = {"cpuset": list(range(8)), "affinity": [0],
                "memory_limit_bytes": 64 * 1024**3}
    with pytest.raises(ValueError, match="cpuset"):
        execution.verify_resources(observed, 1, 1024)
    observed["cpuset"] = [0]
    with pytest.raises(ValueError, match="memory"):
        execution.verify_resources(observed, 1, 1024)
    observed["memory_limit_bytes"] = 1024**3
    execution.verify_resources(observed, 1, 1024)


def security():
    return {"uid": 65534, "euid": 65534, "cap_effective": 0, "cap_permitted": 0,
            "no_new_privileges": 1, "nproc_soft": 64, "nproc_hard": 64,
            "uid_map": "0 100000 65534\n65534 1234 1\n"}


@pytest.mark.parametrize("key,value", [
    ("uid", 0), ("euid", 0), ("cap_effective", 1), ("cap_permitted", 1),
    ("no_new_privileges", 0), ("nproc_soft", -1), ("nproc_hard", -1),
    ("nproc_hard", 128), ("uid_map", "0 0 4294967295\n"),
])
def test_nproc_requires_hard_limit_and_no_privilege_exemption(key, value):
    observed = security()
    observed[key] = value
    with pytest.raises(ValueError):
        execution.verify_security(observed, 64)


def test_nproc_security_pass_is_explicitly_uid_scoped():
    execution.verify_security(security(), 64)


def fixture_cgroups(tmp_path, monkeypatch, *, unified=False):
    proc = tmp_path / "proc"
    proc.mkdir()
    root = tmp_path / "cgroup"
    root.mkdir()
    leaf = root / "job_3920" / "step_10" / "task_0"
    leaf.mkdir(parents=True)
    mounts = []
    controllers = ["unified"] if unified else ["memory", "cpuset"]
    membership = "/job_3920/step_10/task_0"
    for controller in controllers:
        index = len(mounts) + 1
        kind = "cgroup2" if unified else "cgroup"
        mount = f"{index} 0 0:{index} / {root} rw - {kind} cgroup rw,{controller}"
        mounts.append(mount)
    (proc / "mountinfo").write_text("\n".join(mounts))
    (proc / "cgroup").write_text("\n".join(
        f"{i}:{'' if unified else controller}:{membership}"
        for i, controller in enumerate(controllers)
    ))
    memory_name = "memory.max" if unified else "memory.limit_in_bytes"
    cpu_name = "cpuset.cpus.effective" if unified else "cpuset.cpus"
    (leaf / memory_name).write_text("max" if unified else str(2**63 - 4096))
    (leaf.parent / memory_name).write_text(str(1024**3))
    (root / memory_name).write_text(str(64 * 1024**3))
    (leaf / cpu_name).write_text("")
    (leaf.parent / cpu_name).write_text("4-5")
    (root / cpu_name).write_text("0-63")
    if not unified:
        (leaf / "memory.use_hierarchy").write_text("1")
    monkeypatch.setattr(execution.os, "sched_getaffinity", lambda _: {4, 5}, raising=False)
    return proc, leaf


@pytest.mark.parametrize("unified", [False, True])
def test_cgroup_observation_includes_enclosing_step_limits(tmp_path, monkeypatch, unified):
    proc, leaf = fixture_cgroups(tmp_path, monkeypatch, unified=unified)
    actual = execution.resource_observation(proc)
    assert actual["memory_limit_bytes"] == 1024**3
    assert actual["cpuset"] == [4, 5]
    assert actual["cgroup_version"] == (2 if unified else 1)
    execution.verify_resources(actual, 2, 1024)
    assert any(str(leaf.parent) in item["path"] for item in actual["memory_limits"])


def test_disabled_v1_hierarchy_does_not_claim_parent_cap(tmp_path, monkeypatch):
    proc, leaf = fixture_cgroups(tmp_path, monkeypatch)
    (leaf / "memory.use_hierarchy").write_text("0")
    with pytest.raises(ValueError, match="finite hierarchical"):
        execution.resource_observation(proc)


def narrow_cgroups(tmp_path, monkeypatch):
    proc, leaf = fixture_cgroups(tmp_path, monkeypatch)
    (proc / "mountinfo").write_text("\n".join(
        f"{index} 0 0:{index} /job_3920/step_10/task_0 {leaf} ro - cgroup cgroup rw,{controller}"
        for index, controller in enumerate(("memory", "cpuset"), start=1)
    ))
    (leaf / "cpuset.cpus").write_text("4")
    monkeypatch.setattr(execution.os, "sched_getaffinity", lambda _: {4}, raising=False)
    return proc, leaf


def test_private_mount_uses_kernel_hierarchical_memory_stat(tmp_path, monkeypatch):
    proc, leaf = narrow_cgroups(tmp_path, monkeypatch)
    (leaf / "memory.stat").write_text("rss 0\nhierarchical_memory_limit 536870912\n")
    observed = execution.resource_observation(proc)
    assert observed["memory_limit_bytes"] == 512 * 1024**2
    assert observed["memory_limits"] == [{
        "path": str(leaf / "memory.stat"),
        "field": "hierarchical_memory_limit",
        "bytes": 512 * 1024**2,
    }]
    execution.verify_resources(observed, 1, 512)


@pytest.mark.parametrize("content", [
    "rss 1\n", "hierarchical_memory_limit 0\n",
    "hierarchical_memory_limit not-a-number\n",
    "hierarchical_memory_limit 536870912\nhierarchical_memory_limit 536870912\n",
    "hierarchical_memory_limit 9223372036854771712\n",
])
def test_invalid_or_missing_private_hierarchy_limit_fails_closed(tmp_path, monkeypatch, content):
    proc, leaf = narrow_cgroups(tmp_path, monkeypatch)
    (leaf / "memory.stat").write_text(content)
    with pytest.raises(ValueError):
        execution.resource_observation(proc)


def test_hierarchy_stat_not_used_when_hierarchy_disabled(tmp_path, monkeypatch):
    proc, leaf = narrow_cgroups(tmp_path, monkeypatch)
    (leaf / "memory.use_hierarchy").write_text("0")
    (leaf / "memory.stat").write_text("hierarchical_memory_limit 536870912\n")
    with pytest.raises(ValueError, match="finite hierarchical"):
        execution.resource_observation(proc)


def test_memory_receipt_distinguishes_physical_cap_from_unbounded_memsw(tmp_path, monkeypatch):
    proc, leaf = narrow_cgroups(tmp_path, monkeypatch)
    (leaf / "memory.stat").write_text(
        "hierarchical_memory_limit 536870912\nhierarchical_memsw_limit 9223372036854771712\n"
    )
    observed = execution.resource_observation(proc)
    assert observed["memory_budget_scope"] == "physical_ram"
    assert observed["memory_limit_bytes"] == 536870912
    assert observed["swap"]["scope"] == "memory_and_swap_combined"
    assert observed["swap"]["status"] == "unbounded"
    assert observed["swap"]["limit_bytes"] is None
    assert observed["swap"]["policy"] == "inherited_not_enforced_by_adapter"


def test_missing_swap_metadata_is_unknown_not_zero(tmp_path, monkeypatch):
    proc, _ = fixture_cgroups(tmp_path, monkeypatch)
    observed = execution.resource_observation(proc)
    assert observed["swap"]["status"] == "unknown"
    assert observed["swap"]["limit_bytes"] is None
    assert observed["swap"]["sources"] == []


def test_unified_swap_zero_is_observed_disabled_bound(tmp_path, monkeypatch):
    proc, leaf = fixture_cgroups(tmp_path, monkeypatch, unified=True)
    (leaf / "memory.swap.max").write_text("0")
    observed = execution.resource_observation(proc)
    assert observed["swap"]["scope"] == "swap_only"
    assert observed["swap"]["status"] == "bounded"
    assert observed["swap"]["limit_bytes"] == 0


def test_guard_cannot_exec_changed_command(monkeypatch):
    monkeypatch.setattr(execution.os, "execvp", lambda *_: pytest.fail("Must not execute"))
    with pytest.raises(ValueError, match="differs"):
        execution.container_guard({"command": ["python", "trusted.py"]}, ["python", "other.py"])


def test_cleanup_cannot_name_another_container(config, monkeypatch):
    monkeypatch.setattr(execution, "load_config", lambda *_, **__: config)
    with pytest.raises(ValueError, match="outside"):
        execution.main(["shim", "--config", config["_path"], "--config-sha256",
                        config["_sha256"], "--", "rm", "--force", "coworker"])


def test_explicit_separator_prevents_workload_adapter_option_override(config, monkeypatch, capsys):
    monkeypatch.setattr(execution, "load_config", lambda *_, **__: config)
    args = command()
    args.extend(["--config", "/candidate/config.json"])
    execution.main(["plan", "--config", config["_path"], "--config-sha256",
                    config["_sha256"], "--", *args])
    actual = json.loads(capsys.readouterr().out)
    assert actual["plan"]["description"]["command"][-2:] == ["--config", "/candidate/config.json"]


def setup_worker(config, monkeypatch, *, emit_guard=True, seconds=1, cleanup=True):
    plan = execution.parse_run(command(), config)
    plan["description"]["seconds"] = seconds
    evidence = {
        "status": "verified",
        "description_sha256": execution.hashlib.sha256(
            execution._json_bytes(plan["description"])
        ).hexdigest(),
    }
    source = "import sys,time\n"
    if emit_guard:
        source += "print(" + repr(execution.RECEIPT_PREFIX + json.dumps(evidence)) + \
                  ",file=sys.stderr,flush=True)\n"
    source += "time.sleep(" + repr(2 if seconds < 1 else 0) + ")\n"
    # Explicitly synthetic lifecycle transport: this is a trusted test helper,
    # not generated research code or evidence of a working Slurm/container stack.
    plan["container_argv"] = [sys.executable, "-c", source]
    monkeypatch.setenv("SLURM_JOB_ID", "3920")
    monkeypatch.setenv("SLURM_STEP_ID", "100")
    monkeypatch.setattr(execution, "resource_observation", lambda: {
        "cpuset": [0, 1], "affinity": [0, 1], "memory_limit_bytes": 1024**3,
        "cgroup_membership": "1:memory:/slurm/uid_1234/job_3920/step_100/task_0",
    })
    monkeypatch.setattr(execution, "cleanup_container", lambda *_: {
        "confirmed_absent": cleanup, "remove_returncode": 0, "exists_returncode": 1 if cleanup else 0,
    })
    execution.acquire_admission(plan, config)
    return plan


def test_worker_preserves_external_receipt_and_verified_cleanup(config, monkeypatch):
    plan = setup_worker(config, monkeypatch)
    assert execution.worker(plan, config) == 0
    receipt = json.loads((Path(config["receipt_dir"]) / (NAME + ".json")).read_text())
    assert receipt["status"] == "completed"
    assert receipt["guard"]["status"] == "verified"
    assert receipt["cleanup"]["confirmed_absent"]
    assert receipt["job_id"] == "3920"
    assert receipt["step_id"] == "100"
    assert receipt["config_sha256"] == config["_sha256"]


def test_worker_success_without_guard_is_rejected(config, monkeypatch):
    plan = setup_worker(config, monkeypatch, emit_guard=False)
    assert execution.worker(plan, config) == 125
    receipt = json.loads((Path(config["receipt_dir"]) / (NAME + ".json")).read_text())
    assert receipt["status"] == "enforcement_failed"
    assert receipt["guard"] is None


def test_worker_unknown_cleanup_is_rejected(config, monkeypatch):
    plan = setup_worker(config, monkeypatch, cleanup=False)
    assert execution.worker(plan, config) == 125


def test_worker_timeout_stops_owned_process_and_records_failure(config, monkeypatch):
    plan = setup_worker(config, monkeypatch, seconds=0.15)
    assert execution.worker(plan, config) == 124
    receipt = json.loads((Path(config["receipt_dir"]) / (NAME + ".json")).read_text())
    assert receipt["timed_out"] is True
    assert receipt["returncode"] == 124
    assert receipt["cleanup"]["confirmed_absent"]


def other_plan(config, digit, *, cpus=2, memory_mb=1024, gpu=False):
    args = command("--device", "nvidia.com/gpu=" + GPU) if gpu else command()
    args[args.index(NAME)] = "scientisttwo-native-" + digit * 18
    args[args.index("--cpus=2")] = "--cpus=" + str(cpus)
    args[args.index("--memory=1024m")] = "--memory=" + str(memory_mb) + "m"
    return execution.parse_run(args, config)


def complete_lease(config, plan, *, confirmed=True, token=None):
    path = Path(config["receipt_dir"]) / (plan["name"] + ".json")
    path.write_text(json.dumps({
        "admission_id": token or plan["admission_id"],
        "config_sha256": config["_sha256"],
        "completed_at": 100,
        "cleanup": {"confirmed_absent": confirmed},
    }))


def test_two_small_timing_arms_admitted_with_controller_reservation(config):
    first, second = other_plan(config, "1"), other_plan(config, "2")
    execution.acquire_admission(first, config)
    execution.acquire_admission(second, config)
    assert second["admission_snapshot"]["previously_charged"] == {
        "cpus": 2, "memory_mb": 1024, "gpu_runs": 0,
    }
    assert second["admission_snapshot"]["controller_memory_mb"] == 512


def test_aggregate_cpu_limits_are_checked_before_srun(config):
    execution.acquire_admission(other_plan(config, "1", cpus=5), config)
    with pytest.raises(ValueError, match="CPU admission"):
        execution.acquire_admission(other_plan(config, "2", cpus=4), config)


def test_aggregate_memory_reserves_controller_headroom(config):
    execution.acquire_admission(other_plan(config, "1", memory_mb=32768), config)
    with pytest.raises(ValueError, match="memory admission"):
        execution.acquire_admission(other_plan(config, "2", memory_mb=32768), config)


def test_gpu_workloads_are_serial_even_when_cpu_and_memory_fit(config):
    execution.acquire_admission(other_plan(config, "1", gpu=True), config)
    with pytest.raises(ValueError, match="Another GPU"):
        execution.acquire_admission(other_plan(config, "2", gpu=True), config)


def test_unknown_dead_owner_remains_charged(config, monkeypatch):
    first = other_plan(config, "1", cpus=8)
    execution.acquire_admission(first, config)
    monkeypatch.setattr(execution, "_process_identity", lambda _: None)
    with pytest.raises(ValueError, match="CPU admission"):
        execution.acquire_admission(other_plan(config, "2"), config)
    complete_lease(config, first, token="wrong-completion-event")
    with pytest.raises(ValueError, match="CPU admission"):
        execution.acquire_admission(other_plan(config, "2"), config)


def test_matching_completion_event_reconciles_dead_owner(config, monkeypatch):
    first = other_plan(config, "1", cpus=8)
    execution.acquire_admission(first, config)
    complete_lease(config, first)
    monkeypatch.setattr(execution, "_process_identity", lambda _: None)
    execution.acquire_admission(other_plan(config, "2"), config)


def test_worker_can_release_matching_lease_after_external_cleanup_receipt(config):
    first = other_plan(config, "1", cpus=8)
    execution.acquire_admission(first, config)
    execution.release_admission(first, config)
    with pytest.raises(ValueError, match="CPU admission"):
        execution.acquire_admission(other_plan(config, "2"), config)
    complete_lease(config, first)
    execution.release_admission(first, config)
    execution.release_admission(first, config)  # Duplicate completion is harmless.
    execution.acquire_admission(other_plan(config, "2"), config)


def test_worker_rejects_missing_or_modified_lease(config):
    first = other_plan(config, "1")
    with pytest.raises(ValueError, match="no matching"):
        execution.verify_admission(first, config)
    execution.acquire_admission(first, config)
    execution.verify_admission(first, config)
    first["description"]["memory_mb"] += 1
    with pytest.raises(ValueError, match="differs"):
        execution.verify_admission(first, config)


def test_competing_admissions_cannot_overbook_shared_budget(config):
    def admit(index):
        try:
            execution.acquire_admission(other_plan(config, format(index, "x")), config)
            return True
        except ValueError as error:
            assert "CPU admission" in str(error)
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(admit, range(8)))
    assert sum(results) == 4
    ledger = json.loads((Path(config["receipt_dir"]) / "admission.json").read_text())
    assert sum(lease["cpus"] for lease in ledger["active"].values()) == 8


def test_expired_window_blocks_new_runs_but_keeps_owned_cleanup_available(config):
    path = Path(config["_path"])
    data = {key: value for key, value in config.items() if not key.startswith("_")}
    data["expires_at"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="expired"):
        execution.load_config(path, execution.sha256(path), check_identity=False)
    execution.load_config(
        path, execution.sha256(path), check_identity=False, allow_expired_cleanup=True,
    )
