"""PDF argv passes the actual Slurm parser; transports and outputs are synthetic."""

import hashlib
import json
from pathlib import Path
import re

import pytest

from scientisttwo import slurm_execution as guard
from scientisttwo.integrations.paperorchestra.pdf_tools import ContainerPdfTools, PdfToolsError
from scientisttwo.process import ProcessResult


IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def config(tmp_path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    return {"podman": "/frozen/podman", "python": "/frozen/python", "srun": "/usr/bin/srun",
            "job_id": "123", "node": "synthetic-node", "receipt_dir": str(receipts),
            "max_cpus": 8, "max_memory_mb": 65536, "max_seconds": 300,
            "expires_at": 9999999999, "gpu_uuid": None, "gpu_gres": "none",
            "_path": "/frozen/config.json", "_sha256": "b" * 64}


class GuardedPdf:
    def __init__(self, config, *, mutation=None, timeout=False):
        self.config, self.mutation, self.timeout = config, mutation, timeout
        self.plans, self.cleanup = [], []

    def __call__(self, argv, timeout, **kwargs):
        assert argv[0] == "podman"
        if argv[1:3] == ["image", "inspect"]:
            return ProcessResult(0, IMAGE, "", .01)
        if argv[1:3] == ["rm", "--force"]:
            assert re.fullmatch(guard.NAME_PATTERN, argv[-1])
            self.cleanup.append(argv[-1])
            return ProcessResult(0, "", "", .01)
        args = list(argv[1:])
        if self.mutation:
            self.mutation(args)
        plan = guard.parse_run(args, self.config)
        self.plans.append(plan)
        if self.timeout:
            return ProcessResult(-9, "", "", timeout, timed_out=True)
        mounts = {args[i + 1].split(":")[1]: Path(args[i + 1].split(":")[0])
                  for i, part in enumerate(args) if part == "-v"}
        job = json.loads((mounts["/control"] / "job.json").read_bytes())
        assert job["timeout_seconds"] == plan["description"]["seconds"] == timeout
        assert kwargs["output_limits"].directories == (mounts["/output"],)
        payloads = {"compile": {"paper.pdf": b"%PDF-synthetic"},
                    "text": {"paper.txt": b"Synthetic extracted text"},
                    "render": {"page-1.png": b"\x89PNG\r\n\x1a\nsynthetic"}}[job["operation"]]
        for name, payload in payloads.items():
            (mounts["/output"] / name).write_bytes(payload)
        report = {"status": "success", "operation": job["operation"], "outputs": {
            name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in payloads.items()}}
        (mounts["/output"] / "result.json").write_text(json.dumps(report))
        return ProcessResult(0, "", "", .01)


def execute(tmp_path, config, *, kind="compile", policy="deny", mutation=None, timeout=False):
    root = tmp_path / "paper"
    root.mkdir()
    (root / "main.tex").write_text("Synthetic TeX fixture")
    pdf = root / "input.pdf"
    pdf.write_bytes(b"%PDF-synthetic")
    transport = GuardedPdf(config, mutation=mutation, timeout=timeout)
    tools = ContainerPdfTools(IMAGE, runtime="podman", executor=transport,
                              staging_root=tmp_path, swap_policy=policy)
    if kind == "compile":
        tools.compile(root, root / "out.pdf", "main", 10)
    elif kind == "text":
        assert tools.text(pdf, timeout_seconds=10) == "Synthetic extracted text"
    else:
        assert len(tools.render(pdf, root / "pages", timeout_seconds=10)) == 1
    return tools, transport


@pytest.mark.parametrize("kind", ["compile", "text", "render"])
@pytest.mark.parametrize("policy", ["deny", "inherit"])
def test_pdf_operations_keep_restricted_profile_and_slurm_cpu_limits(tmp_path, config, kind, policy):
    tools, transport = execute(tmp_path, config, kind=kind, policy=policy)
    plan = transport.plans[0]
    description, argv = plan["description"], plan["container_argv"]
    assert description["profile"] == "paperorchestra-pdf-v1"
    assert description["command"] == ["python", "/worker.py"]
    assert description["cpus"] == 1 and description["memory_mb"] == 768
    assert description["pids"] == 64 and description["gpu_uuid"] is None
    assert description["swap_policy"] == policy
    assert tools.receipts[0]["swap_policy"] == policy
    assert description["memory_budget_scope"] == tools.receipts[0]["memory_budget_scope"]
    assert argv[argv.index(IMAGE) + 1] == "/scientisttwo-guard.py"
    assert argv[argv.index("--entrypoint") + 1] == "python"
    assert {"--read-only", "no-new-privileges", "65534:65534", "--ulimit=nproc=64:64"} <= set(argv)
    assert "--memory-swap" not in argv
    assert "--ipc" in argv and argv[argv.index("--ipc") + 1] == "none"
    assert not any(value.startswith("--shm-size") for value in argv)
    assert "--device" not in argv and "NVIDIA_VISIBLE_DEVICES=void" in argv
    assert "--gres=none" in guard.slurm_argv(plan, config)
    assert "--mem=768M" in guard.slurm_argv(plan, config)
    assert not list(tmp_path.glob("scientisttwo-pdf-*")), "Staging must be removed"


def replace(args, old, new):
    args[args.index(old)] = new


def mount_change(args, target, mode):
    index = next(i + 1 for i, part in enumerate(args) if part == "-v"
                 and args[i + 1].split(":")[1] == target)
    path, destination, _ = args[index].split(":")
    args[index] = f"{path}:{destination}:{mode}"


@pytest.mark.parametrize("mutation", [
    lambda a: replace(a, "--memory=768m", "--memory=1024m"),
    lambda a: replace(a, "--cpus=1", "--cpus=2"),
    lambda a: replace(a, "--ipc=none", "--ipc=host"),
    lambda a: a.__setitem__(slice(a.index(IMAGE), a.index(IMAGE)), ['--shm-size=16m']),
    lambda a: replace(a, "nofile=256:256", "nofile=1024:1024"),
    lambda a: replace(a, "fsize=100000000:100000000", "fsize=500000001:500000001"),
    lambda a: replace(a, "SCIENTISTTWO_PDF_SWAP_POLICY=deny", "SCIENTISTTWO_PDF_SWAP_POLICY=inherit"),
    lambda a: a.remove("--memory-swap=768m"),
    lambda a: mount_change(a, "/input", "rw"),
    lambda a: mount_change(a, "/control", "rw"),
    lambda a: replace(a, "/worker.py", "/other.py"),
    lambda a: a.__setitem__(a.index("--entrypoint") + 1, "sh"),
    lambda a: a.__setitem__(slice(a.index(IMAGE), a.index(IMAGE)),
                           ["-v", "/tmp:/extra:ro"]),
    lambda a: a.__setitem__(slice(a.index(IMAGE), a.index(IMAGE)),
                           ["--device", "nvidia.com/gpu=all"]),
])
def test_pdf_profile_rejects_wider_commands_mounts_and_resources(tmp_path, config, mutation):
    with pytest.raises(PdfToolsError):
        execute(tmp_path, config, mutation=mutation)


def test_pdf_timeout_requires_label_and_respects_window(tmp_path, config):
    def remove_label(args):
        i = args.index("--label")
        del args[i:i + 2]
    with pytest.raises(PdfToolsError, match="--label"):
        execute(tmp_path, config, mutation=remove_label)


def test_default_no_swap_rejects_unknown_or_unbounded_kernel_policy():
    description = {"swap_policy": "deny", "memory_mb": 768, "file_size_limit": 100000000}
    for swap in ({}, {"status": "unbounded"}, {"status": "unknown", "limit_bytes": None},
                 {"status": "bounded", "scope": "swap_only", "limit_bytes": 1},
                 {"status": "bounded", "scope": "memory_and_swap_combined", "limit_bytes": 769 * 1024**2}):
        with pytest.raises(ValueError, match="no-swap"):
            guard.verify_pdf_resources(description, {"swap": swap}, (256, 256), (100000000, 100000000))
    for swap in ({"status": "bounded", "scope": "swap_only", "limit_bytes": 0},
                 {"status": "bounded", "scope": "memory_and_swap_combined", "limit_bytes": 768 * 1024**2}):
        guard.verify_pdf_resources(description, {"swap": swap}, (256, 256), (100000000, 100000000))


def test_inherited_swap_requires_explicit_configuration_and_preserves_kernel_file_limits():
    with pytest.raises(ValueError, match="explicit Podman"):
        ContainerPdfTools(IMAGE, swap_policy="inherit")
    for policy in ("auto", "unlimited", None):
        with pytest.raises(ValueError):
            ContainerPdfTools(IMAGE, runtime="podman", swap_policy=policy)
    description = {"swap_policy": "inherit", "memory_mb": 768, "file_size_limit": 100000000}
    guard.verify_pdf_resources(description, {"swap": {"status": "unknown"}}, (256, 256), (100000000, 100000000))
    with pytest.raises(ValueError, match="open-file"):
        guard.verify_pdf_resources(description, {}, (1024, 1024), (100000000, 100000000))
    with pytest.raises(ValueError, match="file-size"):
        guard.verify_pdf_resources(description, {}, (256, 256), (-1, -1))


def test_new_options_do_not_widen_normal_experiment_parser(tmp_path, config):
    def rename(args):
        args[args.index("--name") + 1] = "scientisttwo-native-" + "b" * 18
    with pytest.raises(PdfToolsError, match="PDF-only"):
        execute(tmp_path, config, mutation=rename)


def test_owned_pdf_cleanup_remains_available_after_timeout(tmp_path, config):
    root = tmp_path / "paper"
    root.mkdir()
    (root / "main.tex").write_text("Synthetic TeX fixture")
    fake = GuardedPdf(config, timeout=True)
    adapter = ContainerPdfTools(IMAGE, runtime="podman", executor=fake, staging_root=tmp_path)
    with pytest.raises(PdfToolsError, match="timed out"):
        adapter.compile(root, root / "output.pdf", "main", 10)
    assert fake.cleanup == [fake.plans[0]["name"]]
    assert not (root / "output.pdf").exists()
