"""Launch offline validation in an existing Slurm allocation; never submit jobs."""
import argparse
import getpass
import json
import os
from pathlib import Path
import platform
import subprocess

from scientisttwo.runtime_profiles import enforce_gpu_allocation, slurm_gpu_allocation


def _subids(path, username, uid):
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return False
    for line in lines:
        parts = line.split(":")
        if len(parts) == 3 and parts[0] in {username, str(uid)}:
            try:
                if int(parts[1]) >= 0 and int(parts[2]) >= 65536:
                    return True
            except ValueError:
                pass
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--gpu-index", type=int, default=0,
                           help="Ordinal within the Slurm step's GPU list; mapped to a verified UUID")
    selection.add_argument("--gpu-uuid", help="Complete GPU UUID assigned to this Slurm step")
    parser.add_argument("--runtime", choices=["podman", "docker"], default="podman")
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--min-memory-mb", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("This launcher must run on the allocated Linux compute node")
    node = os.environ.get("SLURMD_NODENAME")
    if not node or node.split(".")[0] != platform.node().split(".")[0]:
        raise SystemExit("Current process is not on its allocated Slurm node; use an existing srun step")
    try:
        allocation = slurm_gpu_allocation(gpu_index=args.gpu_index, gpu_uuid=args.gpu_uuid)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.runtime == "podman" and os.geteuid() != 0:
        if not all(_subids(path, getpass.getuser(), os.getuid()) for path in ("/etc/subuid", "/etc/subgid")):
            raise SystemExit("Rootless Podman prerequisite blocked: current user lacks sufficient subuid/subgid ranges. Ask the cluster administrator; this launcher will not modify system configuration.")
    if not args.workspace.is_dir() or not args.workspace.is_absolute():
        raise SystemExit("Workspace must be an existing absolute directory on this worker")
    if args.min_memory_mb < 0:
        raise SystemExit("Minimum host memory must be nonnegative")
    if args.min_memory_mb:
        memory = os.environ.get("SLURM_MEM_PER_NODE")
        if memory is None and os.environ.get("SLURM_MEM_PER_CPU") and os.environ.get("SLURM_CPUS_ON_NODE"):
            memory = str(int(os.environ["SLURM_MEM_PER_CPU"]) * int(os.environ["SLURM_CPUS_ON_NODE"]))
        if memory is None or (int(memory) != 0 and int(memory) < args.min_memory_mb):
            raise SystemExit("Slurm host-memory allocation is missing or below the declared resident-worker requirement")
    for image in args.image:
        if not (image.startswith("sha256:") or "@sha256:" in image):
            raise SystemExit("Provide resolved immutable image digests, not mutable tags")
        subprocess.run([args.runtime, "image", "inspect", image], check=True,
                       stdout=subprocess.DEVNULL, timeout=30)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("Supply the offline validation command after --")
    if Path(command[0]).name != "scientisttwo" or command[1:3] not in [["benchmark", "validate"], ["benchmark", "generate-bundle"]]:
        raise SystemExit("This launcher accepts only offline scientisttwo benchmark validate or generate-bundle commands")
    if any(arg == "--allow-live" for arg in command):
        raise SystemExit("This launcher is for offline testbed validation; live model calls are disabled")
    gpu_options = [arg for arg in command if arg == "--gpus" or arg.startswith("--gpus=")]
    if len(gpu_options) > 1:
        raise SystemExit("Declare the command GPU selection exactly once")
    if "--gpus" in command:
        index = command.index("--gpus")
        if index + 1 >= len(command) or enforce_gpu_allocation(command[index + 1]) != allocation:
            raise SystemExit("Command GPU selection differs from the allocated physical GPU")
        command[index + 1] = allocation
    elif any(arg.startswith("--gpus=") for arg in command):
        selections = [arg.removeprefix("--gpus=") for arg in command if arg.startswith("--gpus=")]
        if len(selections) != 1 or enforce_gpu_allocation(selections[0]) != allocation:
            raise SystemExit("Command GPU selection differs from the allocated physical GPU")
        command[command.index("--gpus=" + selections[0])] = "--gpus=" + allocation
    else:
        command += ["--gpus", allocation]
    if "--runtime" in command:
        index = command.index("--runtime")
        if index + 1 >= len(command) or command[index + 1] != args.runtime:
            raise SystemExit("Validation runtime differs from the checked container runtime")
    elif any(arg.startswith("--runtime=") for arg in command):
        if "--runtime=" + args.runtime not in command:
            raise SystemExit("Validation runtime differs from the checked container runtime")
    else:
        command += ["--runtime", args.runtime]
    environment = os.environ.copy()
    environment["SCIENTISTTWO_GPU_ALLOCATION"] = allocation
    environment["SCIENTISTTWO_GPU_UUID"] = allocation.removeprefix("device=")
    environment["SCIENTISTTWO_CONTAINER_RUNTIME"] = args.runtime
    print(json.dumps({"status": "launching", "gpu_allocation": allocation,
                      "runtime": args.runtime, "minimum_host_memory_mb": args.min_memory_mb,
                      "live_calls": False}), flush=True)
    os.chdir(args.workspace)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
