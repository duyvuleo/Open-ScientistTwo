"""Declared container environments; declarations never imply runtime validation."""

import csv
import json
import hashlib
import logging
import os
from pathlib import Path
import re
import subprocess
from typing import Literal
import xml.etree.ElementTree as ET

from pydantic import Field, model_validator

from .models import Record


class RuntimeAsset(Record):
    """One read-only cached file, addressed relative to the asset mount."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)

    @model_validator(mode="after")
    def safe_path(self):
        from .benchmarks.models import relative_path

        relative_path(self.path)
        return self


class RuntimeRequirements(Record):
    imports: list[str] = []
    package_versions: dict[str, str] = {}
    python_minor: str | None = None
    resolution_platform: str | None = None
    allow_adapted_cpu_platform: bool = False
    framework: Literal["numpy", "torch", "tensorflow", "jax"] = "numpy"
    cuda: bool = False
    min_vram_mb: int = Field(default=0, ge=0)
    min_disk_mb: int = Field(default=1024, ge=0)
    min_shm_mb: int = Field(default=64, ge=0)
    min_tmp_mb: int = Field(default=256, ge=0)
    max_processes: int = Field(default=512, ge=16)
    cpus: float = Field(default=4, gt=0)
    memory_mb: int = Field(default=8192, ge=256)
    max_output_bytes: int = Field(default=16 * 1024**3, gt=0)
    max_output_files: int = Field(default=10000, gt=0)
    assets: list[RuntimeAsset] = []


class RuntimeProfile(Record):
    name: str
    python_version: str
    image: str
    lock_status: Literal["resolved", "requested_unverified"]
    dependency_lock_sha256: str | None = None
    lock_target_platform: str = "x86_64-manylinux_2_28"
    image_status: Literal["previously_validated", "unverified"]
    requirements: RuntimeRequirements
    build_directory: str
    notes: list[str] = []


_PROFILES = {
    "cpu-tabular": RuntimeProfile(
        name="cpu-tabular", python_version="3.12",
        image="scientisttwo-cpu-tabular:audited-v1",
        lock_status="requested_unverified", image_status="unverified",
        requirements=RuntimeRequirements(
            imports=["numpy", "pandas", "scipy", "sklearn"],
            package_versions={"numpy": "1.26.4", "pandas": "2.2.3", "scipy": "1.14.1", "scikit-learn": "1.5.2"},
        ), build_directory="runtime-profiles/cpu-tabular",
        notes=["Build the resolved CPU lock for the target platform and probe the resulting immutable image before execution; no prebuilt image is distributed."],
    ),
    "torch-cuda": RuntimeProfile(
        name="torch-cuda", python_version="3.11", image="scientisttwo-torch-cuda:audited-v1",
        lock_status="requested_unverified", image_status="unverified",
        requirements=RuntimeRequirements(
            imports=["torch", "torchvision", "transformers", "datasets", "torch_geometric", "ogb"],
            package_versions={"torch": "2.5.1", "torchvision": "0.20.1", "transformers": "4.46.3", "datasets": "3.1.0", "torch-geometric": "2.6.1", "ogb": "1.3.6"},
            framework="torch", cuda=True, min_vram_mb=8192, min_disk_mb=32768,
            min_shm_mb=2048, min_tmp_mb=4096,
        ), build_directory="runtime-profiles/torch-cuda",
    ),
    "tensorflow-cuda": RuntimeProfile(
        name="tensorflow-cuda", python_version="3.11", image="scientisttwo-tensorflow-cuda:audited-v1",
        lock_status="requested_unverified", image_status="unverified",
        requirements=RuntimeRequirements(
            imports=["tensorflow", "keras"],
            package_versions={"tensorflow": "2.15.1", "keras": "2.15.0"},
            framework="tensorflow", cuda=True, min_vram_mb=8192,
            min_disk_mb=16384, min_shm_mb=2048, min_tmp_mb=4096,
        ), build_directory="runtime-profiles/tensorflow-cuda",
    ),
    "jax-clrs": RuntimeProfile(
        name="jax-clrs", python_version="3.10", image="scientisttwo-jax-clrs:audited-v1",
        lock_status="requested_unverified", image_status="unverified",
        requirements=RuntimeRequirements(
            imports=["jax", "haiku", "optax", "chex", "clrs", "tensorflow_datasets"],
            package_versions={"jax": "0.4.26", "jaxlib": "0.4.26", "dm-haiku": "0.0.12", "optax": "0.2.2", "chex": "0.1.86", "dm-clrs": "1.0.0"},
            framework="jax", cuda=True, min_vram_mb=8192,
            min_disk_mb=16384, min_shm_mb=2048, min_tmp_mb=4096,
        ), build_directory="runtime-profiles/jax-clrs",
        notes=["Legacy upstream source compatibility must be validated; package pins alone do not prove it."],
    ),
}


def runtime_profile(name: str) -> RuntimeProfile:
    if name not in _PROFILES:
        raise ValueError(f"Unknown runtime profile: {name}")
    profile = _PROFILES[name].model_copy(deep=True)
    receipt = json.loads(Path(__file__).with_name("runtime_locks.json").read_text())[name]
    source_lock = Path(__file__).parents[2] / "runtime-profiles" / name / "requirements.lock"
    if source_lock.is_file() and hashlib.sha256(source_lock.read_bytes()).hexdigest() != receipt["lock_sha256"]:
        raise ValueError("Resolved dependency lock differs from packaged provenance; finish the resolver refresh before running")
    if receipt["status"] == "resolved_not_runtime_validated":
        profile.lock_status = "resolved"
        profile.dependency_lock_sha256 = receipt["lock_sha256"]
        profile.requirements.package_versions = receipt["package_versions"]
        profile.requirements.python_minor = receipt["python"]
        profile.requirements.resolution_platform = receipt["target_platform"]
        profile.requirements.allow_adapted_cpu_platform = name == "cpu-tabular"
    return profile


def runtime_profiles() -> list[RuntimeProfile]:
    return [runtime_profile(name) for name in _PROFILES]


def probe_source() -> Path:
    """Portable probe copied into the container; importing this module loads no frameworks."""
    return Path(__file__).with_name("task_families") / "templates" / "runtime_probe.py"


def probe_argv(requirements: str = "/runtime/requirements.json") -> list[str]:
    return ["python", "/runtime/probe.py", "--requirements", requirements,
            "--assets", "/assets", "--workspace", "/workspace"]


GPU_UUID_PATTERN = r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


def validate_gpu_selector(value: str | None) -> str | None:
    """Only complete UUIDs or unambiguous numeric container indices are accepted."""
    if value is None or value == "all":
        return value
    if not isinstance(value, str) or not value.startswith("device="):
        raise ValueError("GPU selector must be all or device=<indices or complete GPU UUIDs>")
    parts = value.removeprefix("device=").split(",")
    numeric = all(re.fullmatch(r"0|[1-9][0-9]*", part) for part in parts)
    uuids = all(re.fullmatch(GPU_UUID_PATTERN, part) for part in parts)
    if not (numeric or uuids) or len(set(parts)) != len(parts):
        raise ValueError("GPU selectors require distinct numeric indices or complete GPU UUIDs")
    return value


class GpuDevice(Record):
    """Current NVIDIA index joined to the Linux minor by stable UUID."""

    index: int = Field(ge=0)
    minor_number: int = Field(ge=0)
    uuid: str = Field(pattern="^" + GPU_UUID_PATTERN + "$")
    pci_bus_id: str = Field(pattern=r"^[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$")


def gpu_inventory() -> list[GpuDevice]:
    """Read real driver metadata without importing CUDA or creating a context.

    NVIDIA 575 does not expose minor_number through --query-gpu. Join its XML
    minor numbers to the CSV indices by UUID, checking PCI identity as well.
    Fail closed if visibility changes between these two read-only observations.
    A timeout permits one fresh XML/CSV pair; no prior partial pair is reused.
    """
    try:
        for attempt in range(2):
            try:
                xml = subprocess.run(
                    ["nvidia-smi", "-q", "-x"], check=True, capture_output=True, text=True, timeout=10,
                ).stdout
                output = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=10,
                ).stdout
                break
            except subprocess.TimeoutExpired:
                if attempt:
                    raise
                logging.getLogger(__name__).warning(
                    "NVIDIA GPU inventory timed out; discarding the partial observation "
                    "and retrying one fresh XML/CSV pair"
                )
        tree = ET.fromstring(xml)
        xml_devices = {}
        for item in tree.findall("./gpu"):
            uuid = item.findtext("uuid")
            if uuid in xml_devices:
                raise ValueError("Duplicate GPU UUID in NVIDIA XML inventory")
            xml_devices[uuid] = (int(item.findtext("minor_number", "")),
                                 item.findtext("pci/pci_bus_id", "").upper())
        result = []
        for row in csv.reader(output.splitlines(), skipinitialspace=True):
            if len(row) != 3:
                raise ValueError("Unexpected NVIDIA GPU inventory row")
            index, uuid, bus = (part.strip() for part in row)
            minor, xml_bus = xml_devices[uuid]
            if bus.upper() != xml_bus:
                raise ValueError("GPU PCI identity changed between NVIDIA inventory observations")
            result.append(GpuDevice(index=int(index), minor_number=minor, uuid=uuid,
                                    pci_bus_id=bus.upper()))
        if (not result or len(result) != len(xml_devices)
                or any(len({getattr(row, field) for row in result}) != len(result)
                       for field in ("index", "minor_number", "uuid", "pci_bus_id"))):
            raise ValueError("NVIDIA GPU inventory is missing, inconsistent, or nonunique")
        if attempt:
            logging.getLogger(__name__).warning(
                "NVIDIA GPU inventory recovered after one timeout; "
                "the fresh XML/CSV pair passed UUID, PCI and device-minor checks"
            )
        return result
    except (OSError, subprocess.SubprocessError, ET.ParseError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Cannot verify NVIDIA GPU UUID/device-minor mapping: " + str(exc)) from exc


def _slurm_devices(environment):
    if not environment.get("SLURM_JOB_ID"):
        raise ValueError("Run on an allocated Slurm worker; no job allocation is present")
    # A CPU-only step may inherit the parent allocation's GPUs. That inheritance
    # does not authorize GPU execution in this step.
    value = environment.get("SLURM_STEP_GPUS")
    if not value:
        raise ValueError("Slurm step did not expose GPU IDs; parent-job GPUs are not authorization")
    devices = []
    for part in value.split(","):
        if re.fullmatch(GPU_UUID_PATTERN, part):
            devices.append(part)
            continue
        if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?", part):
            raise ValueError("Unsupported Slurm GPU selector; device minors or complete UUIDs are required")
        bounds = [int(number) for number in part.split("-")]
        first, last = bounds[0], bounds[-1]
        if first > last or last - first > 1024:
            raise ValueError("Invalid Slurm GPU range")
        devices.extend(range(first, last + 1))
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Slurm GPU allocation must be nonempty and unique")
    return devices


def _allocated_uuids(devices, inventory):
    by_minor = {row.minor_number: row.uuid for row in inventory}
    known = {row.uuid for row in inventory}
    try:
        mapped = [value if isinstance(value, str) and value in known else by_minor[value]
                  for value in devices]
    except KeyError as exc:
        raise ValueError("Slurm step GPU is absent from the verified NVIDIA device inventory") from exc
    if len(set(mapped)) != len(mapped):
        raise ValueError("Slurm step contains duplicate identities for the same GPU")
    return mapped


def _selector_uuids(selector, inventory):
    validate_gpu_selector(selector)
    if selector in {None, "all"}:
        raise ValueError("Slurm requires explicit GPU devices; gpus=all is forbidden")
    by_index = {str(row.index): row.uuid for row in inventory}
    known = {row.uuid for row in inventory}
    try:
        requested = [value if value in known else by_index[value]
                     for value in selector.removeprefix("device=").split(",")]
    except KeyError as exc:
        raise ValueError("Requested GPU is absent from the verified NVIDIA device inventory") from exc
    return requested


def _enforce_selected(requested, environment, inventory):
    selected = environment.get("SCIENTISTTWO_GPU_ALLOCATION")
    if selected and requested != _selector_uuids(selected, inventory):
        raise ValueError("Manifest GPU devices differ from the launcher-selected allocation")
    selected_uuid = environment.get("SCIENTISTTWO_GPU_UUID")
    if selected_uuid:
        validate_gpu_selector("device=" + selected_uuid)
        if requested != [selected_uuid]:
            raise ValueError("Requested GPU differs from the launcher's bound GPU UUID")


def slurm_gpu_allocation(
    environ: dict[str, str] | None = None, *, gpu_index: int = 0, gpu_uuid: str | None = None,
) -> str:
    """Select one allocated GPU by UUID; Slurm numbers are device minors, not NVML indices."""
    environment = os.environ if environ is None else environ
    devices = _slurm_devices(environment)
    inventory = gpu_inventory()
    allocated = _allocated_uuids(devices, inventory)
    requested_uuid = gpu_uuid or environment.get("SCIENTISTTWO_GPU_UUID")
    if requested_uuid is not None:
        validate_gpu_selector("device=" + requested_uuid)
        if "," in requested_uuid or requested_uuid not in allocated:
            raise ValueError("Requested GPU UUID is outside this Slurm step's allocation")
        selected = requested_uuid
    elif not 0 <= gpu_index < len(allocated):
        raise ValueError("GPU index is outside the current Slurm allocation")
    else:
        selected = allocated[gpu_index]
    _enforce_selected([selected], environment, inventory)
    return "device=" + selected


def enforce_gpu_allocation(gpus: str | None, environ: dict[str, str] | None = None) -> str | None:
    """Reject an overbroad manifest before the launcher drops allocation env vars."""
    environment = os.environ if environ is None else environ
    validate_gpu_selector(gpus)
    if gpus is None:
        return None
    if not environment.get("SLURM_JOB_ID"):
        return gpus
    devices = _slurm_devices(environment)
    inventory = gpu_inventory()
    allocated = set(_allocated_uuids(devices, inventory))
    requested = _selector_uuids(gpus, inventory)
    if not set(requested) <= allocated:
        raise ValueError("Requested GPU is outside this Slurm step's allocation")
    _enforce_selected(requested, environment, inventory)
    return "device=" + ",".join(requested)
