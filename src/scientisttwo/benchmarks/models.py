from pathlib import Path, PurePosixPath
from typing import Literal
from pydantic import Field, field_validator, model_validator

from ..models import Record
from ..runtime_contracts import RuntimeAsset
from ..runtime_profiles import GPU_UUID_PATTERN

Benchmark = Literal["mlagentbench", "mle-bench"]


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path == PurePosixPath(".")
    ):
        raise ValueError(f"Expected a safe relative artifact path: {value}")
    return str(path)


class BenchmarkTask(Record):
    benchmark: Benchmark
    task_id: str
    folder: str
    name: str
    revision: str
    repository: str
    description: str = ""
    grader: str | None = None
    grader_available: bool = False
    baseline_command: list[str] = []
    metric: str | None = None
    direction: Literal["maximize", "minimize"] | None = None
    splits: list[str] = []
    warnings: list[str] = []
    required_artifacts: list[str] = []


class NativeResources(Record):
    cpus: float = Field(default=1, gt=0, le=128)
    memory_mb: int = Field(default=2048, ge=256)
    seconds: float = Field(default=1800, gt=0)
    grading_seconds: float = Field(default=300, gt=0)
    gpus: str | None = Field(
        default=None,
        pattern=rf"^(?:all|device=(?:[0-9]+(?:,[0-9]+)*|{GPU_UUID_PATTERN}(?:,{GPU_UUID_PATTERN})*))$",
    )
    shm_mb: int = Field(default=64, ge=16)
    tmp_mb: int = Field(default=512, ge=16)
    pids_limit: int = Field(default=256, ge=16)
    threads: int = Field(default=1, ge=1, le=128)
    output_max_bytes: int = Field(default=64_000_000_000, ge=1)
    output_max_files: int = Field(default=100_000, ge=1)
    log_max_bytes: int = Field(default=16_777_216, ge=1024)

    @field_validator("gpus")
    @classmethod
    def valid_gpus(cls, value):
        from ..runtime_profiles import validate_gpu_selector

        return validate_gpu_selector(value)


class NativeSpec(Record):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    benchmark: Benchmark
    task_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    repository: Path
    revision: str
    mode: Literal["baseline", "prepared_candidate", "grade_only"]
    data_provenance: Literal[
        "upstream_no_data", "official_prepared", "user_prepared", "contract_fixture"
    ]
    public_dir: Path
    private_dir: Path | None = None
    candidate_dir: Path | None = None
    command: list[str] = []
    submission: str = "submission.csv"
    final_seed: int = Field(default=0, ge=0)
    image: str = "scientisttwo-native:0.1"
    grader_image: str = "scientisttwo-native:0.1"
    runtime: Literal["docker", "podman"] = "docker"
    context: str | None = None
    resources: NativeResources = Field(default_factory=NativeResources)
    assets: list[RuntimeAsset] = []
    required_artifacts: list[str] = []
    public_manifest: dict = {}
    private_manifest: dict = {}
    candidate_manifest: dict = {}
    implementation: dict = {}
    live_calls: Literal[False] = False

    @model_validator(mode="after")
    def valid_modes(self):
        relative_path(self.submission)
        for output in self.required_artifacts:
            relative_path(output)
        for path in (self.repository, self.public_dir, self.private_dir, self.candidate_dir):
            if path is not None and not path.is_absolute():
                raise ValueError("Native manifests require absolute input paths")
        if self.mode == "prepared_candidate" and (self.candidate_dir is None or not self.command):
            raise ValueError(
                "Prepared candidates require a code directory and explicit argv command"
            )
        if self.mode == "grade_only" and self.candidate_dir is None:
            raise ValueError("grade_only requires an existing submission/artifact directory")
        if self.runtime != "docker" and self.context:
            raise ValueError("Named contexts apply only to Docker")
        return self


class NativeResult(Record):
    benchmark: Benchmark
    task_id: str
    status: Literal[
        "completed",
        "blocked",
        "execution_failed",
        "grading_failed",
        "invalid_submission",
        "interrupted",
    ]
    mode: str
    data_provenance: str
    score: float | None = None
    metric: str | None = None
    direction: str | None = None
    official_report_ref: str | None = None
    public_data_checksums: Literal["pass", "fail", "unavailable"] = "unavailable"
    benchmark_data_verified: bool = False
    leaderboard_eligible: bool = False
    agent_evaluation: Literal[False] = False
    live_model_calls: Literal[0] = 0
    reason: str = ""
    warnings: list[str] = []
    elapsed_seconds: float = Field(default=0, ge=0)
    specification_ref: str | None = None
    execution_ref: str | None = None
    artifacts_ref: str | None = None
