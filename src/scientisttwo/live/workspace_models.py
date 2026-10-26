"""Portable task descriptions independent of an upstream benchmark registry."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..models import Record
from ..benchmarks.models import NativeResources, relative_path
from ..runtime_contracts import RuntimeAsset
from ..artifacts import ArtifactSpec
from .pipeline import EvaluationPipeline


class WorkspaceSpec(Record):
    """User-prepared code/data and a separately isolated trusted final evaluator.

    The evaluator is task infrastructure, never an agent-generated score file.
    Development evaluation is supplied independently by a development adapter.
    """

    schema_version: Literal[1] = 1
    benchmark: Literal["workspace"] = "workspace"
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    task_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    description: str = ""
    repository: None = None
    mode: Literal["baseline", "prepared_candidate"] = "baseline"
    data_provenance: Literal["user_prepared", "contract_fixture"] = "user_prepared"
    public_dir: Path
    private_dir: Path | None = None
    candidate_dir: Path
    evaluator_dir: Path
    command: list[str] = Field(min_length=1, max_length=32)
    evaluator_command: list[str] = Field(min_length=1, max_length=32)
    submission: str = "submission.csv"
    metric: str = Field(min_length=1)
    direction: Literal["maximize", "minimize"]
    final_seed: int = Field(default=0, ge=0)
    image: str
    grader_image: str
    runtime: Literal["docker", "podman"] = "docker"
    context: str | None = None
    resources: NativeResources = Field(default_factory=NativeResources)
    assets: list[RuntimeAsset] = []
    output_artifacts: list[ArtifactSpec] = []
    evaluation_pipeline: EvaluationPipeline | None = None
    public_manifest: dict = {}
    private_manifest: dict = {}
    candidate_manifest: dict = {}
    evaluator_manifest: dict = {}
    implementation: dict = {}
    live_calls: Literal[False] = False

    @model_validator(mode="after")
    def paths_and_commands(self):
        relative_path(self.submission)
        if self.evaluation_pipeline:
            if not self.output_artifacts:
                raise ValueError("Staged final evaluation requires declared candidate artifacts")
            if self.evaluation_pipeline.private_dir != self.private_dir:
                raise ValueError("Final scoring must use the declared protected task data")
        roots = [
            root
            for root in (self.public_dir, self.private_dir, self.candidate_dir, self.evaluator_dir)
            if root is not None
        ]
        for root in roots:
            if not root.is_absolute():
                raise ValueError("Workspace task input paths must be absolute")
        for index, root in enumerate(roots):
            for other in roots[index + 1 :]:
                if root.resolve().is_relative_to(other.resolve()) or other.resolve().is_relative_to(
                    root.resolve()
                ):
                    raise ValueError(
                        "Public, private, candidate and evaluator trees must be separate"
                    )
        for command in (self.command, self.evaluator_command):
            if any(not arg or "\x00" in arg for arg in command):
                raise ValueError("Execution commands must be explicit nonempty argv")
        if self.runtime != "docker" and self.context:
            raise ValueError("Named contexts apply only to Docker")
        return self
