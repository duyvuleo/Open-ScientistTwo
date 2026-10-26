"""Versioned offline cross-domain suite. Definitions are fixed before execution."""

from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator

from .catalog import CLASSIFICATION, REGRESSION, OPTIMIZATION
from .models import Record, TaskSpec, Profile
from .runner import evaluator_identity
from .store import canonical, digest
from .tasks import load_task_adapter


def implementation_identity():
    return {
        "sources": {
            p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "evaluator": evaluator_identity(),
    }


class CampaignSpec(Record):
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    task_id: str
    task_version: str
    repetition: int = Field(ge=0)
    task: TaskSpec
    split_manifest_hash: str
    data_origin: Literal["bundled_dataset", "generated_control"]
    expected_control: str = "empirical_outcome"


class BatchManifest(Record):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-z0-9_-]+$")
    suite_version: str = "offline-cross-domain-v1"
    runtime: Literal["docker", "podman"] = "docker"
    context: str | None = None
    image: str = "scientisttwo-worker:0.1"
    workers: int = Field(default=2, ge=1, le=4)
    wall_seconds: float = Field(default=7200, gt=0)
    screening_sample_size: int = Field(default=1, ge=0, le=8)
    profile: Profile = Field(default_factory=Profile)
    implementation: dict
    campaigns: list[CampaignSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_campaigns(self):
        if len({c.id for c in self.campaigns}) != len(self.campaigns):
            raise ValueError("Campaign IDs must be unique")
        by_task = {}
        repetitions = set()
        for c in self.campaigns:
            if c.task.schema_version != 2:
                raise ValueError("Batch campaigns require v2 fixed partitions")
            version = digest(canonical(c.task.model_dump(exclude={"seeds"})))
            if version != c.task_version:
                raise ValueError("Task version does not match its specification")
            if c.task_id in by_task and by_task[c.task_id] != (version, c.split_manifest_hash):
                raise ValueError(
                    "Repetitions of a task must share the same data and split manifests"
                )
            if (c.task_id, c.repetition) in repetitions:
                raise ValueError("Duplicate repetition")
            repetitions.add((c.task_id, c.repetition))
            by_task[c.task_id] = (version, c.split_manifest_hash)
        return self


def make_suite(
    *, smoke=False, workers=2, context=None, image="scientisttwo-worker:0.1"
):
    entries = []
    groups = [
        ("classification", CLASSIFICATION),
        ("regression", REGRESSION),
        ("optimization", tuple(OPTIMIZATION)),
    ]
    for family, cases in groups:
        for index, name in enumerate(cases):
            if smoke and (family, name) not in {
                ("classification", "moons"),
                ("regression", "quadratic"),
                ("optimization", "box-mild"),
            }:
                continue
            metric = {
                "classification": "macro_f1",
                "regression": "mse",
                "optimization": "objective_gap",
            }[family]
            directions = {
                "classification": {"macro_f1": "maximize", "accuracy": "maximize"},
                "regression": {"mse": "minimize", "mae": "minimize"},
                "optimization": {"objective_gap": "minimize", "max_violation": "minimize"},
            }[family]
            guardrails = {
                "classification": {"accuracy": 0.01},
                "regression": {},
                "optimization": {"max_violation": 1e-8},
            }[family]
            task = TaskSpec(
                name=f"{family}-{name}",
                adapter=family,
                question=f"Validate the research workflow on the declared {family}/{name} problem.",
                datasets=[name],
                subset_datasets=[name],
                data_seed=1729 + 97 * index,
                split_seed=31415,
                primary_metric=metric,
                metric_directions=directions,
                guardrails=guardrails,
                improvement_epsilon=1e-4 if family == "classification" else 1e-6,
                settings={
                    "exercise_meta_refinement": family != "classification"
                    and name in {"quadratic", "box-mild"}
                },
            )
            adapter = load_task_adapter(family, task)
            split_hash = digest(canonical(adapter.split_manifest(task)))
            version = digest(canonical(task.model_dump(exclude={"seeds"})))
            for repetition in range(1 if smoke else 3):
                concrete = task.model_copy(
                    update={"seeds": [repetition * 10 + x for x in range(3)]}
                )
                entries.append(
                    CampaignSpec(
                        id=f"{task.name}-r{repetition + 1}",
                        task_id=task.name,
                        task_version=version,
                        repetition=repetition,
                        task=concrete,
                        split_manifest_hash=split_hash,
                        data_origin="bundled_dataset"
                        if family == "classification" and name in {"iris", "wine", "digits"}
                        else "generated_control",
                        expected_control="neutral_exact_baseline"
                        if family == "optimization" and name.endswith("isotropic")
                        else "positive_capacity_control"
                        if (family, name)
                        in {("regression", "quadratic"), ("optimization", "box-mild")}
                        else "empirical_outcome",
                    )
                )
    return BatchManifest(
        name="offline-cross-domain-smoke" if smoke else "offline-cross-domain-90",
        workers=workers,
        context=context,
        image=image,
        implementation=implementation_identity(),
        campaigns=entries,
    )


def write_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(manifest.model_dump_json(indent=2) + "\n")
