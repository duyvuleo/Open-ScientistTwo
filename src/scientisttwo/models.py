"""Versioned, strict contracts shared by the controller and adapters."""

from enum import StrEnum
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Profile(Record):
    name: Literal["offline", "paper"] = "offline"
    limitation_rounds: int = Field(default=2, ge=1)
    seed_ideas: int = Field(default=3, ge=2)
    search_rounds: int = Field(default=2, ge=1)
    engineering_rounds: int = Field(default=1, ge=0)
    ablation_rounds: int = Field(default=1, ge=0)
    peer_rounds: int = Field(default=1, ge=0)
    meta_rounds: int = Field(default=1, ge=0)
    success_target: int = Field(default=4, ge=1)
    review_threshold: float = Field(default=8, ge=1, le=10)
    experiment_seconds: float = Field(default=120, gt=0)
    campaign_seconds: float = Field(default=1800, gt=0)
    live_calls: Literal[False] = False

    @classmethod
    def named(cls, name: str) -> "Profile":
        if name == "paper":
            return cls(
                name="paper",
                limitation_rounds=16,
                seed_ideas=5,
                search_rounds=4,
                engineering_rounds=2,
                peer_rounds=2,
            )
        if name != "offline":
            raise ValueError(f"Unknown profile: {name}")
        return cls()


class TaskSpec(Record):
    schema_version: Literal[1, 2] = 2
    name: str = Field(default="pilot", pattern=r"^[a-z0-9_-]+$")
    question: str = "Compare small classification pipelines under a fixed evaluation protocol."
    adapter: str = "classification"
    settings: dict = {}
    datasets: list[str] = ["iris", "wine", "digits"]
    subset_datasets: list[str] = ["iris", "wine"]
    seeds: list[int] = [0, 1, 2]
    data_seed: int = Field(default=1729, ge=0)
    split_seed: int = Field(default=0, ge=0)
    subset_fraction: float = Field(default=0.5, gt=0, le=1)
    improvement_epsilon: float = Field(default=1e-4, ge=0)
    primary_metric: str = "macro_f1"
    metric_directions: dict[str, Literal["maximize", "minimize"]] = {
        "macro_f1": "maximize",
        "accuracy": "maximize",
    }
    guardrails: dict[str, float] = {"accuracy": 0.01}
    quality_tolerance: float = Field(default=1e-8, gt=0)

    @model_validator(mode="after")
    def check_sets(self):
        if not self.datasets or not self.seeds or not self.subset_datasets:
            raise ValueError("Datasets and seeds cannot be empty")
        if not set(self.subset_datasets) <= set(self.datasets):
            raise ValueError("Screening datasets must belong to the full benchmark")
        if len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("Seeds must be distinct nonnegative integers")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("Datasets must be distinct")
        if self.primary_metric not in self.metric_directions:
            raise ValueError("Primary metric requires an explicit direction")
        if not set(self.guardrails) <= set(self.metric_directions):
            raise ValueError("Guardrails require explicit metric directions")
        if any(not math.isfinite(v) or v < 0 for v in self.guardrails.values()):
            raise ValueError("Guardrail tolerances must be finite and nonnegative")
        return self


class Idea(Record):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    hypothesis: str
    method: str
    parameters: dict = {}
    parent_ids: list[str] = []
    novelty: float = Field(default=0, ge=0, le=10)
    references: list[str] = []
    origin: Literal["baseline", "seed", "evolved", "engineered", "ablation", "meta"] = "seed"


class IdeaBatch(Record):
    ideas: list[Idea]


class Limitations(Record):
    items: list[str]
    sufficient: bool = False


class Novelty(Record):
    score: float = Field(ge=0, le=10)
    references: list[str]
    scope: str


class CodeProposal(Record):
    source: str
    description: str


class EvalSpec(Record):
    task: TaskSpec
    phase: Literal["subset", "full", "rebuttal", "final"]
    seeds: list[int]
    datasets: list[str]
    train_fraction: float = Field(gt=0, le=1)
    split: Literal["development", "test"] = "development"
    split_manifest_hash: str = ""

    @classmethod
    def for_phase(cls, task: TaskSpec, phase: str) -> "EvalSpec":
        return cls(
            task=task,
            phase=phase,
            seeds=task.seeds[:1] if phase == "subset" else task.seeds,
            datasets=task.subset_datasets if phase == "subset" else task.datasets,
            train_fraction=task.subset_fraction if phase == "subset" else 1,
            split="test" if phase == "final" else "development",
        )


class MetricRow(Record):
    dataset: str
    seed: int
    metrics: dict[str, float]
    training_seconds: float = Field(ge=0)
    inference_seconds: float = Field(ge=0)
    count: int = Field(gt=0)


class ExperimentResult(Record):
    id: str
    idea_id: str
    status: Literal["success", "execution_failed", "invalid", "timeout", "interrupted"]
    protocol_hash: str
    code_ref: str
    spec_ref: str
    split_manifest_ref: str | None = None
    rows: list[MetricRow] = []
    observations_ref: str | None = None
    logs_ref: str | None = None
    reason: str = ""
    measurement_origin: Literal["container", "synthetic_test"] = "container"
    duration_seconds: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_rows(self):
        if self.status == "success" and (not self.rows or not self.observations_ref):
            raise ValueError("Successful experiments require observations and metrics")
        if self.status != "success" and self.rows:
            raise ValueError("Failed experiments cannot carry scientific metrics")
        keys = [(r.dataset, r.seed) for r in self.rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate metric rows")
        return self


class Comparison(Record):
    valid: bool
    improved: bool = False
    guardrails_pass: bool = False
    delta: float = 0
    reason: str


class CriticDecision(Record):
    decision: Literal["Good", "Bad", "Engineer"]
    reason: str


class Refinement(Record):
    decision: Literal["Good", "Refine"]
    reason: str


class Selection(Record):
    idea_id: str
    reason: str


class ExperimentPlan(Record):
    idea: Idea
    purpose: str
    changed_component: str


class Plans(Record):
    plans: list[ExperimentPlan]


class Review(Record):
    score: float = Field(ge=1, le=10)
    concerns: list[str]
    synthetic: bool = True


class MetaReview(Record):
    decision: Literal["Accept", "Refine"]
    reason: str


class EvidenceClaim(Record):
    experiment_id: str
    dataset: str
    seed: int
    metric: str
    value: float


class ManuscriptDraft(Record):
    title: str
    claims: list[EvidenceClaim] = []
    synthetic: bool = True


class AgentResult(Record):
    payload: dict
    backend: str
    model: str
    prompt_version: str = "scientisttwo-v1"
    synthetic: bool
    usage: dict = {}


class Stage(StrEnum):
    CREATED = "created"
    LIMITATIONS = "limitations"
    BASELINE = "baseline"
    SEARCH = "search"
    ABLATION = "ablation"
    DRAFT = "draft"
    REVIEW = "review"
    META = "meta"
    FINAL = "final"
    COMPLETE = "complete"


class CampaignState(Record):
    run_id: str
    stage: Stage = Stage.CREATED
    status: Literal["running", "blocked", "interrupted", "complete"] = "running"
    outcome: str | None = None
    best_idea_id: str | None = None
    reason: str = "Created"
