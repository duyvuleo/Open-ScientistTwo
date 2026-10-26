from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StringConstraints, model_validator

from ..models import Record
from ..benchmarks.models import NativeSpec, relative_path
from ..benchmarks.files import FORBIDDEN
from .workspace_models import WorkspaceSpec
from ..research.policy import ResearchPolicy
from ..providers.config import ModelProvider


class SourceFile(Record):
    path: str
    content: str = Field(max_length=250_000)

    @model_validator(mode="after")
    def safe(self):
        relative_path(self.path)
        parts = Path(self.path).parts
        if any(p.startswith(".") or p.lower() in FORBIDDEN for p in parts):
            raise ValueError("Hidden or credential-like candidate paths are forbidden")
        suffixes = {
            "",
            ".py",
            ".r",
            ".jl",
            ".js",
            ".ts",
            ".sh",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".toml",
            ".yaml",
            ".yml",
            ".json",
            ".txt",
            ".md",
        }
        if Path(self.path).suffix.lower() not in suffixes:
            raise ValueError("Candidate source/configuration file extension is unsupported")
        if "\x00" in self.content or len(self.content.encode("utf-8")) > 250_000:
            raise ValueError("Candidate files must be bounded UTF-8 text without NUL bytes")
        return self


class Candidate(Record):
    files: list[SourceFile] = Field(default_factory=list, max_length=24)
    # When present, files are bounded edits to an immutable full source snapshot.
    base_snapshot_ref: str | None = Field(
        default=None, pattern=r"^artifacts/[a-f0-9]{64}\.json$"
    )
    command: list[str] = Field(min_length=1, max_length=32)
    description: str

    @model_validator(mode="after")
    def valid(self):
        if not self.files and self.base_snapshot_ref is None:
            raise ValueError("Candidates require source files or a frozen base snapshot")
        paths = [f.path for f in self.files]
        if (
            len(paths) != len(set(paths))
            or sum(len(f.content.encode("utf-8")) for f in self.files) > 500_000
        ):
            raise ValueError("Duplicate files or oversized candidate")
        # The task adapter validates its execution contract; argv is passed
        # directly to the isolated runtime, never interpreted by a host shell.
        if not self.command[0].strip() or any("\x00" in a for a in self.command):
            raise ValueError("Invalid argv")
        return self


class Hypothesis(Record):
    title: str
    hypothesis: str
    novelty_score: float = Field(ge=0, le=10)
    novelty_scope: str
    source_ids: list[str] = Field(
        description="IDs of supplied task/literature documents only; experiment IDs go in evidence_ids."
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Only committed non-final trial/observation IDs offered in this invocation's reference_contract."
    )
    adapter_payload: dict = Field(default_factory=dict,
                                 description="Optional task-specific hypothesis parameters; never measured scores.")


class Seeds(Record):
    ideas: list[Hypothesis] = Field(min_length=1, max_length=8)


class Revision(Record):
    idea: Hypothesis
    candidate: Candidate


class LimitationAssessment(Record):
    limitations: list[str] = Field(min_length=1, max_length=16)
    relevant: bool
    reason: str


class ControlPlan(Record):
    purpose: str
    changed_file: str | None
    candidate: Candidate


class ControlPlans(Record):
    plans: list[ControlPlan] = Field(max_length=3)


class LiveReview(Record):
    score: float = Field(ge=1, le=10)
    concerns: list[str]
    review_ref: str | None = Field(default=None, pattern=r"^artifacts/[a-f0-9]{64}\.json$")


class LiveMetaReview(Record):
    decision: Literal["Accept", "ReviseManuscript", "ReviseMethod"] = Field(
        description="Accept the evidence-scoped draft; ReviseManuscript changes only claims, wording or limitations using existing evidence; ReviseMethod requests an executable method change and new experiments."
    )
    reason: str = Field(min_length=1)


class Claim(Record):
    experiment_id: str
    metric: str
    value: float


class ConcernResponse(Record):
    concern_id: str
    status: Literal["executed", "limited", "unaddressed"]
    experiment_ids: list[str] = Field(default_factory=list)
    explanation: str


class Draft(Record):
    title: str
    claims: list[Claim]
    limitations: list[str]
    source_ids: list[str] = Field(
        description="Supplied task/literature document IDs only. Numerical claims identify experiments separately."
    )
    concern_responses: list[ConcernResponse] = Field(
        default_factory=list,
        description="Address every supplied supplementary concern. Executed means exact structured actions completed, not scientific resolution. Unsupported or unexecuted requests cannot be marked executed."
    )
    manuscript_ref: str | None = Field(default=None, pattern=r"^artifacts/[a-f0-9]{64}\.json$")


class Source(Record):
    id: str
    title: str
    text: str


MetricName = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S(?:.*\S)?$")]
MetricValue = Annotated[float, Field(strict=True)]
Direction = Literal["minimize", "maximize"]
LEGACY_METRIC_DIRECTIONS: dict[str, Direction] = {
    "roc_auc": "maximize",
    "accuracy": "maximize",
    "log_loss": "minimize",
    "rmse": "minimize",
    "process_seconds": "minimize",
}


class MetricSpec(Record):
    direction: Direction
    hard_min: MetricValue | None = None
    hard_max: MetricValue | None = None
    max_degradation: float | None = Field(default=None, ge=0)
    tolerance: float = Field(default=1e-10, ge=0)

    @model_validator(mode="after")
    def ordered_bounds(self):
        if self.hard_min is not None and self.hard_max is not None:
            if self.hard_min > self.hard_max:
                raise ValueError("Metric hard_min cannot exceed hard_max")
        return self


class DevelopmentSpec(Record):
    adapter: str = "vectorization"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    target: str = "Insult"
    prediction_columns: list[str] = ["Insult"]
    classes: list[str] = []
    text_column: str = "Comment"
    metric: MetricName = "process_seconds"
    direction: Direction | None = None
    aggregation: Literal["mean"] = "mean"
    metric_specs: dict[MetricName, MetricSpec] = {}
    split_seed: int = Field(default=1729, ge=0)
    seeds: list[int] = [0, 1, 2]
    validation_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    subset_fraction: float = Field(default=0.5, gt=0, le=1)
    epsilon: float = Field(default=0.0001, ge=0)
    settings: dict = {}

    @model_validator(mode="after")
    def valid(self):
        relative_path(self.train_file)
        relative_path(self.test_file)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("Distinct nonnegative seeds required")
        primary = self.metric_specs.get(self.metric)
        if primary is not None and self.direction is not None:
            if primary.direction != self.direction:
                raise ValueError("Primary direction conflicts with its metric specification")
        if (
            primary is None
            and self.direction is None
            and self.metric not in LEGACY_METRIC_DIRECTIONS
        ):
            raise ValueError("Custom primary metrics require an explicit direction")
        if not self.prediction_columns or len(set(self.prediction_columns)) != len(
            self.prediction_columns
        ):
            raise ValueError("Distinct prediction columns required")
        return self


class LiveSpec(Record):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    schema_version: Literal[1] = 1
    engine_version: Literal["legacy-live-v1", "research-v2"] = "legacy-live-v1"
    workflow: ResearchPolicy = Field(default_factory=ResearchPolicy)
    native: NativeSpec | WorkspaceSpec
    development: DevelopmentSpec
    model: str = Field(min_length=1)
    provider: ModelProvider = Field(default_factory=ModelProvider)
    mode: Literal["smoke", "research"] = "smoke"
    max_calls: int = Field(ge=1)
    campaign_seconds: float = Field(gt=0)
    call_seconds: float = Field(default=120, gt=0)
    max_prompt_chars: int = Field(default=200_000, ge=1000)
    history_code_chars: int = Field(default=4000, ge=0)
    log_tail_chars: int = Field(default=4000, ge=0)
    max_experiments: int = Field(default=30, ge=1)
    # Usage is observed after a call, so this is a stop threshold, not a hard token cap.
    token_stop_threshold: int | None = Field(default=None, gt=0)
    model_retries: int = Field(default=1, ge=0, le=3)
    # Semantic correction is distinct from transport retries and engineering.
    # It is off unless explicitly configured, charged to the existing budgets.
    agent_output_retries: int = Field(default=0, ge=0, le=1)
    engineering_retries: int = Field(default=1, ge=0, le=2)
    search_rounds: int = Field(default=2, ge=1, le=4)
    limitation_rounds: int = Field(default=1, ge=1, le=16)
    ablation_refinements: int = Field(default=1, ge=0, le=1)
    review_rounds: int = Field(default=1, ge=0, le=2)
    supplementary_seed_count: int = Field(default=2, ge=1, le=32)
    meta_refinements: int = Field(default=1, ge=0, le=1)
    # Separate campaign-wide allowance; editorial work must never spend an
    # experimental refinement slot or trigger training by itself.
    manuscript_revisions: int = Field(default=1, ge=0, le=2)
    success_target: int = Field(default=4, ge=1, le=4)
    sources: list[Source] = []
    implementation: dict[str, str] = {}

    @model_validator(mode="after")
    def valid(self):
        from ..providers.config import assert_credential_free
        assert_credential_free(self.provider, self.model_dump(mode="json"))
        if self.native.mode != "baseline":
            raise ValueError("Live campaigns start from a frozen baseline manifest")
        if len({s.id for s in self.sources}) != len(self.sources) or any(
            s.id == "task" for s in self.sources
        ):
            raise ValueError("Source IDs must be unique; task is reserved")
        return self


class Observation(Record):
    id: str
    phase: Literal["subset", "full", "rebuttal", "final"]
    status: Literal["success", "execution_failed", "invalid", "timeout", "interrupted"]
    protocol_hash: str
    protocol_ref: str | None = None
    code_ref: str
    metric: MetricName
    value: MetricValue | None = None
    values: list[MetricValue] = []
    metrics: dict[MetricName, MetricValue] = {}
    metric_values: dict[MetricName, list[MetricValue]] = {}
    reason: str = ""
    log_refs: list[str] = []
    prediction_refs: list[str] = []
    artifact_refs: list[dict[str, str]] = []
    bundle_refs: list[str] = []
    evaluation_refs: list[str] = []
    source_tree_ref: str | None = None
    timing_refs: list[str] = []
    metric_rows: list[dict] = Field(default_factory=list)
    legacy_result_ref: str | None = None
    seconds: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def evidence(self):
        if self.status == "success" and (
            self.value is None or not self.values or not self.prediction_refs
        ):
            raise ValueError("Successful results require trusted metrics and prediction evidence")
        if self.status != "success" and (
            self.value is not None or self.values or self.metrics or self.metric_values
        ):
            raise ValueError("Execution failures cannot carry scientific scores")
        if self.metrics or self.metric_values:
            if set(self.metrics) != set(self.metric_values) or self.metric not in self.metrics:
                raise ValueError(
                    "Metric aggregates and execution values must contain the primary metric"
                )
            if (
                self.metrics[self.metric] != self.value
                or self.metric_values[self.metric] != self.values
            ):
                raise ValueError("Primary metric aliases differ from the complete metric evidence")
            if any(len(values) != len(self.values) for values in self.metric_values.values()):
                raise ValueError("Every metric requires one value per execution")
        return self
