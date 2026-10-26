"""Structured scientific actions; descriptive prose does not authorize execution."""
from typing import Literal
from pydantic import Field, model_validator
from ..models import Record
from ..live.parent_edits import ParentChangePlan, ParentPatch
from ..live.models import Hypothesis
from ..live.supplementary import SupplementaryDiagnostic, ConcernDisposition


class BaselinePlan(Record):
    mode: Literal["prepared_reference", "reconstruct_from_sources"]
    reason: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)


class FullSetHandoff(Record):
    action: Literal["ReuseSnapshot", "ScalingPatch", "ScientificChange"]
    reason: str = Field(min_length=1)
    patch: ParentPatch | None = None
    change_plan: ParentChangePlan | None = None
    revised_idea: Hypothesis | None = None

    @model_validator(mode="after")
    def coherent(self):
        if self.action == "ReuseSnapshot":
            if self.patch is not None or self.change_plan is not None:
                raise ValueError("ReuseSnapshot cannot contain source edits")
        elif self.patch is None or self.change_plan is None:
            raise ValueError("Source changes require a bound patch and declared scope")
        if (self.action == "ScientificChange") != (self.revised_idea is not None):
            raise ValueError("Scientific changes require a revised hypothesis, and other handoffs cannot change it")
        return self


class SelectionReview(Record):
    selected_id: str
    reason: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class AblationDesign(Record):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    component: str = Field(min_length=1)
    question: str = Field(min_length=1)
    changed_files: list[str] = Field(min_length=1, max_length=24)
    invariants: list[str] = Field(min_length=1)
    known_confounds: list[str] = Field(default_factory=list)


class AblationDesigns(Record):
    plans: list[AblationDesign] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def unique(self):
        if len({p.id for p in self.plans}) != len(self.plans):
            raise ValueError("Ablation plan IDs must be distinct")
        return self


class ControlReview(Record):
    decision: Literal["Accept", "Reject", "DiagnosticOnly"]
    reason: str = Field(min_length=1)
    confounds: list[str] = Field(default_factory=list)


class MethodDisposition(Record):
    action: Literal["ChangeMethod", "NoMethodChange"]
    reason: str = Field(min_length=1)


class SupplementaryDesign(Record):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    concern_ids: list[str] = Field(min_length=1)
    purpose: str
    seed_count: int = Field(ge=1, le=32, strict=True)
    execution_kind: Literal["repeat", "paired_control"]
    evaluation_split: Literal["development", "final"] = "development"
    diagnostics: list[SupplementaryDiagnostic] = Field(min_length=1)
    change_plan: ParentChangePlan | None = None

    @model_validator(mode="after")
    def scoped(self):
        if self.execution_kind == "repeat" and self.change_plan is not None:
            raise ValueError("A repetition must reuse the frozen method")
        if self.execution_kind == "paired_control" and (
            self.change_plan is None or len(self.change_plan.changed_files) != 1
        ):
            raise ValueError("Supplementary controls require one declared component file")
        return self


class SupplementaryDesigns(Record):
    plans: list[SupplementaryDesign] = Field(max_length=3)
    unaddressed_concerns: list[ConcernDisposition] = Field(default_factory=list)
