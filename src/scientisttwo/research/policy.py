"""Frozen workflow choices; model backend and experiment task are independent."""
from typing import Literal

from pydantic import Field, model_validator

from ..models import Record


class ResearchPolicy(Record):
    version: Literal[1] = 1
    profile: Literal["offline-pilot", "research-pilot", "paper-methodology"] = "research-pilot"
    seed_ideas: int = Field(default=3, ge=2, le=8)
    novelty_attempts: int = Field(default=2, ge=1, le=3)
    max_ablation_plans: int = Field(default=3, ge=1, le=6)
    writing_refinements: int = Field(default=1, ge=0, le=2)
    baseline_mode: Literal["prepared_reference", "reconstruct_from_sources"] = "prepared_reference"
    literature: dict = Field(default_factory=dict)
    network_literature: bool = False
    closing_fraction: float = Field(default=0.2, ge=0, lt=0.5)
    estimated_tokens_per_call: int = Field(default=15000, ge=1)
    compiler_image: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$")
    require_pdf: bool = False
    template_directory: str | None = None
    # The schema default preserves previously frozen manifests. New live research
    # CLI manifests explicitly select PaperOrchestra; offline fixtures stay light.
    writer_backend: Literal["canonical", "paperorchestra"] = "canonical"
    paperorchestra_root: str | None = None
    paperorchestra_reflections: int = Field(default=1, ge=1, le=3)
    paperorchestra_ensemble: int = Field(default=3, ge=1, le=3)
    paperorchestra_max_calls: int = Field(default=24, ge=1, le=120)
    # Optional adapter mitigation, not a model token cap. None retains the
    # original complete-response request and its existing 250000-character schema.
    paperorchestra_max_response_chars: int | None = Field(default=None, ge=1, le=250_000, strict=True)
    # A soft scope choice; complete source and evidence validation remain mandatory.
    paperorchestra_writing_scope: Literal["upstream", "concise_note"] = "upstream"
    # Some Slurm Podman profiles enforce physical RAM but inherit the host's
    # swap policy. This must be selected and recorded explicitly.
    paperorchestra_swap_policy: Literal["deny", "inherit"] = "deny"

    @model_validator(mode="after")
    def pdf_contract(self):
        if self.writer_backend == "paperorchestra":
            if self.compiler_image is None:
                raise ValueError("PaperOrchestra requires a pinned PDF image")
            if not self.paperorchestra_root:
                from ..integrations.paperorchestra.runtime import bundled_upstream_root
                # Preserve explicit frozen paths. Actual byte verification is a
                # writer/preflight operation, not a model-schema side effect.
                self.paperorchestra_root = str(bundled_upstream_root())
            if self.template_directory is not None:
                raise ValueError("PaperOrchestra uses its pinned upstream template")
            self.require_pdf = True
        if self.require_pdf and self.compiler_image is None:
            raise ValueError("Required PDF generation needs an explicitly pinned compiler image")
        return self


def profile_updates(name):
    if name not in {"offline-pilot", "research-pilot", "paper-methodology"}:
        raise ValueError("Unknown research workflow profile")
    paper = name == "paper-methodology"
    return {
        "engine_version": "research-v2",
        "workflow": ResearchPolicy(profile=name, seed_ideas=5 if paper else 3,
                                   max_ablation_plans=6 if paper else 3),
        "limitation_rounds": 16 if paper else 2,
        "search_rounds": 4 if paper else 2,
        "engineering_retries": 2 if paper else 1,
        "ablation_refinements": 1,
        "review_rounds": 2 if paper else 1,
        "meta_refinements": 1,
        "manuscript_revisions": 1,
        "success_target": 4,
    }
