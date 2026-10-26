"""Model-free, immutable acceptance schedules and conservative evidence gates.

This module never starts a process, imports a model backend, or opens a network
connection.  A plan is not execution authorization.  Assessments bind supplied
receipts to files and identities; their scientific conclusions still require the
independent audit that produced those receipts.
"""

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


SHA256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Family = Literal["cpu_frames", "paired_timing", "checkpoint_pipeline"]
Status = Literal["pass", "fail", "blocked", "not_exercised"]
WriterBackend = Literal["canonical", "paperorchestra"]

# Baseline reconstruction, engineering and method/control revisions must be
# implemented and controlled-tested, but a natural successful run need not take
# these conditional branches. Missing mandatory LIVE roles need a preregistered
# challenge; branch tests are never relabeled as naturally occurring live work.
REQUIRED_LIVE_ROLES = (
    "limitations", "verify_limitations", "initial_idea", "additional_idea",
    "novelty_checker", "baseline_plan", "implement", "critic", "full_set_coder",
    "selection_review", "evolve", "ablation_planner", "ablation_coder",
    "control_reviewer", "ablation_critic", "initial_drafter", "writing_critic",
    "draft_enhancer", "scientific_review", "rebuttal_planner", "meta",
)
REQUIRED_ROLES = (*REQUIRED_LIVE_ROLES, "baseline_coder", "engineer", "refine",
                  "method_disposition", "rebuttal_coder")


def required_role_sets(writer_backend: WriterBackend = "canonical"):
    """Use actual controller invocation names for the frozen writing adapter."""
    if writer_backend not in {"canonical", "paperorchestra"}:
        raise ValueError("Unknown acceptance writer backend")
    replacements = ({"initial_drafter": "paperorchestra_model",
                     "draft_enhancer": "paperorchestra_bindings"}
                    if writer_backend == "paperorchestra" else {})
    return tuple(replacements.get(role, role) for role in REQUIRED_ROLES), tuple(
        replacements.get(role, role) for role in REQUIRED_LIVE_ROLES)


REQUIRED_BRANCHES = (
    "scientific_rejection", "engineering_failure", "no_successful_idea",
    "successful_method_refinement", "unsuccessful_refinement_preserves_incumbent",
    "manuscript_revision_zero_experiments", "exact_supplementary_request",
    "unsupported_request_unfulfilled", "interrupted_resume_no_duplicate_work",
    "completed_resume_no_operations", "budget_exhaustion_explicit",
)
TASKS = {
    "cpu_frames": "bundled-iris-wine-digits",
    "paired_timing": "vectorization",
    "checkpoint_pipeline": "cifar10",
}


class FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def identity(value) -> str:
    """Canonical JSON identity, also used to bind a complete implementation map."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


class EngineIdentity(FrozenRecord):
    engine_version: Literal["research-v2"] = "research-v2"
    implementation_sha256: SHA256


class EvidenceFile(FrozenRecord):
    path: str = Field(min_length=1)
    sha256: SHA256

    def verify(self) -> str | None:
        path = Path(self.path)
        if not path.is_absolute():
            return f"Evidence path must be absolute: {path}"
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            return f"Evidence path contains a symlink: {path}"
        if not path.is_file():
            return f"Evidence file missing: {path}"
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as error:
            return f"Evidence cannot be read: {path}: {error}"
        return None if digest.hexdigest() == self.sha256 else f"Evidence modified: {path}"


class TaskBinding(FrozenRecord):
    family: Family
    task_id: str
    variant: str = "audited-v1"
    prepared_manifest_sha256: SHA256 | None = None
    protocol_sha256: SHA256 | None = None
    runtime_sha256: SHA256 | None = None
    prepared_path: str | None = None
    data_provenance: Literal["actual_data", "bundled_real_data", "synthetic", "unverified"] = "unverified"
    datasets: tuple[str, ...] = ()


class AcceptanceBudget(FrozenRecord):
    max_calls: int = Field(ge=0)
    observed_token_stop: int = Field(ge=0)
    seconds: int = Field(gt=0)
    model: str | None = None
    model_retries: Literal[0] = 0
    output_corrections: int = Field(default=1, ge=0, le=1)
    closing_fraction: float = Field(default=0.2, ge=0.2, lt=0.5)

    @model_validator(mode="after")
    def live_budget(self):
        if self.model is None and (self.max_calls or self.observed_token_stop):
            raise ValueError("Scripted validation has zero research-model calls and tokens")
        if self.model is not None and (not self.max_calls or not self.observed_token_stop):
            raise ValueError("Live validation requires explicit call and observed-token limits")
        return self


class StageEstimate(FrozenRecord):
    stage: str
    calls: int = Field(ge=0)


class CostEstimate(FrozenRecord):
    stage_calls: tuple[StageEstimate, ...]
    estimated_calls: int
    estimated_tokens: int
    reserved_closing_calls: int
    reserved_closing_tokens: int
    reserved_closing_seconds: int
    fits_limits: bool
    assumptions: tuple[str, ...]


def estimate_acceptance(budget: AcceptanceBudget, *, scope: str = "full",
                        profile: str = "research-pilot", estimated_tokens_per_call: int = 15000,
                        stage_calls: Mapping[str, int] | None = None,
                        writer_backend: WriterBackend = "canonical") -> CostEstimate:
    """Auditable planning arithmetic, not a prediction or hard token guarantee."""
    if estimated_tokens_per_call <= 0 or profile not in {
        "research-pilot", "offline-pilot", "paper-methodology"
    }:
        raise ValueError("Positive per-call estimate and a supported workflow profile required")
    required_role_sets(writer_backend)
    writer_assumptions = ()
    if stage_calls is None:
        if budget.model is None:
            stage_calls = {}
        elif scope == "probe":
            stage_calls = {"proposal_and_execution_review": 8, "evidence_and_reporting": 4}
        elif scope == "challenge":
            stage_calls = {"concern_and_rebuttal": 6, "writing_and_review": 6}
            if writer_backend == "paperorchestra":
                stage_calls["writing_and_review"] = (24 + 2) * (1 + 1) + 2
        elif scope == "full":
            paper = profile == "paper-methodology"
            stage_calls = {
                "limitations": 32 if paper else 4,
                "seed_and_novelty": 20 if paper else 12,
                "baseline_and_search": 44 if paper else 20,
                "ablations_and_refinement": 18,
                "writing_review_and_meta": 22 if paper else 14,
            }
            if writer_backend == "paperorchestra":
                # Match the controller's configured default admission ceiling:
                # 24 upstream calls + indexing + independent critic per writing
                # pass, with one writing refinement. Include initial drafting,
                # review/rebuttal redrafts, one editorial and one method restart.
                review_rounds = 2 if paper else 1
                manuscript_cycles = 1 + review_rounds + 1 + 1
                stage_calls["writing_review_and_meta"] = (
                    ((24 + 2) * (1 + 1) + 1) * manuscript_cycles + 2)
        else:
            raise ValueError("Unknown acceptance scope")
        if budget.model is not None and writer_backend == "paperorchestra" and scope in {"full", "challenge"}:
            writer_assumptions = (
                "PaperOrchestra structural estimate: at most 24 upstream calls plus indexing and "
                "writing criticism per pass, one writing refinement; full includes review redrafts "
                "and one editorial and method restart. These are planning ceilings, not calibrated usage.",
            )
    stages = tuple(StageEstimate(stage=k, calls=v) for k, v in sorted(stage_calls.items()))
    calls = sum(stage.calls for stage in stages)
    tokens = calls * estimated_tokens_per_call
    return CostEstimate(
        stage_calls=stages, estimated_calls=calls, estimated_tokens=tokens,
        reserved_closing_calls=math.ceil(budget.max_calls * budget.closing_fraction),
        reserved_closing_tokens=math.ceil(budget.observed_token_stop * budget.closing_fraction),
        reserved_closing_seconds=math.ceil(budget.seconds * budget.closing_fraction),
        fits_limits=calls <= budget.max_calls and tokens <= budget.observed_token_stop,
        assumptions=(
            f"{estimated_tokens_per_call} observed tokens per admitted call; not calibrated usage",
            "Stage estimates include closing calls; reserves are portions of the total, not extra calls",
            "Conditional repairs may exhaust limits; no automatic retry until a favorable result",
            "Observed tokens are known after a response and may exceed the stop threshold",
            "Wall time is a ceiling; framework/GPU calibration remains a separate prerequisite",
        ) + writer_assumptions,
    )


class AcceptanceSlot(FrozenRecord):
    id: str
    family: Family
    task_id: str
    scope: Literal["scripted_full", "live_probe", "live_full", "live_challenge", "compatibility"]
    mandatory: bool = True
    conditional: bool = False
    profile: str = "research-pilot"
    budget: AcceptanceBudget
    estimate: CostEstimate
    prerequisite_slot: str | None = None


class VerificationProof(FrozenRecord):
    """A verifier-produced receipt; file hashes attest bytes, not truth of prose."""
    engine: EngineIdentity
    binding_sha256: SHA256 | None = None
    status: Literal["pass", "fail", "unverified"] = "unverified"
    implemented_roles: tuple[str, ...] = ()
    controlled_roles: tuple[str, ...] = ()
    controlled_branches: tuple[str, ...] = ()
    evidence: tuple[EvidenceFile, ...] = ()


class AcceptancePlan(FrozenRecord):
    schema_version: Literal[1] = 1
    engine: EngineIdentity
    bindings: tuple[TaskBinding, ...]
    slots: tuple[AcceptanceSlot, ...]
    implementation_proof: VerificationProof | None = None
    prerequisite_proofs: tuple[VerificationProof, ...] = ()
    writer_backend: WriterBackend = "canonical"
    required_roles: tuple[str, ...] = REQUIRED_ROLES
    required_live_roles: tuple[str, ...] = REQUIRED_LIVE_ROLES
    required_branches: tuple[str, ...] = REQUIRED_BRANCHES
    automatic_execution: Literal[False] = False
    authorization: Literal["separate_explicit_live_and_worker_configuration_required"] = "separate_explicit_live_and_worker_configuration_required"
    all_13_readiness: Literal["separate_release_gate"] = "separate_release_gate"
    blockers: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def writer_role_defaults(cls, value):
        if isinstance(value, Mapping):
            value = dict(value)
            required, live = required_role_sets(value.get("writer_backend", "canonical"))
            value.setdefault("required_roles", required)
            value.setdefault("required_live_roles", live)
        return value

    @model_validator(mode="after")
    def writer_role_contract(self):
        required, live = required_role_sets(self.writer_backend)
        if self.required_roles != required or self.required_live_roles != live:
            raise ValueError("Frozen required roles differ from the selected writer backend")
        return self

    @model_serializer(mode="wrap")
    def preserve_legacy_canonical_identity(self, handler):
        value = handler(self)
        # The absence of this newly introduced field means canonical writing.
        # Keep historical schema-v1 plan hashes and their attempt links intact.
        if self.writer_backend == "canonical":
            value.pop("writer_backend", None)
        return value

    @property
    def sha256(self):
        return identity(self)


class AttemptRecord(FrozenRecord):
    slot_id: str
    attempt_id: str
    plan_sha256: SHA256
    engine: EngineIdentity
    binding_sha256: SHA256
    scope: Literal["scripted_full", "live_probe", "live_full", "live_challenge", "compatibility"]
    status: Literal["scheduled", "running", "pass", "fail", "blocked", "skipped", "unknown"]
    reason: str = ""
    model: str | None = None
    data_provenance: Literal["actual_data", "bundled_real_data", "synthetic", "unverified"] = "unverified"
    execution_origin: Literal["isolated_real", "synthetic", "unverified"] = "unverified"
    datasets: tuple[str, ...] = ()
    model_calls: int = Field(default=0, ge=0)
    observed_tokens: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0, ge=0)
    natural_live_roles: tuple[str, ...] = ()
    challenge_live_roles: tuple[str, ...] = ()
    controlled_roles: tuple[str, ...] = ()
    real_experiments: Status = "not_exercised"
    manuscript_pdf: Status = "not_exercised"
    final_evaluation: Status = "not_exercised"
    independent_audit: Status = "not_exercised"
    completed_resume: Status = "not_exercised"
    runtime_probe: Status = "not_exercised"
    workflow_complete: bool = False
    scientific_outcome: Literal["supported_candidate", "no_improvement", "incomplete"] = "incomplete"
    evidence: tuple[EvidenceFile, ...] = ()
    # Additional attempts are never implicitly permitted by a previous failure.
    supersedes_attempt_id: str | None = None
    retry_authorization: EvidenceFile | None = None


def _verify_proof(proof, engine, binding=None):
    problems = []
    if proof.engine != engine:
        problems.append("Stale proof: implementation/engine identity differs")
    if binding is not None and proof.binding_sha256 != identity(binding):
        problems.append("Stale proof: prepared data/protocol/runtime identity differs")
    if proof.status != "pass":
        problems.append("Verification proof is not a pass")
    if not proof.evidence:
        problems.append("Verification proof has no immutable evidence files")
    problems.extend(error for item in proof.evidence if (error := item.verify()))
    return problems


def validate_fresh_manifest(plan: AcceptancePlan, *, engine_identity=None,
                            bindings: Sequence[TaskBinding | Mapping] | None = None) -> tuple[str, ...]:
    """Check current identities and receipt bytes without upgrading old evidence."""
    engine = _engine(engine_identity) if engine_identity is not None else plan.engine
    errors = []
    required, live = required_role_sets(plan.writer_backend)
    if plan.required_roles != required or plan.required_live_roles != live:
        errors.append("Frozen required roles differ from the selected writer backend")
    if engine != plan.engine:
        errors.append("Acceptance manifest belongs to a different implementation")
    if bindings is not None:
        current = tuple(TaskBinding.model_validate(v) for v in bindings)
        if {v.family: v for v in current} != {v.family: v for v in plan.bindings}:
            errors.append("Acceptance manifest belongs to different data/protocol/runtime bindings")
    if plan.implementation_proof is not None:
        errors.extend(_verify_proof(plan.implementation_proof, engine))
    for proof in plan.prerequisite_proofs:
        matches = [v for v in plan.bindings if identity(v) == proof.binding_sha256]
        if not matches:
            errors.append("Prerequisite proof does not bind a current task")
        else:
            errors.extend(_verify_proof(proof, engine, matches[0]))
    for binding in plan.bindings:
        if binding.prepared_path:
            manifest = Path(binding.prepared_path) / "manifest.json"
            try:
                value = _read_json(manifest)
            except (OSError, ValueError) as error:
                errors.append(f"Prepared manifest unavailable: {manifest}: {error}")
            else:
                if identity(value) != binding.prepared_manifest_sha256:
                    errors.append(f"Prepared manifest changed: {manifest}")
    return tuple(errors)


def _engine(value):
    if isinstance(value, EngineIdentity):
        return value
    if not isinstance(value, Mapping) or not value:
        raise ValueError("A current engine identity or nonempty implementation map is required")
    if "implementation_sha256" in value:
        return EngineIdentity.model_validate(value)
    return EngineIdentity(implementation_sha256=identity(dict(value)))


def _read_json(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Metadata must be a bounded regular JSON file, not a symlink")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Prepared metadata must be a JSON object")
    return value


def prepare_acceptance_plan(*, engine_identity, registry: str | Path | Mapping | None = None,
                            paths: Mapping[str, str | Path] | None = None,
                            profiles: Mapping[str, str] | None = None,
                            bindings: Sequence[TaskBinding | Mapping] = (),
                            implementation_proof: VerificationProof | Mapping | None = None,
                            prerequisite_proofs: Sequence[VerificationProof | Mapping] = (),
                            include_ogbn: bool = True,
                            writer_backend: WriterBackend = "canonical") -> AcceptancePlan:
    """Prepare a bounded schedule; never infer validation from available data.

    ``registry`` may be a testbed root or family-to-TaskBinding mapping. ``paths``
    maps family names to prepared directories. Explicit ``bindings`` override
    discovered paths; this lets callers provide frozen bundled-data and runtime
    descriptors. No framework probes or data downloads occur here.
    """
    engine = _engine(engine_identity)
    required_roles, required_live_roles = required_role_sets(writer_backend)
    paths, profiles = dict(paths or {}), dict(profiles or {})
    if set(paths) - TASKS.keys() or set(profiles) - TASKS.keys():
        raise ValueError("Paths and profiles must name declared representative families")
    configured = dict(registry) if isinstance(registry, Mapping) else {}
    configured.update({v.family: v for item in bindings if (v := TaskBinding.model_validate(item))})
    result = []
    blockers = []
    for family, task in TASKS.items():
        if family in configured:
            binding = TaskBinding.model_validate(configured[family])
        else:
            directory = paths.get(family)
            if directory is None and registry is not None and not isinstance(registry, Mapping) and family != "cpu_frames":
                directory = Path(registry) / task / "prepared"
            binding = TaskBinding(family=family, task_id=task,
                                  variant="bundled-real-v1" if family == "cpu_frames" else "audited-v1")
            if directory is not None:
                manifest_path = Path(directory).absolute() / "manifest.json"
                try:
                    manifest = _read_json(manifest_path)
                    if manifest.get("task_id") != task:
                        raise ValueError("Prepared manifest names a different task")
                    binding = binding.model_copy(update={
                        "prepared_path": str(manifest_path.parent),
                        "prepared_manifest_sha256": identity(manifest),
                        "data_provenance": "actual_data" if manifest.get("provenance") == "actual_data" else "unverified",
                    })
                except (OSError, ValueError) as error:
                    blockers.append(f"{family}: prepared metadata unavailable: {error}")
        if binding.family != family or binding.task_id != task:
            raise ValueError(f"Canonical representative binding mismatch: {family}/{task}")
        for key in ("prepared_manifest_sha256", "protocol_sha256", "runtime_sha256"):
            if getattr(binding, key) is None:
                blockers.append(f"{family}: missing frozen {key}")
        if binding.data_provenance not in {"actual_data", "bundled_real_data"}:
            blockers.append(f"{family}: actual data provenance is unverified")
        if family == "cpu_frames" and set(binding.datasets) != {"iris", "wine", "digits"}:
            blockers.append("cpu_frames: must bind actual bundled Iris, Wine and Digits")
        result.append(binding)
    slots = []
    for family, task in TASKS.items():
        budget = AcceptanceBudget(max_calls=0, observed_token_stop=0, seconds=4 * 3600)
        slots.append(AcceptanceSlot(
            id=f"scripted-{family}", family=family, task_id=task, scope="scripted_full",
            profile="offline-pilot", budget=budget,
            estimate=estimate_acceptance(budget, profile="offline-pilot", writer_backend=writer_backend),
        ))
        full = family == "checkpoint_pipeline"
        budget = AcceptanceBudget(max_calls=120 if full else 20,
                                  observed_token_stop=1_500_000 if full else 250_000,
                                  seconds=4 * 3600 if full else 3600, model="gpt-6-astra")
        profile = profiles.get(family, "research-pilot")
        estimate = estimate_acceptance(budget, scope="full" if full else "probe", profile=profile,
                                       writer_backend=writer_backend)
        if not estimate.fits_limits:
            blockers.append(f"live-{family}: selected profile estimate exceeds frozen limits")
        slots.append(AcceptanceSlot(
            id=f"live-{family}", family=family, task_id=task,
            scope="live_full" if full else "live_probe", budget=budget, estimate=estimate,
            profile=profile, prerequisite_slot=f"scripted-{family}",
        ))
    budget = AcceptanceBudget(max_calls=20, observed_token_stop=300_000, seconds=90 * 60,
                              model="gpt-6-astra")
    slots.append(AcceptanceSlot(
        id="live-conditional-challenge", family="checkpoint_pipeline", task_id="cifar10",
        scope="live_challenge", conditional=True, budget=budget,
        estimate=estimate_acceptance(budget, scope="challenge", writer_backend=writer_backend),
        prerequisite_slot="live-checkpoint_pipeline",
    ))
    if include_ogbn:
        budget = AcceptanceBudget(max_calls=0, observed_token_stop=0, seconds=4 * 3600)
        slots.append(AcceptanceSlot(
            id="compatibility-ogbn", family="checkpoint_pipeline", task_id="ogbn-arxiv",
            scope="compatibility", mandatory=False, budget=budget,
            estimate=estimate_acceptance(budget, profile="offline-pilot", writer_backend=writer_backend),
        ))
    proof = VerificationProof.model_validate(implementation_proof) if implementation_proof is not None else None
    if proof is None:
        blockers.append("Implemented roles and controlled branch coverage lack a current proof")
    proofs = tuple(VerificationProof.model_validate(v) for v in prerequisite_proofs)
    plan = AcceptancePlan(engine=engine, bindings=tuple(result), slots=tuple(slots),
                          implementation_proof=proof, prerequisite_proofs=proofs,
                          writer_backend=writer_backend, required_roles=required_roles,
                          required_live_roles=required_live_roles,
                          blockers=tuple(blockers))
    fresh = validate_fresh_manifest(plan)
    return plan.model_copy(update={"blockers": (*plan.blockers, *fresh)})


def _record_errors(plan, slot, record, binding):
    errors = []
    if record.plan_sha256 != plan.sha256 or record.engine != plan.engine:
        errors.append("Attempt belongs to a different plan or implementation")
    if record.binding_sha256 != identity(binding):
        errors.append("Attempt binds different prepared data/protocol/runtime")
    if record.scope != slot.scope:
        errors.append("Probe/full/challenge scope does not match its scheduled slot")
    if record.data_provenance != binding.data_provenance or record.data_provenance not in {
        "actual_data", "bundled_real_data"
    } or record.execution_origin != "isolated_real":
        errors.append("Synthetic or unverified data/execution cannot establish real validation")
    if slot.family == "cpu_frames" and set(record.datasets) != {"iris", "wine", "digits"}:
        errors.append("CPU validation does not cover all three bundled real datasets")
    if record.model != slot.budget.model:
        errors.append("Research model differs from the frozen schedule")
    if slot.budget.model is None and (record.model_calls or record.observed_tokens
                                      or record.natural_live_roles or record.challenge_live_roles):
        errors.append("Scripted validation cannot contain live research calls or live role claims")
    if slot.budget.model is not None and not record.model_calls:
        errors.append("Live validation has no executed model calls")
    if record.model_calls > slot.budget.max_calls or record.elapsed_seconds > slot.budget.seconds:
        errors.append("Attempt exceeded its call or wall-time ceiling")
    if record.scope != "live_challenge" and record.challenge_live_roles:
        errors.append("Controlled challenge roles cannot be relabeled as natural campaign work")
    if record.scope == "live_challenge" and record.natural_live_roles:
        errors.append("Challenge roles must remain explicitly challenge-labeled")
    if not record.evidence:
        errors.append("Attempt has no immutable evidence files")
    errors.extend(error for item in record.evidence if (error := item.verify()))
    for field in ("real_experiments", "independent_audit", "runtime_probe"):
        if getattr(record, field) != "pass":
            errors.append(f"{field} is not pass")
    if slot.scope in {"scripted_full", "live_full"}:
        for field in ("final_evaluation", "completed_resume"):
            if getattr(record, field) != "pass":
                errors.append(f"{field} is not pass")
        if not record.workflow_complete or record.scientific_outcome == "incomplete":
            errors.append("Full workflow did not reach a documented terminal outcome")
    if slot.scope in {"live_full", "live_challenge"} and record.manuscript_pdf != "pass":
        errors.append("Complete manuscript PDF is not independently verified")
    return errors


def assess_acceptance(plan: AcceptancePlan | Mapping, records: Sequence[AttemptRecord | Mapping]):
    """Assess every scheduled attempt; return JSON-ready facts, never execute work."""
    plan = AcceptancePlan.model_validate(plan)
    records = tuple(AttemptRecord.model_validate(v) for v in records)
    if len({v.attempt_id for v in records}) != len(records):
        raise ValueError("Duplicate attempt completion/accounting records are not permitted")
    slots = {slot.id: slot for slot in plan.slots}
    unknown = {v.slot_id for v in records} - slots.keys()
    if unknown:
        raise ValueError(f"Unscheduled attempts are not accepted: {sorted(unknown)}")
    errors = [*plan.blockers, *validate_fresh_manifest(plan)]
    proof = plan.implementation_proof
    missing_implemented = sorted(set(plan.required_roles) - set(proof.implemented_roles if proof else ()))
    missing_controlled = sorted(set(plan.required_roles) - set(proof.controlled_roles if proof else ()))
    missing_branches = sorted(set(plan.required_branches) - set(proof.controlled_branches if proof else ()))
    if missing_implemented or missing_controlled or missing_branches:
        errors.append("Mandatory implementation/controlled role or branch coverage is incomplete")
    results = {}
    natural, challenge, attempts = set(), set(), []
    bindings = {v.family: v for v in plan.bindings}
    for slot in plan.slots:
        relevant = [v for v in records if v.slot_id == slot.id]
        latest = None
        for index, record in enumerate(relevant):
            row_errors = []
            if index:
                previous = relevant[index - 1]
                if record.supersedes_attempt_id != previous.attempt_id or record.retry_authorization is None:
                    row_errors.append("Additional attempts need an explicit recorded retry authorization")
                elif error := record.retry_authorization.verify():
                    row_errors.append(error)
                if previous.status in {"scheduled", "running", "unknown"}:
                    row_errors.append("Prior attempt completion is unresolved; duplicate work is not authorized")
            elif record.supersedes_attempt_id is not None:
                row_errors.append("Retry references an absent prior attempt")
            # Compatibility is reported independently and cannot fill CIFAR gates.
            if slot.scope == "compatibility":
                row_errors.append("OGBN compatibility requires its own task-specific assessment")
            elif record.status == "pass":
                row_errors.extend(_record_errors(plan, slot, record, bindings[slot.family]))
            state = "pass" if record.status == "pass" and not row_errors else (
                "fail" if record.status in {"pass", "fail"} or row_errors else "blocked"
            )
            if slot.scope == "compatibility" and record.status != "fail":
                state = "blocked"
            row = {"attempt_id": record.attempt_id, "slot_id": slot.id,
                   "reported_status": record.status, "status": state, "errors": row_errors,
                   "reason": record.reason, "model_calls": record.model_calls,
                   "observed_tokens": record.observed_tokens, "elapsed_seconds": record.elapsed_seconds,
                   "token_threshold_exceeded": record.observed_tokens > slot.budget.observed_token_stop}
            attempts.append(row)
            latest = row
            if state == "pass":
                natural.update(record.natural_live_roles)
                challenge.update(record.challenge_live_roles)
        results[slot.id] = {"status": latest["status"] if latest else "not_exercised",
                            "attempt_count": len(relevant), "mandatory": slot.mandatory,
                            "conditional": slot.conditional, "scope": slot.scope}
    for slot in plan.slots:
        if slot.prerequisite_slot and results[slot.id]["status"] == "pass":
            if results[slot.prerequisite_slot]["status"] != "pass":
                results[slot.id]["status"] = "blocked"
                errors.append(f"{slot.id}: required predecessor has not passed")
    missing_natural = sorted(set(plan.required_live_roles) - natural)
    missing_live = sorted(set(plan.required_live_roles) - natural - challenge)
    challenge_required = bool(missing_natural)
    if missing_live:
        errors.append("Mandatory roles lack real live execution evidence")
    if challenge_required and results["live-conditional-challenge"]["status"] != "pass":
        errors.append("A separately preregistered real-evidence challenge is required")
    for slot in plan.slots:
        if slot.mandatory and (not slot.conditional or challenge_required):
            if results[slot.id]["status"] != "pass":
                errors.append(f"{slot.id}: required gate is {results[slot.id]['status']}")
    return {
        "schema_version": 1, "plan_sha256": plan.sha256,
        "status": "pass" if not errors else "blocked",
        "release_ready": not errors,
        "scope": "Identity and immutable-receipt gate assessment; independent audits establish execution evidence, not certified scientific validity",
        "engine": plan.engine.model_dump(mode="json"),
        "writer_backend": plan.writer_backend,
        "errors": list(dict.fromkeys(errors)), "gates": results, "attempts": attempts,
        "coverage": {"missing_implemented_roles": missing_implemented,
                     "missing_controlled_roles": missing_controlled,
                     "missing_controlled_branches": missing_branches,
                     "natural_live_roles": sorted(natural), "challenge_live_roles": sorted(challenge),
                     "missing_natural_live_roles": missing_natural, "missing_live_roles": missing_live},
        "challenge_required": challenge_required,
        "accounting": {"scheduled_slots": len(plan.slots), "attempts": len(records),
                       "failed_attempts": sum(v["status"] == "fail" for v in attempts),
                       "model_calls": sum(v.model_calls for v in records),
                       "observed_tokens": sum(v.observed_tokens for v in records),
                       "elapsed_seconds": sum(v.elapsed_seconds for v in records)},
        "all_13_readiness": "not_assessed_separate_release_gate",
        "scientific_validity": "requires_independent_judgment",
        "automatic_retry": False,
    }
