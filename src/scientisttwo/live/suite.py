"""Frozen, sequential live campaigns with a persistent aggregate budget.

The suite is an orchestration layer: each assignment retains its own scientific
protocol and outcome. It never pools heterogeneous benchmark scores.
"""

import json
import math
from pathlib import Path
import shutil
import sqlite3
import time
import random
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from ..models import Record
from ..store import IntegrityError, Store, canonical, digest, file_digest
from .models import LiveSpec
from .runtime import AmbiguousCall, LiveBudgetExhausted


def suite_identity():
    return {
        p.name: file_digest(p) for p in (Path(__file__), Path(__file__).with_name("suite_cli.py"))
    }


class SuiteAssignment(Record):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    repetition: int = Field(ge=0)
    campaign: LiveSpec


class SuiteSpec(Record):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    wall_seconds: float = Field(gt=0)
    max_calls: int = Field(ge=1)
    max_experiments: int = Field(ge=1)
    token_stop_threshold: int | None = Field(default=None, gt=0)
    assignments: list[SuiteAssignment] = Field(min_length=1)
    implementation: dict[str, str]
    schedule_seed: int = Field(default=1729, ge=0)
    order_policy: Literal["declared", "interleaved"] = "declared"

    @model_validator(mode="after")
    def unique_assignments(self):
        if len({a.id for a in self.assignments}) != len(self.assignments):
            raise ValueError("Suite assignment IDs must be unique")
        declarations = [(digest(canonical(a.campaign)), a.repetition) for a in self.assignments]
        if len(set(declarations)) != len(declarations):
            raise ValueError("Duplicate campaign/repetition declaration")
        return self


def make_suite(
    manifests,
    *,
    name,
    repetitions,
    wall_seconds,
    max_calls,
    max_experiments,
    token_stop_threshold=None,
    schedule_seed=1729,
    order_policy="interleaved",
):
    if repetitions < 1:
        raise ValueError("At least one repetition is required")
    campaigns = [LiveSpec.model_validate_json(Path(p).read_text()) for p in manifests]
    assignments = [SuiteAssignment(id=f"c{i + 1:03d}-r{r + 1:03d}", repetition=r, campaign=c) for i, c in enumerate(campaigns) for r in range(repetitions)]
    if order_policy == "interleaved":
        rng = random.Random(schedule_seed)
        ordered = []
        tasks = sorted({a.campaign.native.task_id for a in assignments})
        for repetition in range(repetitions):
            for task in tasks:
                block = [a for a in assignments if a.repetition == repetition and a.campaign.native.task_id == task]
                rng.shuffle(block)
                ordered.extend(block)
        assignments = ordered
    return SuiteSpec(
        name=name,
        wall_seconds=wall_seconds,
        max_calls=max_calls,
        max_experiments=max_experiments,
        token_stop_threshold=token_stop_threshold,
        implementation=suite_identity(),
        assignments=assignments, schedule_seed=schedule_seed, order_policy=order_policy,
    )


class SuiteBudget:
    """Reservations count failed attempts; completion only updates their receipts."""

    def __init__(self, store, config, clock=time.time):
        self.store, self.config, self.clock = store, config, clock
        store.db.execute("""CREATE TABLE IF NOT EXISTS suite_operations(
            assignment TEXT NOT NULL, key TEXT NOT NULL, kind TEXT NOT NULL,
            request_hash TEXT NOT NULL, status TEXT NOT NULL,
            seconds REAL NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0,
            result_ref TEXT, PRIMARY KEY(assignment,key))""")
        store.db.commit()

    def for_assignment(self, assignment):
        if assignment not in {a.id for a in self.config.assignments}:
            raise ValueError("Assignment is outside the frozen suite")
        return AssignmentBudget(self, assignment)

    def remaining(self):
        deadline = self.store.get("suite_deadline")
        if deadline is None:
            deadline = self.clock() + self.config.wall_seconds
            self.store.set("suite_deadline", deadline)
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise LiveBudgetExhausted("Suite wall-clock budget exhausted, including interruption")
        return remaining

    def ensure_available(self, kind):
        if kind not in {"model", "experiment", "probe"}:
            raise ValueError("Unknown suite operation kind")
        self.remaining()
        if kind == "probe":
            return
        if kind == "model":
            for operation in self.store.db.execute(
                "SELECT status,result_ref FROM suite_operations WHERE kind='model'"
            ):
                if operation["result_ref"]:
                    receipt = self.store.json(operation["result_ref"])
                    if isinstance(receipt, dict) and receipt.get("usage_known") is False:
                        raise AmbiguousCall("Suite API usage remains unknown; no further model dispatch is permitted")
        count = self.store.db.execute(
            "SELECT count(*) FROM suite_operations WHERE kind=?", (kind,)
        ).fetchone()[0]
        limit = self.config.max_calls if kind == "model" else self.config.max_experiments
        if count >= limit:
            raise LiveBudgetExhausted(f"Suite {kind} operation limit exhausted")
        tokens = self.store.db.execute(
            "SELECT coalesce(sum(tokens),0) FROM suite_operations"
        ).fetchone()[0]
        if (
            kind == "model"
            and self.config.token_stop_threshold is not None
            and tokens >= self.config.token_stop_threshold
        ):
            raise LiveBudgetExhausted("Suite observed token stop threshold reached")

    def summary(self):
        return [
            dict(r) for r in self.store.db.execute("SELECT * FROM suite_operations ORDER BY rowid")
        ]


class AssignmentBudget:
    def __init__(self, suite, assignment):
        self.suite, self.assignment = suite, assignment

    def remaining(self):
        return self.suite.remaining()

    def reserve(self, key, kind, request):
        db = self.suite.store.db
        row = db.execute(
            "SELECT * FROM suite_operations WHERE assignment=? AND key=?",
            (self.assignment, key),
        ).fetchone()
        request_hash = digest(canonical(request))
        if row:
            if row["request_hash"] != request_hash or row["kind"] != kind:
                raise IntegrityError("Suite operation inputs changed on resume")
            return dict(row)
        self.suite.ensure_available(kind)
        with db:
            db.execute(
                "INSERT INTO suite_operations(assignment,key,kind,request_hash,status) "
                "VALUES(?,?,?,?,'running')",
                (self.assignment, key, kind, request_hash),
            )
        return None

    def finish(self, key, status, result, seconds, tokens=0):
        if status == "running" or not math.isfinite(seconds) or seconds < 0 or tokens < 0:
            raise ValueError("Invalid suite completion accounting")
        db = self.suite.store.db
        row = db.execute(
            "SELECT * FROM suite_operations WHERE assignment=? AND key=?",
            (self.assignment, key),
        ).fetchone()
        if row is None:
            raise IntegrityError("Suite completion has no reservation")
        ref = self.suite.store.put_json(result, "suite-operation")
        receipt = (status, seconds, tokens, ref)
        if row["status"] != "running":
            if receipt != (row["status"], row["seconds"], row["tokens"], row["result_ref"]):
                raise IntegrityError("Conflicting suite completion receipt")
            return
        with db:
            db.execute(
                "UPDATE suite_operations SET status=?,seconds=?,tokens=?,result_ref=? "
                "WHERE assignment=? AND key=?",
                (*receipt, self.assignment, key),
            )

    def summary(self):
        return [r for r in self.suite.summary() if r["assignment"] == self.assignment]


def validate_output(config, output):
    output = Path(output).resolve()
    for assignment in config.assignments:
        native = assignment.campaign.native
        for name in ("repository", "public_dir", "private_dir", "candidate_dir", "evaluator_dir"):
            value = getattr(native, name, None)
            if value is not None:
                source = Path(value).resolve()
                if output.is_relative_to(source) or source.is_relative_to(output):
                    raise ValueError("Suite output must not overlap any benchmark input")
        child = output / "campaigns" / assignment.id
        if child.is_symlink() or (output / "campaigns").is_symlink():
            raise ValueError("Suite campaign outputs cannot be symlinks")
    return output


def check_assignment(config):
    """Read-only prerequisite inspection; never instantiate an agent backend."""
    from .adapters import implementation_identity, load_development
    from .workspace_task import preflight
    from ..process import run_process
    from .workspace import validate_baseline_contract, live_readiness

    try:
        checks = preflight(config.native)
    except (ValueError, OSError, IntegrityError) as error:
        return {"ready": False, "errors": [str(error)]}
    checks["implementation_matches"] = config.implementation == implementation_identity()
    try:
        adapter = load_development(config.native, config.development)
        validate_baseline_contract(adapter)
        checks["adapter_live_readiness"] = live_readiness(adapter)
        if not checks["adapter_live_readiness"]["ready"]:
            raise ValueError("Adapter's measured live readiness gate has not passed")
        checks["adapter_ready"] = True
    except (ValueError, OSError, KeyError) as error:
        checks["adapter_ready"] = False
        checks.setdefault("errors", []).append(str(error))
    checks["ready"] = (
        checks["ready"] and checks["implementation_matches"] and checks["adapter_ready"]
    )
    native = config.native
    prefix = [native.runtime] + (["--context", native.context] if native.context else [])
    images = {}
    wanted_images = {native.image, native.grader_image}
    if config.workflow.compiler_image:
        wanted_images.add(config.workflow.compiler_image)
    for name in wanted_images:
        try:
            result = run_process(prefix + ["image", "inspect", name, "--format", "{{.Id}}"], 15)
            images[name] = result.stdout.strip() if result.code == 0 else None
        except OSError:
            images[name] = None
    checks["images"] = images
    checks["provider"] = config.provider.kind
    if config.provider.kind == "codex":
        checks["codex_installed"] = shutil.which("codex") is not None
        checks["provider_ready"] = checks["codex_installed"]
    else:
        from ..providers.config import readiness
        checks["provider_configuration"] = readiness(config.provider)
        checks["provider_ready"] = checks["provider_configuration"]["ready"]
        if (config.engine_version != "research-v2" or config.token_stop_threshold is None
                or config.model_retries != 0):
            checks["provider_ready"] = False
            checks.setdefault("errors", []).append(
                "API live execution requires research-v2, a token threshold and model_retries=0")
        if config.workflow.writer_backend == "paperorchestra" and not config.provider.vision:
            checks["provider_ready"] = False
            checks.setdefault("errors", []).append("PaperOrchestra requires provider vision capability")
    checks["writer_ready"] = True
    if config.workflow.writer_backend == "paperorchestra":
        from ..integrations.paperorchestra.runtime import verify_upstream
        try:
            verify_upstream(Path(config.workflow.paperorchestra_root))
        except (OSError, ValueError):
            checks["writer_ready"] = False
    checks["ready"] = (checks["ready"] and checks["provider_ready"]
                       and checks["writer_ready"] and all(images.values()))
    checks["model_access"] = "unverified_without_inference"
    checks["authentication"] = "not_checked"
    return checks


def check_suite(config, checker=check_assignment):
    checks = [{"id": a.id, **checker(a.campaign)} for a in config.assignments]
    identity = config.implementation == suite_identity()
    return {
        "ready": identity and all(c["ready"] for c in checks),
        "implementation_matches": identity,
        "assignments": checks,
        "live_model_calls": 0,
    }


def execute_assignment(config, output, shared, *, allow_literature_network=False):
    from ..backends import CodexBackend
    from ..cli import locked
    from .adapters import load_development
    from .controller import LiveController
    from .runtime import Budget, DevelopmentRunner
    from .workspace import live_readiness

    if config.provider.kind != "codex" and config.engine_version != "research-v2":
        raise ValueError("API providers require research-v2; legacy suites remain Codex-only")
    with locked(output):
        if not (output / "campaign.sqlite").exists() and any(
            p.name != ".campaign.lock" for p in output.iterdir()
        ):
            raise ValueError("Campaign output contains unrelated files")
        store = Store(output)
        try:
            binding = {
                "suite": str(shared.suite.store.root),
                "assignment": shared.assignment,
                "spec_sha256": digest(canonical(shared.suite.config)),
            }
            owner = store.get("suite_owner")
            if owner != binding:
                if (
                    owner is not None
                    or store.db.execute("SELECT count(*) FROM metadata").fetchone()[0]
                ):
                    raise IntegrityError("Child output belongs to a different campaign or suite")
                store.set("suite_owner", binding)
            adapter = load_development(config.native, config.development)
            live_readiness(adapter, require=True)
            budget = Budget(store, config, shared=shared)
            runner = DevelopmentRunner(store, config, adapter, budget)
            if config.engine_version == "research-v2":
                from ..research.live_pilot import run_prepared_live
                return run_prepared_live(store, config, adapter, runner, allow_live=True,
                                         allow_network_literature=allow_literature_network)
            backend = CodexBackend(
                output / "agent", config.model, allow_live=True, timeout=config.call_seconds
            )
            return LiveController(store, config, adapter, backend, runner, budget=budget,
                                  allow_network_literature=allow_literature_network).run()
        finally:
            store.close()


def _summary(config, states, operations, *, status, reason=""):
    rows = []
    for a in config.assignments:
        state = states.get(a.id, {"status": "not_started", "outcome": "not_started"})
        rows.append(
            {
                "id": a.id,
                "model": a.campaign.model,
                "provider": a.campaign.provider.kind,
                "repetition": a.repetition,
                "benchmark": a.campaign.native.benchmark,
                "task_id": a.campaign.native.task_id,
                **state,
            }
        )
    improved = sum(r.get("outcome") == "development_improvement" for r in rows)
    return {
        "name": config.name,
        "status": status,
        "reason": reason,
        "declared_assignments": len(rows),
        "assignments": rows,
        "development_improvements": improved,
        "improvement_denominator": len(rows),
        "improvement_rate_over_all_declared_assignments": improved / len(rows),
        "model_attempts": sum(r["kind"] == "model" for r in operations),
        "experiment_attempts": sum(r["kind"] == "experiment" for r in operations),
        "observed_tokens": sum(r["tokens"] for r in operations),
        "failed_or_unresolved_model_attempts": sum(
            r["kind"] == "model" and r["status"] != "success" for r in operations
        ),
        "token_accounting_note": "Observed usage only; failed or interrupted usage may be unknown",
        "unresolved_operations": sum(r["status"] == "running" for r in operations),
        "scope": "Workflow outcomes; no pooled benchmark score or conference acceptance",
        "schedule_seed": config.schedule_seed,
        "order_policy": config.order_policy,
        "schedule": [a.id for a in config.assignments],
    }


def run_suite(config, output, *, allow_live=False, executor=execute_assignment, allow_literature_network=False):
    if not allow_live:
        raise ValueError("Suite live calls require --allow-live and explicit frozen budgets")
    output = validate_output(config, output)
    if config.implementation != suite_identity():
        raise IntegrityError("Suite implementation changed; create a new manifest")
    from ..cli import locked

    with locked(output):
        if not (output / "campaign.sqlite").exists() and any(
            p.name != ".campaign.lock" for p in output.iterdir()
        ):
            raise ValueError("Suite output contains unrelated files")
        store = Store(output)
        try:
            ref = store.get("suite_spec_ref")
            if ref:
                if store.json(ref) != config.model_dump(mode="json"):
                    raise IntegrityError("Frozen suite changed on resume")
            elif store.get("live_spec_ref") or store.get("task_ref"):
                raise ValueError("Output already belongs to a non-suite campaign")
            else:
                store.set("suite_spec_ref", store.put_json(config, "live-suite-spec"))
            if store.verify():
                raise IntegrityError("Suite evidence was modified")
            budget = SuiteBudget(store, config)
            states = store.get("suite_assignments", {})
            status, reason = "complete", "All declared assignments have recorded outcomes"
            for assignment in config.assignments:
                prior = states.get(assignment.id, {})
                if prior.get("status") in {"complete", "blocked", "failed"}:
                    if executor is execute_assignment:
                        _verify_child(
                            assignment,
                            output / "campaigns" / assignment.id,
                            required=prior.get("status") == "complete",
                        )
                    continue
                # Unknown model outcomes must never silently trigger another invocation.
                shared = budget.for_assignment(assignment.id)
                unresolved = [r for r in shared.summary() if r["status"] == "running"]
                if unresolved:
                    # The child controller can recover a committed local completion receipt.
                    child_db = output / "campaigns" / assignment.id / "campaign.sqlite"
                    if not child_db.is_file():
                        states[assignment.id] = {
                            "status": "blocked",
                            "outcome": "ambiguous_operation",
                            "reason": "Unresolved reservation has no child recovery evidence",
                        }
                        store.set("suite_assignments", states)
                        continue
                if not prior:
                    try:
                        budget.ensure_available("model")
                        budget.ensure_available("experiment")
                    except AmbiguousCall:
                        status = "blocked"
                        reason = "Unknown model usage requires investigation before another assignment"
                        break
                    except LiveBudgetExhausted as error:
                        status, reason = "budget_exhausted", str(error)
                        break
                states[assignment.id] = {"status": "running", "outcome": "pending"}
                store.set("suite_assignments", states)
                try:
                    state = executor(
                        assignment.campaign, output / "campaigns" / assignment.id, shared,
                        **({"allow_literature_network": allow_literature_network} if executor is execute_assignment else {})
                    )
                    if not isinstance(state, dict) or not {"status", "outcome"} <= state.keys():
                        raise ValueError("Campaign executor returned no outcome")
                    states[assignment.id] = state
                except AmbiguousCall:
                    states[assignment.id] = {
                        "status": "blocked",
                        "outcome": "ambiguous_operation",
                        "reason": "Unknown operation outcome requires manual investigation",
                    }
                except LiveBudgetExhausted as error:
                    states[assignment.id] = {
                        "status": "complete",
                        "outcome": "budget_exhausted",
                        "reason": str(error),
                    }
                except KeyboardInterrupt:
                    states[assignment.id] = {"status": "interrupted", "outcome": "interrupted"}
                    store.set("suite_assignments", states)
                    store.set(
                        "suite_state", {"status": "interrupted", "reason": "User interrupted"}
                    )
                    raise
                except (ValueError, OSError, IntegrityError) as error:
                    states[assignment.id] = {
                        "status": "blocked",
                        "outcome": "execution_error",
                        "reason": f"Assignment raised {type(error).__name__}; inspect child evidence",
                    }
                store.set("suite_assignments", states)
            if status == "complete" and any(
                s.get("status") not in {"complete", "failed"} for s in states.values()
            ):
                status, reason = "blocked", "Some assignments require attention; outcomes preserved"
            result = _summary(config, states, budget.summary(), status=status, reason=reason)
            store.set("suite_state", {"status": status, "reason": reason})
            store.set("suite_summary_ref", store.put_json(result, "live-suite-summary"))
            (output / "suite-summary.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        finally:
            store.close()


def inspect_suite(output):
    """Read saved records without opening a write connection or invoking models."""
    root = Path(output).resolve()
    db = sqlite3.connect((root / "campaign.sqlite").as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        metadata = {r["key"]: json.loads(r["value"]) for r in db.execute("SELECT * FROM metadata")}
        ref = metadata.get("suite_spec_ref")
        if not ref:
            raise ValueError("Output is not a live suite")
        errors = []
        for row in db.execute("SELECT path,sha FROM artifacts"):
            p = (root / row["path"]).resolve()
            if (
                not p.is_relative_to(root / "artifacts")
                or not p.is_file()
                or file_digest(p) != row["sha"]
            ):
                errors.append(row["path"])
        if errors:
            raise IntegrityError("Missing or modified suite artifacts")
        config = SuiteSpec.model_validate_json((root / ref).read_text())
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='suite_operations'"
        ).fetchone()
        operations = (
            [dict(r) for r in db.execute("SELECT * FROM suite_operations ORDER BY rowid")]
            if exists
            else []
        )
        state = metadata.get(
            "suite_state", {"status": "interrupted", "reason": "No terminal receipt"}
        )
        return config, _summary(config, metadata.get("suite_assignments", {}), operations, **state)
    finally:
        db.close()


def _verify_child(assignment, output, *, required=False):
    """Completed runs stay reusable only while their code/config/evidence remain intact."""
    from .adapters import implementation_identity

    if assignment.campaign.implementation != implementation_identity():
        raise IntegrityError("Child implementation changed; create a new suite manifest")
    if not (output / "campaign.sqlite").is_file():
        if required:
            raise IntegrityError("Completed child campaign evidence is missing")
        # A blocked prerequisite may never have created a child store.
        return
    db = sqlite3.connect((output / "campaign.sqlite").as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        for row in db.execute("SELECT path,sha FROM artifacts"):
            p = (output / row["path"]).resolve()
            if (
                not p.is_relative_to(output / "artifacts")
                or not p.is_file()
                or file_digest(p) != row["sha"]
            ):
                raise IntegrityError("Completed child evidence was modified")
        row = db.execute("SELECT value FROM metadata WHERE key='live_spec_ref'").fetchone()
        if row:
            stored = json.loads((output / json.loads(row["value"])).read_text())
            if stored != assignment.campaign.model_dump(mode="json"):
                raise IntegrityError("Completed child specification differs from suite")
    finally:
        db.close()
