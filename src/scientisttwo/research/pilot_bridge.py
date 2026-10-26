"""Bridge bundled TaskSpec pilots to the shared research state machine.

The existing container worker and trusted task scorer remain the execution
boundary. Candidate Python is parsed as text here, never imported or evaluated.
One legacy container executes the declared dataset/seed grid; evidence says so
explicitly rather than inventing separate container launches for each seed.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from statistics import fmean
import time

from ..controller import initialize
from ..live.adapters import implementation_identity
from ..live.models import (
    Candidate,
    DevelopmentSpec,
    Hypothesis,
    LiveSpec,
    MetricSpec,
    Observation,
    SourceFile,
)
from ..live.runtime import Budget, LiveBudgetExhausted
from ..live.source_tree import candidate_snapshot
from ..live.workspace_models import WorkspaceSpec
from ..models import EvalSpec, ExperimentPlan, ExperimentResult, Idea, Profile, TaskSpec
from ..process import run_process
from ..runner import BudgetExhausted, ContainerRunner, evaluator_identity
from ..splits import bind_spec
from ..store import IntegrityError, Store, canonical, digest
from ..tasks import load_task_adapter
from .policy import ResearchPolicy


PILOT_VERSION = "research-pilot-bridge-v1"
COMMAND = ["python", "/worker.py"]


class PilotAdapter:
    """Preserve the original task's protocols, parameter fixtures and guardrails."""

    def __init__(self, store: Store, task: TaskSpec, adapter, *, frozen_literature=()):
        self.store, self.task, self.legacy = store, task, adapter
        self.frozen_literature = list(frozen_literature)

    def identity(self):
        identity = {
            "bridge": digest(Path(__file__).read_bytes()),
            "version": PILOT_VERSION,
            "task": digest(canonical(self.task)),
            "legacy": self.legacy.identity(),
        }
        if self.frozen_literature:
            identity["frozen_literature"] = digest(canonical(self.frozen_literature))
        return identity

    def source_files(self):
        directory = Path(__file__).parents[1]
        paths = {"research/pilot_bridge.py": Path(__file__), "worker.py": self.legacy.worker_path}
        for name in self.legacy.identity():
            if (directory / name).is_file():
                paths[name] = directory / name
        return paths

    def sources(self):
        return [*self.legacy.sources(), *self.frozen_literature]

    def editable_paths(self):
        return ["candidate.py"]

    def diagnostic_capabilities(self):
        return ["paired_differences", "paired_interval"]

    def validate(self, candidate: Candidate):
        candidate = Candidate.model_validate(candidate.model_dump())
        if candidate.base_snapshot_ref is not None or [f.path for f in candidate.files] != [
            "candidate.py"
        ]:
            raise ValueError("Bundled pilots permit exactly one candidate.py source file")
        if candidate.command != COMMAND:
            raise ValueError("The pilot's trusted worker command cannot be changed")
        tree = ast.parse(candidate.files[0].content)
        functions = [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build"
        ]
        if len(functions) != 1 or len(functions[0].args.args) != 1:
            raise ValueError("Pilot candidate.py must expose exactly one build(seed) factory")

    def candidate_for(self, idea: Idea | dict) -> Candidate:
        idea = Idea.model_validate(idea)
        proposal = self.legacy.fixture_response("implement", {"idea": idea.model_dump()})
        candidate = Candidate(
            files=[SourceFile(path="candidate.py", content=proposal["source"])],
            command=COMMAND,
            description=proposal["description"],
        )
        self.validate(candidate)
        key = "pilot/source-idea/" + digest(candidate.files[0].content.encode())
        # Different provenance IDs may generate identical code. Retain all declarations.
        values = self.store.get(key, [])
        if idea.model_dump(mode="json") not in values:
            values.append(idea.model_dump(mode="json"))
            self.store.set(key, values)
        return candidate

    def idea_for(self, candidate: Candidate) -> Idea:
        self.validate(candidate)
        sha = digest(candidate.files[0].content.encode())
        known = self.store.get("pilot/source-idea/" + sha)
        if known:
            return Idea.model_validate(known[0])
        return Idea(
            id="submitted-" + sha[:16],
            method="submitted-source",
            hypothesis="Submitted build(seed) implementation; method attribution unverified",
        )

    def hypothesis_for(self, idea: Idea | dict) -> Hypothesis:
        idea = Idea.model_validate(idea)
        return Hypothesis(
            title=idea.id,
            hypothesis=idea.hypothesis,
            novelty_score=idea.novelty,
            novelty_scope="Scripted established-method proposal, not a novelty finding",
            source_ids=["task"],
            adapter_payload=idea.model_dump(mode="json"),
        )

    def seed_ideas(self, count: int) -> list[Idea]:
        return [
            Idea.model_validate(idea)
            for idea in self.legacy.fixture_response("seed_ideas", {"count": count})["ideas"]
        ]

    def baseline(self):
        return self.candidate_for(self.legacy.baseline())

    def context(self):
        return {
            "contract": "Edit only candidate.py exposing build(seed), which returns an estimator with fit/predict. The trusted worker imports candidate code only inside an offline restricted container. Training rows and labels are available; evaluation labels are withheld. The controller computes scores from predictions.",
            "task": self.task.model_dump(mode="json"),
            "editable_files": ["candidate.py"],
            "scaling_paths": [],
            "baseline_modes": ["prepared_reference"],
            "baseline_provenance": "Explicit scripted reference supplied by the installed TaskAdapter",
            "baseline_source": self.baseline().files[0].content,
            "execution": {
                "command": COMMAND,
                "kind": "pilot_batch",
                "one_container_per_evaluation": True,
                "training_seed_policy": "A fresh build(seed) and fit per dataset/seed cell",
            },
            "effective_settings": {
                "source": "Complete candidate.py; constructor parameters are explicit",
                "preprocessing": "Any supplied scaler fits within the training pipeline",
                "threads": 1,
                "cpus": 1,
                "memory_mb": 1024,
                "equal_permitted_budget": True,
            },
            "comparison": {
                "primary": self.task.primary_metric,
                "aggregation": "mean over matched datasets and seeds",
                "guardrails": self.task.guardrails,
                "guardrail_scope": "each dataset mean",
                "epsilon": self.task.improvement_epsilon,
            },
            "measurement_scope": (
                "Real container measurements when executed. Research and writer roles use the explicitly configured live backend; the prepared baseline is a supplied reference, not an agent discovery."
                if self.store.get("live_pilot_options_ref") else
                "Real container measurements when executed; scripted proposals are synthetic agents"
            ),
        }

    def compare(self, result: Observation, reference: Observation):
        invalid = {"valid": False, "improved": False, "guardrails_pass": False}
        if result.protocol_hash != reference.protocol_hash or result.phase != reference.phase:
            return {**invalid, "reason": "Incomparable pilot protocols or phases"}
        if result.status != "success" or reference.status != "success":
            return {**invalid, "reason": "Execution failed; no scientific comparison"}
        try:
            left, right = self._original(result), self._original(reference)
        except (ValueError, IntegrityError) as error:
            return {**invalid, "reason": str(error)}
        comparison = self.legacy.compare(left, right, self.task).model_dump(mode="json")
        comparison["constraint_failures"] = (
            [] if comparison["guardrails_pass"] else ["Per-dataset guardrail failed"]
        )
        # Generic controller constraints must not be bypassed by a custom critic.
        comparison["valid"] = comparison["valid"] and comparison["guardrails_pass"]
        return comparison

    def _original(self, observation: Observation) -> ExperimentResult:
        if not observation.legacy_result_ref:
            raise ValueError("Pilot observation is missing its authoritative grid")
        original = ExperimentResult.model_validate(self.store.json(observation.legacy_result_ref))
        if (
            original.status != observation.status
            or [r.model_dump(mode="json") for r in original.rows] != observation.metric_rows
        ):
            raise IntegrityError("Pilot observation differs from the original metric grid")
        protocol = self.store.json(observation.protocol_ref)
        if digest(canonical(protocol)) != observation.protocol_hash:
            raise IntegrityError("Pilot observation protocol changed")
        metrics, values = _aggregate(
            original, self.task, protocol["seeds"], protocol["evaluation"]["datasets"]
        )
        if observation.metrics != metrics or observation.metric_values != values:
            raise IntegrityError("Pilot aggregate differs from authoritative per-dataset values")
        return original

    def _trial_idea(self, trial):
        if trial.get("candidate"):
            return self.idea_for(Candidate.model_validate(trial["candidate"]))
        value = trial.get("idea", trial)
        if value.get("adapter_payload"):
            return Idea.model_validate(value["adapter_payload"])
        return self.legacy.baseline()

    def _new_id(self, role, inputs):
        return role + "-" + digest(canonical(inputs))[:16]

    def _patch(self, inputs, idea):
        target = self.candidate_for(idea)
        parent = inputs["parent"]
        original = next(f["content"] for f in parent["files"] if f["path"] == "candidate.py")
        return {
            "parent_source_ref": parent["parent_source_ref"],
            "edits": [
                {
                    "path": "candidate.py",
                    "before": original,
                    "after": target.files[0].content,
                    "reason": "Apply the declared scripted method parameters to the exact parent",
                }
            ],
            "description": "Scripted parent-bound method change",
        }

    def fixture_response(self, role, inputs):
        """Translate legacy scripted decisions; never substitute fabricated metrics."""
        if role in {"limitations", "verify_limitations"}:
            value = self.legacy.fixture_response(
                "extract_limitations" if role == "limitations" else role, inputs
            )
            return {
                "limitations": value["items"],
                "relevant": value["sufficient"],
                "reason": "Scripted TaskAdapter limitation assessment",
            }
        if role == "seeds":
            return {
                "ideas": [
                    self.hypothesis_for(i).model_dump() for i in self.seed_ideas(inputs["count"])
                ]
            }
        if role in {"implement", "engineer"}:
            idea = self._trial_idea({"idea": inputs["idea"]})
            if role == "engineer":
                idea = Idea.model_validate(
                    self.legacy.fixture_response(
                        "engineer",
                        {"idea": idea.model_dump(), "new_id": self._new_id(role, inputs)},
                    )
                )
            return (
                self._patch(inputs, idea)
                if "parent" in inputs
                else self.candidate_for(idea).model_dump()
            )
        if role == "critic":
            return self.legacy.fixture_response(role, inputs)
        if role in {"evolve", "refine"}:
            if role == "refine":
                idea = self._trial_idea(inputs["incumbent"])
            else:
                parent_source = next(
                    f["content"] for f in inputs["parent"]["files"] if f["path"] == "candidate.py"
                )
                candidate = Candidate(
                    files=[SourceFile(path="candidate.py", content=parent_source)],
                    command=COMMAND,
                    description="Exact active parent",
                )
                idea = self.idea_for(candidate)
            revised = Idea.model_validate(
                self.legacy.fixture_response(
                    role, {"idea": idea.model_dump(), "new_id": self._new_id(role, inputs)}
                )
            )
            return {
                "idea": self.hypothesis_for(revised).model_dump(),
                "change_plan": {
                    "changed_files": ["candidate.py"],
                    "reason": "Test the scripted parameter revision using the recorded successes and failures",
                },
            }
        if role == "ablations":
            incumbent = inputs["incumbent"]
            idea = self._trial_idea(incumbent)
            planned = self.legacy.fixture_response(
                "plan_ablations", {"idea": idea.model_dump(), "new_id": self._new_id(role, inputs)}
            )
            results = []
            for item in planned["plans"]:
                plan = ExperimentPlan.model_validate(item)
                self.legacy.validate_plan(idea, plan)
                results.append(
                    {
                        "purpose": plan.purpose,
                        "changed_file": "candidate.py",
                        "candidate": self.candidate_for(plan.idea).model_dump(),
                    }
                )
            return {"plans": results}
        if role == "ablation_critic":
            evidence = [Observation.model_validate(item) for item in inputs.get("evidence", [])]
            comparisons = [
                self.compare(evidence[index + 1], evidence[index])
                for index in range(0, len(evidence) - 1, 2)
            ]
            return self.legacy.fixture_response(role, {**inputs, "comparisons": comparisons})
        if role == "rebuttal":
            candidate = inputs["incumbent"]["candidate"]
            concerns = [item["id"] for item in inputs["concerns"]]
            return {
                "plans": [
                    {
                        "id": "repeat-fixed-development",
                        "concern_ids": concerns,
                        "purpose": "Repeat the unchanged method on exact additional training seeds",
                        "seed_count": inputs["action_contract"]["max_seed_count"],
                        "execution_kind": "repeat",
                        "evaluation_split": "development",
                        "diagnostics": [{"kind": "metric", "metric": self.task.primary_metric}],
                        "changed_file": None,
                        "candidate": candidate,
                    }
                ]
                if concerns
                else [],
                "unaddressed_concerns": [],
            }
        if role == "draft":
            coverage = inputs.get("supplementary_coverage", [])
            return {
                "title": "Scripted pilot research workflow",
                "claims": [
                    {"experiment_id": item["id"], "metric": item["metric"], "value": item["value"]}
                    for item in inputs["evidence"]
                    if item["status"] == "success"
                ],
                "limitations": [
                    "Established scripted methods; no claim of autonomous discovery or global novelty"
                ],
                "source_ids": ["task"],
                "concern_responses": [
                    {
                        "concern_id": item["concern_id"],
                        "status": "executed"
                        if item.get("execution_status") == "fulfilled"
                        else "unaddressed",
                        "experiment_ids": item.get("experiment_ids", []),
                        "explanation": "Status follows the controller's execution coverage; scientific sufficiency remains separate",
                    }
                    for item in coverage
                ],
            }
        if role == "review":
            result = self.legacy.fixture_response(
                role, {**inputs, "has_rebuttal": inputs.get("has_rebuttal", False)}
            )
            return {key: result[key] for key in ("score", "concerns")}
        if role == "meta":
            result = self.legacy.fixture_response("meta_review", inputs)
            return {
                "decision": "ReviseMethod" if result["decision"] == "Refine" else "Accept",
                "reason": result["reason"],
            }
        raise ValueError("Unsupported pilot fixture role: " + role)


def _aggregate(result, task, seeds, datasets):
    if result.status != "success":
        return {}, {}
    rows = {(row.dataset, row.seed): row for row in result.rows}
    expected = {(dataset, seed) for dataset in datasets for seed in seeds}
    if set(rows) != expected:
        raise IntegrityError("Pilot metric grid differs from the declared datasets and seeds")
    required = set(task.metric_directions)
    if any(set(row.metrics) != required for row in result.rows):
        raise IntegrityError("Pilot metric grid differs from declared metric names")
    values = {
        metric: [
            float(fmean(rows[dataset, seed].metrics[metric] for dataset in datasets))
            for seed in seeds
        ]
        for metric in sorted(required)
    }
    return {name: float(fmean(series)) for name, series in values.items()}, values


class PilotRunner:
    def __init__(
        self,
        store,
        config,
        adapter,
        profile,
        *,
        budget=None,
        executor=run_process,
        container_runner=None,
    ):
        self.store, self.config, self.adapter, self.profile = store, config, adapter, profile
        self.budget = budget or Budget(store, config)
        self.raw_executor = executor
        self.legacy = container_runner or ContainerRunner(
            store,
            profile,
            config.native.runtime,
            config.native.image,
            adapter=adapter.legacy,
            context=config.native.context,
            executor=self.executor,
        )
        self.image = None

    def executor(self, argv, timeout, **kwargs):
        prefix = [self.config.native.runtime] + (
            ["--context", self.config.native.context] if self.config.native.context else []
        )
        cleanup = argv[: len(prefix) + 2] == prefix + ["rm", "--force"]
        return self.raw_executor(
            argv, timeout if cleanup else min(timeout, self.budget.remaining()), **kwargs
        )

    def preflight(self):
        request = {
            "runtime": self.config.native.runtime,
            "image": self.config.native.image,
            "adapter": self.adapter.identity(),
            "task": self.adapter.task.model_dump(mode="json"),
        }
        key = "pilot/preflight"
        index = 0
        while True:
            row = self.store.db.execute("SELECT status,result_ref FROM live_operations WHERE key=?", (key,)).fetchone()
            if row is None or row["status"] != "failed" or not row["result_ref"]:
                break
            index += 1
            key = f"pilot/preflight/{index}"
        prior = self.budget.reserve(key, "probe", request)
        started = time.monotonic()
        try:
            self.legacy.preflight()
        except Exception as error:
            if not prior or not prior["result_ref"]:
                self.budget.finish(key, "failed", {"error_type": type(error).__name__, "reason": str(error)}, time.monotonic() - started)
            raise
        self.image = self.legacy.image_id
        images = {
            "execution": self.image,
            "grading": "trusted-host-scorer:" + digest(canonical(evaluator_identity())),
        }
        if self.store.get("live_images") not in (None, images):
            raise IntegrityError("Pilot execution image or trusted scorer changed")
        self.store.set("live_images", images)
        if not prior or not prior["result_ref"]:
            self.budget.finish(
                key,
                "success",
                {"images": images, "kind": "pilot_container_preflight"},
                time.monotonic() - started,
            )

    def materialize(self, key, candidate):
        self.adapter.validate(candidate)
        tree = candidate_snapshot(self.store, candidate)
        code_ref = self.store.put_json(candidate, "candidate-source")
        tree_ref = self.store.put_json(tree, "materialized-source-tree")
        request = {
            "proposal_ref": self.store.put_json(candidate, "candidate-proposal"),
            "adapter": self.adapter.identity(),
        }
        evidence = {
            "proposal_ref": request["proposal_ref"],
            "code_ref": code_ref,
            "source_tree_ref": tree_ref,
        }
        saved = self.store.cache("materialization/" + key, request, evidence)
        if saved != evidence:
            raise IntegrityError("Pilot materialization changed on resume")
        return candidate, code_ref, tree_ref

    def execute(self, key, candidate, phase="full", seeds=None):
        if phase == "final":
            raise ValueError("Use final() after candidate freeze for protected evaluation")
        return self._execute(key, candidate, phase, seeds)

    def _execute(self, key, candidate, phase, seeds=None):
        if self.image is None:
            self.preflight()
        if seeds is not None and (
            not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("Pilot seeds must be distinct nonnegative integers")
        spec = self.adapter.legacy.evaluation(self.adapter.task, phase)
        if seeds is not None:
            spec = spec.model_copy(update={"seeds": list(seeds)})
        manifest = self.adapter.legacy.split_manifest(self.adapter.task)
        spec = bind_spec(spec, manifest)
        split_ref = self.store.put_json(manifest, "split-manifest")
        if split_ref != self.store.get("split_manifest_ref"):
            raise IntegrityError("Pilot partitions changed")
        candidate, code_ref, tree_ref = self.materialize(
            ("final/" if phase == "final" else "development/") + key, candidate
        )
        protocol = {
            "phase": phase,
            "seeds": spec.seeds,
            "adapter": self.adapter.identity(),
            "data": {"split_manifest_ref": split_ref, "hash": spec.split_manifest_hash},
            "evaluation": spec.model_dump(mode="json"),
            "runtime": {"images": self.store.get("live_images"), "evaluator": evaluator_identity()},
            "execution": {
                "kind": "pilot_batch",
                "command": COMMAND,
                "datasets": spec.datasets,
                "fresh_build_per_dataset_seed": True,
                "worker_source_sha256": digest(self.adapter.legacy.worker_path.read_bytes()),
            },
        }
        request = {
            "candidate": candidate.model_dump(mode="json"),
            "protocol": digest(canonical(protocol)),
        }
        cache_key = ("pilot-final/" if phase == "final" else "development/") + key
        saved = self.store.cached(cache_key, request)
        if saved:
            if self.store.verify():
                raise IntegrityError("Pilot evidence changed before replay")
            return Observation.model_validate(saved)
        operation = ("final/" if phase == "final" else "experiment/") + key
        prior = self.budget.reserve(operation, "experiment", request)
        if prior and prior["result_ref"]:
            committed = self.store.json(prior["result_ref"])
            recovered = Observation.model_validate(
                self.store.json(committed["observation_ref"]) if phase == "final" else committed
            )
            self.store.cache(cache_key, request, recovered)
            return recovered
        legacy_key = "pilot/" + operation
        job = self.store.db.execute("SELECT * FROM jobs WHERE key=?", (legacy_key,)).fetchone()
        source = candidate.files[0].content
        idea = self.adapter.idea_for(candidate)
        if prior and job is None:
            original = ExperimentResult(
                id=legacy_key,
                idea_id=idea.id,
                status="interrupted",
                protocol_hash=digest(canonical(spec)),
                code_ref=self.store.put(source.encode(), "py", "candidate-code"),
                spec_ref=self.store.put_json(spec, "evaluation-spec"),
                split_manifest_ref=split_ref,
                reason="Bridge admission has no committed worker job; refusing a possibly duplicated execution",
            )
        else:
            deadline = self.store.get("live_deadline")
            if deadline is not None:
                self.store.set("deadline", deadline)
            try:
                original = self.legacy.execute(legacy_key, idea, source, spec)
            except BudgetExhausted as error:
                raise LiveBudgetExhausted(str(error)) from error
        legacy_ref = self.store.put_json(original, "experiment-result")
        status, reason = original.status, original.reason
        try:
            metrics, values = _aggregate(original, self.adapter.task, spec.seeds, spec.datasets)
        except IntegrityError as error:
            status, reason = "invalid", str(error)
            metrics, values = {}, {}
        log = original.logs_ref or self.store.put(
            b"No separate worker output was retained.", "log", "execution-log"
        )
        committed_job = self.store.db.execute("SELECT * FROM jobs WHERE key=?", (legacy_key,)).fetchone()
        completion_ref = (self.store.put_json({
            "schema_version": "pilot-job-completion-v1", "job": dict(committed_job),
            "worker_image_id": self.image,
            "worker_source_sha256": digest(self.adapter.legacy.worker_path.read_bytes()),
            "observations_ref": original.observations_ref,
            "measurement_origin": original.measurement_origin,
        }, "pilot-job-completion") if committed_job is not None and committed_job["result_ref"] else None)
        batch = {
            "kind": "pilot_batch",
            "operation_key": operation,
            "legacy_job_key": legacy_key,
            "legacy_result_ref": legacy_ref,
            "legacy_completion_ref": completion_ref,
            "seeds": spec.seeds,
            "datasets": spec.datasets,
            "log_ref": log,
            "status": original.status,
            "source_tree_ref": tree_ref,
            "spec_ref": original.spec_ref,
            "seconds": original.duration_seconds,
            "scope": "One container for the full grid; fresh model build and fit per dataset/seed, not separate container launches",
        }
        result = Observation(
            id=key,
            phase=phase,
            status=status,
            protocol_hash=request["protocol"],
            protocol_ref=self.store.put_json(protocol, "evaluation-protocol"),
            code_ref=code_ref,
            source_tree_ref=tree_ref,
            legacy_result_ref=legacy_ref,
            metric=self.adapter.task.primary_metric,
            metrics=metrics,
            metric_values=values,
            value=metrics.get(self.adapter.task.primary_metric),
            values=values.get(self.adapter.task.primary_metric, []),
            metric_rows=[row.model_dump(mode="json") for row in original.rows],
            prediction_refs=[
                self.store.put_json(
                    {
                        "kind": "pilot_prediction_selection",
                        "observations_ref": original.observations_ref,
                        "seed": seed,
                        "datasets": spec.datasets,
                    },
                    "pilot-predictions",
                )
                for seed in spec.seeds
            ]
            if status == "success"
            else [],
            log_refs=[self.store.put_json(batch, "pilot-execution-receipt")],
            seconds=original.duration_seconds,
            reason=reason or "Trusted pilot scorer accepted the complete grid",
        )
        completion = self._final_envelope(result) if phase == "final" else result
        status = completion["status"] if phase == "final" else result.status
        self.budget.finish(operation, status, completion, result.seconds)
        self.store.cache(cache_key, request, result)
        return result

    def final(self, key, candidate):
        if key not in {"baseline", "selected"}:
            raise ValueError("Final evaluation is limited to baseline and selected frozen sources")
        frozen = self.store.get("live_frozen")
        if not frozen:
            raise IntegrityError("Protected pilot evaluation requires a frozen campaign")
        accepted = frozen["baseline"] if key == "baseline" else frozen["best"]["candidate"]
        if candidate.model_dump(mode="json") != accepted:
            raise IntegrityError("Final candidate differs from the frozen source")
        result = self._execute(key, candidate, "final")
        return self._final_envelope(result)

    def _final_envelope(self, result):
        return {
            "status": "completed" if result.status == "success" else result.status,
            "benchmark": "bundled-pilot",
            "task_id": self.adapter.task.name,
            "score": result.value,
            "metric": result.metric,
            "direction": self.adapter.task.metric_directions[result.metric],
            "per_seed_values": result.values,
            "metric_rows": result.metric_rows,
            "observation_ref": self.store.put_json(result, "pilot-final-observation"),
            "legacy_result_ref": result.legacy_result_ref,
            "elapsed_seconds": result.seconds,
            "reason": result.reason,
            "live_model_calls": 0,
            "leaderboard_eligible": False,
            "scope": "One frozen final evaluation using the original task's declared seeds and test partition",
        }


def create_pilot_campaign(
    store: Store,
    task: TaskSpec,
    profile: Profile,
    runtime="docker",
    image="scientisttwo-worker:0.1",
    context=None,
    *,
    adapter=None,
    executor=run_process,
    container_runner=None,
    live_options=None,
):
    """Create/reconstruct the shared-engine pilot without constructing a live backend."""
    # Import lazily so ordinary offline construction keeps its existing contract.
    saved_options = store.get("live_pilot_options_ref")
    if live_options is not None or saved_options is not None:
        from .live_pilot import LivePilotOptions
        options = LivePilotOptions.model_validate(
            live_options if live_options is not None else store.json(saved_options))
        import re
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
            raise ValueError("Live pilots require an explicitly pinned worker image SHA256 digest")
        if saved_options is not None and options.model_dump(mode="json") != store.json(saved_options):
            raise IntegrityError("Live pilot options changed on resume")
        if saved_options is None and store.get("pilot_live_spec_ref") is not None:
            raise IntegrityError("An offline frozen pilot cannot be converted into a live campaign")
        if saved_options is None:
            store.set("live_pilot_options_ref", store.put_json(options, "live-pilot-options"))
    else:
        options = None
    legacy = adapter or load_task_adapter(task.adapter, task)
    if store.get("task_ref") is None:
        initialize(store, task, profile, legacy)
    elif store.json(store.get("task_ref")) != task.model_dump(mode="json") or store.json(
        store.get("profile_ref")
    ) != profile.model_dump(mode="json"):
        raise IntegrityError("Pilot task or profile changed on resume")
    previous = store.get("runtime")
    runtime_spec = {"runtime": runtime, "image": image, "context": context}
    if previous not in (None, runtime_spec):
        raise IntegrityError("Pilot runtime changed on resume")
    store.set("runtime", runtime_spec)
    bridge = PilotAdapter(store, task, legacy,
        frozen_literature=[source.model_dump(mode="json") for source in options.frozen_literature] if options else ())
    if options is not None:
        source_ids = [source["id"] for source in bridge.sources()] + [source.id for source in options.sources]
        if len(source_ids) != len(set(source_ids)) or "task" in source_ids:
            raise ValueError("Live pilot sources conflict with the task's frozen literature")
    roots = {
        name: store.root / "pilot-inputs" / name
        for name in ("public", "private", "candidate", "evaluator")
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    native = WorkspaceSpec(
        name=task.name,
        task_id=task.name,
        description=task.question,
        public_dir=roots["public"],
        private_dir=roots["private"],
        candidate_dir=roots["candidate"],
        evaluator_dir=roots["evaluator"],
        command=COMMAND,
        evaluator_command=["trusted-host-task-scorer"],
        metric=task.primary_metric,
        direction=task.metric_directions[task.primary_metric],
        image=image,
        grader_image=image,
        runtime=runtime,
        context=context,
        data_provenance="user_prepared",
        final_seed=task.seeds[0],
        resources={
            "cpus": 1,
            "memory_mb": 1024,
            "seconds": profile.experiment_seconds,
            "grading_seconds": profile.experiment_seconds,
            "gpus": None,
        },
    )
    config = LiveSpec(
        native=native,
        development=DevelopmentSpec(
            adapter="bundled-pilot",
            metric=task.primary_metric,
            direction=task.metric_directions[task.primary_metric],
            metric_specs={
                name: MetricSpec(direction=direction)
                for name, direction in task.metric_directions.items()
            },
            seeds=task.seeds,
            split_seed=task.split_seed,
            subset_fraction=task.subset_fraction,
            epsilon=task.improvement_epsilon,
            settings={
                "pilot_bridge": PILOT_VERSION,
                "task_ref": store.get("task_ref"),
                "comparison_scope": "Per-dataset guardrails; complete original metric grid retained",
            },
        ),
        model="scripted-fixture-v1",
        mode="research",
        engine_version="research-v2",
        workflow=ResearchPolicy(
            profile="paper-methodology" if profile.name == "paper" else "offline-pilot",
            seed_ideas=profile.seed_ideas,
            network_literature=False,
            require_pdf=False,
        ),
        max_calls=200,
        max_experiments=100,
        campaign_seconds=profile.campaign_seconds,
        call_seconds=30,
        model_retries=0,
        agent_output_retries=0,
        engineering_retries=profile.engineering_rounds,
        search_rounds=profile.search_rounds,
        limitation_rounds=profile.limitation_rounds,
        ablation_refinements=profile.ablation_rounds,
        review_rounds=profile.peer_rounds,
        meta_refinements=profile.meta_rounds,
        success_target=profile.success_target,
        implementation=implementation_identity(),
    )
    writer_options = store.get("pilot_workflow_options", {})
    controller_options = store.get("pilot_controller_options", {})
    if not isinstance(controller_options, dict) or set(controller_options) - {"max_prompt_chars"}:
        raise ValueError("Unknown pilot controller options")
    if set(writer_options) - {"compiler_image", "require_pdf", "writer_backend", "paperorchestra_root",
                       "paperorchestra_reflections", "paperorchestra_ensemble", "paperorchestra_max_calls",
                       "paperorchestra_swap_policy"}:
        raise ValueError("Unknown pilot workflow options")
    if options is not None:
        if controller_options:
            raise IntegrityError("Offline controller options cannot alter an explicit live pilot")
        if writer_options and any(options.workflow.model_dump().get(key) != value for key, value in writer_options.items()):
            raise IntegrityError("Legacy writer options conflict with the explicit live specification")
        config = LiveSpec.model_validate({**config.model_dump(mode="json"), **options.config_values()})
        store.set("live_pilot_execution_policy", {
            "model": options.model, "backend": options.provider.kind, "synthetic_agents": False,
            "authorization": "Explicit allow-live required by the separate live-pilot entrypoint",
            "legacy_profile_scope": "Experiment resources only; Profile.live_calls remains False and offline CLI cannot launch this campaign",
        })
    else:
        config.workflow = ResearchPolicy.model_validate({**config.workflow.model_dump(), **writer_options})
        config = LiveSpec.model_validate({**config.model_dump(mode="json"), **controller_options})
    if store.get("pilot_live_spec_ref"):
        old = LiveSpec.model_validate(store.json(store.get("pilot_live_spec_ref")))
        if old != config:
            raise IntegrityError(
                "Shared pilot specification changed; execution requires its frozen implementation"
            )
    else:
        store.set("pilot_live_spec_ref", store.put_json(config, "pilot-research-spec"))
    runner = PilotRunner(
        store, config, bridge, profile, executor=executor, container_runner=container_runner
    )
    return config, bridge, runner


def audit_pilot_campaign(
    store, adapter: PilotAdapter, runner=None, *, reexecute=False, include_final=False
):
    """Replay trusted scoring from frozen predictions without importing candidates.

    Optional development reexecution consumes the campaign's existing operation
    and wall-clock budget. It never grants a fresh campaign allowance. Test labels
    are read only with include_final=True and are never sent to agent roles.
    """
    errors = store.verify()
    current_identity = adapter.identity()
    identity_ok = store.get("live_adapter_identity") in (None, current_identity)
    evaluator_ok = store.get("evaluator") == evaluator_identity()
    if not identity_ok:
        errors.append("Pilot task adapter differs from the frozen implementation")
    if not evaluator_ok:
        errors.append("Trusted evaluator or dependencies differ from the frozen implementation")
    records = []
    query = "SELECT key,output_ref FROM calls WHERE key LIKE 'development/%'"
    if include_final:
        query += " OR key LIKE 'pilot-final/%'"
    for row in store.db.execute(query + " ORDER BY key"):
        record = {
            "key": row["key"],
            "observation_ref": row["output_ref"],
            "integrity": "unverified",
            "scoring": "unverified",
            "ledger": "unverified",
        }
        try:
            observed = Observation.model_validate(store.json(row["output_ref"]))
            if not observed.legacy_result_ref:
                raise IntegrityError("Missing authoritative pilot result reference")
            original = ExperimentResult.model_validate(store.json(observed.legacy_result_ref))
            spec = EvalSpec.model_validate(store.json(original.spec_ref))
            record.update(
                id=observed.id,
                phase=observed.phase,
                execution_status=observed.status,
                measurement_origin=original.measurement_origin,
            )
            if spec.task != adapter.task or spec.phase != observed.phase:
                raise IntegrityError("Pilot task or phase differs from original evaluation")
            if bind_spec(spec, adapter.legacy.split_manifest(adapter.task)) != spec:
                raise IntegrityError("Original pilot split binding changed")
            if original.protocol_hash != digest(canonical(spec)):
                raise IntegrityError("Original evaluation protocol hash changed")
            protocol = store.json(observed.protocol_ref)
            if (
                digest(canonical(protocol)) != observed.protocol_hash
                or protocol["evaluation"] != spec.model_dump(mode="json")
                or protocol["adapter"] != current_identity
            ):
                raise IntegrityError("Bridge protocol differs from original evaluation")
            candidate = Candidate.model_validate(store.json(observed.code_ref))
            adapter.validate(candidate)
            if candidate.files[0].content.encode() != store.read(original.code_ref):
                raise IntegrityError("Observed candidate source differs from executed pilot source")
            if store.json(observed.source_tree_ref) != candidate_snapshot(
                store, candidate
            ).model_dump(mode="json"):
                raise IntegrityError("Pilot materialized source tree differs")
            if observed.metric_rows != [r.model_dump(mode="json") for r in original.rows]:
                raise IntegrityError("Bridge discarded or modified original metric rows")
            if observed.status == "success":
                adapter._original(observed)
                for seed, ref in zip(protocol["seeds"], observed.prediction_refs, strict=True):
                    expected = {
                        "kind": "pilot_prediction_selection",
                        "observations_ref": original.observations_ref,
                        "seed": seed,
                        "datasets": spec.datasets,
                    }
                    if store.json(ref) != expected:
                        raise IntegrityError("Prediction selection is not bound to the scored grid")
            operation_key = ("final/" if observed.phase == "final" else "experiment/") + observed.id
            operation = store.db.execute(
                "SELECT * FROM live_operations WHERE key=?", (operation_key,)
            ).fetchone()
            completion = (
                store.json(operation["result_ref"])
                if operation and operation["result_ref"]
                else None
            )
            wrapped_final = observed.phase == "final" and completion is not None
            expected_status = (
                "completed" if wrapped_final and observed.status == "success" else observed.status
            )
            committed_observation = (
                store.json(completion["observation_ref"]) if wrapped_final else completion
            )
            if (
                not operation
                or canonical(committed_observation) != canonical(observed)
                or operation["status"] != expected_status
                or operation["seconds"] != observed.seconds
            ):
                raise IntegrityError("Shared operation ledger differs from the observation")
            job = store.db.execute("SELECT * FROM jobs WHERE key=?", (original.id,)).fetchone()
            if job:
                if (
                    job["result_ref"] != observed.legacy_result_ref
                    or job["status"] != original.status
                    or job["charged"] != original.duration_seconds
                ):
                    raise IntegrityError(
                        "Underlying worker ledger differs from the original result"
                    )
            elif original.status != "interrupted":
                raise IntegrityError("Original pilot result has no worker execution admission")
            record["integrity"], record["ledger"] = "pass", "pass"
            if (
                observed.status == "success"
                and original.measurement_origin == "container"
                and evaluator_ok
            ):
                replayed = adapter.legacy.score(spec, store.json(original.observations_ref))
                if len(replayed) != len(original.rows):
                    raise IntegrityError("Trusted metric replay produced a different grid size")
                actual = {(r.dataset, r.seed): r for r in replayed}
                if len(actual) != len(replayed):
                    raise IntegrityError("Trusted metric replay duplicated grid cells")
                for expected in original.rows:
                    value = actual.get((expected.dataset, expected.seed))
                    if (
                        value is None
                        or value.count != expected.count
                        or value.metrics.keys() != expected.metrics.keys()
                    ):
                        raise IntegrityError("Trusted metric replay differs from original grid")
                    if any(
                        not math.isclose(
                            value.metrics[m],
                            expected.metrics[m],
                            rel_tol=0,
                            abs_tol=adapter.task.quality_tolerance,
                        )
                        for m in expected.metrics
                    ):
                        raise IntegrityError(
                            "Trusted metric replay does not reproduce quality scores"
                        )
                    if (
                        value.training_seconds != expected.training_seconds
                        or value.inference_seconds != expected.inference_seconds
                    ):
                        raise IntegrityError("Stored timings differ from frozen prediction receipt")
                record["scoring"] = "pass"
            elif original.measurement_origin == "synthetic_test":
                record["reason"] = "Synthetic test outcomes cannot establish real scoring replay"
            else:
                record["reason"] = (
                    "Failed execution remains in the audit denominator; no successful score to replay"
                )
        except (ValueError, KeyError, TypeError, OSError, IntegrityError) as error:
            record["integrity"] = "fail"
            record["reason"] = str(error)
            errors.append(f"{row['key']}: {error}")
        records.append(record)
    successful = [r for r in records if r.get("execution_status") == "success"]
    result = {
        "schema_version": 1,
        "artifacts": "fail" if errors else "pass",
        "evaluator_identity": "pass" if evaluator_ok and identity_ok else "fail",
        "scoring": "pass"
        if successful and all(r["scoring"] == "pass" for r in successful)
        else "unverified",
        "ledger": "pass"
        if records and all(r["ledger"] == "pass" for r in records)
        else "unverified",
        "observations": records,
        "observation_count": len(records),
        "failed_observations": sum(r.get("execution_status") != "success" for r in records),
        "errors": errors,
        "reexecution": "unverified",
        "final_included": include_final,
        "timing_comparison": "Frozen timings verified; rerun timing equivalence is informational only",
    }
    if reexecute and not errors and runner is not None:
        # A negative campaign freezes the reference as its final selection, but
        # deliberately never promotes it to live_best. Audit that exact frozen
        # development snapshot too; absence of an improvement is not an audit skip.
        frozen = store.get("live_frozen") or {}
        best = frozen.get("best") or store.get("live_best")
        if best:
            saved = Observation.model_validate(best["result"])
            if saved.phase == "final":
                raise IntegrityError("Post-run reexecution must use development evidence")
            protocol = store.json(saved.protocol_ref)
            try:
                repeated = runner.execute(
                    "audit/selected/" + digest(saved.id.encode())[:16],
                    Candidate.model_validate(best["candidate"]),
                    saved.phase,
                    protocol["seeds"],
                )
            except (LiveBudgetExhausted, ValueError) as error:
                result["reexecution_reason"] = str(error)
            else:
                if repeated.status == "success":
                    from ..evaluation import quality_matches

                    left, right = adapter._original(saved), adapter._original(repeated)
                    result["reexecution"] = (
                        "pass"
                        if quality_matches(left, right, adapter.task.quality_tolerance)
                        else "fail"
                    )
                    result["reexecution_measurement_origin"] = right.measurement_origin
                else:
                    result["reexecution"] = "fail"
                result["reexecution_ref"] = store.put_json(repeated, "pilot-audit-observation")
                if result["reexecution"] == "fail":
                    errors.append("Selected development snapshot failed reproduction within quality tolerance")
        else:
            result["reexecution_reason"] = "No committed selected development snapshot is available"
    result["audit_ref"] = store.put_json(result, "pilot-reproduction-audit")
    store.set("latest_pilot_audit", result)
    return result
