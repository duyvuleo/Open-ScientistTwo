"""Real isolated experiments with scripted agents and zero inference calls."""

import json
from pathlib import Path

from ..backends import FixtureBackend
from ..benchmarks.files import tree_manifest, file_hash
from ..benchmarks.models import NativeResources
from ..live.adapters import implementation_identity, load_development
from ..live.audit import audit_live
from ..live.controller import LiveController
from ..live.models import Candidate, DevelopmentSpec, LiveSpec
from ..live.source_tree import candidate_snapshot
from ..live.runtime import Budget, DevelopmentRunner
from ..live.workspace_models import WorkspaceSpec
from ..live.workspace_task import freeze_workspace
from ..runtime_contracts import RuntimeAsset
from ..runtime_profiles import runtime_profile
from ..store import Store
from .catalog import get_recipe
from .preparation import CPU_TASKS, canonical, manifest_hash, verify_prepared


def sealed_evidence(root):
    """Bind durable stores without following execution-workspace input symlinks."""
    root = Path(root)
    result = {}
    for database in sorted(root.rglob("campaign.sqlite")):
        if database.is_symlink():
            raise ValueError("Evidence database must not be a symlink")
        result[database.relative_to(root).as_posix()] = {
            "sha256": file_hash(database),
            "size": database.stat().st_size,
        }
        artifacts = database.parent / "artifacts"
        for name, item in tree_manifest(artifacts).items():
            result[(artifacts / name).relative_to(root).as_posix()] = item
    return result


def _final_once(runner, label, candidate):
    """Reuse an already frozen final when its exact source and argv are identical."""
    store = runner.store
    expected = candidate_snapshot(store, candidate)
    frozen, results = store.get("live_frozen", {}), store.get("live_final", {})
    possibilities = {
        "baseline": frozen.get("baseline"),
        "selected": frozen.get("best", {}).get("candidate"),
    }
    for name, value in possibilities.items():
        if value is None or name not in results:
            continue
        previous = Candidate.model_validate(value)
        actual = candidate_snapshot(store, previous)
        if actual.bundle_ref == expected.bundle_ref and previous.command == candidate.command:
            return results[name], "final/" + name
    key = "validation-" + label
    return runner.final(key, candidate), "final/" + key


class ScriptedResearch(FixtureBackend):
    """All proposed code is prepared; all empirical values come from the runner."""

    def __init__(self, adapter):
        self.adapter = adapter
        super().__init__(self.respond)

    @staticmethod
    def idea(title):
        return {
            "title": title,
            "hypothesis": "Measure a declared prepared control against the reference",
            "novelty_score": 0,
            "novelty_scope": "Scripted workflow validation; no novelty claim",
            "source_ids": ["task"],
        }

    def _prepared_targets(self, name, parent):
        """Read prepared source in the exact proposal representation, never execute it."""
        candidate = self.adapter.baseline() if name == "reference" else self.adapter.control()
        files = {item.path: item.content for item in candidate.files}
        for item in parent["files"]:
            if item["path"] not in files:
                # Checkpoint candidates omit unchanged files from their overlays.
                # The frozen prepared source supplies their complete bytes.
                files[item["path"]] = (
                    (self.adapter.root / name / item["path"]).read_bytes().decode("utf-8")
                )
        return files

    def _method_change(self, inputs, role):
        parent = inputs["parent"]
        current = {item["path"]: item["content"] for item in parent["files"]}
        for name in ("reference", "control"):
            target = self._prepared_targets(name, parent)
            changed = [
                path for path in parent["editable_paths"] if target.get(path) != current.get(path)
            ]
            if changed:
                title = "Restore reference" if name == "reference" else "Restore prepared control"
                title += (
                    " as a revision control" if role == "refine" else " after observed attempts"
                )
                return {
                    "idea": self.idea(title),
                    "change_plan": {
                        "changed_files": changed,
                        "reason": "Measure the distinct prepared source against the exact active parent",
                    },
                }
        raise ValueError("Prepared reference and control provide no distinct method revision")

    def _parent_patch(self, inputs):
        parent = inputs["parent"]
        name = (
            "control"
            if inputs["idea"]["title"].startswith("Restore prepared control")
            else "reference"
        )
        target = self._prepared_targets(name, parent)
        current = {item["path"]: item["content"] for item in parent["files"]}
        edits = [
            {
                "path": path,
                "before": current.get(path),
                "after": target[path],
                "reason": "Restore this declared prepared component and preserve all other parent bytes",
            }
            for path in inputs["change_plan"]["changed_files"]
        ]
        return {
            "parent_source_ref": parent["parent_source_ref"],
            "edits": edits,
            "description": f"Scripted restoration of prepared {name} through exact anchored parent edits",
        }

    def respond(self, role, inputs):
        if role in {"limitations", "verify_limitations"}:
            return {
                "limitations": ["Reference hyperparameters are fixed without task optimization"],
                "relevant": True,
                "reason": "Scripted validation assessment",
            }
        if role == "seeds":
            return {"ideas": [self.idea(f"Prepared control {i}") for i in range(inputs["count"])]}
        if role == "evolve":
            return self._method_change(inputs, role)
        if role in {"implement", "engineer"} and "parent" in inputs:
            return self._parent_patch(inputs)
        if role == "implement" and inputs["idea"]["title"] == "Prepared control 1":
            return self.adapter.baseline().model_dump()
        if role in {"implement", "engineer"}:
            return self.adapter.control().model_dump()
        if role == "critic":
            return {
                "decision": "Good",
                "reason": "Controller must independently enforce measured comparison and validity",
            }
        if role == "ablations":
            changed = self.adapter.control_component()
            return {
                "plans": [
                    {
                        "purpose": "Remove the prepared control component under matched budgets",
                        "changed_file": changed,
                        "candidate": self.adapter.baseline().model_dump(),
                    }
                ]
                if changed
                else []
            }
        if role == "rebuttal":
            changed = self.adapter.control_component()
            concerns = [item["id"] for item in inputs["concerns"]]
            if changed is None:
                return {
                    "plans": [],
                    "unaddressed_concerns": [
                        {
                            "concern_ids": concerns,
                            "reason": "Prepared controls have no distinct component for a paired comparison",
                        }
                    ]
                    if concerns
                    else [],
                }
            return {
                "plans": [
                    {
                        "id": "prepared-reference-control",
                        "concern_ids": concerns,
                        "purpose": "Measure the prepared reference against the frozen incumbent under matched supplementary seeds",
                        "seed_count": inputs["action_contract"]["max_seed_count"],
                        "execution_kind": "paired_control",
                        "evaluation_split": "development",
                        "diagnostics": [
                            {"kind": "metric", "metric": inputs["action_contract"]["metrics"][0]}
                        ],
                        "changed_file": changed,
                        "candidate": self.adapter.baseline().model_dump(),
                    }
                ],
                "unaddressed_concerns": [],
            }
        if role == "ablation_critic":
            return {
                "decision": "Good",
                "reason": "Preserve evidence; no unsupported mechanism attribution",
            }
        if role == "refine":
            return self._method_change(inputs, role)
        if role == "draft":
            evidence = [row for row in inputs["evidence"] if row["status"] == "success"]
            return {
                "title": "Scripted validation of a prepared MLAgentBench task",
                "claims": [
                    {"experiment_id": row["id"], "metric": row["metric"], "value": row["value"]}
                    for row in evidence
                ],
                "limitations": [
                    "Synthetic agents; real measured experiments; no live autonomous research assessment"
                ],
                "source_ids": ["task"],
                "concern_responses": [
                    {
                        "concern_id": item["concern_id"],
                        "status": {
                            "executed": "executed",
                            "partial": "limited",
                            "unfulfilled": "unaddressed",
                        }[item["status"]],
                        "experiment_ids": item["observation_ids"],
                        "explanation": "Recorded execution coverage only; resolution of the scientific concern remains unverified",
                    }
                    for item in inputs.get("supplementary_coverage", [])
                ],
            }
        if role == "review":
            return {"score": 7, "concerns": ["Repeat the reference as a supplementary control"]}
        if role == "meta":
            return {
                "decision": "ReviseMethod",
                "reason": "Measure a reference-restoration revision and verify that an unsuccessful revision preserves the incumbent",
            }
        raise ValueError(f"Unknown scripted validation role: {role}")


def create_validation_spec(
    prepared,
    *,
    image,
    grader_image=None,
    context=None,
    seconds=1800,
    campaign_seconds=18000,
    max_prompt_chars=200_000,
    level="full",
    runtime="docker",
    gpus=None,
    compiler_image=None,
    require_pdf=False,
):
    root = Path(prepared).resolve()
    value = verify_prepared(root)
    recipe = get_recipe(value["task_id"])
    family = None
    if recipe.task_id not in CPU_TASKS | {"vectorization"}:
        from ..task_families import get_family

        family = get_family(
            recipe.task_id, settings=value["split_metadata"].get("resource_limits", {})
        )
        if gpus is None:
            raise ValueError(
                "Framework validation requires an explicit allocated --gpus device; no automatic all-GPU selection"
            )
    if level not in {"smoke", "full"}:
        raise ValueError("Validation level must be smoke or full")
    assets = []
    if (root / "assets").is_dir():
        for path in sorted((root / "assets").iterdir()):
            if not path.is_dir():
                raise ValueError("Asset caches must be separated into named directories")
            assets.append(
                RuntimeAsset(
                    name=path.name,
                    path=path,
                    role="tokenizer" if "tokenizer" in path.name else "runtime",
                    manifest=tree_manifest(path),
                )
            )
    native = WorkspaceSpec(
        name=recipe.task_id,
        task_id=recipe.task_id,
        public_dir=root / "phases/final/public",
        private_dir=root / "phases/final/private",
        candidate_dir=root / "reference",
        evaluator_dir=root / "evaluator",
        command=["python", "train.py"],
        evaluator_command=[
            "python",
            "timing_worker.py" if recipe.task_id == "vectorization" else "evaluate.py",
        ],
        metric=recipe.metric,
        direction=recipe.direction,
        image=image,
        grader_image=grader_image or image,
        context=context,
        runtime=runtime,
        data_provenance="user_prepared"
        if value["provenance"] == "actual_data"
        else "contract_fixture",
        resources=NativeResources(
            seconds=seconds,
            grading_seconds=min(seconds, 300),
            gpus=gpus,
            memory_mb=32768 if recipe.task_id == "llama-inference" else 8192 if family else 2048,
            shm_mb=2048 if family else 64,
            # Profile imports create small temporary files
            # before the free-space probe. Capacity must
            # exceed the unchanged minimum-free contract.
            tmp_mb=(runtime_profile(recipe.runtime_profile).requirements.min_tmp_mb + 64)
            if family
            else 512,
        ),
        assets=assets,
    )
    development = DevelopmentSpec(
        adapter="testbed",
        metric=recipe.metric,
        direction=recipe.direction,
        seeds=recipe.seeds if level == "full" else [0],
        prediction_columns=recipe.targets or [recipe.metric],
        settings={
            "prepared_root": str(root),
            "runtime_profile": recipe.runtime_profile,
            **(
                {"minimum_free_vram_mb": recipe.training_constraints["minimum_free_vram_mb"]}
                if "minimum_free_vram_mb" in recipe.training_constraints
                else {}
            ),
            **(
                {
                    "required_free_disk_mb": value["split_metadata"]["resource_forecast"][
                        "required_free_disk_mb"
                    ]
                }
                if value["split_metadata"].get("resource_forecast")
                else {}
            ),
        },
    )
    if family:
        from .adapter import PreparedDevelopment

        adapter = PreparedDevelopment(native, development)
        native.evaluation_pipeline = adapter.final_evaluation()
        native.output_artifacts = adapter.artifacts(None)
    native = freeze_workspace(native)
    from ..research.policy import ResearchPolicy
    return LiveSpec(
        engine_version="research-v2",
        workflow=ResearchPolicy(profile="offline-pilot", compiler_image=compiler_image, require_pdf=require_pdf),
        native=native,
        development=development,
        model="scripted-fixture-v1",
        mode="research" if level == "full" else "smoke",
        max_calls=120,
        campaign_seconds=campaign_seconds,
        max_prompt_chars=max_prompt_chars,
        max_experiments=60,
        search_rounds=2 if level == "full" else 1,
        model_retries=0,
        engineering_retries=1,
        implementation=implementation_identity(),
    )


def validate(
    prepared,
    output,
    *,
    image,
    grader_image=None,
    context=None,
    seconds=1800,
    campaign_seconds=18000,
    max_prompt_chars=200_000,
    level="full",
    runtime="docker",
    gpus=None,
    compiler_image=None,
    require_pdf=False,
):
    """Run/resume one bounded validation. Container unavailability never falls back to host."""
    from ..cli import locked

    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if output.is_relative_to(prepared) or prepared.is_relative_to(output):
        raise ValueError("Validation output cannot overlap immutable prepared data")
    value = verify_prepared(prepared)
    spec = create_validation_spec(
        prepared,
        image=image,
        grader_image=grader_image,
        context=context,
        seconds=seconds,
        campaign_seconds=campaign_seconds,
        max_prompt_chars=max_prompt_chars,
        level=level,
        runtime=runtime,
        gpus=gpus,
        compiler_image=compiler_image,
        require_pdf=require_pdf,
    )
    with locked(output):
        spec_path = output / "validation-spec.json"
        content = canonical(spec.model_dump(mode="json"))
        if spec_path.exists() and spec_path.read_bytes() != content:
            raise ValueError("Validation configuration changed on resume")
        if not spec_path.exists():
            spec_path.write_bytes(content)
        receipt_path = output / "receipt.json"
        if receipt_path.exists():
            saved = json.loads(receipt_path.read_text())
            if saved.get("gates") and all(saved["gates"].values()):
                from .readiness import verify_validation

                return verify_validation(prepared, output)
        adapter = load_development(spec.native, spec.development)
        store = Store(output / "campaign")
        try:
            backend = ScriptedResearch(adapter)
            if spec.engine_version == "research-v2":
                from ..research.fixtures import ResearchFixtures
                backend = ResearchFixtures(backend.respond, store=store,
                                           materialize=getattr(adapter, "materialize", None))
            budget = Budget(store, spec)
            runner = DevelopmentRunner(store, spec, adapter, budget)
            controller = LiveController(store, spec, adapter, backend, runner, budget=budget)
            outcome = controller.run()
            # Regardless of scientific improvement, validate the prepared control
            # on full development and protected final data. This occurs after the
            # controller's last agent call; scores never feed method refinement.
            observations, finals, final_operation_keys = {}, {}, {}
            if outcome["status"] == "complete" and outcome["outcome"] not in {
                "baseline_failed",
                "budget_exhausted",
            }:
                if runner.image is None:
                    runner.preflight()
                for name, candidate in (
                    ("reference", adapter.baseline()),
                    ("control", adapter.control()),
                ):
                    observations[name] = runner.execute(
                        f"validation/{name}/full", candidate, "full"
                    ).model_dump()
                    finals[name], final_operation_keys[name] = _final_once(runner, name, candidate)
                before = budget.summary()
                resumed = controller.run()
                for name, candidate in (
                    ("reference", adapter.baseline()),
                    ("control", adapter.control()),
                ):
                    runner.execute(f"validation/{name}/full", candidate, "full")
                    _final_once(runner, name, candidate)
                after = budget.summary()
                recovery = resumed == outcome and after == before
                store.set(
                    "validation_recovery",
                    store.put_json(
                        {
                            "before": before,
                            "after": after,
                            "state_before": outcome,
                            "state_after": resumed,
                            "no_duplicate_operations": recovery,
                        },
                        "validation-recovery-proof",
                    ),
                )
            else:
                recovery = False
            operations = budget.summary()
            model_calls = sum(
                not store.json(row["result_ref"]).get("synthetic", True)
                for row in operations
                if row["kind"] == "model" and row["result_ref"]
            )
            roles = sorted(
                {
                    store.json(json.loads(row[0]))["role"]
                    for row in store.db.execute(
                        "SELECT value FROM metadata WHERE key LIKE 'request/%'"
                    )
                }
            )
            images = store.get("live_images")
            probe_ref = store.get("runtime_probe_ref")
            probe = store.json(probe_ref) if probe_ref else None
            hardware = store.get("runtime_hardware")
        finally:
            store.close()
        audit = audit_live(output / "campaign", reexecute_evaluators=True)
        gates = {
            "development": len(observations) == 2
            and all(row["status"] == "success" for row in observations.values()),
            "final": len(finals) == 2
            and all(row["status"] == "completed" for row in finals.values()),
            "audit": audit["status"] == "pass",
            "recovery": recovery,
            "zero_live_calls": model_calls == 0,
            "runtime": bool(probe and probe.get("report", {}).get("status") == "pass"),
            "hardware": bool(
                probe
                and probe.get("report", {}).get("status") == "pass"
                and hardware
                and hardware.get("status") == "verified"
            ),
        }
        receipt = {
            "schema_version": 1,
            "task_id": value["task_id"],
            "recipe_sha256": value["recipe_sha256"],
            "prepared_manifest_sha256": manifest_hash(value),
            "provenance": value["provenance"],
            "dataset_identity": value.get("dataset_identity", {"status": "unverified"}),
            "level": level,
            "outcome": outcome,
            "gates": gates,
            "images": images,
            "runtime_probe_ref": probe_ref,
            "runtime_probe": probe,
            "hardware": hardware,
            "observations": observations,
            "finals": finals,
            "audit": audit,
            "final_operation_keys": final_operation_keys,
            "roles": roles,
            "live_model_calls": model_calls,
            "synthetic_agents": True,
            "research_tokens": 0,
            "operations": len(operations),
            "actual_task_validated": (
                value["provenance"] == "actual_data"
                and value.get("dataset_identity", {}).get("status") == "verified"
                and level == "full"
                and all(gates.values())
            ),
            "evidence": sealed_evidence(output / "campaign"),
        }
        (output / "receipt.json").write_bytes(canonical(receipt))
        return receipt
