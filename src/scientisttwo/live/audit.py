"""Recompute trusted metrics from sealed predictions; never call agents or private graders."""

import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from ..store import Store, canonical, digest, IntegrityError
from .adapters import implementation_identity, load_development
from .models import Candidate, LiveSpec, Observation
from .evaluation import aggregate_scores, normalize_scores, observation_metrics
from .workspace import artifact_contracts
from .runtime import validate_prepared
from .source_tree import candidate_snapshot
from ..artifacts import (
    ArtifactFile,
    load_bundle,
    restore_bundle,
    freeze_bundle,
    safe_path,
    verified_file,
)
from .pipeline import EvaluationPipeline, evaluate_pipeline
from ..benchmarks.runner import NativeRunner
from ..process import run_process


def _observation_binding(store, row, result, *, final=False):
    namespace = "final-paired/" if final else "development/"
    if row["key"] != namespace + result.id or (result.phase == "final") != final:
        raise ValueError("Observation phase or ID differs from its evidence namespace")
    if result.status == "success":
        materialization = store.db.execute(
            "SELECT output_ref FROM calls WHERE key=?",
            ("materialization/" + namespace + result.id,),
        ).fetchone()
        evidence = store.json(materialization[0]) if materialization else None
        if (
            evidence is None
            or evidence["code_ref"] != result.code_ref
            or (
                evidence.get("source_tree_ref")
                and evidence["source_tree_ref"] != result.source_tree_ref
            )
        ):
            raise ValueError("Observation source differs from its materialization receipt")
        request = {
            "candidate": store.json(evidence["proposal_ref"]),
            "protocol": result.protocol_hash,
        }
        if row["input_hash"] != digest(canonical(request)):
            raise ValueError("Observation request binding differs from source or protocol")
    operation = store.db.execute(
        "SELECT * FROM live_operations WHERE key=?",
        (("final/" if final else "experiment/") + result.id,),
    ).fetchone()
    if (
        operation is None
        or operation["result_ref"] != row["output_ref"]
        or operation["request_hash"] != row["input_hash"]
        or operation["status"] != result.status
        or operation["seconds"] != result.seconds
    ):
        raise ValueError(
            "Observation/evaluation completion binding differs from its operation ledger"
        )


def _prepared_partition(store, adapter, phase):
    ref = store.get("prepared/" + phase)
    if ref is None:
        raise ValueError("Observation has no prepared evaluation partition")
    root = store.root / "final-data" if phase == "final" else store.root / "development" / phase
    prepared = store.json(ref)
    validate_prepared(prepared, root)
    if prepared.get("private"):
        from ..benchmarks.files import verify_tree

        private = Path(prepared["private"])
        if (
            private.is_symlink()
            or not private.is_dir()
            or not private.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("Protected evaluation labels escaped their owned partition")
        public = Path(prepared["public"]).resolve()
        if private.resolve().is_relative_to(public) or public.is_relative_to(private.resolve()):
            raise ValueError("Protected evaluation labels overlap the public partition")
        verify_tree(private, prepared["private_files"])
    with tempfile.TemporaryDirectory(prefix="scientisttwo-audit-data-") as temp:
        rebuilt = validate_prepared(adapter.prepare(phase, Path(temp)), Path(temp))
        if {k: v for k, v in rebuilt.items() if k not in {"public", "private"}} != {
            k: v for k, v in prepared.items() if k not in {"public", "private"}
        }:
            raise ValueError("Evaluation split or labels differ from deterministic reconstruction")
    return prepared


def audit_live(output, *, reexecute_evaluators=False):
    output = Path(output)
    if not (output / "campaign.sqlite").is_file():
        raise ValueError("Live campaign not found")
    store = Store(output)
    checks, errors = [], []
    verification = {
        "required_container_replays": [],
        "replay_sources": {},
        "container_replays": [],
        "host_metric_recomputations": [],
        "paired_correctness_recomputations": [],
    }
    try:
        errors += store.verify()
        if errors:
            return {"status": "fail", "errors": errors, "live_model_calls": 0}
        config = LiveSpec.model_validate(store.json(store.get("live_spec_ref")))
        if config.implementation != implementation_identity():
            raise ValueError("Evaluator implementation differs from frozen campaign")
        adapter = load_development(config.native, config.development)
        if store.get("authorized_source_snapshot"):
            adapter.bind_source(store.get("authorized_source_snapshot"))
        if adapter.identity() != store.get("live_adapter_identity"):
            raise ValueError("Development adapter/source data identity differs")
        for row in store.db.execute(
            "SELECT output_ref FROM calls WHERE key LIKE 'materialization/%'"
        ):
            evidence = store.json(row[0])
            proposal = Candidate.model_validate(store.json(evidence["proposal_ref"]))
            transform = getattr(adapter, "materialize", None)
            rebuilt = transform(proposal) if transform is not None else proposal
            if rebuilt.base_snapshot_ref and rebuilt.base_snapshot_ref != store.get(
                "authorized_source_snapshot"
            ):
                raise ValueError("Materialized proposal uses an unauthorized source snapshot")
            adapter.validate(rebuilt)
            tree = candidate_snapshot(store, rebuilt)
            if evidence.get("source_tree_ref") and tree.model_dump(mode="json") != store.json(
                evidence["source_tree_ref"]
            ):
                raise ValueError("Materialized source tree differs from declared proposal/base")
            expected = (
                tree.model_dump(mode="json")
                if rebuilt.base_snapshot_ref
                else rebuilt.model_dump(mode="json")
            )
            if expected != store.json(evidence["code_ref"]):
                raise ValueError("Materialized candidate differs from declared proposal/template")
            checks.append({"materialization": evidence["code_ref"], "scope": "pass"})
        from .parent_audit import audit_parent_edits

        checks.extend(audit_parent_edits(store, adapter))
        for phase in ("subset", "full"):
            ref = store.get("prepared/" + phase)
            if not ref:
                continue
            _prepared_partition(store, adapter, phase)
        observations = {}
        for row in store.db.execute("SELECT * FROM calls WHERE key LIKE 'development/%'"):
            result = Observation.model_validate(store.json(row["output_ref"]))
            _observation_binding(store, row, result)
            observations[result.id] = result
            if result.status != "success":
                continue
            if result.timing_refs:
                from .paired_evaluation import audit_paired_observation

                prepared = store.json(
                    store.get("prepared/" + ("subset" if result.phase == "subset" else "full"))
                )
                check = audit_paired_observation(store, result, config, adapter, prepared)
                checks.append(check)
                verification["paired_correctness_recomputations"].append(
                    _metric_proof(
                        store,
                        "development/" + result.id,
                        "paired_output_recomputation",
                        row["output_ref"],
                        result,
                        check,
                    )
                )
                continue
            materialization = store.db.execute(
                "SELECT output_ref FROM calls WHERE key=?",
                ("materialization/development/" + result.id,),
            ).fetchone()
            if (
                materialization is None
                or store.json(materialization[0])["code_ref"] != result.code_ref
            ):
                raise ValueError("Observation source differs from its materialization receipt")
            if not (len(result.values) == len(result.prediction_refs) == len(result.log_refs)):
                raise ValueError("Metric/evidence cardinality mismatch")
            if result.evaluation_refs and (
                not result.protocol_ref
                or len(result.evaluation_refs) != len(result.values)
                or len(result.bundle_refs) != len(result.values)
            ):
                raise ValueError("Staged prediction/evaluation evidence cardinality mismatch")
            prepared = store.json(
                store.get("prepared/" + ("subset" if result.phase == "subset" else "full"))
            )
            if result.protocol_ref:
                protocol = store.json(result.protocol_ref)
                if (
                    digest(canonical(protocol)) != result.protocol_hash
                    or protocol["adapter"] != adapter.identity()
                    or protocol["evaluation"] != config.development.model_dump(mode="json")
                    or protocol["data"] != prepared["files"]
                    or protocol["phase"] != result.phase
                    or len(protocol["seeds"]) != len(result.values)
                ):
                    raise ValueError("Observation protocol differs from declared evaluation")
            recomputed = []
            contracts = artifact_contracts(adapter, prepared)
            names = [s.path for s in contracts]
            bundles = result.artifact_refs
            if not bundles:
                if len(names) != 1:
                    raise ValueError("Legacy observation cannot describe multiple artifacts")
                bundles = [{names[0]: ref} for ref in result.prediction_refs]
            if len(bundles) != len(result.values):
                raise ValueError("Output artifact/evidence cardinality mismatch")
            for index, (value, ref, log, bundle) in enumerate(
                zip(result.values, result.prediction_refs, result.log_refs, bundles)
            ):
                manifest = (
                    load_bundle(store, result.bundle_refs[index]) if result.bundle_refs else None
                )
                if manifest:
                    expected = {name: f.ref for name, f in manifest.files.items()}
                    first = (
                        contracts[0].path
                        if contracts[0].kind == "file"
                        else next(
                            name for name in expected if name.startswith(contracts[0].path + "/")
                        )
                    )
                    if (manifest.specs != contracts or bundle != expected
                            or (not result.evaluation_refs and bundle[first] != ref)):
                        raise ValueError(
                            "Prediction artifacts differ from declared output contract"
                        )
                elif set(bundle) != set(names) or bundle[names[0]] != ref:
                    raise ValueError("Prediction artifacts differ from declared output contract")
                receipt = store.json(log)
                if receipt["returncode"] or receipt["timed_out"]:
                    raise ValueError("Successful metric references failed execution")
                if result.evaluation_refs:
                    evaluated = store.json(result.evaluation_refs[index])
                    seed = protocol["seeds"][index]
                    cached_evaluation = store.db.execute(
                        "SELECT output_ref FROM calls WHERE key=?",
                        ("pipeline/" + result.id + "/" + str(seed),),
                    ).fetchone()
                    if (
                        cached_evaluation is None
                        or store.json(cached_evaluation[0]) != evaluated
                        or evaluated["input_bundle"] != result.bundle_refs[index]
                        or evaluated["prediction_bundle"] != ref
                        or evaluated["pipeline"] != protocol.get("pipeline")
                        or evaluated["seed"] != seed
                        or evaluated["public_manifest"] != prepared["files"]
                    ):
                        raise ValueError(
                            "Evaluation binding belongs to a different candidate or protocol"
                        )
                    predictions = load_bundle(store, ref)
                    if (not any(spec.role == "prediction" for spec in predictions.specs)
                            or any(spec.role not in {"prediction", "diagnostic"}
                                   for spec in predictions.specs)):
                        raise ValueError("Scored prediction bundle contains non-prediction artifacts")
                    inference = evaluated["pipeline"].get("inference")
                    expected_predictions = (
                        inference["outputs"] if inference else
                        [spec.model_dump(mode="json") for spec in contracts]
                    )
                    if [spec.model_dump(mode="json") for spec in predictions.specs] != expected_predictions:
                        raise ValueError("Scored prediction bundle differs from the inference contract")
                    request = store.json(evaluated["request_ref"])
                    if (
                        request["bundle"] != result.bundle_refs[index]
                        or request["pipeline"] != evaluated["pipeline"]
                        or request["seed"] != seed
                    ):
                        raise ValueError("Evaluator request binding differs")
                    grade = store.json(evaluated["grade_ref"])
                    if evaluated.get("diagnostics_bundle"):
                        diagnostics = load_bundle(store, evaluated["diagnostics_bundle"])
                        if [s.model_dump(mode="json") for s in diagnostics.specs] != evaluated[
                            "pipeline"
                        ]["score"]["outputs"]:
                            raise ValueError("Scoring diagnostic contract differs from protocol")
                    if (
                        grade.get("valid_submission") is not True
                        or grade["metrics"] != evaluated["metrics"]
                    ):
                        raise ValueError("Stored scoring response differs from evaluator grade")
                    for stage_ref in evaluated["receipts"]:
                        stage = store.json(stage_ref)
                        if stage["returncode"] or stage["timed_out"]:
                            raise ValueError("Successful score references failed evaluator")
                    actual = normalize_scores(config.development, grade["metrics"])
                    replay_id = "development/" + result.id + "/" + str(seed)
                    _require_replay(verification, replay_id, evaluated["grade_ref"], actual)
                    if reexecute_evaluators:
                        replayed, replay_proof = replay_evaluator(
                            store, config, evaluated, prepared, proof_id=replay_id
                        )
                        if any(
                            not math.isclose(actual[name], v, rel_tol=1e-10, abs_tol=1e-10)
                            for name, v in replayed.items()
                        ):
                            raise ValueError("Replayed trusted evaluator differs")
                        verification["container_replays"].append(replay_proof)
                else:
                    with tempfile.TemporaryDirectory(
                        prefix="scientisttwo-audit-prediction-"
                    ) as temp:
                        if manifest:
                            restore_bundle(store, result.bundle_refs[index], Path(temp).resolve())
                        else:
                            for name, output_ref in bundle.items():
                                target = Path(temp) / name
                                target.parent.mkdir(parents=True, exist_ok=True)
                                target.write_bytes(store.read(output_ref))
                        actual = normalize_scores(
                            config.development,
                            adapter.score(result.phase, prepared, Path(temp), receipt["seconds"]),
                        )
                    verification["host_metric_recomputations"].append(
                        _metric_proof(
                            store,
                            "development/" + result.id + "/" + str(index),
                            "trusted_host_metric_recomputation",
                            row["output_ref"],
                            result,
                            {"metrics": actual, "prediction_refs": list(bundle.values())},
                        )
                    )
                if not math.isclose(value, actual[result.metric], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError("Stored metric differs from trusted prediction score")
                recomputed.append(actual)
            metrics, values = aggregate_scores(config.development, recomputed)
            recorded = observation_metrics(config.development, result)
            if any(
                not math.isclose(recorded[name], v, rel_tol=1e-10, abs_tol=1e-10)
                for name, v in metrics.items()
            ):
                raise ValueError("Stored aggregate metric differs")
            if result.metric_values and any(
                len(result.metric_values[name]) != len(series)
                or any(
                    not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
                    for a, b in zip(result.metric_values[name], series)
                )
                for name, series in values.items()
            ):
                raise ValueError("Stored per-execution metrics differ")
            checks.append({"experiment": result.id, "predictions": "pass", "metric": "pass"})
        # Final evidence is audited in its own namespace. It is deliberately not
        # added to observations, which is the sole source of development draft claims.
        final_rows = list(store.db.execute("SELECT * FROM calls WHERE key LIKE 'final-paired/%'"))
        if final_rows or store.get("prepared/final"):
            if not store.get("live_frozen"):
                raise ValueError("Protected final evidence exists without a frozen research method")
            prepared_final = _prepared_partition(store, adapter, "final")
            for row in final_rows:
                result = Observation.model_validate(store.json(row["output_ref"]))
                _observation_binding(store, row, result, final=True)
                if result.status != "success":
                    checks.append({"final_experiment": result.id, "status": result.status})
                    continue
                if not result.timing_refs:
                    raise ValueError("Successful paired final result has no timing evidence")
                from .paired_evaluation import audit_paired_observation

                check = audit_paired_observation(store, result, config, adapter, prepared_final)
                check["final_experiment"] = check.pop("experiment")
                check["scope"] = "protected_final_only"
                checks.append(check)
                verification["paired_correctness_recomputations"].append(
                    _metric_proof(
                        store,
                        "final/" + result.id,
                        "paired_output_recomputation",
                        row["output_ref"],
                        result,
                        check,
                    )
                )
        if store.get("live_draft"):
            draft = store.json(store.get("live_draft"))
            for claim in draft["claims"]:
                e = observations.get(claim["experiment_id"])
                metrics = (
                    observation_metrics(config.development, e)
                    if e is not None and e.status == "success"
                    else {}
                )
                if (
                    e is None
                    or e.status != "success"
                    or claim["metric"] not in metrics
                    or not math.isclose(
                        claim["value"], metrics[claim["metric"]], rel_tol=1e-10, abs_tol=1e-10
                    )
                ):
                    raise ValueError("Unsupported draft claim")
        for nested in (output / "final").glob("*/campaign.sqlite"):
            child = Store(nested.parent)
            try:
                errors += child.verify()
                if child.get("workspace_spec"):
                    checks.append(
                        _audit_workspace_final(
                            child,
                            config,
                            reexecute_evaluators,
                            proof_store=store,
                            verification=verification,
                        )
                    )
            finally:
                child.close()
    except (ValueError, OSError, KeyError, TypeError, IntegrityError) as error:
        errors.append(str(error))
    finally:
        store.close()
    verification["container_replay_count"] = len(verification["container_replays"])
    verification["complete"] = (
        not errors
        and set(verification["required_container_replays"])
        == {item["id"] for item in verification["container_replays"]}
        and any(
            verification[name]
            for name in (
                "container_replays",
                "host_metric_recomputations",
                "paired_correctness_recomputations",
            )
        )
    )
    return {
        "status": "fail" if errors else "pass",
        "checks": checks,
        "errors": errors,
        "live_model_calls": 0,
        "scope": "Artifact integrity and metric recomputation; no experiment rerun",
        "method_code_consistency": "unverified",
        "container_evaluator_reexecution": "requested" if reexecute_evaluators else "not_requested",
        "verification": verification,
    }


def _audit_workspace_final(store, config, reexecute, *, proof_store, verification):
    """Bind frozen final metrics to the separately executed trusted grade."""
    from .workspace_models import WorkspaceSpec
    from .workspace_task import preflight

    spec = WorkspaceSpec.model_validate(store.json(store.get("workspace_spec")))
    if not preflight(spec)["ready"]:
        raise ValueError("Frozen final workspace prerequisites changed")
    result_ref = store.get("workspace_result")
    if result_ref is None:
        return {"final_workspace": store.root.name, "status": "uncommitted"}
    result = store.json(result_ref)
    check = {"final_workspace": store.root.name, "status": result["status"]}
    if result["status"] != "completed":
        return check
    if result["metric"] != spec.metric or result["direction"] != spec.direction:
        raise ValueError("Final workspace objective differs from its frozen specification")
    if result["execution_ref"] != store.get("workspace_execution") or result[
        "artifacts_ref"
    ] != store.get("workspace_artifacts"):
        raise ValueError("Final workspace execution/artifact binding differs")
    execution = store.json(result["execution_ref"])
    if execution["returncode"] or execution["timed_out"] or execution.get("limit_exceeded"):
        raise ValueError("Successful final score references failed execution")
    report = store.json(result["official_report_ref"])
    if spec.evaluation_pipeline:
        cached = store.db.execute(
            "SELECT output_ref FROM calls WHERE key='pipeline/final'"
        ).fetchone()
        if (
            cached is None
            or cached[0] != result["official_report_ref"]
            or report["seed"] != spec.final_seed
        ):
            raise ValueError("Final scoring pipeline binding differs")
        bundle = load_bundle(store, report["input_bundle"])
        outputs = store.json(result["artifacts_ref"])
        if bundle.specs != spec.output_artifacts or any(
            outputs["references"].get(name) != file.ref for name, file in bundle.files.items()
        ):
            raise ValueError("Final scoring input is not the frozen execution artifact")
        request = store.json(report["request_ref"])
        if (
            request["bundle"] != report["input_bundle"]
            or request["source"] != spec.candidate_manifest
            or request["public"] != spec.public_manifest
        ):
            raise ValueError("Final evaluator request differs from frozen source/data")
        grade = store.json(report["grade_ref"])
        if grade.get("valid_submission") is not True or grade["metrics"] != report["metrics"]:
            raise ValueError("Final pipeline score differs from its sealed grade")
        for ref in report["receipts"]:
            stage = store.json(ref)
            if stage["returncode"] or stage["timed_out"] or stage.get("limit_exceeded"):
                raise ValueError("Successful final score references a failed evaluation stage")
        replay_id = "final/" + store.root.name
        _require_replay(verification, replay_id, report["grade_ref"], grade["metrics"])
        if reexecute:
            prepared = {"public": str(spec.public_dir), "files": spec.public_manifest}
            replayed, replay_proof = replay_evaluator(
                store,
                SimpleNamespace(native=spec, development=config.development),
                report,
                prepared,
                evidence_store=proof_store,
                proof_id=replay_id,
            )
            if set(replayed) != set(grade["metrics"]) or any(
                not math.isclose(value, grade["metrics"][name], rel_tol=1e-10, abs_tol=1e-10)
                for name, value in replayed.items()
            ):
                raise ValueError("Replayed final scoring metrics differ")
            verification["container_replays"].append(replay_proof)
    else:
        grade = report
        grading = store.json(store.get("workspace_grading"))
        if grading["returncode"] or grading["timed_out"] or grading.get("limit_exceeded"):
            raise ValueError("Successful final score references a failed trusted grader")
        replay_id = "final/" + store.root.name
        _require_replay(verification, replay_id, result["official_report_ref"], grade["metrics"])
        if reexecute:
            replayed, replay_proof = _replay_legacy_final(
                store, proof_store, spec, result, replay_id
            )
            if set(replayed) != set(grade["metrics"]) or any(
                not math.isclose(value, grade["metrics"][name], rel_tol=1e-10, abs_tol=1e-10)
                for name, value in replayed.items()
            ):
                raise ValueError("Replayed final scoring metrics differ")
            verification["container_replays"].append(replay_proof)
    metrics = grade.get("metrics")
    if (
        grade.get("valid_submission") is not True
        or not isinstance(metrics, dict)
        or spec.metric not in metrics
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in metrics.values()
        )
    ):
        raise ValueError("Final workspace has no valid finite trusted metrics")
    if result["metrics"] != metrics or result["score"] != metrics[spec.metric]:
        raise ValueError("Final workspace metrics differ from trusted grading evidence")
    check.update(metric="pass", final_only=True)
    return check


def replay_evaluator(store, config, evaluated, prepared, *, evidence_store=None, proof_id=None):
    """Replay trusted scoring only, in a separate bounded audit store and container."""

    class ReplayRunner:
        _launch = NativeRunner._launch
        _image = NativeRunner._image
        executor = staticmethod(run_process)

    root = store.root / "audit-work"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scorer-", dir=root) as temp:
        destination = Path(temp)
        predictions = destination / "predictions"
        bundle = restore_bundle(store, evaluated["prediction_bundle"], predictions)
        audit_store = Store(destination / "evidence")
        try:
            ref = freeze_bundle(audit_store, predictions, bundle.specs)
            runner = ReplayRunner()
            runner.store, runner.spec = audit_store, config.native
            runner.prefix = [config.native.runtime] + (
                ["--context", config.native.context] if config.native.context else []
            )
            declaration = {**evaluated["pipeline"], "inference": None}
            pipeline = EvaluationPipeline.model_validate(declaration)
            result = evaluate_pipeline(
                runner, "replay", pipeline, ref, Path(prepared["public"]), seed=evaluated["seed"]
            )
            metrics = normalize_scores(config.development, result["metrics"])
            if proof_id is None:
                return metrics
            proof = _seal_replay(
                audit_store,
                evidence_store or store,
                proof_id,
                evaluated["grade_ref"],
                metrics,
                result["receipts"],
                result["grade_ref"],
            )
            return metrics, proof
        finally:
            audit_store.close()


def _require_replay(verification, identifier, grade_ref, metrics):
    verification["required_container_replays"].append(identifier)
    verification["replay_sources"][identifier] = {"grade_ref": grade_ref, "metrics": metrics}


def _metric_proof(store, identifier, kind, observation_ref, observation, detail):
    metrics = detail.get("metrics", observation.metrics)
    ref = store.put_json(
        {
            "verification_kind": kind,
            "id": identifier,
            "status": "pass",
            "observation_ref": observation_ref,
            "phase": observation.phase,
            "metrics": metrics,
            "detail": detail,
        },
        "audit-metric-recomputation",
    )
    return {"id": identifier, "evidence_ref": ref, "metrics": metrics}


def _seal_replay(source, destination, identifier, source_grade_ref, metrics, executions, grade_ref):
    errors = source.verify()
    if errors:
        raise IntegrityError(errors[0])
    for row in source.db.execute("SELECT path,kind FROM artifacts"):
        copied = destination.put_file(source.root / row["path"], row["kind"])
        if copied != row["path"]:
            raise IntegrityError("Audit replay artifact identity changed during transfer")
    ref = destination.put_json(
        {
            "verification_kind": "container_evaluator_replay",
            "id": identifier,
            "status": "pass",
            "source_grade_ref": source_grade_ref,
            "metrics": metrics,
            "execution_refs": executions,
            "replayed_grade_ref": grade_ref,
        },
        "audit-container-replay",
    )
    return {"id": identifier, "evidence_ref": ref, "metrics": metrics}


def _replay_legacy_final(source, destination, spec, result, identifier):
    """Replay a declared legacy workspace grader against sealed submission files."""
    from ..runtime_contracts import asset_mounts
    from .workspace_task import preflight

    class ReplayRunner:
        _launch = NativeRunner._launch
        executor = staticmethod(run_process)

    root = source.root / "audit-work"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="final-scorer-", dir=root) as temporary:
        work = Path(temporary)
        submission, output = work / "submission", work / "grade-output"
        submission.mkdir()
        output.mkdir()
        output.chmod(0o777)
        artifacts = source.json(result["artifacts_ref"])
        for name, metadata in artifacts["files"].items():
            safe_path(name)
            artifact = ArtifactFile(
                ref=artifacts["references"][name], sha256=metadata["sha256"], size=metadata["size"]
            )
            path = submission / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(verified_file(source, artifact), path)
        evidence = Store(work / "evidence")
        try:
            runner = ReplayRunner()
            runner.store, runner.spec = evidence, spec
            runner.prefix = [spec.runtime] + (["--context", spec.context] if spec.context else [])
            mounts = [
                (submission, "/submission", "ro"),
                (spec.public_dir, "/data/public", "ro"),
                (spec.evaluator_dir, "/evaluator", "ro"),
                (output, "/grade-output", "rw"),
            ] + ([(spec.private_dir, "/data/private", "ro")] if spec.private_dir else [])
            mounts += asset_mounts(spec, "score", protected=[spec.private_dir, spec.evaluator_dir])
            images = source.get("workspace_images")
            process, execution = runner._launch(
                "replay-final",
                images["grading"],
                {"command": spec.evaluator_command, "cwd": "/evaluator", "seed": spec.final_seed},
                mounts,
                spec.resources.grading_seconds,
                gpus=spec.resources.gpus,
                bridge=Path(__file__).with_name("workspace_worker.py"),
            )
            if process.code or process.timed_out or process.limit_exceeded:
                raise ValueError("Final evaluator replay failed or exceeded a resource limit")
            grade_path = output / "grade.json"
            if (
                grade_path.is_symlink()
                or not grade_path.is_file()
                or grade_path.stat().st_size > 1_000_000
            ):
                raise ValueError("Final evaluator replay omitted a bounded regular grade")
            grade_ref = evidence.put_file(grade_path, "replayed-final-grade")
            grade = evidence.json(grade_ref)
            metrics = grade.get("metrics")
            if (
                grade.get("valid_submission") is not True
                or not isinstance(metrics, dict)
                or spec.metric not in metrics
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in metrics.values()
                )
            ):
                raise ValueError("Final evaluator replay did not produce valid finite metrics")
            if not preflight(spec)["ready"]:
                raise ValueError("Final evaluator inputs changed during replay")
            proof = _seal_replay(
                evidence,
                destination,
                identifier,
                result["official_report_ref"],
                metrics,
                [execution],
                grade_ref,
            )
            return metrics, proof
        finally:
            evidence.close()
