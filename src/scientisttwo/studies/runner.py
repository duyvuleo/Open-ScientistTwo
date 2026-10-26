"""Study-specific receipt enforcement and three-seed, grant-gated final evaluation."""

import math
from contextlib import nullcontext
import re
import time

from ..artifacts import load_bundle
from ..benchmarks.files import tree_manifest, verify_tree
from ..live.runtime import DevelopmentRunner, LiveBudgetExhausted
from ..live.source_tree import candidate_snapshot, restore_source_tree
from ..live.workspace_models import WorkspaceSpec
from ..live.workspace_task import WorkspaceRunner
from ..store import IntegrityError, Store, canonical, digest
from .cifar_baseline.training_contract import validate_overrides
from .cifar_recipe import validate_training_receipt

FINAL_SEEDS = [0, 1, 2]


def _valid_development_seeds(seeds, phase, *, supplementary_limit=32):
    if (phase not in {"subset", "full", "rebuttal"}
            or not isinstance(seeds, (list, tuple)) or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        return False
    if phase == "rebuttal":
        return len(seeds) <= supplementary_limit
    return all(seed in FINAL_SEEDS for seed in seeds)


def _small_json(store, bundle, name):
    member = bundle.files.get(name)
    if member is None or member.size > 65_536:
        raise IntegrityError("Missing or oversized immutable study configuration/receipt: " + name)
    value = store.json(member.ref)
    if not isinstance(value, dict):
        raise IntegrityError("Study configuration and receipts must be JSON objects")
    return value


def candidate_source_hash(store, candidate):
    tree = candidate_snapshot(store, candidate)
    manifest = load_bundle(store, tree.bundle_ref)
    return digest(canonical({"files": {name: {"sha256": item.sha256, "size": item.size,
                                                "executable": item.executable}
                                        for name, item in manifest.files.items()},
                             "command": candidate.command}))


def _source_configuration(store, tree_ref):
    tree = store.json(tree_ref) if isinstance(tree_ref, str) else tree_ref.model_dump()
    source = load_bundle(store, tree["bundle_ref"])
    base = _small_json(store, source, "training.json")
    overrides = validate_overrides(_small_json(store, source, "training_config.json"))
    return {**base, **overrides}


def validate_checkpoint_receipt(store, bundle_ref, expected_configuration, *, seed, phase):
    """Read bounded JSON only; model weights remain streaming verified artifacts."""
    bundle = load_bundle(store, bundle_ref)
    config = _small_json(store, bundle, "checkpoint/config.json")
    receipt = _small_json(store, bundle, "checkpoint/training.json")
    expected = {**expected_configuration, "seed": seed}
    if canonical(config) != canonical(expected):
        raise IntegrityError("Checkpoint configuration differs from the materialized candidate and declared seed")
    try:
        result = validate_training_receipt(receipt, config, phase=phase)
    except ValueError as error:
        raise IntegrityError(str(error)) from error
    return {**result, "seed": seed, "bundle_ref": bundle_ref,
            "configuration_ref": bundle.files["checkpoint/config.json"].ref,
            "training_receipt_ref": bundle.files["checkpoint/training.json"].ref}


def validate_observation_receipts(store, observation, *, expected_seeds=None):
    """Bind every successful development seed to the fixed executed training budget."""
    if observation.status != "success" or observation.phase not in {"subset", "full", "rebuttal"}:
        raise IntegrityError("Only successful non-final observations have development receipt proofs")
    if not observation.protocol_ref or not observation.source_tree_ref:
        raise IntegrityError("Successful study observation lacks its protocol or materialized source")
    protocol = store.json(observation.protocol_ref)
    seeds = protocol.get("seeds")
    if not isinstance(seeds, list) or not _valid_development_seeds(seeds, observation.phase):
        raise IntegrityError("Study protocol has invalid or repeated training seeds")
    if expected_seeds is not None and seeds != list(expected_seeds):
        raise IntegrityError("Study observation seeds differ from the requested evaluation")
    if protocol.get("phase") != observation.phase or digest(canonical(protocol)) != observation.protocol_hash:
        raise IntegrityError("Study observation protocol/phase binding differs")
    if len(observation.bundle_refs) != len(seeds) or len(observation.values) != len(seeds):
        raise IntegrityError("Successful study observation lacks one checkpoint and score per seed")
    expected = _source_configuration(store, observation.source_tree_ref)
    phase = "screen" if observation.phase == "subset" else "full"
    proofs = [validate_checkpoint_receipt(store, bundle, expected, seed=seed, phase=phase)
              for seed, bundle in zip(seeds, observation.bundle_refs, strict=True)]
    return {"status": "pass", "observation_id": observation.id,
            "observation_sha256": digest(canonical(observation)), "protocol_ref": observation.protocol_ref,
            "source_tree_ref": observation.source_tree_ref, "seeds": seeds, "receipts": proofs}


class StudyDevelopmentRunner(DevelopmentRunner):
    """Keep raw outcomes immutable; prevent invalid budgets from reaching selection."""

    def __init__(self, *args, final_grant_ref=None, final_grant_verifier=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_grant_ref = final_grant_ref
        self.final_grant_verifier = final_grant_verifier

    def execute(self, key, candidate, phase="full", seeds=None):
        if self.store.get("study_final_release_ref"):
            raise IntegrityError("Non-final experiments cannot run after the study final release")
        scope = getattr(self.budget, "experiment_scope", nullcontext)
        with scope():
            return self._execute_development(key, candidate, phase, seeds)

    def _execute_development(self, key, candidate, phase="full", seeds=None):
        expected = seeds if seeds is not None else (self.config.development.seeds[:1]
                                                   if phase == "subset" else self.config.development.seeds)
        if not _valid_development_seeds(expected, phase,
                                        supplementary_limit=self.config.supplementary_seed_count):
            raise ValueError("Study execution requires valid, unique, phase-bounded training seeds")
        observed = super().execute(key, candidate, phase, seeds)
        if observed.status != "success":
            return observed
        request = {"observation_sha256": digest(canonical(observed)), "seeds": list(expected)}
        proof_key = "study-training-validation/" + key
        try:
            proof = validate_observation_receipts(self.store, observed, expected_seeds=expected)
        except (IntegrityError, ValueError, OSError, KeyError) as error:
            proof = {"status": "fail", **request, "reason": str(error),
                     "raw_observation_ref": self.store.put_json(observed, "rejected-study-observation"),
                     "promotion": "blocked", "raw_execution_unchanged": True}
            self.store.cache(proof_key, request, proof)
            raise IntegrityError("Study training contract failed; immutable execution retained: " + str(error)) from error
        self.store.cache(proof_key, request, proof)
        return observed

    def _grant(self, label, source_hash):
        if not self.final_grant_ref or not callable(self.final_grant_verifier):
            raise IntegrityError("Final evaluation requires the verified BOTH_ARMS_FROZEN coordinator grant")
        value = self.final_grant_verifier(self, self.final_grant_ref, label, source_hash)
        if (not isinstance(value, dict) or value.get("status") != "BOTH_ARMS_FROZEN"
                or value.get("seeds") != FINAL_SEEDS or value.get("candidate_source_hash") != source_hash
                or not isinstance(value.get("grant_digest"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", value["grant_digest"])):
            raise IntegrityError("Final coordinator grant is absent, malformed or bound to another candidate")
        activate = getattr(self.budget, "activate_final_grant", None)
        if callable(activate):
            if "final_deadline" not in value:
                raise IntegrityError("Final coordinator grant omitted its separately bounded wall deadline")
            activate(self.final_grant_ref, value["final_deadline"])
        return value

    def final(self, key, candidate):
        if not isinstance(self.spec, WorkspaceSpec):
            raise ValueError("Study finals require the isolated WorkspaceSpec inference/scoring protocol")
        if key not in {"baseline", "selected"}:
            raise ValueError("Study finals use only the baseline and selected labels")
        source_hash = candidate_source_hash(self.store, candidate)
        grant = self._grant(key, source_hash)  # No final data or native runner access before this gate.
        request = {"candidate": candidate.model_dump(), "native": self.spec.model_dump(mode="json"),
                   "source_sha256": source_hash, "grant_ref": self.final_grant_ref,
                   "grant_digest": grant["grant_digest"], "seeds": FINAL_SEEDS,
                   "execution_type": "aggregation"}
        operation = "final/" + key
        prior = self.budget.reserve(operation, "experiment", request)
        if prior and prior["result_ref"]:
            result = self.store.json(prior["result_ref"])
            verify_final_aggregation(self.store, result)
            return result
        candidate, _ = self.materialize("study-final/" + key, candidate)
        tree = candidate_snapshot(self.store, candidate)
        expected = _source_configuration(self.store, tree)
        path = self.store.root / "frozen" / ("study-" + key)
        if path.exists():
            manifest = load_bundle(self.store, tree.bundle_ref)
            verify_tree(path, {name: {"sha256": item.sha256, "size": item.size}
                               for name, item in manifest.files.items()})
        else:
            restore_source_tree(self.store, tree, path)
        rows = []
        for seed in FINAL_SEEDS:
            current_grant = self._grant(key, source_hash)
            if current_grant != grant:
                raise IntegrityError("Coordinator grant changed during protected final evaluation")
            try:
                row = self._final_seed(key, seed, candidate, path, expected, source_hash, grant)
            except LiveBudgetExhausted as error:
                row = {"seed": seed, "status": "timeout", "reason": str(error), "elapsed_seconds": 0.0}
            rows.append(row)
            if row["status"] != "completed":
                break
        complete = len(rows) == len(FINAL_SEEDS) and all(row["status"] == "completed" for row in rows)
        values = [row["score"] for row in rows if row["status"] == "completed"]
        result = {"status": "completed" if complete else "interrupted" if any(row["status"] == "interrupted" for row in rows) else "failed",
                  "execution_type": "aggregation", "task_id": "cifar10", "metric": "accuracy",
                  "direction": "maximize", "score": sum(values) / 3 if complete else None,
                  "metrics": {"accuracy": sum(values) / 3} if complete else {},
                  "per_seed_values": values if complete else [],
                  "partial_seed_values": [] if complete else values,
                  "seeds": FINAL_SEEDS, "seed_results": rows,
                  "seed_result_refs": [row["result_ref"] for row in rows if row.get("result_ref")],
                  "source_sha256": source_hash, "grant_ref": self.final_grant_ref,
                  "source_tree_ref": self.store.put_json(tree, "study-final-materialized-source"),
                  "grant_digest": grant["grant_digest"], "live_model_calls": 0,
                  "elapsed_seconds": sum(row.get("elapsed_seconds", 0.0) for row in rows),
                  "charged_seconds": 0.0,
                  "reason": "Three independently executed matched final seeds" if complete else "Incomplete final evaluation; no aggregate score"}
        self.budget.finish(operation, result["status"], result, 0.0)
        verify_final_aggregation(self.store, result)
        return result

    def _final_seed(self, label, seed, candidate, path, expected, source_hash, grant):
        scope = getattr(self.budget, "experiment_scope", None)
        with (scope(final_label=label) if callable(scope) else nullcontext()):
            return self._execute_final_seed(label, seed, candidate, path, expected, source_hash, grant)

    def _execute_final_seed(self, label, seed, candidate, path, expected, source_hash, grant):
        key = f"{label}-seed-{seed}"
        operation = "final/" + key
        spec = self.spec.model_copy(update={"mode": "prepared_candidate", "candidate_dir": path,
            "candidate_manifest": tree_manifest(path), "command": candidate.command, "final_seed": seed,
            "image": self.store.get("live_images")["execution"],
            "grader_image": self.store.get("live_images")["grading"]})
        request = {"native": spec.model_dump(mode="json"), "source_sha256": source_hash,
                   "seed": seed, "grant_ref": self.final_grant_ref, "grant_digest": grant["grant_digest"]}
        prior = self.budget.reserve(operation, "experiment", request)
        if prior:
            if prior["result_ref"]:
                saved = self.store.json(prior["result_ref"])
                return {**saved, "result_ref": prior["result_ref"]}
            return {"seed": seed, "status": "interrupted", "elapsed_seconds": 0.0,
                    "reason": "Unknown prior final-seed completion; execution will not be repeated"}
        started = time.monotonic()
        child = WorkspaceRunner(spec, self.store.root / "final" / key, executor=self.executor)
        try:
            raw = child.run().model_dump(mode="json")
            result = {**raw, "seed": seed, "source_sha256": source_hash,
                      "child_store": "final/" + key, "raw_result_ref": child.store.get("workspace_result")}
            if result["status"] == "completed":
                try:
                    if type(result.get("score")) not in (int, float) or not math.isfinite(result["score"]) or not 0 <= result["score"] <= 1:
                        raise IntegrityError("Final accuracy must be finite and between zero and one")
                    report = child.store.json(result["official_report_ref"])
                    if report.get("seed") != seed or report.get("metrics", {}).get("accuracy") != result["score"]:
                        raise IntegrityError("Final seed score differs from its trusted evaluation report")
                    proof = validate_checkpoint_receipt(child.store, report["input_bundle"], expected,
                                                        seed=seed, phase="final")
                    result["training_validation_ref"] = child.store.put_json(proof, "study-final-training-proof")
                except (IntegrityError, ValueError, OSError, KeyError) as error:
                    failure = {"status": "fail", "reason": str(error), "raw_result": raw,
                               "promotion": "blocked", "raw_execution_unchanged": True}
                    result.update(status="invalid", score=None, metrics={},
                                  failure_proof_ref=child.store.put_json(failure, "study-final-training-failure"),
                                  reason=str(error))
            result["elapsed_seconds"] = time.monotonic() - started
        finally:
            child.close()
        self.budget.finish(operation, result["status"], result, result["elapsed_seconds"])
        return {**result, "result_ref": self.store.db.execute(
            "SELECT result_ref FROM live_operations WHERE key=?", (operation,)).fetchone()[0]}


def verify_final_aggregation(store, result):
    """Check complete aggregates against immutable seed operations without rerunning them."""
    if result.get("execution_type") != "aggregation" or result.get("seeds") != FINAL_SEEDS:
        raise IntegrityError("Invalid matched final aggregation contract")
    values = []
    source = store.json(result["source_tree_ref"])
    source_bundle = load_bundle(store, source["bundle_ref"])
    source_manifest = {name: {"sha256": item.sha256, "size": item.size}
                       for name, item in source_bundle.files.items()}
    source_hash = digest(canonical({"files": {name: {**source_manifest[name], "executable": item.executable}
                                                for name, item in source_bundle.files.items()},
                                    "command": source["command"]}))
    if source_hash != result.get("source_sha256"):
        raise IntegrityError("Final aggregate source identity differs from its frozen candidate")
    rows = result.get("seed_results", [])
    if not isinstance(rows, list) or len(rows) > 3:
        raise IntegrityError("Final aggregation contains an invalid number of seeds")
    if (result.get("seed_result_refs") != [row["result_ref"] for row in rows if row.get("result_ref")]
            or result.get("live_model_calls") != 0 or result.get("charged_seconds") != 0.0
            or result.get("elapsed_seconds") != sum(row.get("elapsed_seconds", 0.0) for row in rows)):
        raise IntegrityError("Final aggregation evidence references or resource accounting differ")
    for index, row in enumerate(rows):
        if row.get("seed") != FINAL_SEEDS[index]:
            raise IntegrityError("Final seed evidence order or identity differs")
        if row.get("result_ref"):
            if row.get("source_sha256") != source_hash:
                raise IntegrityError("Final aggregate mixes different candidate sources")
            saved = store.json(row["result_ref"])
            if canonical(saved) != canonical({key: value for key, value in row.items() if key != "result_ref"}):
                raise IntegrityError("Final aggregate differs from its immutable seed result")
            location = row.get("child_store", "")
            if not re.fullmatch(r"final/(baseline|selected)-seed-[012]", location):
                raise IntegrityError("Invalid child evidence location")
            ledger = store.db.execute("SELECT status,result_ref FROM live_operations WHERE key=?", (location,)).fetchone()
            if ledger is None or ledger["result_ref"] != row["result_ref"] or ledger["status"] != row["status"]:
                raise IntegrityError("Final seed result differs from its charged operation")
        if row.get("status") == "completed":
            if not row.get("result_ref") or not row.get("training_validation_ref"):
                raise IntegrityError("Completed final seed lacks immutable training/evaluation evidence")
            location = store.root / row["child_store"]
            if location.is_symlink() or not (location / "campaign.sqlite").is_file():
                raise IntegrityError("Final child evidence store is missing or unsafe")
            child = Store(location)
            try:
                raw = child.json(row["raw_result_ref"])
                spec = child.json(child.get("workspace_spec"))
                report = child.json(raw["official_report_ref"])
                proof = child.json(row["training_validation_ref"])
                if raw["status"] != "completed" or raw["score"] != row["score"] or report["seed"] != row["seed"] or proof["seed"] != row["seed"]:
                    raise IntegrityError("Final seed receipt does not support the aggregate")
                if (spec["final_seed"] != row["seed"] or spec["candidate_manifest"] != source_manifest
                        or spec["command"] != source["command"]):
                    raise IntegrityError("Final seed executed a different source, command or seed")
                load_bundle(child, report["input_bundle"])
                values.append(row["score"])
            finally:
                child.close()
    if result.get("status") == "completed":
        if (len(values) != 3 or result.get("per_seed_values") != values or result.get("score") != sum(values) / 3
                or result.get("metrics") != {"accuracy": sum(values) / 3} or result.get("partial_seed_values")):
            raise IntegrityError("Completed final result does not aggregate exactly three executed seed scores")
    elif (result.get("score") is not None or result.get("per_seed_values") or result.get("metrics")
          or result.get("partial_seed_values") != values):
        raise IntegrityError("Incomplete final evaluation cannot supply an aggregate score")
    return {"status": "pass", "completed_seeds": len(values), "live_model_calls": 0}
