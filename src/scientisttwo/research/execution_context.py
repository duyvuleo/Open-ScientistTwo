"""Read-only effective execution evidence, without executing source or reading labels.

Configured settings, worker-reported realization, and independent launch limits
are separate facts. Missing values remain unknown. A source hash is not proof of
causal isolation or an initialization mechanism.
"""

import ast
import json
import math

from ..artifacts import load_bundle
from ..store import IntegrityError, canonical, digest


_SETTINGS = {"epochs", "max_epochs", "max_steps", "batch_size", "learning_rate", "optimizer",
             "weight_decay", "momentum", "dropout", "hidden_size", "sequence_length",
             "inference_batch_size", "gradient_clip_norm", "seed"}
_MAX_TEXT = 500_000


class ConfigurationDomainError(IntegrityError):
    """Invalid configuration values after immutable-file verification succeeds."""

    def __init__(self, message, code, provenance):
        super().__init__(message)
        self.code = code
        self.provenance = provenance


def _small_file(store, manifest, path):
    item = manifest.files.get(path)
    if item is None:
        return None, None
    if item.size > _MAX_TEXT:
        raise IntegrityError("Execution descriptor exceeds bounded text size")
    data = store.read(item.ref)
    if len(data) != item.size or digest(data) != item.sha256:
        raise IntegrityError("Execution descriptor differs from its manifest")
    return data.decode("utf-8"), {"ref": item.ref, "sha256": item.sha256}


def _configuration(store, manifest, path):
    text, proof = _small_file(store, manifest, path)
    if text is None:
        return {}, None
    try:
        value = json.loads(text)
    except (ValueError, RecursionError) as error:
        raise ConfigurationDomainError(
            "Recorded training configuration is not valid bounded JSON", "invalid_json", proof
        ) from error
    if not isinstance(value, dict):
        raise ConfigurationDomainError(
            "Recorded training configuration must be an object", "invalid_shape", proof
        )
    values = {key: value[key] for key in sorted(_SETTINGS & value.keys())}
    if any((key == "optimizer" and (not isinstance(item, str) or item not in {"sgd", "adam", "adamw"})) or
           (key != "optimizer" and (type(item) not in {int, float}))
           for key, item in values.items()):
        raise ConfigurationDomainError(
            "Recorded training settings contain undeclared value types", "invalid_value_type", proof
        )
    if any(type(item) is float and not math.isfinite(item) for item in values.values()):
        raise ConfigurationDomainError(
            "Recorded training settings must be finite", "nonfinite_value", proof
        )
    canonical(values)
    return values, proof


def _literal(node, task):
    """Interpret only literal task choices; no eval/import or candidate execution."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
        return node.value
    if isinstance(node, ast.Name) and node.id == "task":
        return task
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [_literal(item, task) for item in node.elts]
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        left, right = _literal(node.left, task), _literal(node.comparators[0], task)
        if isinstance(node.ops[0], ast.Eq):
            return left == right
        if isinstance(node.ops[0], ast.In):
            return left in right
    if isinstance(node, ast.IfExp):
        return _literal(node.body if _literal(node.test, task) else node.orelse, task)
    raise ValueError("Default expression is not a supported literal task choice")


def _worker_defaults(source, task):
    """Only unambiguous config.get literals inside the frozen train function."""
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"]
    if len(functions) != 1 or task is None:
        return {}
    values = {}
    for node in ast.walk(functions[0]):
        if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute)
                or not isinstance(node.func.value, ast.Name) or node.func.value.id != "config"
                or node.func.attr != "get" or len(node.args) != 2 or node.keywords
                or not isinstance(node.args[0], ast.Constant) or node.args[0].value not in _SETTINGS):
            continue
        name = node.args[0].value
        try:
            value = _literal(node.args[1], task)
        except ValueError:
            continue
        values.setdefault(name, []).append(value)
    # Different defaults in different paths are intentionally not guessed.
    return {key: values[0] for key, values in values.items()
            if all(value == values[0] for value in values)}


def _wrapper_task(source):
    choices = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.List, ast.Tuple)):
            for left, right in zip(node.elts, node.elts[1:]):
                if (isinstance(left, ast.Constant) and left.value == "--task"
                        and isinstance(right, ast.Constant) and isinstance(right.value, str)):
                    choices.add(right.value)
    return next(iter(choices)) if len(choices) == 1 else None


def _stored_observation(store, offered):
    if isinstance(offered, str):
        raw, ref = store.json(offered), offered
    else:
        raw = offered.model_dump(mode="json") if hasattr(offered, "model_dump") else offered
        if raw.get("phase") == "final":
            return None, None
        row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("development/" + raw["id"],)).fetchone()
        if row is None:
            raise IntegrityError("Execution context requires a committed observation")
        ref = row[0]
        if store.json(ref) != raw:
            raise IntegrityError("Offered observation differs from its committed execution")
    if raw.get("phase") == "final":
        return None, None
    row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("development/" + raw["id"],)).fetchone()
    if row is None or row[0] != ref:
        raise IntegrityError("Execution context references an uncommitted observation")
    return raw, ref


def build_execution_context(store, observations):
    """Return prompt-safe descriptions for actual development observations only."""
    if len(observations) > 256:
        raise ValueError("Execution context exceeds its record bound")
    output = []
    for offered in observations:
        observation, ref = _stored_observation(store, offered)
        if observation is None:
            continue
        item = {"id": observation["id"], "observation_ref": ref, "status": observation["status"],
                "phase": observation["phase"], "source": {}, "executions": [],
                "configuration_status": "unverified", "unknown_settings": sorted(_SETTINGS - {"seed"})}
        if not observation.get("protocol_ref"):
            output.append(item)
            continue
        protocol = store.json(observation["protocol_ref"])
        if digest(canonical(protocol)) != observation["protocol_hash"] or protocol.get("phase") != observation["phase"]:
            raise IntegrityError("Execution context protocol identity mismatch")
        seeds = protocol.get("seeds", [])
        if (not isinstance(seeds, list) or len(seeds) != len(set(seeds))
                or any(type(seed) is not int or seed < 0 for seed in seeds)):
            raise IntegrityError("Execution context protocol seeds are invalid")
        item["protocol_hash"] = observation["protocol_hash"]
        tree_ref = observation.get("source_tree_ref")
        defaults, overrides, proofs, inferred, task = {}, {}, {}, {}, None
        invalid_candidate_configuration = None
        if tree_ref:
            row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("materialization/development/" + observation["id"],)).fetchone()
            if row is None:
                item["source"]["status"] = "unverified_materialization"
            else:
                materialization = store.json(row[0])
                if materialization.get("source_tree_ref") != tree_ref or materialization.get("code_ref") != observation["code_ref"]:
                    raise IntegrityError("Execution context source differs from materialization")
                tree = store.json(tree_ref)
                manifest = load_bundle(store, tree["bundle_ref"], verify=False)
                defaults, proofs["base_configuration"] = _configuration(store, manifest, "training.json")
                try:
                    overrides, proofs["candidate_overrides"] = _configuration(store, manifest, "training_config.json")
                except ConfigurationDomainError as error:
                    # Only candidate-owned, already verified source-domain errors
                    # from an unsuccessful execution become diagnostic context.
                    # Corrupt artifacts and trusted base/worker records are not
                    # caught here. Accepted outputs cannot coexist with a source
                    # configuration the frozen loader could not interpret.
                    if observation["status"] == "success" or any(observation.get(name) for name in (
                            "bundle_refs", "evaluation_refs", "prediction_refs", "artifact_refs", "timing_refs")):
                        raise
                    proofs["candidate_overrides"] = error.provenance
                    invalid_candidate_configuration = {
                        "field": "candidate_overrides", "status": "invalid",
                        "code": error.code, "message": str(error),
                        "artifact_ref": error.provenance["ref"],
                        "interpretation": "Failed candidate configuration; effective settings unknown. Original execution failure remains authoritative.",
                    }
                base_files = {}
                if tree.get("base_snapshot_ref"):
                    base = store.json(tree["base_snapshot_ref"])
                    base_files = load_bundle(store, base["bundle_ref"], verify=False).files
                worker = manifest.files.get("family_worker.py")
                wrapper = manifest.files.get("train.py")
                # An editable or changed worker cannot supply trusted defaults.
                if (worker and worker == base_files.get("family_worker.py")
                        and wrapper and wrapper == base_files.get("train.py")):
                    worker_source, proofs["worker_defaults"] = _small_file(store, manifest, "family_worker.py")
                    wrapper_source, proofs["task_binding"] = _small_file(store, manifest, "train.py")
                    task = _wrapper_task(wrapper_source)
                    inferred = _worker_defaults(worker_source, task)
                item["source"] = {"status": "recorded", "source_tree_ref": tree_ref,
                    "bundle_ref": tree["bundle_ref"], "method_isolation": "unverified",
                    "component_hashes": {path: value.sha256 for path, value in manifest.files.items()
                                         if path in {"model.py", "transforms.py", "training_contract.py", "family_worker.py"}}}
        item["base_configuration"] = defaults
        item["candidate_overrides"] = None if invalid_candidate_configuration else overrides
        if invalid_candidate_configuration:
            item["configuration_errors"] = [invalid_candidate_configuration]
        item["configuration_provenance"] = {key: proof for key, proof in proofs.items() if proof}
        for index, seed in enumerate(seeds):
            configured = ({"seed": seed} if invalid_candidate_configuration
                          else {**inferred, **defaults, **overrides, "seed": seed})
            execution = {"seed": seed, "status": "unverified", "configured": configured,
                         "resolved_literal_defaults": inferred,
                         "realized": {}, "unknown_realization": ["epochs_completed", "steps", "initialization_mechanism", "augmentation_effect"]}
            refs = observation.get("log_refs", [])
            if index < len(refs):
                receipt = store.json(refs[index])
                execution["receipt_ref"] = refs[index]
                if receipt.get("job_ref") and receipt.get("stage"):
                    job = store.json(receipt["job_ref"])
                    expected = "dev-" + digest((observation["id"] + str(seed)).encode())[:24]
                    if job.get("seed") != seed or receipt["stage"] != expected:
                        raise IntegrityError("Execution descriptor receipt seed or stage mismatch")
                    execution.update(status="recorded", resource_caps=receipt.get("resources", {}),
                                     image_id=receipt.get("image_id"), process_seconds=receipt.get("seconds"),
                                     returncode=receipt.get("returncode"), timed_out=receipt.get("timed_out"))
            bundle_refs = observation.get("bundle_refs", [])
            if index < len(bundle_refs):
                bundle = load_bundle(store, bundle_refs[index], verify=False)
                config, config_proof = _configuration(store, bundle, "checkpoint/config.json")
                if config_proof:
                    if config.get("seed") != seed:
                        raise IntegrityError("Checkpoint configuration seed differs from execution")
                    if any(config[key] != expected for key, expected in execution["configured"].items() if key in config):
                        raise IntegrityError("Checkpoint configuration contradicts frozen defaults or candidate overrides")
                    execution["worker_reported_configuration"] = config
                    execution["worker_configuration_ref"] = config_proof["ref"]
                    # Missing receipt values remain missing, never filled from expectations.
                    text, proof = _small_file(store, bundle, "checkpoint/training.json")
                    if text is not None:
                        training = json.loads(text)
                        if not isinstance(training, dict) or training.get("seed") != seed:
                            raise IntegrityError("Training metadata seed differs from execution")
                        if task is not None and training.get("task") != task:
                            raise IntegrityError("Training metadata task differs from frozen task wrapper")
                        for name in ("steps", "epochs_started", "training_seconds"):
                            if name in training and (type(training[name]) not in {int, float} or training[name] < 0):
                                raise IntegrityError("Training realization metadata must be finite nonnegative numbers")
                        numeric = {key: training[key] for key in ("steps", "epochs_started", "training_seconds") if key in training}
                        canonical(numeric)
                        execution["realized"] = numeric
                        if training.get("initialization") in {"scratch", "declared_reference"}:
                            execution["realized"]["initialization"] = training["initialization"]
                        execution["realization_ref"] = proof["ref"]
                        execution["realization_authority"] = "worker-reported; not independent mechanism verification"
                        if "steps" in training:
                            execution["unknown_realization"].remove("steps")
            execution["unknown_settings"] = sorted(_SETTINGS - execution["configured"].keys())
            execution["augmentation_source_hash"] = item["source"].get("component_hashes", {}).get("transforms.py")
            execution["initialization_source_hash"] = item["source"].get("component_hashes", {}).get("model.py")
            configured = execution["configured"]
            execution["optimizer_parameters"] = {key: configured[key] for key in
                ("learning_rate", "weight_decay", *( ["momentum"] if configured.get("optimizer") == "sgd" else []))
                if key in configured}
            item["executions"].append(execution)
        if item["executions"]:
            item["unknown_settings"] = sorted(set().union(*(set(value["unknown_settings"]) for value in item["executions"])))
            item["configuration_status"] = ("invalid_candidate_overrides" if invalid_candidate_configuration
                                            else "recorded_partial" if item["unknown_settings"] else "recorded")
        output.append(item)
    return {"schema_version": 2, "observations": output, "interpretation_rules": [
        "Invalid overrides from an unsuccessful candidate remain explicit diagnostics; base defaults are not substituted as accepted effective settings",
        "Equal permitted resource caps do not imply equal realized training settings",
        "Configured epochs and worker-reported epochs_started do not establish epochs_completed",
        "Default values are resolved only from unambiguous literal expressions in unchanged frozen worker source",
        "Initialization and augmentation source hashes identify code; they do not certify mechanism equivalence",
        "Absent configuration or metadata remains unknown; no labels, raw logs, model binaries or host paths are forwarded",
    ]}
