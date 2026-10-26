"""Bounded, non-final method/protocol context for the upstream writer.

Only declared task fields, partition counts and the existing safe development
presentation are exposed. Artifact pointers are never recursively dereferenced.
Dataset arrays, split indices, labels, checkpoints and raw logs are excluded.
"""
import ast
import json
import math
from pathlib import Path
import re

from ...live.evidence_context import build_evidence_context
from ...live.models import LiveSpec, Observation
from ...models import TaskSpec, EvalSpec
from ...splits import validate_partitions
from ...store import IntegrityError, canonical, digest


CONTEXT_VERSION = "paperorchestra-method-protocol-context-v1"
_TASK_FIELDS = (
    "schema_version", "name", "question", "adapter", "datasets", "subset_datasets",
    "seeds", "data_seed", "split_seed", "subset_fraction", "primary_metric",
    "metric_directions", "guardrails", "improvement_epsilon", "quality_tolerance",
)
_NONFINAL = {"subset", "full", "rebuttal"}
_CONSTRUCTORS = {"LogisticRegression", "SVC", "NuSVC", "LinearSVC", "SVR", "LinearSVR",
    "Ridge", "Lasso", "ElasticNet", "KNeighborsClassifier", "KNeighborsRegressor",
    "RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor", "GradientBoostingClassifier", "GradientBoostingRegressor",
    "HistGradientBoostingClassifier", "HistGradientBoostingRegressor", "MLPClassifier", "MLPRegressor",
    "PolynomialFeatures", "StandardScaler"}
_PARAMETERS = {"C", "gamma", "degree", "n_neighbors", "n_estimators", "max_depth", "alpha",
    "tol", "max_iter", "learning_rate", "min_samples_split", "min_samples_leaf", "max_features",
    "l1_ratio", "n_jobs", "random_state", "coef0", "epsilon", "leaf_size"}


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _python_literals(text):
    """Direct constructor keyword literals only; no execution or name inference."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and (node.module == "sklearn" or node.module.startswith("sklearn.")):
            imports.update({item.asname or item.name: node.module + "." + item.name for item in node.names})
        elif isinstance(node, ast.Import):
            imports.update({item.asname or item.name.split(".")[0]: item.name if item.asname else item.name.split(".")[0]
                            for item in node.names if item.name == "sklearn" or item.name.startswith("sklearn.")})
    # Conservative: any static rebinding/shadowing makes the imported root
    # unavailable, including nested scopes. We do not infer assignment values.
    imported_counts = {}
    shadowed = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for item in node.names:
                key = item.asname or (item.name.split(".")[0] if isinstance(node, ast.Import) else item.name)
                imported_counts[key] = imported_counts.get(key, 0) + 1
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            shadowed.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            shadowed.add(node.name)
        elif isinstance(node, ast.arg):
            shadowed.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            shadowed.add(node.name)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                shadowed.add(root.id)
    imports = {key: value for key, value in imports.items()
               if key not in shadowed and imported_counts.get(key) == 1}

    def name(node):
        if isinstance(node, ast.Name):
            return imports.get(node.id, "")
        if isinstance(node, ast.Attribute):
            parent = name(node.value)
            return parent + "." + node.attr if parent else ""
        return ""

    values = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        target = name(call.func)
        if not target.startswith("sklearn.") or target.rsplit(".", 1)[-1] not in _CONSTRUCTORS:
            continue
        for keyword in call.keywords:
            if keyword.arg not in _PARAMETERS:
                continue
            node, sign = keyword.value, 1
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                sign, node = (-1 if isinstance(node.op, ast.USub) else 1), node.operand
            if not isinstance(node, ast.Constant) or type(node.value) not in (int, float):
                continue
            value = sign * node.value
            if not _finite_number(value):
                continue
            literal = keyword.value
            values.append({"call_target": target, "keyword": keyword.arg, "value": value,
                "source_path": ["python_keyword_literal", target, keyword.arg, literal.lineno,
                                literal.col_offset, literal.end_lineno, literal.end_col_offset],
                "literal_text": ast.get_source_segment(text, literal)})
    return sorted(values, key=lambda item: (item["source_path"][3], item["source_path"][4]))


def _source_parameters(safe):
    values = []
    for source in safe["sources"]:
        for file in source["files"]:
            if not file.get("text_complete") or not file["path"].endswith(".py"):
                continue
            for literal in _python_literals(file["text"]):
                values.append({**literal, "source_ref": file["ref"], "source_sha256": file["sha256"],
                    "source_tree_ref": source["source_tree_ref"], "file_path": file["path"],
                    "scope": "declared_source_parameter_not_runtime_or_mechanism_verification"})
    if len(values) > 2048:
        raise IntegrityError("Too many declared source parameter literals")
    return values


def _dataset_metadata(task):
    declared = task.settings.get("dataset_metadata", {})
    if not isinstance(declared, dict) or set(declared) - set(task.datasets):
        raise IntegrityError("Dataset metadata must name only declared task datasets")
    result = {}
    for dataset, record in sorted(declared.items()):
        if not isinstance(record, dict):
            raise IntegrityError("Dataset metadata must be a typed object")
        selected = {}
        for name in ("n_samples", "n_features"):
            if name in record:
                if type(record[name]) is not int or not 1 <= record[name] <= 1_000_000_000:
                    raise IntegrityError("Dataset size metadata must be positive bounded integers")
                selected[name] = record[name]
        if "noise_standard_deviation" in record:
            value = record["noise_standard_deviation"]
            if not _finite_number(value) or not 0 <= value <= 1_000_000:
                raise IntegrityError("Dataset noise metadata must be finite and nonnegative")
            selected["noise_standard_deviation"] = value
        if "generator" in record:
            if not isinstance(record["generator"], str) or not re.fullmatch(r"sklearn\.datasets\.[A-Za-z_][A-Za-z0-9_]{0,95}", record["generator"]):
                raise IntegrityError("Dataset generator declaration must be a public sklearn dataset function name")
            selected["generator"] = record["generator"]
        result[dataset] = selected
    return result


def _native_task(store, protocols):
    """Typed declaration only: no native argv, mounts, raw settings or data."""
    ref = store.get("live_spec_ref")
    if not ref:
        return {"status": "unavailable"}
    config = LiveSpec.model_validate(store.json(ref))
    native = config.native.model_dump(mode="json")
    value = {"status": "recorded_native_task_declaration", "ref": ref,
        "fields": {name: native[name] for name in
            ("name", "benchmark", "task_id", "data_provenance", "final_seed")},
        "development": {"adapter": config.development.adapter, "metric": config.development.metric,
            "direction": config.development.direction, "seeds": config.development.seeds},
        "seed_semantics": "Protocol seeds and native.final_seed identify timing/training repetitions; they are not workload-generation seeds",
        "workload_generation": {"status": "unavailable"}}
    if "revision" in native:
        value["fields"]["revision"] = native["revision"]
    # The adapter identity already freezes this manifest. Seal the declaration
    # once, and subsequently use that artifact so historical verification never
    # requires mutable external asset paths or reopens raw datasets.
    hashes = {store.json(item["ref"]).get("adapter", {}).get("prepared_manifest")
              for item in protocols.values()}
    hashes.discard(None)
    if hashes and config.development.adapter == "testbed":
        if len(hashes) != 1:
            raise IntegrityError("Native writer protocols refer to different prepared manifests")
        expected = hashes.pop()
        manifest_ref = next((row["path"] for row in store.db.execute(
            "SELECT path FROM artifacts WHERE kind='writer_prepared_manifest'")
            if digest(canonical(store.json(row["path"]))) == expected), None)
        if manifest_ref is None:
            path = Path(config.development.settings["prepared_root"]) / "manifest.json"
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 4_000_000:
                raise IntegrityError("Prepared writer manifest is missing or oversized")
            manifest = json.loads(path.read_bytes())
            if digest(canonical(manifest)) != expected:
                raise IntegrityError("Prepared writer manifest differs from the frozen adapter")
            manifest_ref = store.put_json(manifest, "writer_prepared_manifest")
        manifest = store.json(manifest_ref)
        if manifest.get("task_id") != config.native.task_id:
            raise IntegrityError("Prepared writer task differs from the native declaration")
        metadata = manifest.get("split_metadata", {})
        if config.native.task_id == "vectorization":
            for name in ("split_seed", "final_seed"):
                if type(metadata.get(name)) is not int or metadata[name] < 0:
                    raise IntegrityError("Prepared workload generation seeds must be nonnegative integers")
            value["workload_generation"] = {"status": "recorded_preparation_declaration",
                "ref": manifest_ref, "development_seed": metadata["split_seed"],
                "held_out_final_seed": metadata["final_seed"],
                "scope": "Fixed input-array generation seeds; not independent timing repetitions; no arrays or final measurements supplied"}
    return value


def _paired_protocol(store, observed, protocol):
    pair = protocol.get("paired_timing")
    if pair is None:
        return {}
    if pair.get("warmups") != 5 or pair.get("pairs") != 10:
        raise IntegrityError("Writer paired timing differs from the frozen 5/10 protocol")
    partition = "subset" if observed.phase == "subset" else "full"
    ref = store.get("prepared/" + partition)
    prepared = store.json(ref)
    if prepared.get("files") != protocol.get("data"):
        raise IntegrityError("Public workload metadata differs from the observation protocol")
    cases = prepared.get("cases", [])
    if not isinstance(cases, list) or len(cases) > 128:
        raise IntegrityError("Public timing workload metadata must be a bounded list")
    public = []
    for case in cases:
        if (not isinstance(case, dict) or type(case.get("stride")) is not int
                or not 1 <= case["stride"] <= 1024 or type(case.get("padding")) is not int
                or not 0 <= case["padding"] <= 1024 or case.get("activation") not in {None, "relu"}):
            raise IntegrityError("Public timing case metadata exceeds its typed projection")
        public.append({name: case[name] for name in ("stride", "padding", "activation")})
    return {"paired_timing": {"warmups": 5, "pairs": 10,
                "selection": pair["selection"],
                "measurement_scope": "Resident worker request/response including IPC and output serialization; excludes startup and independent correctness checks",
                "seeds_scope": "Timing repetitions; fixed workloads shared by both arms",
                "public_workloads": ({"ref": ref, "case_count": len(public), "cases": public,
                    "contains_arrays_or_labels": False} if "cases" in prepared else
                    {"status": "unavailable", "contains_arrays_or_labels": False})}}


def build_writer_context(store, ledger, *, max_source_chars=32000):
    """Resolve only the exact non-final observations in this frozen ledger."""
    if ledger.get("includes_final") is not False:
        raise IntegrityError("Writer context requires explicitly non-final evidence")
    records = ledger.get("observations")
    if not records:
        return {"schema_version": CONTEXT_VERSION, "status": "unavailable",
                "reason": "No frozen observation inventory supplied", "includes_final": False}
    if not isinstance(records, dict) or len(records) > 256:
        raise IntegrityError("Invalid or oversized writer observation inventory")
    task_ref = store.get("task_ref")
    task = TaskSpec.model_validate(store.json(task_ref)) if task_ref else None
    observations, protocols = [], {}
    for identifier, entry in sorted(records.items()):
        if not isinstance(entry, dict) or entry.get("phase") not in _NONFINAL:
            raise IntegrityError("Writer context cannot include final or untyped observations")
        observed = Observation.model_validate(store.json(entry["observation_ref"]))
        if (observed.id != identifier or observed.phase != entry["phase"]
                or observed.protocol_ref != entry["protocol_ref"]
                or observed.protocol_hash != entry["protocol_hash"]
                or observed.code_ref != entry["code_ref"]
                or observed.source_tree_ref != entry["source_tree_ref"]):
            raise IntegrityError("Writer context inventory differs from the frozen observation")
        protocol = store.json(observed.protocol_ref)
        if digest(canonical(protocol)) != observed.protocol_hash or protocol.get("phase") != observed.phase:
            raise IntegrityError("Writer protocol differs from its frozen identity")
        evaluation = protocol.get("evaluation", {})
        if not isinstance(evaluation, dict):
            raise IntegrityError("Writer evaluation metadata must be an object")
        if evaluation and "task" in evaluation:
            spec = EvalSpec.model_validate(evaluation)
            if (task is None or spec.task != task or spec.phase != observed.phase
                    or spec.split == "test"):
                raise IntegrityError("Writer evaluation protocol differs from the frozen task or is final")
        # Never copy the embedded TaskSpec.settings, native argv, mounts, input
        # paths, final outputs or an arbitrary nested evaluation dictionary.
        projection = {name: evaluation[name] for name in
                      ("phase", "split", "datasets", "seeds", "train_fraction") if name in evaluation}
        if ("phase" in projection and (not isinstance(projection["phase"], str) or projection["phase"] not in _NONFINAL)
                or "split" in projection and (not isinstance(projection["split"], str) or projection["split"] not in {"development", "train"})
                or "datasets" in projection and (not isinstance(projection["datasets"], list)
                    or len(projection["datasets"]) > 256 or any(not isinstance(name, str)
                        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name) for name in projection["datasets"]))
                or "seeds" in projection and (not isinstance(projection["seeds"], list)
                    or len(projection["seeds"]) > 256 or any(type(seed) is not int or seed < 0 for seed in projection["seeds"]))
                or "train_fraction" in projection and (not _finite_number(projection["train_fraction"])
                    or not 0 < projection["train_fraction"] <= 1)):
            raise IntegrityError("Writer evaluation metadata exceeds its typed projection")
        protocols[identifier] = {"ref": observed.protocol_ref, "hash": observed.protocol_hash,
            "phase": observed.phase, "seeds": protocol.get("seeds", []),
            "evaluation": projection, "status": "recorded_protocol_metadata",
            **_paired_protocol(store, observed, protocol)}
        observations.append(observed)
    safe = build_evidence_context(store, observations, max_source_chars=max_source_chars)
    task_info, counts = {"status": "unavailable"}, {"status": "unavailable"}
    if task_ref:
        raw = task.model_dump(mode="json")
        task_info = {"status": "recorded_task_declaration", "ref": task_ref,
                     "fields": {name: raw[name] for name in _TASK_FIELDS},
                     "dataset_metadata": _dataset_metadata(task),
                     "dataset_metadata_scope": "Task-declared public generator settings; not independently inferred effective settings",
                     "settings": "Unstructured settings are intentionally not exported"}
        split_ref = store.get("split_manifest_ref")
        if split_ref:
            split = store.json(split_ref)
            if (split.get("version") != 2 or split.get("adapter") != task.adapter
                    or split.get("data_seed") != task.data_seed
                    or split.get("split_seed") != task.split_seed
                    or set(split.get("datasets", {})) != set(task.datasets)):
                raise IntegrityError("Partition metadata differs from the frozen task")
            datasets = {}
            for name, record in sorted(split["datasets"].items()):
                validate_partitions(record)
                datasets[name] = {"total_rows": record["count"],
                    **{part + "_rows": len(record[part]) for part in ("train", "development", "test")}}
                declared_count = task_info["dataset_metadata"].get(name, {}).get("n_samples")
                if declared_count is not None and declared_count != record["count"]:
                    raise IntegrityError("Declared dataset sample count differs from the frozen partitions")
            counts = {"status": "recorded_partition_counts", "ref": split_ref,
                      "sha256": digest(canonical(split)), "stratified": split["stratified"],
                      "datasets": datasets, "contains_indices_or_labels": False}
    else:
        task_info = _native_task(store, protocols)
    value = {"schema_version": CONTEXT_VERSION, "status": "available", "includes_final": False,
             "task": task_info, "partition_counts": counts, "protocols": protocols,
             "execution_and_source": safe,
             "declared_source_parameters": _source_parameters(safe),
             "scope": "Declared protocol and frozen editable code, not new metrics or independently verified mechanism claims",
             "rules": [
                 "Use supplied task declarations, partition counts and source text instead of claiming they are unavailable.",
                 "Only text_complete source files are complete; excerpts cannot certify omitted behavior.",
                 "Report settings as recorded declarations/code; semantic method-code agreement and causal support still require review.",
                 "Partition counts and seeds are protocol metadata, not outcome measurements or independent statistical replications.",
                 "Source parameter metadata covers direct literals in allowlisted constructor keywords only; assignment-derived, positional or omitted defaults are not inferred. Never bind an unsupported constructor value to an equal-valued outcome metric.",
                 "No dataset arrays, row identifiers, labels, checkpoint payloads, raw logs, credentials or protected final results are supplied.",
             ]}
    value["numeric_metadata"] = numeric_metadata(value)
    if len(canonical(value)) > 400000:
        raise IntegrityError("Writer method/protocol context exceeds its bounded size")
    return value


def numeric_metadata(context):
    """Typed allowlist of nonempirical numbers; never visit metrics or source text."""
    values = []

    def add(path, value, ref, source_path, operation="identity"):
        if not _finite_number(value):
            raise IntegrityError("Method metadata numbers must be typed scalars")
        matched = re.fullmatch(r"artifacts/([a-f0-9]{64})\.[A-Za-z0-9_-]+", ref)
        if not matched:
            raise IntegrityError("Method metadata requires a content-addressed JSON source")
        values.append({"context_path": path, "value": value, "source_ref": ref,
                       "source_sha256": matched[1],
                       "source_path": source_path, "operation": operation,
                       "scope": "nonempirical_method_protocol_only"})

    task = context.get("task", {})
    if task.get("status") == "recorded_task_declaration":
        fields = task["fields"]
        for name in ("data_seed", "split_seed", "subset_fraction", "improvement_epsilon", "quality_tolerance"):
            add(["task", "fields", name], fields[name], task["ref"], [name])
        for index, seed in enumerate(fields["seeds"]):
            add(["task", "fields", "seeds", index], seed, task["ref"], ["seeds", index])
        for metric, threshold in sorted(fields["guardrails"].items()):
            add(["task", "fields", "guardrails", metric], threshold, task["ref"], ["guardrails", metric])
        for dataset, record in sorted(task.get("dataset_metadata", {}).items()):
            for name in ("n_samples", "n_features", "noise_standard_deviation"):
                if name in record:
                    add(["task", "dataset_metadata", dataset, name], record[name], task["ref"],
                        ["settings", "dataset_metadata", dataset, name])
    if task.get("status") == "recorded_native_task_declaration":
        add(["task", "fields", "final_seed"], task["fields"]["final_seed"], task["ref"], ["native", "final_seed"])
        for index, seed in enumerate(task["development"]["seeds"]):
            add(["task", "development", "seeds", index], seed, task["ref"], ["development", "seeds", index])
        generation = task["workload_generation"]
        if generation.get("status") == "recorded_preparation_declaration":
            for name, source in (("development_seed", "split_seed"), ("held_out_final_seed", "final_seed")):
                add(["task", "workload_generation", name], generation[name], generation["ref"], ["split_metadata", source])
    counts = context.get("partition_counts", {})
    if counts.get("status") == "recorded_partition_counts":
        for dataset, record in sorted(counts["datasets"].items()):
            add(["partition_counts", "datasets", dataset, "total_rows"], record["total_rows"],
                counts["ref"], ["datasets", dataset, "count"])
            for part in ("train", "development", "test"):
                add(["partition_counts", "datasets", dataset, part + "_rows"], record[part + "_rows"],
                    counts["ref"], ["datasets", dataset, part], "length")
    for identifier, protocol in sorted(context.get("protocols", {}).items()):
        evaluation = protocol["evaluation"]
        if "train_fraction" in evaluation:
            add(["protocols", identifier, "evaluation", "train_fraction"], evaluation["train_fraction"],
                protocol["ref"], ["evaluation", "train_fraction"])
        for index, seed in enumerate(protocol["seeds"]):
            add(["protocols", identifier, "seeds", index], seed, protocol["ref"], ["seeds", index])
        if "paired_timing" in protocol:
            pair = protocol["paired_timing"]
            for name in ("warmups", "pairs"):
                add(["protocols", identifier, "paired_timing", name], pair[name], protocol["ref"], ["paired_timing", name])
            public = pair["public_workloads"]
            if "ref" not in public:
                continue
            add(["protocols", identifier, "paired_timing", "public_workloads", "case_count"],
                public["case_count"], public["ref"], ["cases"], "length")
            for index, case in enumerate(public["cases"]):
                for name in ("stride", "padding"):
                    add(["protocols", identifier, "paired_timing", "public_workloads", "cases", index, name],
                        case[name], public["ref"], ["cases", index, name])
    for index, parameter in enumerate(context.get("declared_source_parameters", [])):
        add(["declared_source_parameters", index, "value"], parameter["value"], parameter["source_ref"],
            parameter["source_path"], "python_keyword_literal")
    if len(values) > 4096:
        raise IntegrityError("Method metadata exceeds its bounded field count")
    return values


def context_method_fact(store, ledger):
    context = build_writer_context(store, ledger)
    ref = store.put_json(context, "paperorchestra-method-protocol-context")
    return {"text": "Declared task, partition counts, evaluation settings and frozen editable code. These metadata values are not experimental outcome measurements; semantic method-code agreement remains unverified.",
            "evidence_ref": ref, "metadata_contract": CONTEXT_VERSION,
            "context_sha256": digest(canonical(context)),
            "numeric_metadata": context.get("numeric_metadata", [])}


def verify_context_method_fact(store, fact, observations):
    """Recheck numeric metadata against exact task/protocol artifacts, not prose."""
    if fact.get("metadata_contract") != CONTEXT_VERSION:
        raise IntegrityError("Unknown writer method-metadata contract")
    context = store.json(fact["evidence_ref"])
    if (context.get("schema_version") != CONTEXT_VERSION or context.get("includes_final") is not False
            or fact.get("context_sha256") != digest(canonical(context))):
        raise IntegrityError("Writer method context identity differs")
    expected = build_writer_context(store, {"includes_final": False, "observations": observations},
        max_source_chars=context.get("execution_and_source", {}).get("source_excerpt_limit", 32000))
    # Later experimental history cannot enlarge the original source/protocol
    # scope. Numeric fields use only these stable allowlisted projections.
    if any(context.get(name) != expected.get(name) for name in ("task", "partition_counts", "protocols", "declared_source_parameters")):
        raise IntegrityError("Writer method metadata differs from the frozen task or observation protocols")
    fields = numeric_metadata(context)
    if context.get("numeric_metadata", []) != fields or fact.get("numeric_metadata") != fields:
        raise IntegrityError("Writer numeric metadata differs from its typed context")
    for field in fields:
        if digest(store.read(field["source_ref"])) != field["source_sha256"]:
            raise IntegrityError("Writer metadata source hash differs")
        if field["operation"] == "python_keyword_literal":
            matches = [item for item in _python_literals(store.read(field["source_ref"]).decode())
                       if item["source_path"] == field["source_path"]]
            if len(matches) != 1 or matches[0]["value"] != field["value"]:
                raise IntegrityError("Writer constructor literal differs from its frozen source span")
            continue
        raw = store.json(field["source_ref"])
        for key in field["source_path"]:
            raw = raw[key]
        actual = len(raw) if field["operation"] == "length" else raw
        if type(actual) not in (int, float) or actual != field["value"]:
            raise IntegrityError("Writer metadata value differs from its exact source path")


def writing_scope_instruction(scope):
    if scope == "upstream":
        return None
    if scope != "concise_note":
        raise ValueError("Unknown PaperOrchestra writing scope")
    return (
        "Controller-approved writing scope: a concise, COMPLETE research note demonstrating this controlled workflow. "
        "Aim for roughly 1000-1500 prose words; this is a soft target, not a character or token cap. "
        "You may consolidate the generated outline, remove redundant sections and omit proposed but unexecuted work. "
        "Preserve the actual method, protocol, measured comparisons, citations, uncertainty and material limitations. "
        "Use the supplied method/protocol context and code; distinguish declarations from executed outcomes. "
        "Put numeric configuration/protocol details in dedicated Methodology, Evaluation Protocol or Reproducibility sections, separate from empirical result tables. "
        "Use anonymous authors, replace template placeholders, and return all required closing syntax and references. "
        "Keep empirical values accurate; full artifact hashes may remain in the supplied provenance rather than fill the prose. "
        "No novelty, broad benchmark readiness, independent-replication or completed-final-evaluation claim is authorized. "
        "Review this note against its stated controlled-demonstration scope, without pretending it is a novel conference contribution."
    )
