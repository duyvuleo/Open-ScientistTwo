"""Bounded, read-only explanations of committed development evidence for agents.

This is presentation, not an audit or a method-equivalence verifier. It never
reads prediction/checkpoint payloads, evaluator code, labels, or execution logs.
Only frozen editable source text is eligible for bounded excerpts. Artifact
manifests and receipts are integrity checked; payload verification stays in audit.
"""

from pathlib import PurePosixPath
import re

from ..artifacts import load_bundle
from ..store import IntegrityError, canonical, digest
from .context import _allowances
from .models import Candidate, Observation
from .source_tree import MaterializedSourceTree, SourceSnapshot

_NONFINAL = {"subset", "full", "rebuttal"}
_MAX_RECORDS = 256
_MAX_SOURCE_BYTES = 500_000
_TEXT_SUFFIXES = {"", ".py", ".r", ".jl", ".js", ".ts", ".sh", ".c", ".cpp",
                  ".h", ".hpp", ".toml", ".yaml", ".yml", ".json", ".txt", ".md"}
_HOST_PATH = re.compile(r"(?<![\w:])/(?:[^\s\"'<>:,;()\[\]{}]+)")


def _plain(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _safe_text(value, limit=1000):
    """Prose is not permission to disclose paths from failed host-side errors."""
    return _HOST_PATH.sub("[path omitted]", str(value))[:limit]


def _cached(store, key):
    row = store.db.execute("SELECT output_ref FROM calls WHERE key=?", (key,)).fetchone()
    return (store.json(row[0]), row[0]) if row else (None, None)


def _offered(values):
    """Only explicit evidence containers, never strings or arbitrary candidate data."""
    output = []

    def visit(value, depth=0):
        if depth > 16:
            raise ValueError("Evidence context nesting exceeds its bound")
        value = _plain(value)
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item, depth + 1)
        elif isinstance(value, dict):
            if value.get("phase") in _NONFINAL | {"final"} and "id" in value:
                if value["phase"] in _NONFINAL:
                    output.append(value)
                return
            for name in ("result", "observation", "observations", "matched", "control"):
                if name in value:
                    visit(value[name], depth + 1)

    visit(values)
    if len(output) > _MAX_RECORDS:
        raise ValueError("Evidence context record count exceeds its bound")
    return output


def _committed(store, offered):
    stored, ref = _cached(store, "development/" + offered["id"])
    if stored is None:
        return None, None
    if canonical(stored) != canonical(offered) or stored.get("phase") not in _NONFINAL:
        raise IntegrityError("Offered development evidence differs from its committed observation")
    return Observation.model_validate(stored), ref


def _protocol(store, observation):
    if not observation.protocol_ref:
        return {"status": "unverified", "reason": "No frozen protocol record", "seeds": []}
    record = store.json(observation.protocol_ref)
    if digest(canonical(record)) != observation.protocol_hash or record.get("phase") != observation.phase:
        raise IntegrityError("Observation protocol hash or phase differs from its frozen record")
    seeds = record.get("seeds")
    if (not isinstance(seeds, list) or len(seeds) > _MAX_RECORDS
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or observation.status == "success" and len(seeds) != len(observation.values)):
        raise IntegrityError("Observation seed/metric cardinality differs from its protocol")
    # Hash complete components rather than forwarding their protected host paths,
    # evaluator source closures, hardware identifiers, or input/output manifests.
    components = {name: digest(canonical(record[name])) for name in (
        "adapter", "data", "evaluation", "runtime", "execution", "artifacts", "pipeline",
        "paired_timing"
    ) if name in record}
    return {"status": "recorded", "ref": observation.protocol_ref,
            "hash": observation.protocol_hash, "phase": observation.phase,
            "seeds": seeds, "component_hashes": components}


def _bundle(store, ref):
    manifest = load_bundle(store, ref, verify=False)
    roles = {}
    for spec in manifest.specs:
        members = [value for path, value in manifest.files.items()
                   if path == spec.path or spec.kind == "directory" and path.startswith(spec.path + "/")]
        role = roles.setdefault(spec.role, {"files": 0, "bytes": 0})
        role["files"] += len(members)
        role["bytes"] += sum(value.size for value in members)
    return {"ref": ref, "roles": roles, "file_count": len(manifest.files),
            "payload_integrity": "not_rechecked_in_prompt_builder"}


def _execution(store, observation, index, seed):
    if observation.log_refs:
        first = store.json(observation.log_refs[0])
        if first.get("kind") == "pilot_batch":
            if len(observation.log_refs) != 1:
                raise IntegrityError("Pilot evidence must contain exactly one shared batch receipt")
            return _pilot_execution(store, observation, first, seed)
    if index >= len(observation.log_refs):
        return {"seed": seed, "status": "unverified", "reason": "No training receipt"}
    ref = observation.log_refs[index]
    receipt = store.json(ref)
    result = {"seed": seed, "receipt_ref": ref, "status": "unverified",
              "reason": "Receipt omits independent execution binding"}
    if observation.timing_refs:
        # Paired latency has two persistent workers, not a training launch. Keep
        # that separate, without forwarding their argv/mounts or raw logs.
        result.update(kind="paired_timing", worker_receipt_refs={
            arm: receipt.get("receipts", {}).get(arm) for arm in ("baseline", "candidate")
        })
        return result
    if not receipt.get("job_ref") or not receipt.get("stage") or seed is None:
        return result
    job = store.json(receipt["job_ref"])
    expected_stage = "dev-" + digest((observation.id + str(seed)).encode())[:24]
    if job.get("seed") != seed or receipt["stage"] != expected_stage:
        raise IntegrityError("Training receipt belongs to a different observation or seed")
    if type(receipt.get("returncode")) is not int or type(receipt.get("timed_out")) is not bool:
        return result
    successful = not receipt["returncode"] and not receipt["timed_out"] and not receipt.get("limit_exceeded")
    if observation.status == "success" and not successful:
        raise IntegrityError("Successful observation references failed training execution")
    result.update(kind="training", execution_id=expected_stage,
                  status="recorded_success" if successful else "recorded_failure",
                  returncode=receipt["returncode"], timed_out=receipt["timed_out"],
                  seconds=receipt.get("seconds"),
                  reason="Distinct bound execution receipts identify separate launches; payload hashes do not")
    return result


def _pilot_execution(store, observation, receipt, seed):
    """Describe a bound worker batch without inventing per-seed containers.

    This checks the same persisted completion/result relationships as pilot
    audit. Prediction/label arrays and raw logs are never read or forwarded.
    Metrics are committed trusted-scorer results; recomputation stays in audit.
    """
    from ..models import EvalSpec, ExperimentResult

    protocol = store.json(observation.protocol_ref)
    try:
        original = ExperimentResult.model_validate(store.json(observation.legacy_result_ref))
        spec = EvalSpec.model_validate(store.json(original.spec_ref))
    except (ValueError, TypeError) as error:
        raise IntegrityError("Invalid authoritative pilot result or evaluation specification") from error
    operation_key = "experiment/" + observation.id
    legacy_key = "pilot/" + operation_key
    if (receipt.get("operation_key") != operation_key or receipt.get("legacy_job_key") != legacy_key
            or receipt.get("legacy_result_ref") != observation.legacy_result_ref
            or original.id != legacy_key or receipt.get("spec_ref") != original.spec_ref
            or receipt.get("source_tree_ref") != observation.source_tree_ref
            or receipt.get("status") != original.status or receipt.get("seconds") != original.duration_seconds
            or original.duration_seconds != observation.seconds
            or original.logs_ref is not None and receipt.get("log_ref") != original.logs_ref
            or receipt.get("seeds") != spec.seeds or receipt.get("datasets") != spec.datasets
            or seed not in spec.seeds or protocol.get("seeds") != spec.seeds
            or protocol.get("evaluation") != spec.model_dump(mode="json")
            or spec.phase != observation.phase or original.protocol_hash != digest(canonical(spec))
            or original.split_manifest_ref != protocol.get("data", {}).get("split_manifest_ref")
            or spec.split_manifest_hash != protocol.get("data", {}).get("hash")
            or observation.metric_rows != [row.model_dump(mode="json") for row in original.rows]):
        raise IntegrityError("Pilot batch/source/protocol/grid binding differs from its committed result")
    operation = store.db.execute("SELECT * FROM live_operations WHERE key=?", (operation_key,)).fetchone()
    candidate = Candidate.model_validate(store.json(observation.code_ref))
    request_hash = digest(canonical({"candidate": candidate.model_dump(mode="json"), "protocol": observation.protocol_hash}))
    if (operation is None or operation["kind"] != "experiment" or operation["request_hash"] != request_hash
            or operation["status"] != observation.status or operation["seconds"] != observation.seconds
            or not operation["result_ref"] or canonical(store.json(operation["result_ref"])) != canonical(observation)
            or len(candidate.files) != 1 or candidate.files[0].path != "candidate.py"
            or candidate.files[0].content.encode() != store.read(original.code_ref)):
        raise IntegrityError("Pilot batch is not bound to its actual source and shared completion")
    job = store.db.execute("SELECT * FROM jobs WHERE key=?", (legacy_key,)).fetchone()
    if job is None:
        if original.status == "interrupted":
            return {"seed": seed, "kind": "pilot_batch", "status": "unverified",
                    "reason": "No committed worker job exists; no batch execution is claimed"}
        raise IntegrityError("Pilot batch worker completion is missing")
    if (job["result_ref"] != observation.legacy_result_ref or job["status"] != original.status
            or job["charged"] != original.duration_seconds
            or not re.fullmatch(r"[a-f0-9]{64}", job["request_hash"])):
        raise IntegrityError("Pilot worker job and authoritative result completion differ")
    if completion_ref := receipt.get("legacy_completion_ref"):
        completion = store.json(completion_ref)
        worker_hash = protocol.get("execution", {}).get("worker_source_sha256") or protocol.get("adapter", {}).get("legacy", {}).get("worker.py")
        if (completion.get("schema_version") != "pilot-job-completion-v1" or completion.get("job") != dict(job)
                or completion.get("observations_ref") != original.observations_ref
                or completion.get("measurement_origin") != original.measurement_origin
                or completion.get("worker_image_id") != protocol.get("runtime", {}).get("images", {}).get("execution")
                or not re.fullmatch(r"[a-f0-9]{64}", str(completion.get("worker_source_sha256")))
                or worker_hash is not None and completion.get("worker_source_sha256") != worker_hash):
            raise IntegrityError("Frozen pilot completion proof differs from its worker/result binding")
    expected_cells = {(dataset, item_seed) for dataset in spec.datasets for item_seed in spec.seeds}
    cells = {(row.dataset, row.seed) for row in original.rows}
    if observation.status == "success":
        if (original.status != "success" or cells != expected_cells or len(original.rows) != len(expected_cells)
                or len(observation.prediction_refs) != len(spec.seeds)):
            raise IntegrityError("Successful pilot batch lacks the complete declared dataset/seed grid")
        from ..research.pilot_bridge import _aggregate
        metrics, values = _aggregate(original, spec.task, spec.seeds, spec.datasets)
        if observation.metrics != metrics or observation.metric_values != values:
            raise IntegrityError("Pilot aggregate differs from its authoritative per-cell metric grid")
        for index, item_seed in enumerate(spec.seeds):
            selected = store.json(observation.prediction_refs[index])
            if selected != {"kind": "pilot_prediction_selection", "observations_ref": original.observations_ref,
                            "seed": item_seed, "datasets": spec.datasets}:
                raise IntegrityError("Pilot prediction selection does not match the authoritative observation artifact")
        if not store.db.execute("SELECT 1 FROM artifacts WHERE path=?", (original.observations_ref,)).fetchone():
            raise IntegrityError("Authoritative pilot observations are not registered")
    return {"seed": seed, "kind": "pilot_batch", "receipt_ref": observation.log_refs[0],
            "status": ("synthetic_test_recorded" if original.measurement_origin == "synthetic_test" else
                       "recorded_success" if observation.status == "success" else "recorded_failure"),
            "measurement_origin": original.measurement_origin,
            "execution_id": legacy_key, "shared_operation_key": operation_key,
            "legacy_request_hash": job["request_hash"], "legacy_result_ref": observation.legacy_result_ref,
            "completion_ref": completion_ref, "observations_ref": original.observations_ref,
            "grid_seeds": spec.seeds, "grid_datasets": spec.datasets, "metric_grid_complete": cells == expected_cells,
            "shared_batch_receipt": True, "separate_container_per_seed": False,
            "fit_policy": "Pinned worker declares a fresh build(seed) and fit per dataset/seed cell inside one batch",
            "statistical_independence": "not established; seed cells may share deterministic predictions and the same fixed data split",
            "seconds": original.duration_seconds,
            "reason": "Shared operation, worker job, frozen source, specification, result grid and prediction references are bound; this is not an independent scoring replay",
            "payload_integrity": "Prediction arrays and logs are not re-read by this presentation helper"}


def _source(store, observation, sources):
    ref = observation.source_tree_ref
    if not ref:
        return {"status": "unverified", "reason": "No materialized source-tree evidence"}
    materialized, _ = _cached(store, "materialization/development/" + observation.id)
    if materialized is None:
        return {"status": "unverified", "reason": "No source materialization receipt"}
    if materialized.get("source_tree_ref") != ref or materialized.get("code_ref") != observation.code_ref:
        raise IntegrityError("Observation differs from its materialized source receipt")
    if ref in sources:
        return {"status": "recorded", "source_tree_ref": ref}
    tree = MaterializedSourceTree.model_validate(store.json(ref)).model_dump(mode="json")
    manifest = load_bundle(store, tree["bundle_ref"], verify=False)
    proposal = Candidate.model_validate(store.json(tree["proposal_ref"])).model_dump(mode="json")
    if tree.get("base_snapshot_ref"):
        snapshot = SourceSnapshot.model_validate(store.json(tree["base_snapshot_ref"])).model_dump(mode="json")
        paths = snapshot["editable_paths"]
    else:
        paths = [value["path"] for value in proposal["files"]]
    files = []
    for path in sorted(paths):
        if path not in manifest.files:
            continue
        file = manifest.files[path]
        entry = {"path": path, "ref": file.ref, "sha256": file.sha256,
                 "bytes": file.size, "status": "unverified"}
        if file.size > _MAX_SOURCE_BYTES or PurePosixPath(path).suffix.lower() not in _TEXT_SUFFIXES:
            entry["reason"] = "Non-text or oversized source omitted"
        else:
            raw = store.read(file.ref)
            if len(raw) != file.size or digest(raw) != file.sha256:
                raise IntegrityError("Editable source differs from its manifest")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                entry["reason"] = "Non-UTF-8 source omitted"
            else:
                if "\x00" in text:
                    entry["reason"] = "Binary source omitted"
                else:
                    entry.update(status="recorded", _content=text)
        files.append(entry)
    sources[ref] = {"source_tree_ref": ref, "bundle_ref": tree["bundle_ref"],
                    "changed_paths_from_starter": sorted(tree.get("changed_paths", [])),
                    "files": files, "method_isolation": "unverified"}
    return {"status": "recorded", "source_tree_ref": ref}


def _summarize(store, observation, ref, sources):
    protocol = _protocol(store, observation)
    seeds = protocol["seeds"]
    result = {"id": observation.id, "phase": observation.phase, "status": observation.status,
              "observation_ref": ref, "commitment": "recorded", "protocol": protocol,
              "metric": observation.metric, "metrics": observation.metrics or (
                  {observation.metric: observation.value} if observation.value is not None else {}),
              "per_seed_metrics": [], "training_executions": [],
              "prediction_evidence": [], "training_output_bundles": [],
              "source": _source(store, observation, sources)}
    for index in range(max(len(seeds), len(observation.log_refs), len(observation.values))):
        seed = seeds[index] if index < len(seeds) else None
        result["training_executions"].append(_execution(store, observation, index, seed))
        if index < len(observation.values):
            result["per_seed_metrics"].append({"seed": seed, "values": {
                name: values[index] for name, values in (observation.metric_values or {
                    observation.metric: observation.values}).items()}})
        if index < len(observation.bundle_refs):
            result["training_output_bundles"].append({"seed": seed, **_bundle(store, observation.bundle_refs[index])})
        if index < len(observation.evaluation_refs):
            evaluated = store.json(observation.evaluation_refs[index])
            saved, _ = _cached(store, "pipeline/" + observation.id + "/" + str(seed))
            if (saved is None or canonical(saved) != canonical(evaluated)
                    or evaluated.get("seed") != seed
                    or index >= len(observation.bundle_refs)
                    or evaluated.get("input_bundle") != observation.bundle_refs[index]):
                raise IntegrityError("Scored prediction evidence differs from the committed pipeline")
            prediction_ref = evaluated["prediction_bundle"]
            prediction = _bundle(store, prediction_ref)
            if ("prediction" not in prediction["roles"]
                    or not set(prediction["roles"]) <= {"prediction", "diagnostic"}):
                raise IntegrityError("Scored prediction bundle contains non-prediction outputs")
            declared_ref = observation.prediction_refs[index] if index < len(observation.prediction_refs) else None
            result["prediction_evidence"].append({"seed": seed, "kind": "scored_prediction_bundle",
                **prediction, "observation_reference_status": "matches" if declared_ref == prediction_ref else
                "legacy_or_incorrect_alias; authoritative prediction bundle taken from bound evaluation"})
        elif index < len(observation.prediction_refs):
            # Older/file/timing adapters use a direct file ref. Do not parse its
            # contents or reinterpret arbitrary JSON as a prediction/checkpoint.
            file_ref = observation.prediction_refs[index]
            artifact = store.db.execute("SELECT sha,size FROM artifacts WHERE path=?", (file_ref,)).fetchone()
            result["prediction_evidence"].append({"seed": seed, "kind": "adapter_file_reference",
                "ref": file_ref, "sha256": artifact["sha"] if artifact else None,
                "bytes": artifact["size"] if artifact else None,
                "status": "recorded" if artifact else "unverified",
                "payload_integrity": "not_rechecked_in_prompt_builder"})
    return result


def _trial_summaries(store, trials):
    if len(trials) > _MAX_RECORDS:
        raise ValueError("Evidence context trial count exceeds its bound")
    summaries = []
    for trial in trials:
        trial = _plain(trial)
        observation = trial.get("result", {})
        if observation.get("phase") not in _NONFINAL:
            continue
        saved, ref = _cached(store, "live-trial/" + trial["id"])
        if saved is not None and canonical(saved) != canonical(trial):
            raise IntegrityError("Offered trial differs from its committed decision")
        summaries.append({"id": trial["id"], "commitment": "recorded" if saved is not None else "unverified",
            "trial_ref": ref, "observation_id": observation.get("id"),
            "execution_status": observation.get("status"), "phase": observation.get("phase"),
            "decision": trial.get("decision", {}).get("decision"),
            "decision_reason": _safe_text(trial.get("decision", {}).get("reason", "")),
            "execution_reason": _safe_text(observation.get("reason", "")),
            "parents": list(trial.get("parents", [])),
            "interpretation": "Execution failure and scientific rejection are separate outcomes"})
    return summaries


def _comparisons(summaries):
    output = []
    # Compare each observation with the first of each distinct protocol, keeping
    # output bounded instead of an O(n^2) all-pairs prompt expansion.
    groups = {}
    for summary in summaries:
        protocol = summary.get("protocol", {})
        if protocol.get("status") == "recorded":
            groups.setdefault(protocol["hash"], summary)
    if not groups:
        return output
    first = next(iter(groups.values()))
    left = first["protocol"]
    for other in list(groups.values())[1:]:
        right = other["protocol"]
        fields = [name for name in ("phase", "seeds") if left[name] != right[name]]
        fields += [name for name in sorted(set(left["component_hashes"]) | set(right["component_hashes"]))
                   if left["component_hashes"].get(name) != right["component_hashes"].get(name)]
        output.append({"left_observation": first["id"], "right_observation": other["id"],
            "exact_protocol_match": False, "different_components": fields,
            "interpretation": "Different phase/seed sets explain expected hash changes; compare each control with its matched protocol, not unmatched original runs"})
    return output


def committed_control_context(store, observations):
    """Explain actual byte changes for supplied, committed matched controls."""
    selected = {value["id"]: value for value in observations}
    result = []
    for row in store.db.execute("SELECT key,output_ref FROM calls WHERE key LIKE 'control/%' ORDER BY key"):
        record = store.json(row["output_ref"])
        matched, control = record.get("matched", {}), record.get("control", {})
        if (matched.get("phase") not in _NONFINAL or control.get("phase") not in _NONFINAL
                or matched.get("id") not in selected or control.get("id") not in selected):
            continue
        for value in (matched, control):
            committed, _ = _committed(store, value)
            if committed is None:
                raise IntegrityError("Matched control refers to an uncommitted observation")
        entry = {"control_record_ref": row["output_ref"], "matched_id": matched["id"],
            "control_id": control["id"],
            "exact_protocol_match": matched["protocol_hash"] == control["protocol_hash"],
            "method_isolation": "unverified", "source_comparison": "unverified"}
        # Preserve the scientific disposition, not only code differences. The
        # old combined control format includes candidate code inside its plan;
        # explicitly select descriptive fields rather than forwarding that code.
        plan = record.get("plan", {})
        entry["plan"] = {name: plan[name] for name in (
            "id", "component", "question", "purpose", "changed_files", "changed_file",
            "invariants", "known_confounds", "execution_kind", "evaluation_split",
        ) if name in plan}
        review = record.get("review")
        entry["review"] = ({name: review[name] for name in ("decision", "reason", "confounds")
                            if name in review} if isinstance(review, dict) else None)
        entry["known_confounds"] = list(dict.fromkeys([
            *plan.get("known_confounds", []),
            *(review.get("confounds", []) if isinstance(review, dict) else []),
        ]))
        reasons = []
        if not entry["exact_protocol_match"]:
            reasons.append("The control and matched reference use different protocols")
        if not isinstance(review, dict) or review.get("decision") != "Accept":
            reasons.append("The control has no accepted isolation review")
        if entry["known_confounds"]:
            reasons.append("The recorded control design or review identifies confounds")
        entry["attribution_restriction"] = {
            "isolated_component": "prohibited" if reasons else "eligible_for_review",
            "reasons": reasons,
            "scientific_support": "unverified; an accepted design does not certify causal isolation",
        }
        if matched.get("source_tree_ref") and control.get("source_tree_ref"):
            left_tree = store.json(matched["source_tree_ref"])
            right_tree = store.json(control["source_tree_ref"])
            left = load_bundle(store, left_tree["bundle_ref"], verify=False).files
            right = load_bundle(store, right_tree["bundle_ref"], verify=False).files
            changed = sorted(path for path in set(left) | set(right) if left.get(path) != right.get(path))
            entry.update(source_comparison="recorded_byte_and_executable_metadata",
                changed_paths=changed,
                command_unchanged=left_tree["command"] == right_tree["command"],
                matched_source_tree_ref=matched["source_tree_ref"],
                control_source_tree_ref=control["source_tree_ref"],
                note="Inspect the matching source entries; unchanged file count does not prove that one scientific component was isolated")
        result.append(entry)
    return result


def build_evidence_context(store, observations, *, controls=(), trials=(), max_source_chars=16_000):
    """Build deterministic non-final evidence explanations without model/tool calls.

    Observations can be model records, dicts, or explicit result/matched/control
    containers. Missing fixture-only provenance remains visibly unverified.
    Existing contradictory or corrupted committed records always fail closed.
    Source excerpts have one total character allowance across distinct source
    files; hashes are provenance, never a substitute for inspecting omitted code.
    """
    if not 0 <= max_source_chars <= 100_000:
        raise ValueError("Evidence source context must be bounded to 0..100000 characters")
    offered = _offered([observations, controls])
    distinct = {}
    for observation in offered:
        if observation["id"] in distinct and canonical(distinct[observation["id"]]) != canonical(observation):
            raise IntegrityError("Conflicting observations have the same evidence ID")
        distinct.setdefault(observation["id"], observation)
    summaries, sources = [], {}
    for offered_observation in distinct.values():
        observation, ref = _committed(store, offered_observation)
        if observation is None:
            summaries.append({"id": offered_observation["id"], "commitment": "unverified",
                              "reason": "Observation has no committed development record"})
        else:
            summaries.append(_summarize(store, observation, ref, sources))
    files = [file for source in sources.values() for file in source["files"] if "_content" in file]
    allowances = _allowances([len(file["_content"]) for file in files], max_source_chars)
    for file, count in zip(files, allowances, strict=True):
        text = file.pop("_content")
        file.update(characters=len(text), omitted_characters=len(text) - count,
                    text_complete=len(text) == count)
        if count == len(text):
            file["text"] = text
        else:
            head, tail = (count + 1) // 2, count // 2
            file.update(excerpt_head=text[:head], excerpt_tail=text[-tail:] if tail else "")
    return {"schema_version": 1, "scope": "committed_non_final_development",
        "observations": summaries, "protocol_differences": _comparisons(summaries),
        "matched_controls": committed_control_context(store, summaries),
        "sources": list(sources.values()), "search_attempts": _trial_summaries(store, trials),
        "source_excerpt_characters": sum(allowances), "source_excerpt_limit": max_source_chars,
        "interpretation_rules": [
            "Content-addressed hashes describe equal bytes, not whether training ran again. Use bound training execution receipts and seed lists to assess repeated execution.",
            "Training/checkpoint bundles and scored prediction bundles are separate artifacts. Shared checkpoint config.json hashes do not imply identical learned weights or predictions.",
            "A seed identifies a training invocation; distinct seeds/receipts do not certify statistical independence or that candidate code used the seed correctly.",
            "Protocol hashes include phase, seeds, runtime, evaluator and data identities. A mismatch requires explanation and matched controls; do not compare unmatched runs as if their protocols matched.",
            "Source excerpts show only frozen editable files. Omitted code, semantic isolation, and mechanism attribution remain unverified; file-level isolation is not semantic equivalence.",
            "This package is a presentation of committed records, not an independent audit, final-test evidence, or proof that an experiment request was fully satisfied.",
        ]}
