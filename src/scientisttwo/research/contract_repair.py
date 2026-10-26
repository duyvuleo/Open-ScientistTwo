"""A source-bound child for review-contract repairs; parent evidence stays intact.

This is deliberately limited to a closed bundled pilot blocked in review after
its manuscript was committed. Scientific execution and writer implementations
must remain identical. Existing rows retain their original source epoch; only
the corrected review roles receive new invocation keys and new accounting.
"""

import ast
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import time

from ..live.adapters import implementation_identity
from ..live.models import LiveSpec
from ..store import IntegrityError, Store, canonical, digest, file_digest
from ..studies import media_relay as relay


VERSION = "scientisttwo-review-contract-repair-v1"
INHERITED_VERSION = "scientisttwo-inherited-model-accounting-v1"
REVIEW_REPAIR_FILES = frozenset({
    "research/controller.py", "research/stages.py", "research/reviews.py",
    "research/contract_repair.py", "research/live_pilot.py", "research/minimal_live_acceptance.py",
})
_NAMESPACED_ROLES = {"scientific_review", "rebuttal_planner"}
_TABLE_KEYS = {"calls": "key", "jobs": "key", "live_operations": "key", "events": "key", "artifacts": "path"}
_PATH_FIELDS = ("public_dir", "private_dir", "candidate_dir", "evaluator_dir")
_MUTABLE_METADATA = {"live_spec_ref", "pilot_live_spec_ref", "live_sources_ref", "live_state",
                     "live_pilot_model_accounting", "live_pilot_launch_ref"}
_FINAL_MARKERS = {"live_frozen", "live_final", "research_final_bundle", "study_final_grant_ref",
                  "study_final_release_ref", "study_research_freeze_ref"}
_ALWAYS_PROTECTED = {"task_ref", "profile_ref", "evaluator", "adapter_identity", "adapter_sources_ref",
    "live_adapter_identity", "live_backend", "engine_version", "runtime", "live_deadline", "deadline",
    "live_pilot_options_ref", "split_manifest_ref", "live_images", "research_writer_identity"}


def _tree(root):
    result = {}
    for current, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(current) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise IntegrityError("Repair fork rejects symlinks and special files")
            if stat.S_ISREG(mode):
                result[path.relative_to(root).as_posix()] = file_digest(path)
    return result


@contextmanager
def _closed_parent(root):
    for suffix in ("-wal", "-shm"):
        if (root / ("campaign.sqlite" + suffix)).exists():
            raise IntegrityError("Close and checkpoint the parent SQLite store before repair")
    lock = root / ".campaign.lock"
    descriptor = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW) if lock.exists() else None
    try:
        if descriptor is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        database = sqlite3.connect(f"file:{root / 'campaign.sqlite'}?mode=ro&immutable=1", uri=True)
        database.row_factory = sqlite3.Row
        try:
            if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise IntegrityError("Parent SQLite integrity check failed")
            yield database
        finally:
            database.close()
    finally:
        if descriptor is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _metadata(db):
    return {row["key"]: json.loads(row["value"]) for row in db.execute("SELECT * FROM metadata")}


def _rows(db):
    return {table: {row[key]: digest(canonical(dict(row))) for row in db.execute("SELECT * FROM " + table)}
            for table, key in _TABLE_KEYS.items()}


def _parent_json(root, db, ref):
    row = db.execute("SELECT sha,size FROM artifacts WHERE path=?", (ref,)).fetchone()
    path = root / ref if isinstance(ref, str) else root
    if (row is None or not path.resolve().is_relative_to((root / "artifacts").resolve())
            or not path.is_file() or path.is_symlink() or path.stat().st_size != row["size"]
            or file_digest(path) != row["sha"]):
        raise IntegrityError("Parent artifact is missing or changed")
    return json.loads(path.read_text())


def _source_relative(name):
    if name.startswith("native/"):
        return "benchmarks/" + name.removeprefix("native/")
    if name.startswith("core/"):
        return name.removeprefix("core/")
    if "/" not in name:
        return "live/" + name
    return name


def _check_scientific_identity(metadata, read_json):
    from ..models import TaskSpec
    from ..runner import evaluator_identity
    from ..tasks import load_task_adapter
    from .pilot_bridge import PilotAdapter

    task = TaskSpec.model_validate(read_json(metadata["task_ref"]))
    if task.adapter not in {"classification", "regression", "optimization"}:
        raise IntegrityError("Review repair supports only built-in pilot scientific adapters")
    legacy = load_task_adapter(task.adapter, task)
    options = read_json(metadata["live_pilot_options_ref"])
    bridge = PilotAdapter(None, task, legacy, frozen_literature=options.get("frozen_literature", []))
    if (metadata.get("evaluator") != evaluator_identity() or metadata.get("adapter_identity") != legacy.identity()
            or metadata.get("live_adapter_identity") != bridge.identity()):
        raise IntegrityError("Scientific adapter, data generator, scorer or worker implementation changed")
    package = Path(__file__).parents[1]
    for relative, ref in read_json(metadata["adapter_sources_ref"]).items():
        path = package / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_file()
                or path.is_symlink() or file_digest(path) != Path(ref).stem):
            raise IntegrityError("Trusted adapter source closure changed")


def _review_ast(source, filename):
    tree = ast.parse(source)
    allowed = ({"ResearchController": {"ask", "write_review_pipeline"}} if filename.endswith("controller.py")
               else {"ResearchStages": {"research_review", "research_supplementary_plans"}})
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in allowed:
            if node.name == "ResearchController":
                _normalize_task_source_calls(node)
            node.body = [child for child in node.body
                         if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                         or child.name not in allowed[node.name]]
        if filename.endswith("stages.py") and isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "ROLE_INSTRUCTIONS" for target in node.targets):
            if not isinstance(node.value, ast.Dict):
                raise IntegrityError("Review role instructions must remain a literal mapping")
            kept = [(key, value) for key, value in zip(node.value.keys, node.value.values, strict=True)
                    if not isinstance(key, ast.Constant) or key.value != "scientific_review"]
            node.value.keys, node.value.values = [key for key, _ in kept], [value for _, value in kept]
    return ast.dump(tree, include_attributes=False)


_TASK_SOURCE_HELPER = '''def _task_source_text(self):
    context = self.adapter.context()
    if self.store.get("contract_repair_manifest_ref"):
        from .contract_repair import recorded_task_source_text
        return recorded_task_source_text(self.store, context)
    return str(context)
'''


def _ast_equal(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _normalize_task_source_calls(controller):
    """Permit only exact task-source serialization substitutions, not method edits."""
    helper = ast.parse(_TASK_SOURCE_HELPER).body[0]
    replacement = ast.parse("self._task_source_text()", mode="eval").body
    initial = ast.parse("str(adapter.context())", mode="eval").body
    running = ast.parse("str(self.adapter.context())", mode="eval").body
    initial_target = ast.parse("self.sources = None").body[0].targets[0]
    running_target = ast.parse("self.sources[0]['text'] = None").body[0].targets[0]
    kept = []
    for method in controller.body:
        if isinstance(method, ast.FunctionDef) and method.name == "_task_source_text":
            if not _ast_equal(method, helper):
                raise IntegrityError("Task-source helper differs from the exact representation-only adapter")
            continue
        if isinstance(method, ast.FunctionDef) and method.name in {"__init__", "run"}:
            for assignment in ast.walk(method):
                if not isinstance(assignment, ast.Assign) or len(assignment.targets) != 1:
                    continue
                if method.name == "run" and _ast_equal(assignment.targets[0], running_target):
                    if _ast_equal(assignment.value, running) or _ast_equal(assignment.value, replacement):
                        assignment.value = ast.Constant(value="verified-task-source-representation")
                    else:
                        raise IntegrityError("Run task-source expression is outside representation repair scope")
                elif method.name == "__init__" and _ast_equal(assignment.targets[0], initial_target):
                    value = assignment.value
                    if not isinstance(value, ast.List) or not value.elts or not isinstance(value.elts[0], ast.Dict):
                        continue
                    entry = value.elts[0]
                    keys = [key.value if isinstance(key, ast.Constant) else None for key in entry.keys]
                    if "id" not in keys or "text" not in keys:
                        continue
                    if not _ast_equal(entry.values[keys.index("id")], ast.Constant(value="task")):
                        continue
                    index = keys.index("text")
                    if _ast_equal(entry.values[index], initial) or _ast_equal(entry.values[index], replacement):
                        entry.values[index] = ast.Constant(value="verified-task-source-representation")
                    else:
                        raise IntegrityError("Constructor task-source expression is outside representation repair scope")
        kept.append(method)
    controller.body = kept


def recorded_task_source_text(store, current_context):
    """Reuse exact old task text only when its typed JSON content is unchanged."""
    manifest = verify_review_contract(store)
    snapshot_ref = manifest["parent_snapshot_ref"]
    store.read(snapshot_ref)
    snapshot = sqlite3.connect((store.root / snapshot_ref).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    snapshot.row_factory = sqlite3.Row
    try:
        row = snapshot.execute("SELECT value FROM metadata WHERE key='request/limitations/0'").fetchone()
        call = snapshot.execute("SELECT input_hash FROM calls WHERE key='live-agent/limitations/0'").fetchone()
        if row is None or call is None:
            raise IntegrityError("Original limitations request is absent from the parent snapshot")
        ref = json.loads(row["value"])
        if ref != manifest["parent_metadata"].get("request/limitations/0"):
            raise IntegrityError("Original task source differs from the parent metadata snapshot")
        request = store.json(ref)
        if request.get("role") != "limitations" or digest(canonical(request)) != call["input_hash"]:
            raise IntegrityError("Original task source is not bound to its committed limitations request")
    finally:
        snapshot.close()
    sources = request.get("inputs", {}).get("sources", [])
    offered = [source for source in sources if isinstance(source, dict) and source.get("id") == "task"]
    if len(offered) != 1 or not isinstance(offered[0].get("text"), str) or len(offered[0]["text"]) > 100000:
        raise IntegrityError("Original task source must be one bounded literal text record")
    text = offered[0]["text"]
    try:
        expression = ast.parse(text, mode="eval")
        nodes = list(ast.walk(expression))
        if len(nodes) > 10000:
            raise ValueError("Literal is too large")
        for node in nodes:
            if isinstance(node, ast.Dict):
                keys = [key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)]
                if len(keys) != len(node.keys) or len(set(keys)) != len(keys):
                    raise ValueError("Ambiguous or non-string literal keys")
        old_context = ast.literal_eval(expression)

        def json_shape(value, depth=0):
            if depth > 64:
                raise ValueError("Literal nesting is too deep")
            if value is None or type(value) in (str, bool, int, float):
                return
            if isinstance(value, dict) and all(isinstance(key, str) for key in value):
                for item in value.values():
                    json_shape(item, depth + 1)
                return
            if isinstance(value, list):
                for item in value:
                    json_shape(item, depth + 1)
                return
            raise ValueError("Non-JSON task source type")

        json_shape(old_context)
        json_shape(current_context)
        if canonical(old_context) != canonical(current_context):
            raise ValueError("Task source values changed")
    except (ValueError, TypeError, SyntaxError, RecursionError):
        raise IntegrityError("Task source is not a representation-only equivalent of the frozen context") from None
    return text


def _source_delta(root, db, metadata, old, current, allowed):
    if not allowed <= REVIEW_REPAIR_FILES:
        raise IntegrityError("Source-repair whitelist cannot be broadened")
    changes = {key: {"before": old.get(key), "after": current.get(key)}
               for key in sorted(old.keys() | current.keys()) if old.get(key) != current.get(key)}
    if not changes or not changes.keys() <= allowed:
        raise IntegrityError("Source changes exceed the explicit review-contract repair scope")
    if old.keys() - current.keys() or current.keys() - old.keys() - {"research/contract_repair.py"}:
        raise IntegrityError("Repair cannot remove source or add unrelated implementation files")
    source_index = _parent_json(root, db, metadata.get("live_sources_ref"))
    package = Path(__file__).parents[1]
    for relative, ref in source_index.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise IntegrityError("Parent source closure contains an unsafe path")
        registered = db.execute("SELECT sha,size FROM artifacts WHERE path=?", (ref,)).fetchone()
        if registered is None or file_digest(root / ref) != registered["sha"]:
            raise IntegrityError("Parent source closure differs from registered bytes")
    for name, values in changes.items():
        relative = _source_relative(name)
        path = package / relative
        if not path.is_file() or file_digest(path) != values["after"]:
            raise IntegrityError("Current reviewed source differs from its declared identity")
        if values["before"] is not None:
            ref = source_index.get(relative)
            if ref is None or file_digest(root / ref) != values["before"]:
                raise IntegrityError("Original source closure does not match its source epoch")
        if name in {"research/controller.py", "research/stages.py"}:
            ref = source_index.get(relative)
            if ref is None or file_digest(root / ref) != values["before"]:
                raise IntegrityError("Review repair requires the exact original controller source")
            if _review_ast((root / ref).read_text(), name) != _review_ast(path.read_text(), name):
                raise IntegrityError("Controller changes include non-review workflow logic")
            values["scope_verification"] = "AST unchanged outside explicitly permitted review functions"
        else:
            values["scope_verification"] = "Explicit review or accounting support module"
    return changes, source_index


def fork_review_contract(parent_root, child_root, *, repair_id, authorization_ref,
                         authorization_files=(), allowed_changed_files=REVIEW_REPAIR_FILES):
    """Create one append-only review repair child; makes no live/runtime calls."""
    parent, child = Path(parent_root).resolve(), Path(child_root).resolve()
    if (not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", repair_id)
            or not isinstance(authorization_ref, str) or not 1 <= len(authorization_ref) <= 2000):
        raise ValueError("Repair ID and explicit existing-authorization reference are required")
    if child.exists() or child.is_relative_to(parent) or parent.is_relative_to(child):
        raise IntegrityError("Repair requires a new disjoint child directory")
    authorization_files = [Path(path) for path in authorization_files]
    if any(path.is_symlink() or not path.is_file() or path.stat().st_size > 65536 for path in authorization_files):
        raise IntegrityError("Authorization evidence must be bounded regular files without credentials")
    with _closed_parent(parent) as database:
        before = _tree(parent)
        meta = _metadata(database)
        state = meta.get("live_state", {})
        if (state.get("status") != "blocked" or state.get("stage") != "review"
                or any(meta.get(key) for key in _FINAL_MARKERS)
                or meta.get("contract_repair_manifest_ref") or meta.get("inherited_model_accounting_ref")):
            raise IntegrityError("Only a first closed pre-final review-contract repair is supported")
        if database.execute("SELECT 1 FROM live_operations WHERE status='running' OR key LIKE 'final/%'").fetchone():
            raise IntegrityError("Running or protected-final operations forbid a repair fork")
        if database.execute("SELECT 1 FROM jobs WHERE status='running'").fetchone():
            raise IntegrityError("A running experiment job forbids a repair fork")
        old_spec = LiveSpec.model_validate(_parent_json(parent, database, meta.get("live_spec_ref")))
        if (old_spec.development.adapter != "bundled-pilot" or meta.get("live_backend") != {
                "backend": "codex", "synthetic": False, "model": old_spec.model}):
            raise IntegrityError("Review repair currently supports explicit live bundled pilots only")
        deadline = meta.get("live_deadline")
        if type(deadline) not in (int, float) or not time.time() < deadline:
            raise IntegrityError("Original campaign deadline expired; repair cannot extend it")
        if not meta.get("research_manuscript_bundle") or not meta.get("research_manuscript"):
            raise IntegrityError("Repair requires a committed manuscript before scientific review")
        summary = _parent_json(parent, database, meta.get("live_pilot_model_accounting"))
        if not summary.get("usage_complete") or summary.get("unknown_calls"):
            raise IntegrityError("This review repair cannot resolve or hide unknown model usage")
        for row in database.execute("SELECT * FROM artifacts"):
            path = parent / row["path"]
            if (not path.resolve().is_relative_to(parent / "artifacts") or not path.is_file()
                    or path.stat().st_size != row["size"] or file_digest(path) != row["sha"]):
                raise IntegrityError("Parent evidence integrity verification failed")
        _check_scientific_identity(meta, lambda ref: _parent_json(parent, database, ref))
        current = implementation_identity()
        changes, source_index = _source_delta(parent, database, meta, old_spec.implementation,
                                               current, frozenset(allowed_changed_files))
        native = old_spec.native.model_dump(mode="json")
        for field in _PATH_FIELDS:
            old_path = Path(native[field])
            if not old_path.is_absolute() or old_path.resolve() != old_path or not old_path.is_relative_to(parent):
                raise IntegrityError("Bundled pilot execution paths must be canonical and local to the parent")
            native[field] = str(child / old_path.relative_to(parent))
        historical = _rows(database)
        operations = [dict(row) for row in database.execute("SELECT * FROM live_operations ORDER BY key")]
        policy = relay.RelayPolicy.from_mapping(relay.read_json(parent / "model-ledger/policy.json", 4096))
        if (policy.model != old_spec.model or policy.source_sha256 != digest(canonical(old_spec.implementation))
                or policy.expires_at != deadline or policy.max_calls != old_spec.max_calls
                or policy.token_stop_threshold != old_spec.token_stop_threshold):
            raise IntegrityError("Original model policy differs from its frozen campaign")
        claims = sorted((parent / "model-ledger/claims").iterdir())
        if len(claims) != summary.get("claimed_calls") or len(claims) != summary.get("completed_receipts"):
            raise IntegrityError("Parent claim inventory does not match known usage summary")
        for claim in claims:
            if not claim.is_dir() or {path.name for path in claim.iterdir()} != {"claim.json", "dispatch.json", "response.json"}:
                raise IntegrityError("Parent model completion is partial or ambiguous")
        from .live_pilot import _validated_dispatches
        triplets = [(relay.read_json(path / "claim.json", relay.REQUEST_LIMIT + 4096),
                     relay.read_json(path / "dispatch.json", 16384),
                     relay.read_json(path / "response.json", relay.RESPONSE_LIMIT)) for path in claims]
        if any(path.name != triplet[0]["request"]["request_id"] for path, triplet in zip(claims, triplets, strict=True)):
            raise IntegrityError("Parent claim directory does not match its request identity")
        totals = _validated_dispatches(triplets, policy)
        if (summary.get("known_observed_tokens") != totals["tokens"]
                or summary.get("ledger") != str(parent / "model-ledger")
                or summary.get("transport") != "direct_local_no_queue"
                or canonical(summary.get("policy")) != canonical(asdict(policy))):
            raise IntegrityError("Parent known-usage summary differs from validated dispatches")
        # All source and parent-evidence checks precede the first child write.
        child.mkdir(parents=True)
        (child / ".contract-repair-parent").mkdir(mode=0o700)
        for name in before:
            destination = child / (".contract-repair-parent/" + name if name.startswith("model-ledger/") else name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent / name, destination)
            if file_digest(destination) != before[name]:
                raise IntegrityError("Parent changed during repair copy")
        archive = child / ".contract-repair-parent"
        archive.mkdir(exist_ok=True)
        shutil.copy2(parent / "campaign.sqlite", archive / "campaign.sqlite")
        store = Store(child)
        try:
            if _rows(store.db) != historical:
                raise IntegrityError("Child checkpoint did not preserve historical evidence rows")
            snapshot_ref = store.put_file(archive / "campaign.sqlite", "contract-repair-parent-snapshot")
            operations_ref = store.put_json(operations, "contract-repair-parent-operations")
            policy_ref = store.put_json(relay.read_json(archive / "model-ledger/policy.json", 4096), "contract-repair-parent-model-policy")
            copied_claims = [{name + "_ref": store.put_file(archive / "model-ledger/claims" / claim.name / (name + ".json"),
                                "contract-repair-parent-model-" + name)
                              for name in ("claim", "dispatch", "response")} for claim in claims]
            inherited = {"schema_version": INHERITED_VERSION, "source_epoch": "parent", "includes_prior_epochs": False,
                "parent_root": str(parent), "parent_spec_ref": meta["live_spec_ref"],
                "parent_policy_ref": policy_ref, "parent_summary_ref": meta["live_pilot_model_accounting"],
                "parent_operations_ref": operations_ref, "parent_snapshot_ref": snapshot_ref,
                "parent_deadline": deadline, "claims": copied_claims}
            inherited_ref = store.put_json(inherited, "contract-repair-inherited-model-accounting")
            new_spec = LiveSpec.model_validate({**old_spec.model_dump(mode="json"), "native": native, "implementation": current})
            new_spec_ref = store.put_json(new_spec, "contract-repair-child-specification")
            sources = dict(source_index)
            package = Path(__file__).parents[1]
            for name in changes:
                relative = _source_relative(name)
                sources[relative] = store.put_file(package / relative, "contract-repair-current-source")
            sources_ref = store.put_json(sources, "contract-repair-current-source-index")
            manifest = {"schema_version": VERSION, "repair_id": repair_id, "authorization_ref": authorization_ref,
                "authorization_artifact_refs": [store.put_file(path, "contract-repair-existing-authorization")
                                                for path in authorization_files],
                "parent_root": str(parent), "child_root": str(child), "created_at": time.time(),
                "module_sha256": file_digest(Path(__file__)), "parent_snapshot_ref": snapshot_ref,
                "parent_tree_hashes": before, "parent_metadata": meta, "historical_rows": historical,
                "parent_implementation": old_spec.implementation, "implementation": current,
                "source_changes": changes, "allowed_source_changes": sorted(allowed_changed_files),
                "parent_spec_ref": meta["live_spec_ref"], "child_spec_ref": new_spec_ref,
                "child_sources_ref": sources_ref, "inherited_model_accounting_ref": inherited_ref,
                "original_deadline": deadline, "review_roles": sorted(_NAMESPACED_ROLES),
                "invocation_suffix": "/contract-repair/" + repair_id,
                "scope": "Review contract repair only; original source epochs, calls, experiments, writer artifacts and resource charges are preserved"}
            manifest_ref = store.put_json(manifest, "contract-repair-manifest")
            store.set("inherited_model_accounting_ref", inherited_ref)
            store.set("contract_repair_manifest_ref", manifest_ref)
            store.set("live_spec_ref", new_spec_ref)
            store.set("pilot_live_spec_ref", new_spec_ref)
            store.set("live_sources_ref", sources_ref)
            store.set("live_state", {**state, "status": "running", "outcome": "contract_repair_pending",
                "reason": "Replay exact committed work, then apply the corrected scientific review contract"})
            # The old launch artifact remains parent provenance, never relabeled.
            store.set("live_pilot_launch_ref", None)
            store.set("live_pilot_model_accounting", None)
            store.event("contract-repair/" + repair_id, state["stage"], manifest["scope"])
            verify_review_contract(store)
            from .live_pilot import validate_inherited_model_accounting
            validate_inherited_model_accounting(store, new_spec)
            if _tree(parent) != before:
                raise IntegrityError("Parent files changed during repair preparation")
            return manifest_ref
        finally:
            store.close()


def verify_review_contract(store):
    """Recheck source, configuration, protected evidence and copied row prefixes."""
    ref = store.get("contract_repair_manifest_ref")
    if not ref:
        raise IntegrityError("Review repair manifest is absent")
    manifest = store.json(ref)
    if (manifest.get("schema_version") != VERSION or manifest.get("child_root") != str(store.root)
            or manifest.get("module_sha256") != file_digest(Path(__file__))
            or manifest.get("implementation") != implementation_identity()
            or store.get("live_spec_ref") != manifest.get("child_spec_ref")
            or store.get("pilot_live_spec_ref") != manifest.get("child_spec_ref")
            or store.get("live_sources_ref") != manifest.get("child_sources_ref")
            or store.get("inherited_model_accounting_ref") != manifest.get("inherited_model_accounting_ref")):
        raise IntegrityError("Repair source epoch or manifest binding changed")
    parent = manifest["parent_metadata"]
    for key, value in parent.items():
        if key in _MUTABLE_METADATA or key.startswith(("raw_request/", "request/", "context/")):
            continue
        # Workflow pointers legitimately advance only after the verified replay gate.
        if key not in _ALWAYS_PROTECTED and store.get("contract_repair_replay_gate_ref"):
            continue
        if store.get(key) != value:
            raise IntegrityError("Protected parent metadata changed before corrected review: " + key)
    spec = store.json(manifest["child_spec_ref"])
    old = store.json(manifest["parent_spec_ref"])
    comparable = {**spec, "implementation": old["implementation"], "native": dict(spec["native"])}
    for field in _PATH_FIELDS:
        expected = str(store.root / Path(old["native"][field]).relative_to(manifest["parent_root"]))
        if spec["native"][field] != expected:
            raise IntegrityError("Repair execution paths changed")
        comparable["native"][field] = old["native"][field]
    if comparable != old or store.get("live_deadline") != manifest["original_deadline"]:
        raise IntegrityError("Repair changed task, budgets, configuration or original deadline")
    _check_scientific_identity(_metadata(store.db), store.json)
    current = _rows(store.db)
    for table, rows in manifest["historical_rows"].items():
        if any(current[table].get(key) != value for key, value in rows.items()):
            raise IntegrityError("Inherited evidence row changed: " + table)
    if store.verify():
        raise IntegrityError("Repair artifact integrity failed")
    return manifest


def repair_invocation_key(store, key, role):
    if not store.get("contract_repair_manifest_ref") or role not in _NAMESPACED_ROLES:
        return key
    manifest = verify_review_contract(store)
    gate_ref = store.get("contract_repair_replay_gate_ref")
    if not gate_ref:
        if role != "scientific_review":
            raise IntegrityError("Corrected scientific review must precede repaired rebuttal planning")
        current = _rows(store.db)
        for table in ("calls", "jobs", "live_operations"):
            if current[table] != manifest["historical_rows"][table]:
                raise IntegrityError("Replay added work before the corrected review")
        claims = store.root / "model-ledger/claims"
        if claims.exists() and any(claims.iterdir()):
            raise IntegrityError("A model dispatch occurred before the verified replay gate")
        gate = {"schema_version": "scientisttwo-review-replay-gate-v1", "manifest_ref": store.get("contract_repair_manifest_ref"),
                "first_role": role, "original_key": key, "verified_at": time.time(),
                "added_model_calls": 0, "added_experiments": 0,
                "historical_rows": {table: current[table] for table in ("calls", "jobs", "live_operations")}}
        store.set("contract_repair_replay_gate_ref", store.put_json(gate, "contract-repair-replay-gate"))
    return key + manifest["invocation_suffix"]


def require_repair_dispatch_gate(store):
    if store.get("contract_repair_manifest_ref"):
        verify_review_contract(store)
        if not store.get("contract_repair_replay_gate_ref"):
            raise IntegrityError("New model dispatch is forbidden before verified cached review replay")


class ReviewReplayReady(RuntimeError):
    """The dry replay reached corrected review before reserving or dispatching it."""


class ContractRepairBudget:
    """Prevent any new experiment/model admission until exact replay is proven."""
    def __init__(self, delegate, *, stop_at_review=False):
        self.delegate, self.store = delegate, delegate.store
        self.stop_at_review = stop_at_review
        verify_review_contract(self.store)

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def reserve(self, key, kind, request):
        exists = self.store.db.execute("SELECT 1 FROM live_operations WHERE key=?", (key,)).fetchone()
        gate_ref = self.store.get("contract_repair_replay_gate_ref")
        if not gate_ref and not exists:
            raise IntegrityError("New operation is forbidden before exact cached replay reaches corrected review")
        if self.stop_at_review and not exists:
            manifest = verify_review_contract(self.store)
            gate = self.store.json(gate_ref)
            expected = "model/" + gate["original_key"] + manifest["invocation_suffix"] + "/0"
            if kind != "model" or key != expected:
                raise IntegrityError("Dry replay attempted unexpected work after review gate")
            raise ReviewReplayReady("Cached replay reached corrected scientific review with no new reservation")
        return self.delegate.reserve(key, kind, request)
