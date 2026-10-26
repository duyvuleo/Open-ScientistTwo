"""Exact parent context and controller-applied anchored edits.

These checks establish source provenance and declared file scope, not that a
proposed edit implements its scientific rationale. No candidate code is run.
"""

import ast
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Callable

from pydantic import Field, model_validator

from ..artifacts import load_bundle
from ..models import Record
from ..store import IntegrityError, Store, canonical, digest
from .models import Candidate, Hypothesis, SourceFile
from .source_tree import MaterializedSourceTree, candidate_snapshot, load_source_snapshot


class ParentEditError(ValueError):
    """A completed agent proposal violates the declared parent-edit contract."""


class ParentChangePlan(Record):
    changed_files: list[str] = Field(min_length=1, max_length=24)
    reason: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def valid(self):
        _paths(self.changed_files)
        return self


class MethodChange(Record):
    idea: Hypothesis
    change_plan: ParentChangePlan


class AnchoredEdit(Record):
    path: str
    before: str | None = Field(
        max_length=32_000,
        description="Exact unique text in the supplied parent edit target; null only for an allowed new file.",
    )
    after: str = Field(max_length=32_000)
    reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def valid(self):
        SourceFile(path=self.path, content=self.after)
        if self.before is not None:
            SourceFile(path=self.path, content=self.before)
            if not self.before:
                raise ValueError("Existing-file anchors must be nonempty")
        if self.before == self.after or self.before is None and not self.after:
            raise ValueError("No-op parent edits are forbidden")
        return self


class ParentPatch(Record):
    parent_source_ref: str = Field(pattern=r"^artifacts/[a-f0-9]{64}\.json$")
    edits: list[AnchoredEdit] = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def bounded(self):
        total = sum(len((edit.before or "").encode()) + len(edit.after.encode()) for edit in self.edits)
        if total > 500_000:
            raise ValueError("Parent edit payload exceeds 500000 UTF-8 bytes")
        return self


Materialize = Callable[[Candidate], Candidate]


def _paths(paths, *, max_count=None):
    paths = list(paths)
    if not paths or len(set(paths)) != len(paths) or max_count is not None and len(paths) > max_count:
        raise ParentEditError("Edit paths must be nonempty, distinct and within their declared count limit")
    for path in paths:
        SourceFile(path=path, content="")
    return paths


def _materialized(store, parent, materialize):
    candidate = deepcopy(parent)
    candidate = materialize(candidate) if materialize else candidate
    if candidate.command != parent.command:
        raise IntegrityError("Adapter materialization changed the parent's command")
    return candidate_snapshot(store, candidate)


def _parent(store, parent_candidate, parent_source_ref, materialize, editable_paths):
    parent = Candidate.model_validate(parent_candidate.model_dump())
    tree = MaterializedSourceTree.model_validate(store.json(parent_source_ref))
    if _materialized(store, parent, materialize) != tree:
        raise IntegrityError("Parent source reference differs from the exact active parent")
    bundle = load_bundle(store, tree.bundle_ref)
    proposal = {file.path: file.content for file in parent.files}
    if parent.base_snapshot_ref:
        allowed = load_source_snapshot(store, parent.base_snapshot_ref).editable_paths
        if editable_paths is not None:
            supplied = _paths(editable_paths)
            if not set(supplied) <= set(allowed):
                raise ParentEditError("Adapter edit scope exceeds the frozen source snapshot")
            allowed = supplied
    else:
        allowed = list(proposal) if editable_paths is None else editable_paths
    allowed = _paths(allowed)
    targets = {}
    for path in allowed:
        entry = bundle.files.get(path)
        if entry is None:
            if path in proposal:
                raise IntegrityError("Materialization omitted a parent proposal file")
            if path in bundle.directories or any(parent.as_posix() in bundle.files for parent in Path(path).parents):
                raise ParentEditError("Allowed new file conflicts with the parent source tree")
            targets[path] = None
            continue
        raw = store.read(entry.ref)
        if digest(raw) != entry.sha256 or len(raw) != entry.size:
            raise IntegrityError("Parent file differs from its materialized manifest")
        try:
            content = raw.decode("utf-8")
            SourceFile(path=path, content=content)
        except (ValueError, UnicodeError) as exc:
            raise ParentEditError("Editable parent source is not bounded UTF-8 text") from exc
        # Body-only adapters must receive body edits again, not a regenerated
        # whole-file proposal. Ordinary overlays edit the complete active bytes.
        target = proposal.get(path, content)
        targets[path] = {
            "path": path,
            "content": target,
            "sha256": digest(target.encode()),
            "materialized_sha256": entry.sha256,
            "representation": "proposal" if target != content else "materialized_file",
        }
        if target != content:
            targets[path]["materialized_content"] = content
    return parent, tree, bundle, allowed, targets


def parent_context(
    store: Store,
    parent_candidate: Candidate,
    parent_source_ref: str,
    *,
    materialize: Materialize | None = None,
    editable_paths: list[str] | None = None,
    max_context_bytes: int = 600_000,
) -> dict:
    """Return complete editable parent text, or fail instead of truncating it.

``files[*].content`` is the exact anchored-edit target. A transformed proposal
also supplies its complete ``materialized_content`` for inspection only.
``new_files`` names allowed paths which do not yet exist. Neither baseline
source nor compacted history is substituted for this verified parent.
    """
    parent, _, _, allowed, targets = _parent(
        store, parent_candidate, parent_source_ref, materialize, editable_paths
    )
    context = {
        "parent_source_ref": parent_source_ref,
        "parent_proposal_sha256": digest(canonical(parent)),
        "command": parent.command,
        "base_snapshot_ref": parent.base_snapshot_ref,
        "editable_paths": allowed,
        "files": [targets[path] for path in allowed if targets[path] is not None],
        "new_files": [path for path in allowed if targets[path] is None],
        "edit_contract": (
            "Plan changed_files before implementation. Return ParentPatch bound to parent_source_ref. "
            "Each before anchor must occur exactly once in files[*].content of the original parent; "
            "anchors in one file must not overlap. Use before=null only for a listed new_file. "
            "Only planned files may change. The controller preserves all other bytes, the base "
            "snapshot and command. materialized_content is complete read-only adapter output, "
            "not the edit target. No source text is truncated. Source fidelity is not a "
            "certification of scientific intent or mechanism attribution."
        ),
    }
    if max_context_bytes < 1 or len(canonical(context)) > max_context_bytes:
        raise ParentEditError("Complete parent context exceeds its explicit byte limit; source was not truncated")
    return context


def apply_parent_patch(
    store: Store,
    parent_candidate: Candidate,
    parent_source_ref: str,
    patch: ParentPatch,
    *,
    declared_paths: list[str],
    materialize: Materialize | None = None,
    editable_paths: list[str] | None = None,
) -> tuple[Candidate, dict]:
    """Apply unique nonoverlapping anchors to a fixed parent and retain proof.

All anchors are resolved against the original parent, not earlier hunks. The
returned candidate preserves full-file, frozen-overlay, or body-only form.
    """
    patch = ParentPatch.model_validate(patch.model_dump())
    if patch.parent_source_ref != parent_source_ref:
        raise ParentEditError("Patch is bound to a different or stale parent source")
    parent, _, before_bundle, allowed, targets = _parent(
        store, parent_candidate, parent_source_ref, materialize, editable_paths
    )
    planned = set(_paths(declared_paths, max_count=24))
    if not planned <= set(allowed):
        raise ParentEditError("Declared changes exceed the parent's editable file scope")
    edited_paths = {edit.path for edit in patch.edits}
    if edited_paths != planned:
        raise ParentEditError("Patch paths must exactly match the independently declared changed_files")
    grouped = defaultdict(list)
    for edit in patch.edits:
        grouped[edit.path].append(edit)
    replacements = {}
    for path, edits in grouped.items():
        target = targets[path]
        if target is None:
            if len(edits) != 1 or edits[0].before is not None:
                raise ParentEditError("New files require exactly one explicit before=null edit")
            replacements[path] = edits[0].after
            continue
        original = target["content"]
        spans = []
        for edit in edits:
            if edit.before is None:
                raise ParentEditError("before=null cannot replace an existing file")
            start = original.find(edit.before)
            if start < 0 or original.find(edit.before, start + 1) >= 0:
                raise ParentEditError("Each before anchor must occur exactly once in the parent")
            spans.append((start, start + len(edit.before), edit.after))
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ParentEditError("Parent edit anchors must not overlap")
        content = original
        for start, end, replacement in reversed(spans):
            content = content[:start] + replacement + content[end:]
        if content == original:
            raise ParentEditError("Combined parent edits have no effect")
        replacements[path] = content
    try:
        files = [SourceFile(path=file.path, content=replacements.pop(file.path, file.content))
                 for file in parent.files]
        files += [SourceFile(path=path, content=content) for path, content in sorted(replacements.items())]
        candidate = Candidate(
            files=files, base_snapshot_ref=parent.base_snapshot_ref,
            command=list(parent.command), description=patch.description,
        )
    except ValueError as exc:
        raise ParentEditError("Patched candidate exceeds the bounded source contract") from exc
    try:
        after_tree = _materialized(store, candidate, materialize)
    except (ValueError, SyntaxError) as exc:
        raise ParentEditError("Patched source violates the adapter's materialization contract") from exc
    after_bundle = load_bundle(store, after_tree.bundle_ref)
    all_paths = set(before_bundle.files) | set(after_bundle.files)
    changed = sorted(path for path in all_paths if before_bundle.files.get(path) != after_bundle.files.get(path))
    if changed != sorted(planned):
        raise ParentEditError("Materialized changes differ from declared paths or contain a no-op")
    after_ref = store.put_json(after_tree, "parent_patch_materialized_source")
    proof = {
        "schema_version": 1,
        "parent_source_ref": parent_source_ref,
        "candidate_source_ref": after_ref,
        "parent_proposal_sha256": digest(canonical(parent)),
        "candidate_proposal_sha256": digest(canonical(candidate)),
        "patch_ref": store.put_json(patch, "parent_patch"),
        "declared_paths": sorted(planned),
        "changed_paths": changed,
        "command_inherited": candidate.command == parent.command,
        "base_snapshot_inherited": candidate.base_snapshot_ref == parent.base_snapshot_ref,
        "source_hashes": {
            path: {"before": before_bundle.files[path].sha256 if path in before_bundle.files else None,
                   "after": after_bundle.files[path].sha256 if path in after_bundle.files else None}
            for path in sorted(all_paths)
        },
        "scope": "Verified parent bytes, anchored edits and declared file scope; scientific intent and mechanism attribution remain unverified",
        "semantic_intent": "unverified",
    }
    return candidate, proof


def validate_executable_parent_change(
    store: Store,
    parent_source_ref: str,
    candidate_source_ref: str,
) -> dict:
    """Reject a new method proposal containing only unchanged Python syntax.

Call this admission check after materialization, separately from
``apply_parent_patch``. Historical patches still replay under their original
source-fidelity contract. Parsing does not import or execute candidate code.
Identical ASTs ignore source locations, comments and formatting; this narrow
check is not a proof of general semantic equivalence. Other languages, changed
file presence or modes, and unparseable Python remain explicitly unverified.
    """
    parent = MaterializedSourceTree.model_validate(store.json(parent_source_ref))
    candidate = MaterializedSourceTree.model_validate(store.json(candidate_source_ref))
    before = load_bundle(store, parent.bundle_ref)
    after = load_bundle(store, candidate.bundle_ref)
    paths = sorted(path for path in set(before.files) | set(after.files)
                   if before.files.get(path) != after.files.get(path))
    if not paths and parent.command == candidate.command:
        raise ParentEditError("Method revision has no changed source files or execution command")
    checks = {}
    for path in paths:
        original, revised = before.files.get(path), after.files.get(path)
        if original is None or revised is None:
            checks[path] = "file_presence_changed"
            continue
        if original.executable != revised.executable:
            checks[path] = "executable_mode_changed"
            continue
        if Path(path).suffix != ".py":
            checks[path] = "non_python_unverified"
            continue
        trees = []
        for entry in (original, revised):
            raw = store.read(entry.ref)
            if digest(raw) != entry.sha256 or len(raw) != entry.size:
                raise IntegrityError("Method admission source differs from its immutable manifest")
            try:
                tree = ast.parse(raw.decode("utf-8"), filename=path, type_comments=True)
                trees.append(ast.dump(tree, include_attributes=False))
            except (SyntaxError, UnicodeError, ValueError, RecursionError):
                checks[path] = "python_parse_unverified"
                break
        else:
            checks[path] = ("python_ast_unchanged" if trees[0] == trees[1]
                            else "python_ast_changed")
    if (parent.command == candidate.command and checks
            and all(value == "python_ast_unchanged" for value in checks.values())):
        raise ParentEditError(
            "Method revision changes only Python comments or formatting (identical parsed ASTs); "
            "route manuscript clarification to manuscript revision without retraining"
        )
    return {
        "schema_version": 1,
        "parent_source_ref": parent_source_ref,
        "candidate_source_ref": candidate_source_ref,
        "status": "python_ast_changed" if "python_ast_changed" in checks.values() else "unverified",
        "changed_paths": paths,
        "file_checks": checks,
        "command_changed": parent.command != candidate.command,
        "scope": "Static method admission only; scientific intent and semantic equivalence remain unverified",
    }
