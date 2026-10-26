"""Immutable full source trees plus bounded agent edits, without executing source.

Candidate.files describes edits when base_snapshot_ref is present. The materialized
tree has a separate evidence record, so large starters never bypass proposal limits.
"""

from pathlib import Path
import stat
from typing import Literal

from pydantic import Field, model_validator

from ..artifacts import (
    ArtifactFile,
    ArtifactSpec,
    BundleManifest,
    freeze_bundle,
    load_bundle,
    restore_bundle,
    safe_path,
)
from ..models import Record
from ..store import IntegrityError, Store, digest
from .models import Candidate


class SourceSnapshot(Record):
    schema_version: Literal[1] = 1
    bundle_ref: str
    editable_paths: list[str]
    max_files: int = Field(default=10_000, ge=1)
    max_bytes: int = Field(default=50_000_000, ge=1)

    @model_validator(mode="after")
    def valid(self):
        for path in self.editable_paths:
            safe_path(path)
        if len(set(self.editable_paths)) != len(self.editable_paths):
            raise ValueError("Editable source paths must be distinct")
        return self


class MaterializedSourceTree(Record):
    schema_version: Literal[1] = 1
    bundle_ref: str
    base_snapshot_ref: str | None
    proposal_ref: str
    command: list[str]
    changed_paths: list[str]


def _within_limits(bundle, snapshot):
    if (
        len(bundle.files) > snapshot.max_files
        or sum(f.size for f in bundle.files.values()) > snapshot.max_bytes
    ):
        raise IntegrityError("Source tree exceeds its declared limits")


def freeze_source_tree(
    store: Store,
    root: Path,
    editable_paths: list[str],
    *,
    max_files: int = 10_000,
    max_bytes: int = 50_000_000,
) -> str:
    """Freeze an already isolated source tree; editable paths may include new files."""
    root = Path(root)
    snapshot = SourceSnapshot(
        bundle_ref="pending",
        editable_paths=editable_paths,
        max_files=max_files,
        max_bytes=max_bytes,
    )
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError("Source root must be a real directory")
    original_root = root.stat()
    specs = []
    total_bytes, total_files = 0, 0
    # Check aggregate quotas before copying a large tree into the store.
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        safe_path(relative)
        if path.is_symlink():
            raise IntegrityError("Source tree symlinks are forbidden")
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISREG(mode):
            total_files += 1
            total_bytes += path.stat().st_size
        elif not stat.S_ISDIR(mode):
            raise IntegrityError("Source tree special files are forbidden")
        if total_files > max_files or total_bytes > max_bytes:
            raise IntegrityError("Source tree exceeds its declared limits")
    for path in sorted(root.iterdir()):
        specs.append(
            ArtifactSpec(
                path=path.name,
                kind="directory" if path.is_dir() else "file",
                role="source",
                max_files=max_files,
                max_bytes=max_bytes,
            )
        )
    snapshot.bundle_ref = freeze_bundle(store, root, specs)
    current_root = root.stat()
    if (original_root.st_ino, original_root.st_mtime_ns, original_root.st_ctime_ns) != (
        current_root.st_ino,
        current_root.st_mtime_ns,
        current_root.st_ctime_ns,
    ):
        raise IntegrityError("Source root changed while its snapshot was frozen")
    _within_limits(load_bundle(store, snapshot.bundle_ref), snapshot)
    return store.put_json(snapshot, "source_snapshot")


def load_source_snapshot(store: Store, ref: str) -> SourceSnapshot:
    snapshot = SourceSnapshot.model_validate(store.json(ref))
    _within_limits(load_bundle(store, snapshot.bundle_ref), snapshot)
    return snapshot


def materialize_source_tree(
    store: Store,
    candidate: Candidate,
    *,
    expected_base_ref: str | None = None,
) -> MaterializedSourceTree:
    """Apply bounded complete-file replacements to a verified base source tree."""
    # Revalidate copies created with model_copy(update=...), which bypass validators.
    candidate = Candidate.model_validate(candidate.model_dump())
    if candidate.base_snapshot_ref is None:
        raise ValueError("Source overlays require base_snapshot_ref")
    if expected_base_ref is not None and candidate.base_snapshot_ref != expected_base_ref:
        raise IntegrityError("Candidate changed the authorized source snapshot")
    snapshot = load_source_snapshot(store, candidate.base_snapshot_ref)
    base = load_bundle(store, snapshot.bundle_ref)
    files = dict(base.files)
    directories = set(base.directories)
    changed = []
    for edit in candidate.files:
        if edit.path not in snapshot.editable_paths:
            raise IntegrityError(f"Source path is not declared editable: {edit.path}")
        if edit.path in directories or any(str(p) in files for p in Path(edit.path).parents):
            raise IntegrityError("Source edit conflicts with a file or directory")
        content = edit.content.encode("utf-8")
        old = files.get(edit.path)
        if old is not None and old.sha256 == digest(content):
            continue
        files[edit.path] = ArtifactFile(
            ref=store.put(
                content, suffix=Path(edit.path).suffix.lstrip(".") or "txt", kind="source"
            ),
            sha256=digest(content),
            size=len(content),
            executable=old.executable if old else False,
        )
        for parent in Path(edit.path).parents:
            if str(parent) != ".":
                directories.add(parent.as_posix())
        changed.append(edit.path)
    # Rebuild nonoverlapping top-level contracts, including explicitly allowed additions.
    top_levels = sorted({path.split("/")[0] for path in [*files, *directories]})
    specs = [
        ArtifactSpec(
            path=path,
            kind="directory" if path in directories else "file",
            role="source",
            max_files=snapshot.max_files,
            max_bytes=snapshot.max_bytes,
        )
        for path in top_levels
    ]
    bundle = BundleManifest(specs=specs, files=files, directories=sorted(directories))
    _within_limits(bundle, snapshot)
    return MaterializedSourceTree(
        bundle_ref=store.put_json(bundle, "materialized_source_bundle"),
        base_snapshot_ref=candidate.base_snapshot_ref,
        proposal_ref=store.put_json(candidate, "source_overlay_proposal"),
        command=candidate.command,
        changed_paths=sorted(changed),
    )


def restore_source_tree(store: Store, tree: MaterializedSourceTree | str, destination: Path):
    """Restore a materialized record or its stored reference, verifying its derivation."""
    if isinstance(tree, str):
        tree = MaterializedSourceTree.model_validate(store.json(tree))
    proposal = Candidate.model_validate(store.json(tree.proposal_ref))
    rebuilt = candidate_snapshot(store, proposal)
    if rebuilt != tree:
        raise IntegrityError("Materialized source does not match its proposal and frozen base")
    return restore_bundle(store, tree.bundle_ref, destination)


def compare_source_trees(store: Store, left_ref: str, right_ref: str) -> list[str]:
    """Compare materialized bytes and executable permissions, for ablation checks."""

    def files(ref):
        value = store.json(ref)
        tree = MaterializedSourceTree.model_validate(value)
        proposal = Candidate.model_validate(store.json(tree.proposal_ref))
        if candidate_snapshot(store, proposal) != tree:
            raise IntegrityError("Materialized source evidence changed")
        return load_bundle(store, tree.bundle_ref).files

    left, right = files(left_ref), files(right_ref)
    return sorted(path for path in set(left) | set(right) if left.get(path) != right.get(path))


def candidate_snapshot(store: Store, candidate: Candidate) -> MaterializedSourceTree:
    """Uniform materialized evidence for legacy candidates and frozen-tree overlays."""
    candidate = Candidate.model_validate(candidate.model_dump())
    if candidate.base_snapshot_ref is not None:
        return materialize_source_tree(store, candidate)
    files, directories = {}, set()
    for source in candidate.files:
        content = source.content.encode("utf-8")
        files[source.path] = ArtifactFile(
            ref=store.put(
                content, suffix=Path(source.path).suffix.lstrip(".") or "txt", kind="source"
            ),
            sha256=digest(content),
            size=len(content),
        )
        directories.update(p.as_posix() for p in Path(source.path).parents if str(p) != ".")
    top_levels = sorted({path.split("/")[0] for path in [*files, *directories]})
    specs = [
        ArtifactSpec(
            path=path,
            kind="directory" if path in directories else "file",
            role="source",
            max_files=24,
            max_bytes=500_000,
        )
        for path in top_levels
    ]
    bundle = BundleManifest(specs=specs, files=files, directories=sorted(directories))
    return MaterializedSourceTree(
        bundle_ref=store.put_json(bundle, "materialized_source_bundle"),
        base_snapshot_ref=None,
        proposal_ref=store.put_json(candidate, "source_proposal"),
        command=candidate.command,
        changed_paths=sorted(files),
    )


def build_source(store: Store, candidate: Candidate, destination: Path) -> MaterializedSourceTree:
    """Materialize either proposal form and restore only verified source bytes."""
    tree = candidate_snapshot(store, candidate)
    restore_source_tree(store, tree, destination)
    return tree
