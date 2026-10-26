"""Typed, streaming artifact bundles shared by runners and checkpoint evaluators.

Only the small manifest is decoded in memory. Payloads are never imported, unpickled,
or passed through Store.read; copying and integrity verification are streaming.
"""

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Literal

from pydantic import Field, model_validator

from .benchmarks.files import FORBIDDEN
from .benchmarks.models import relative_path
from .models import Record
from .store import IntegrityError, Store, file_digest


def safe_path(value: str) -> str:
    """Require canonical portable paths, excluding credential-bearing names."""
    normalized = relative_path(value)
    if (
        normalized != value
        or "\x00" in value
        or any(
            part.startswith(".") or part.lower() in FORBIDDEN for part in PurePosixPath(value).parts
        )
    ):
        raise ValueError(f"Unsafe or noncanonical bundle path: {value}")
    return value


class ArtifactSpec(Record):
    path: str
    kind: Literal["file", "directory"] = "file"
    role: str = Field(default="prediction", min_length=1, max_length=128)
    required: bool = True
    max_bytes: int = Field(default=1_000_000_000, ge=0)
    max_files: int = Field(default=10_000, ge=1, le=1_000_000)

    @model_validator(mode="after")
    def valid(self):
        safe_path(self.path)
        return self


class ArtifactFile(Record):
    ref: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size: int = Field(ge=0)
    executable: bool = False


class BundleManifest(Record):
    schema_version: Literal[1] = 1
    specs: list[ArtifactSpec] = Field(max_length=100_000)
    files: dict[str, ArtifactFile] = {}
    directories: list[str] = []
    missing: list[str] = []

    @model_validator(mode="after")
    def valid(self):
        _validate_specs(self.specs)
        specs = {spec.path: spec for spec in self.specs}
        counts = dict.fromkeys(specs, 0)
        sizes = dict.fromkeys(specs, 0)
        for path in [*self.files, *self.directories, *self.missing]:
            safe_path(path)
        if len(set(self.directories)) != len(self.directories):
            raise ValueError("Duplicate bundle directories")
        if len(set(self.missing)) != len(self.missing):
            raise ValueError("Duplicate missing artifacts")
        if set(self.files) & set(self.directories):
            raise ValueError("Bundle file/directory collision")
        for path in [*self.files, *self.directories]:
            if any(str(p) in self.files for p in PurePosixPath(path).parents):
                raise ValueError("A bundle file cannot contain another artifact")
            owners = [
                specs[str(parent)]
                for parent in [PurePosixPath(path), *PurePosixPath(path).parents]
                if str(parent) in specs and _contains(specs[str(parent)], path)
            ]
            if len(owners) != 1:
                raise ValueError("Every bundle entry must have exactly one artifact contract")
            owner = owners[0]
            if owner.path in self.missing:
                raise ValueError("Missing artifact contains entries")
            if path in self.files:
                counts[owner.path] += 1
                sizes[owner.path] += self.files[path].size
        for path in self.missing:
            if path not in specs or specs[path].required:
                raise ValueError("Only declared optional artifacts can be missing")
        for spec in self.specs:
            present = spec.path in (self.files if spec.kind == "file" else self.directories)
            if present == (spec.path in self.missing):
                raise ValueError("Artifact must be present or explicitly absent")
            if spec.kind == "file" and counts[spec.path] > 1:
                raise ValueError("File contract contains multiple payloads")
            if counts[spec.path] > spec.max_files or sizes[spec.path] > spec.max_bytes:
                raise ValueError("Bundle exceeds declared artifact quotas")
        return self


def _contains(spec: ArtifactSpec, path: str) -> bool:
    return path == spec.path or (spec.kind == "directory" and path.startswith(spec.path + "/"))


def _validate_specs(specs):
    paths = [spec.path for spec in specs]
    path_set = set(paths)
    if len(paths) != len(path_set):
        raise ValueError("Duplicate artifact contracts")
    for path in paths:
        if any(str(parent) in path_set for parent in PurePosixPath(path).parents):
            raise ValueError("Artifact contracts cannot overlap")


def _checked_path(root: Path, relative: str) -> Path:
    safe_path(relative)
    current = root
    for part in PurePosixPath(relative).parts:
        current /= part
        if current.is_symlink():
            raise IntegrityError("Artifact paths must not contain symlinks")
    if not current.resolve().is_relative_to(root.resolve()):
        raise IntegrityError("Artifact path escaped its root")
    return current


def _identity(path):
    info = path.stat(follow_symlinks=False)
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def verified_file(store: Store, artifact: ArtifactFile) -> Path:
    """Verify registration, size and digest without loading payloads into RAM."""
    row = store.db.execute("SELECT * FROM artifacts WHERE path=?", (artifact.ref,)).fetchone()
    try:
        path = _checked_path(store.root, artifact.ref)
    except (ValueError, OSError) as exc:
        raise IntegrityError("Unsafe artifact reference") from exc
    if (row is not None and artifact.ref.startswith("artifacts/")
            and row["sha"] == artifact.sha256 and row["size"] == artifact.size
            and not path.exists()):
        store.materialize_artifact(artifact.ref)
        # An overlay hook cannot exempt the resulting path from normal checks.
        path = _checked_path(store.root, artifact.ref)
    if (
        row is None
        or not artifact.ref.startswith("artifacts/")
        or not path.is_file()
        or row["sha"] != artifact.sha256
        or row["size"] != artifact.size
        or path.stat().st_size != artifact.size
        or file_digest(path) != artifact.sha256
    ):
        raise IntegrityError(f"Missing or modified bundle artifact: {artifact.ref}")
    return path


def freeze_bundle(store: Store, work: Path, specs: list[ArtifactSpec]) -> str:
    """Freeze declared outputs, rejecting escapes, special files and quota excesses."""
    work = Path(work)
    if work.is_symlink() or not work.is_dir():
        raise IntegrityError("Bundle workspace must be a real directory")
    specs = [ArtifactSpec.model_validate(spec) for spec in specs]
    _validate_specs(specs)
    files, directories, missing, snapshots = {}, [], [], {}
    for spec in specs:
        target = _checked_path(work, spec.path)
        if not target.exists():
            if spec.required:
                raise IntegrityError(f"Required artifact missing: {spec.path}")
            missing.append(spec.path)
            continue
        mode = target.stat(follow_symlinks=False).st_mode
        if (spec.kind == "file" and not stat.S_ISREG(mode)) or (
            spec.kind == "directory" and not stat.S_ISDIR(mode)
        ):
            raise IntegrityError(f"Wrong artifact kind: {spec.path}")
        entries = [target]
        if spec.kind == "directory":
            entries.extend(sorted(target.rglob("*")))
        members, byte_count = [], 0
        for path in entries:
            relative = path.relative_to(work).as_posix()
            _checked_path(work, relative)
            info = path.stat(follow_symlinks=False)
            snapshots[relative] = _identity(path)
            if stat.S_ISDIR(info.st_mode):
                directories.append(relative)
            elif stat.S_ISREG(info.st_mode):
                byte_count += info.st_size
                members.append((relative, path, info))
                if len(members) > spec.max_files or byte_count > spec.max_bytes:
                    raise IntegrityError(f"Artifact quota exceeded: {spec.path}")
            else:
                raise IntegrityError(f"Special files are forbidden in bundles: {relative}")
        for relative, path, info in members:
            ref = store.put_file(path, kind=spec.role)
            row = store.db.execute("SELECT sha,size FROM artifacts WHERE path=?", (ref,)).fetchone()
            record = ArtifactFile(
                ref=ref,
                sha256=row["sha"],
                size=info.st_size,
                executable=bool(info.st_mode & 0o111),
            )
            if _identity(path) != snapshots[relative] or row["size"] != info.st_size:
                raise IntegrityError("Artifact changed while its bundle was frozen")
            verified_file(store, record)
            files[relative] = record
    # Directory identities catch additions/removals while individual files were copied.
    for relative, expected in snapshots.items():
        if _identity(_checked_path(work, relative)) != expected:
            raise IntegrityError("Artifact tree changed while its bundle was frozen")
    manifest = BundleManifest(
        specs=specs, files=files, directories=sorted(directories), missing=sorted(missing)
    )
    return store.put_json(manifest, "artifact_bundle")


def load_bundle(store: Store, manifest_ref: str, *, verify: bool = True) -> BundleManifest:
    row = store.db.execute("SELECT size FROM artifacts WHERE path=?", (manifest_ref,)).fetchone()
    if row is None or row["size"] > 64_000_000:
        raise IntegrityError("Bundle manifest is unregistered or oversized")
    manifest = BundleManifest.model_validate(store.json(manifest_ref))
    if verify:
        for artifact in manifest.files.values():
            verified_file(store, artifact)
    return manifest


def restore_bundle(store: Store, manifest_ref: str, destination: Path) -> BundleManifest:
    """Restore atomically into a new or empty directory; never merge stale outputs."""
    manifest = load_bundle(store, manifest_ref)
    destination = Path(destination).absolute()
    if any(path.is_symlink() for path in [destination, *destination.parents]):
        raise IntegrityError("Restore destination must not contain symlinks")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise IntegrityError("Restore destination must be absent or empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="bundle-", dir=destination.parent))
    try:
        for relative in manifest.directories:
            (temporary / relative).mkdir(parents=True, exist_ok=True)
        for relative, artifact in manifest.files.items():
            source = verified_file(store, artifact)
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum = hashlib.sha256()
            with source.open("rb") as reader, target.open("xb") as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    checksum.update(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            if checksum.hexdigest() != artifact.sha256 or target.stat().st_size != artifact.size:
                raise IntegrityError("Artifact changed during streaming restoration")
            target.chmod(0o755 if artifact.executable else 0o644)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest
