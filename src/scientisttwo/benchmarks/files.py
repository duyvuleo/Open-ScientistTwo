"""Streaming dataset manifests and bounded code snapshots; no credentials or symlink escapes."""

import hashlib
from pathlib import Path
import shutil


FORBIDDEN = {
    ".env",
    "kaggle.json",
    "auth.json",
    "credentials",
    "credentials.json",
    "openai_api_key.txt",
    "claude_api_key.txt",
    "crfm_api_key.txt",
}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv"}
CODE_SUFFIXES = {".py", ".sh", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".ipynb"}


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_manifest(root):
    root = Path(root)
    if not root.is_dir():
        return {}
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Materialize input symlinks before freezing data: {relative}")
        if path.is_file():
            if path.name.lower() in FORBIDDEN or path.name.startswith(".env."):
                raise ValueError(
                    f"Credential-like file must not enter benchmark inputs: {relative}"
                )
            files[relative.as_posix()] = {"sha256": file_hash(path), "size": path.stat().st_size}
    return files


def verify_tree(root, expected):
    actual = tree_manifest(root)
    if actual != expected:
        raise ValueError(
            "Frozen input tree changed or was not prepared when the manifest was created"
        )


def native_identity():
    return {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def workspace(public, target, candidate=None):
    """Copy editable text code, link data to a separate read-only container mount."""
    public, target = Path(public), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    linked = set()
    for relative in tree_manifest(public):
        source, destination = public / relative, target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix in CODE_SUFFIXES and source.stat().st_size < 2_000_000:
            shutil.copy2(source, destination)
        else:
            destination.symlink_to("/data/public/" + relative)
            linked.add(relative)
    if candidate:
        for relative in tree_manifest(candidate):
            source, destination = Path(candidate) / relative, target / relative
            if relative in linked:
                raise ValueError("A candidate cannot replace a frozen input data file")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return linked


def artifact_manifest(root, input_links):
    root = Path(root)
    for relative in input_links:
        path = root / relative
        if not path.is_symlink() or str(path.readlink()) != "/data/public/" + relative:
            raise ValueError("Candidate modified or removed a protected input-data link")
    records = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(part in SKIP_DIRS for part in Path(relative).parts):
            continue
        if relative in input_links:
            if not path.is_symlink() or str(path.readlink()) != "/data/public/" + relative:
                raise ValueError("Candidate modified a protected input-data link")
            continue
        if path.is_symlink():
            raise ValueError("Unexpected output symlink")
        if path.is_file():
            if path.name.lower() in FORBIDDEN:
                raise ValueError("Credential-like artifact cannot be exported")
            records[relative] = {"sha256": file_hash(path), "size": path.stat().st_size}
    return records
