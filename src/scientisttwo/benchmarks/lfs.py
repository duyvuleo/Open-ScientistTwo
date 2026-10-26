"""Read-only retrieval of pinned public Git LFS objects, verified against their pointers."""

import hashlib
from pathlib import Path
import re
import subprocess
import urllib.request


def parse_pointer(data):
    if not data.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        return None
    match = re.search(rb"oid sha256:([a-f0-9]{64})\nsize ([0-9]+)", data)
    if not match:
        raise ValueError("Malformed Git LFS pointer")
    return {"sha256": match[1].decode(), "size": int(match[2])}


def pointer_at_head(repo, relative):
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"HEAD:{relative}"], capture_output=True
    )
    return parse_pointer(result.stdout) if result.returncode == 0 else None


def matches_pointer(path, pointer):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != pointer["size"]:
        return False
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == pointer["sha256"]


def materialize_leaderboard(repo, revision, task_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", task_id) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Invalid task or revision identifier")
    relative = f"mlebench/competitions/{task_id}/leaderboard.csv"
    pointer = pointer_at_head(repo, relative)
    if pointer is None:
        raise ValueError("Requested leaderboard is not an upstream LFS object")
    target = Path(repo) / relative
    if matches_pointer(target, pointer):
        return {"path": str(target), "sha256": pointer["sha256"], "cached": True}
    if parse_pointer(target.read_bytes()) != pointer:
        raise ValueError("Refusing to replace a locally modified leaderboard")
    if pointer["size"] > 20_000_000:
        raise ValueError("Leaderboard exceeds the bounded metadata download size")
    url = f"https://media.githubusercontent.com/media/openai/mle-bench/{revision}/{relative}"
    temporary = target.with_suffix(".download")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > pointer["size"]:
                    raise ValueError("LFS response exceeds its declared size")
                output.write(chunk)
        if not matches_pointer(temporary, pointer):
            raise ValueError("LFS object hash verification failed")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(target), "sha256": pointer["sha256"], "cached": False}
