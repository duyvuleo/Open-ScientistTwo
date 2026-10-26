"""Fetch one hash-verified public leaderboard required by integration fixtures.

This is an explicit network preparation step. It never fetches benchmark data,
models, credentials, or other Git LFS objects. Runtime execution stays offline.
"""

import argparse
import json
from pathlib import Path

from scientisttwo.benchmarks.lfs import (
    materialize_leaderboard,
    matches_pointer,
    pointer_at_head,
)
from scientisttwo.benchmarks.registry import PINS, verify_repository


TASK = "detecting-insults-in-social-commentary"
RELATIVE_PATH = f"mlebench/competitions/{TASK}/leaderboard.csv"
MAX_METADATA_BYTES = 1_000_000


def prepare(root):
    root = Path(root).resolve()
    revision = PINS["mle-bench"]["revision"]
    verify_repository(root, revision)
    target = root / RELATIVE_PATH
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError("Leaderboard path must stay inside the pinned source checkout")
    pointer = pointer_at_head(root, RELATIVE_PATH)
    if pointer is None or not 0 < pointer["size"] <= MAX_METADATA_BYTES:
        raise ValueError("Expected a bounded tracked leaderboard LFS pointer")
    # The existing downloader uses only official GitHub media at this exact
    # revision and rejects oversized responses, SHA-256 mismatches, and edits.
    result = materialize_leaderboard(root, revision, TASK)
    if not matches_pointer(target, pointer):
        raise ValueError("Downloaded metadata does not match the tracked SHA-256 and size")
    verify_repository(root, revision)
    return {"status": "pass", "repository": PINS["mle-bench"]["url"],
            "revision": revision, "file": RELATIVE_PATH, "bytes": pointer["size"],
            "sha256": pointer["sha256"], "cached": result["cached"],
            "scope": "One public leaderboard metadata file; no benchmark data or models"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlebench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.mlebench_root)
    except (OSError, ValueError):
        # Network exceptions may contain redirect URLs; never echo them.
        parser.exit(1, "Pinned public leaderboard preparation or integrity verification failed\n")
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
