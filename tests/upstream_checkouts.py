"""Portable optional upstream fixtures; required CI coverage fails closed.

Set SCIENTISTTWO_REQUIRE_UPSTREAMS to ``all`` or a comma-separated selection
of paperorchestra, mlagentbench and mle-bench to forbid missing-checkout skips.
Source revision verification remains the responsibility of each integration.
"""

import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKOUTS = {
    "paperorchestra": ("paper-orchestra", "SCIENTISTTWO_PAPERORCHESTRA_ROOT", None),
    "mlagentbench": ("MLAgentBench", "SCIENTISTTWO_MLAGENTBENCH_ROOT", "SCIENTISTTWO_MLAB_REPO"),
    "mle-bench": ("mle-bench", "SCIENTISTTWO_MLEBENCH_ROOT", "SCIENTISTTWO_MLE_REPO"),
}


def upstream_path(name):
    """Resolve an explicit path, ignored .upstream checkout, or public sibling."""
    folder, variable, legacy = CHECKOUTS[name]
    configured = os.environ.get(variable) or (os.environ.get(legacy) if legacy else None)
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_dir():
            pytest.fail(f"Configured {variable} is not a directory: {path}", pytrace=False)
        return path
    candidates = [PROJECT_ROOT / ".upstream" / folder, PROJECT_ROOT.parent / folder]
    if name == "paperorchestra":
        candidates.append(PROJECT_ROOT / "src" / "scientisttwo" / "_vendor" / "paperorchestra")
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    required = {item.strip().lower() for item in os.environ.get(
        "SCIENTISTTWO_REQUIRE_UPSTREAMS", "").split(",") if item.strip()}
    unknown = required - set(CHECKOUTS) - {"1", "true", "all"}
    if unknown:
        pytest.fail("Unknown required upstream selection: " + ", ".join(sorted(unknown)), pytrace=False)
    if name in required or required & {"1", "true", "all"}:
        pytest.fail(f"Required upstream {name} is missing; configure {variable} or .upstream/{folder}",
                    pytrace=False)
    return candidates[0]


def require_upstream(name):
    path = upstream_path(name)
    if not path.is_dir():
        pytest.skip(f"Optional pinned {name} checkout unavailable; set {CHECKOUTS[name][1]}")
    return path
