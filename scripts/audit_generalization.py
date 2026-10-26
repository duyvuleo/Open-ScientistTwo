"""Replay saved fixture evidence with its explicit synthetic test plugin; no execution."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "tests"))
    # This trusted fixture is the same plugin implementation used during validation.
    # It is not imported from a candidate or evidence directory.
    from test_workspace_task import install_plugin
    import pytest
    from scientisttwo.live.audit import audit_live
    from scientisttwo.live.adapters import implementation_identity
    from scientisttwo.live.models import LiveSpec
    from scientisttwo.store import Store

    checks = []
    with pytest.MonkeyPatch.context() as patch:
        install_plugin(patch)
        for case in sorted((args.output / "cases").iterdir()):
            campaign = case / "campaign"
            if not (campaign / "campaign.sqlite").is_file():
                continue
            store = Store(campaign)
            try:
                errors = store.verify()
                ref = store.get("live_spec_ref")
                if ref:
                    config = LiveSpec.model_validate(store.json(ref))
                    if config.implementation != implementation_identity():
                        errors.append("Current implementation differs from the validated source")
            finally:
                store.close()
            audit = audit_live(campaign) if ref and not errors else None
            checks.append(
                {
                    "case": case.name,
                    "artifact_errors": errors,
                    "development_audit": audit,
                    "status": "fail" if errors or (audit and audit["status"] != "pass") else "pass",
                }
            )
    result = {
        "status": "pass" if checks and all(c["status"] == "pass" for c in checks) else "fail",
        "scope": "Artifact hashes for all cases; score recomputation for four complete research campaigns using the recorded synthetic plugin",
        "live_model_calls": 0,
        "experiment_executions": 0,
        "cases": checks,
    }
    print(json.dumps(result, indent=2))
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
