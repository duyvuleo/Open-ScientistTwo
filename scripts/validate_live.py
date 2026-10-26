"""Save a reproducible fake-Codex/real-container smoke run. Never invokes a model.

Run from the source checkout: uv run python scripts/validate_live.py OUTPUT_DIR
The tests module supplies a visibly labeled recorded executable and known method.
"""

import argparse
import json
import os
from pathlib import Path
import sys

from scientisttwo.benchmarks.manifest import create_spec
from scientisttwo.live.adapters import implementation_identity, load_development
from scientisttwo.live.audit import audit_live
from scientisttwo.live.controller import LiveController
from scientisttwo.live.models import DevelopmentSpec, LiveSpec
from scientisttwo.live.runtime import Budget, DevelopmentRunner
from scientisttwo.store import Store

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "tests"))
from test_live_container import RecordedCodex, fake_codex, vector_candidate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", type=Path,
                        default=os.environ.get("SCIENTISTTWO_MLAGENTBENCH_ROOT"),
                        help="Pinned MLAgentBench checkout (default: .upstream/MLAgentBench)")
    parser.add_argument("--context", default=None, help="Optional Docker context")
    args = parser.parse_args()
    repository = Path(args.repository or project / ".upstream" / "MLAgentBench").resolve()
    if not repository.is_dir():
        parser.error("Prepare a pinned MLAgentBench checkout and pass --repository")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    native = create_spec(
        "mlagentbench",
        "vectorization",
        repository,
        context=args.context,
        seconds=120,
    )
    config = LiveSpec(
        native=native,
        development=DevelopmentSpec(epsilon=0.05),
        model="recorded-cli-no-inference",
        mode="smoke",
        max_calls=10,
        campaign_seconds=600,
        search_rounds=1,
        implementation=implementation_identity(),
    )
    (root / "manifest.json").write_text(config.model_dump_json(indent=2) + "\n")
    adapter = load_development(native, config.development)
    backend = RecordedCodex(
        root / "agent",
        config.model,
        executable=fake_codex(root / "fake-codex", vector_candidate(adapter.source)),
        allow_live=True,
    )
    store = Store(root / "campaign")
    try:
        budget = Budget(store, config)
        runner = DevelopmentRunner(store, config, adapter, budget)
        controller = LiveController(store, config, adapter, backend, runner, budget=budget)
        state = controller.run()
        before = budget.summary()
        assert controller.run() == state
        assert budget.summary() == before
        audit = audit_live(store.root)
        assert state["status"] == "complete" and audit["status"] == "pass", (state, audit)
        summary = store.json(store.get("live_summary_ref"))
        proof = {
            "state": state,
            "audit": audit,
            "resume_repeated_work": False,
            "synthetic_agents": True,
            "live_model_calls": 0,
            "model_attempts": summary["model_attempts"],
            "experiment_attempts": summary["experiment_attempts"],
            "images": store.get("live_images"),
        }
        (root / "verification.json").write_text(json.dumps(proof, indent=2) + "\n")
        print(json.dumps(proof, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
