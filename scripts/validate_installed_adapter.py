"""Inspect a real installed entry point; no registry mocks or model execution."""

import argparse
from importlib.metadata import entry_points
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pandas as pd

from scientisttwo.live.adapters import load_development
from scientisttwo.live.models import DevelopmentSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = list(entry_points(group="scientisttwo.native_development", name="installed-test-csv"))
    if len(entries) != 1 or entries[0].dist.metadata["Name"] != "scientisttwo-test-adapter":
        raise ValueError("Install examples/test-adapter as a real distribution first")
    with tempfile.TemporaryDirectory() as temporary:
        public = Path(temporary)
        pd.DataFrame({"id": range(40), "x": range(40), "y": range(40)}).to_csv(public / "train.csv", index=False)
        native = SimpleNamespace(public_dir=public, benchmark="workspace", candidate_dir=None,
                                 task_id="installed-fixture", submission="submission.csv", command=[])
        config = DevelopmentSpec(adapter="installed-test-csv", target="y", prediction_columns=["y"],
                                 metric="mae", direction="minimize",
                                 settings={"problem_type": "regression", "features": {"numeric": ["x"]},
                                           "id_columns": ["id"], "submission_columns": ["id", "y"]})
        adapter = load_development(native, config)
        adapter.validate(adapter.baseline())
        identity = adapter.identity()
        if identity.get("plugin_distribution") != "scientisttwo-test-adapter":
            raise ValueError("Loaded adapter lacks installed distribution provenance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "pass", "entry_point": entries[0].value,
                                      "identity": identity, "live_model_calls": 0,
                                      "scope": "Real installed-package discovery and baseline contract; no empirical task score"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
