"""Container entry point. No labels for the evaluation rows are mounted here."""

import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np


def main():
    root = Path("/input")
    manifest = json.loads((root / "manifest.json").read_text())
    spec = importlib.util.spec_from_file_location("candidate", "/candidate/candidate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictions = {}
    for entry in manifest["cells"]:
        arrays = np.load(root / entry["file"], allow_pickle=False)
        estimator = module.build(entry["seed"])
        started = time.monotonic()
        estimator.fit(arrays["x_train"], arrays["y_train"])
        training = time.monotonic() - started
        started = time.monotonic()
        values = estimator.predict(arrays["x_eval"])
        inference = time.monotonic() - started
        predictions[entry["key"]] = {
            "row_ids": arrays["row_ids"].tolist(),
            "predictions": values.tolist(),
            "training_seconds": training,
            "inference_seconds": inference,
        }
    # Completion marker is written last and atomically: interrupted runs cannot look complete.
    output = Path("/output/observations.json")
    temp = output.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump(predictions, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, output)
    marker = Path("/output/completed.tmp")
    marker.write_text(json.dumps({"request_hash": manifest["request_hash"]}))
    os.replace(marker, Path("/output/completed.json"))


if __name__ == "__main__":
    main()
