"""Container-only entry point to unchanged official grading code. No model SDK calls."""

import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    job = json.loads(Path("/control/job.json").read_text())
    os.environ["SCIENTISTTWO_SEED"] = str(job.get("seed", 0))
    if job["stage"] == "execute":
        if job.get("sample_submission"):
            destination = Path("/workspace") / job["submission"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path("/data/public") / job["sample_submission"], destination)
            return
        result = subprocess.run(job["command"], cwd="/workspace")
        raise SystemExit(result.returncode)
    sys.path.insert(0, "/upstream")
    if job["benchmark"] == "mle-bench":
        from mlebench.registry import Registry
        from mlebench.grade import grade_csv

        competition = Registry(Path("/data")).get_competition(job["task_id"])
        report = grade_csv(Path("/submission") / job["submission"], competition).to_dict()
        if report["score"] is not None and not math.isfinite(report["score"]):
            report["score"] = None
            report["valid_submission"] = False
        payload = {
            "official_report": report,
            "score": report["score"],
            "valid_submission": report["valid_submission"],
            "direction": "minimize" if report["is_lower_better"] else "maximize",
        }
    else:
        os.chdir("/task/scripts")
        sys.path.insert(0, "/task/env")
        spec = importlib.util.spec_from_file_location(
            "official_mlagentbench_eval", "/task/scripts/eval.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        score = float(module.get_score("/submission"))
        required = job["required_artifacts"]
        valid = math.isfinite(score) and all((Path("/submission") / p).exists() for p in required)
        payload = {
            "official_report": {
                "get_score": score if math.isfinite(score) else None,
                "entry_point": "scripts/eval.py:get_score",
            },
            "score": score if valid else None,
            "valid_submission": valid,
            "direction": job["direction"],
        }
    out = Path("/grade-output/grade.json")
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False))
    temporary.replace(out)


if __name__ == "__main__":
    main()
