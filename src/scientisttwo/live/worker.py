"""Task-independent execution bridge. Mounted read-only; never run on the host."""

import json
import os
from pathlib import Path
import subprocess


def main():
    job = json.loads(Path("/control/job.json").read_text())
    execution = job["execution"]
    environment = os.environ.copy()
    environment.update(execution.get("environment", {}))
    environment["SCIENTISTTWO_SEED"] = str(job["seed"])
    if execution["kind"] == "command":
        command = job["command"]
    elif execution["kind"] == "python_driver":
        command = ["python", "/control/driver.py"]
    else:
        raise ValueError("Unknown trusted execution descriptor")
    raise SystemExit(subprocess.run(command, cwd="/workspace", env=environment).returncode)


if __name__ == "__main__":
    main()
