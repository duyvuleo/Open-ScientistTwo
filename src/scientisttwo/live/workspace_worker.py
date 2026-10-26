"""Generic argv bridge, executed only inside a restricted container."""

import json
import os
from pathlib import Path
import subprocess


def main():
    job = json.loads(Path("/control/job.json").read_text())
    environment = os.environ.copy()
    environment["SCIENTISTTWO_SEED"] = str(job["seed"])
    raise SystemExit(subprocess.run(job["command"], cwd=job["cwd"], env=environment).returncode)


if __name__ == "__main__":
    main()
