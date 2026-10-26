"""Run no-LLM container validation and preserve evidence; never start/build a runtime."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--image", default="scientisttwo-native:0.1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = {
        **os.environ,
        "SCIENTISTTWO_NATIVE_CONTEXT": args.context,
        "SCIENTISTTWO_WORKSPACE_IMAGE": args.image,
        "SCIENTISTTWO_GENERALIZATION_OUTPUT": str(output / "cases"),
    }
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_workspace_task.py",
        "tests/test_workspace_container.py",
        "--junitxml",
        str(output / "tests.xml"),
    ]
    result = subprocess.run(command, cwd=project, env=environment, text=True, capture_output=True)
    (output / "pytest.log").write_text(result.stdout + result.stderr)
    findings = [
        json.loads(path.read_text())
        for path in sorted((output / "cases").glob("*/verification.json"))
    ]
    (output / "verification.json").write_text(
        json.dumps(
            {
                "status": "pass" if result.returncode == 0 else "fail",
                "scope": "Portable controller validation with synthetic agents/data and real container execution",
                "live_model_calls": 0,
                "context": args.context,
                "image": args.image,
                "cases": findings,
            },
            indent=2,
        )
        + "\n"
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
