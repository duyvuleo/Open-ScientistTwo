"""Resident child RPC driver, mounted read-only inside each isolated container.

The handler is trusted adapter code but loads candidate code inside this process.
It cannot access the external parent's timer, independent oracle, or other arm.
"""

import contextlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    output = sys.stdout

    def send(message):
        output.write(json.dumps(message, allow_nan=False) + "\n")
        output.flush()

    job = json.loads(Path("/control/timing_job.json").read_text())
    with contextlib.redirect_stdout(sys.stderr):
        spec = importlib.util.spec_from_file_location(
            "timing_handler", "/control/timing_handler.py"
        )
        handler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(handler)
        handler.initialize(job["parameters"])
        device = job.get("device", "cpu")
        if device not in {"cpu", "cuda"}:
            raise ValueError("Unknown timing device")
        if device == "cuda" and not callable(getattr(handler, "synchronize", None)):
            raise ValueError("CUDA handler requires explicit synchronization")
        synchronize = getattr(handler, "synchronize", lambda: None)
        synchronize()
    send({"type": "ready", "protocol": 1, "device": device})
    while True:
        line = sys.stdin.buffer.readline(65_537)
        if not line:
            break
        if len(line) > 65_536 or not line.endswith(b"\n"):
            raise ValueError("Oversized timing request")
        request = json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                synchronize()
                result = handler.invoke(request["payload"])
                synchronize()
            send({"request_id": request["request_id"], "status": "ok", "result": result})
        except Exception:
            # Detailed failures stay in container logs; no arbitrary model error strings enter RPC.
            send({"request_id": request.get("request_id"), "status": "failed"})
            raise


if __name__ == "__main__":
    main()
