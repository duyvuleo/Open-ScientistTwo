"""Standard-library process profiler, copied into an offline container.

Sampling starts before the workload process, including framework imports and model
loading. These are observed lower bounds, never exact allocator high-water marks.
"""

import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time


def process_tree_rss(root_pid, proc_root=Path("/proc")):
    records = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            records[int(entry.name)] = (int(fields[1]), int(fields[21]) * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    while True:
        additional = {pid for pid, (parent, _) in records.items() if parent in descendants}
        updated = descendants | additional
        if updated == descendants:
            break
        descendants = updated
    return sum(records[pid][1] for pid in descendants if pid in records), len(descendants & records.keys())


def cgroup_peak(memory_limit_bytes, cgroup=Path("/sys/fs/cgroup")):
    """Only use a cgroup peak when its limit matches this declared container."""
    for maximum, peak in [("memory.max", "memory.peak"),
                          ("memory/memory.limit_in_bytes", "memory/memory.max_usage_in_bytes")]:
        try:
            if int((cgroup / maximum).read_text()) == memory_limit_bytes:
                return int((cgroup / peak).read_text())
        except (OSError, ValueError):
            pass
    return None


def run(job, output):
    started = time.monotonic()
    records = []
    stopped = threading.Event()
    environment = os.environ.copy()
    environment["SCIENTISTTWO_PHASE_EVENTS"] = str(output / "phases.jsonl")
    environment["SCIENTISTTWO_SEED"] = str(job.get("seed", 0))
    # Framework preallocation is part of the requested configuration, not hidden.
    environment.update(job.get("environment", {}))
    child = subprocess.Popen(job["command"], cwd=job["cwd"], env=environment)

    def sample():
        while not stopped.is_set():
            try:
                rss, processes = process_tree_rss(child.pid)
                records.append({"elapsed_seconds": time.monotonic() - started,
                                "rss_bytes": rss, "process_count": processes})
            except OSError:
                pass
            stopped.wait(job["sample_interval_seconds"])

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    code = child.wait()
    stopped.set()
    thread.join()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = {
        "schema_version": 1,
        "returncode": code,
        "elapsed_seconds": time.monotonic() - started,
        "measurement_scope": "workload process tree, including startup and model loading",
        "sample_interval_seconds": job["sample_interval_seconds"],
        "sample_count": len(records),
        "sampled_process_tree_peak_rss_bytes": max((r["rss_bytes"] for r in records), default=None),
        "direct_children_high_water_rss_bytes": int(usage.ru_maxrss) * 1024,
        "container_cgroup_peak_bytes": cgroup_peak(job["memory_limit_bytes"]),
        "peak_semantics": "RSS samples are lower bounds and may miss short spikes; cgroup peak includes container processes when available",
        "samples": records,
    }
    (output / "process-memory.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    return code


if __name__ == "__main__":
    specification = json.loads(Path("/control/job.json").read_text())
    raise SystemExit(run(specification, Path("/output")))
