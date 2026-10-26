"""Resumable local process queue for independent offline campaigns."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .audit import audit_campaign
from .backends import FixtureBackend
from .batch_report import atomic_json, summarize, read_receipt
from .controller import Controller, initialize
from .process import clean_environment
from .runner import ContainerRunner
from .store import Store, IntegrityError, canonical, digest
from .suite import BatchManifest, implementation_identity
from .tasks import load_task_adapter


def read_manifest(path):
    return BatchManifest.model_validate_json(Path(path).read_text())


def validate_manifest(manifest):
    if manifest.implementation != implementation_identity():
        raise IntegrityError(
            "Implementation changed after manifest creation; create a new versioned batch manifest"
        )
    for entry in manifest.campaigns:
        adapter = load_task_adapter(entry.task.adapter, entry.task)
        if digest(canonical(adapter.split_manifest(entry.task))) != entry.split_manifest_hash:
            raise IntegrityError(f"Dataset or partition changed: {entry.id}")


def campaign_worker(manifest_path, root, campaign_id):
    from .cli import locked

    root = Path(root).resolve()
    manifest = read_manifest(manifest_path)
    entry = next(c for c in manifest.campaigns if c.id == campaign_id)
    metadata = json.loads((root / "batch-state.json").read_text())
    run_dir = root / "campaigns" / campaign_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with locked(run_dir):
        store = Store(run_dir)
        started = time.time()
        try:
            if manifest.implementation != implementation_identity():
                raise IntegrityError("Implementation changed while batch was executing")
            adapter = load_task_adapter(entry.task.adapter, entry.task)
            if store.get("state") is None:
                initialize(store, entry.task, manifest.profile, adapter)
                store.set(
                    "runtime",
                    {
                        "runtime": manifest.runtime,
                        "context": manifest.context,
                        "image": metadata["image_id"],
                    },
                )
                store.set("batch_entry", entry.model_dump(mode="json"))
                store.set(
                    "deadline",
                    min(started + manifest.profile.campaign_seconds, metadata["deadline"]),
                )
                store.set("batch_deadline", metadata["deadline"])
            if store.get("batch_entry") != entry.model_dump(mode="json"):
                raise IntegrityError("Campaign assignment differs from its original manifest")
            runner = ContainerRunner(
                store, manifest.profile, **store.get("runtime"), adapter=adapter
            )
            controller = Controller(
                store,
                entry.task,
                manifest.profile,
                adapter,
                FixtureBackend(adapter.fixture_response),
                runner,
            )
            state = controller.run()
            if state.status != "complete":
                result = {"status": state.status, "error": state.reason}
            else:
                audit = audit_campaign(
                    store,
                    adapter,
                    runner,
                    include_final=True,
                    screening_sample_size=manifest.screening_sample_size,
                )
                checks = [
                    audit[k]
                    for k in (
                        "artifact_integrity",
                        "partition_isolation",
                        "specification_compliance",
                        "score_recalculation",
                        "reexecution",
                    )
                ]
                if store.get("freeze_ref"):
                    checks.append(audit["final_reexecution"])
                for optional in ("manuscript_claims", "method_code_alignment"):
                    if audit[optional] == "fail":
                        checks.append("fail")
                checks.extend(
                    "pass" if item["result"]["status"] == "success" else "fail"
                    for item in audit["screening_checks"]
                )
                summary = store.json(store.get("summary_ref"))
                jobs = [
                    dict(row) for row in store.db.execute("SELECT key,status,charged FROM jobs")
                ]
                split_ref = store.get("split_manifest_ref")
                observed_hash = digest(store.read(split_ref))
                if observed_hash != entry.split_manifest_hash:
                    raise IntegrityError("Campaign completed under the wrong partitions")
                result = {
                    "status": "complete",
                    "outcome": state.outcome,
                    "audit_pass": all(c == "pass" for c in checks),
                    "audit_ref": audit["audit_ref"],
                    "final_delta": store.get("final_comparison", {}).get("delta"),
                    "observed_split_hash": observed_hash,
                    "worker_image_id": store.get("worker_image_id"),
                    "experiment_jobs": sum(not j["key"].startswith("audit/") for j in jobs),
                    "audit_jobs": sum(j["key"].startswith("audit/") for j in jobs),
                    "job_statuses": {
                        status: sum(j["status"] == status for j in jobs)
                        for status in {j["status"] for j in jobs}
                    },
                    "execution_seconds": sum(j["charged"] for j in jobs),
                    "live_model_calls": summary["live_calls"],
                    "screening_audited": len(audit["screening_checks"]),
                    "screening_missed_development_improvements": sum(
                        item["comparison"]["improved"] for item in audit["screening_checks"]
                    ),
                    "stages": [
                        row[0] for row in store.db.execute("SELECT DISTINCT stage FROM events")
                    ],
                    "elapsed_seconds": time.time() - started,
                    "feasibility_rejections": store.db.execute(
                        "SELECT count(*) FROM events WHERE reason LIKE '%Hard feasibility%'"
                    ).fetchone()[0],
                }
            store.set("batch_result_ref", store.put_json(result, "batch-campaign-result"))
            atomic_json(run_dir / "batch-result.json", result)
            return 0 if result["status"] == "complete" and result["audit_pass"] else 1
        except KeyboardInterrupt:
            atomic_json(
                run_dir / "batch-result.json",
                {"status": "interrupted", "error": "Batch worker interrupted"},
            )
            return 130
        except Exception as error:
            result = {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.time() - started,
            }
            store.set("batch_result_ref", store.put_json(result, "batch-campaign-result"))
            atomic_json(run_dir / "batch-result.json", result)
            return 1
        finally:
            store.close()


def run_batch(manifest_path, root, *, max_campaigns=None, retry_errors=False):
    from .cli import locked

    manifest = read_manifest(manifest_path)
    validate_manifest(manifest)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with locked(root):
        frozen = root / "manifest.json"
        if frozen.exists() and frozen.read_bytes() != canonical(manifest):
            raise IntegrityError("Batch directory belongs to a different manifest")
        if not frozen.exists():
            frozen.write_bytes(canonical(manifest))
        state_path = root / "batch-state.json"
        if not state_path.exists():
            probe = Store(root / "preflight")
            try:
                entry = manifest.campaigns[0]
                runner = ContainerRunner(
                    probe,
                    manifest.profile,
                    runtime=manifest.runtime,
                    context=manifest.context,
                    image=manifest.image,
                    adapter=load_task_adapter(entry.task.adapter, entry.task),
                )
                runner.preflight()
                atomic_json(
                    state_path,
                    {
                        "manifest_hash": digest(canonical(manifest)),
                        "image_id": runner.image_id,
                        "started_at": time.time(),
                        "deadline": time.time() + manifest.wall_seconds,
                    },
                )
            finally:
                probe.close()
        metadata = json.loads(state_path.read_text())
        if metadata["manifest_hash"] != digest(canonical(manifest)):
            raise IntegrityError("Batch state does not match its manifest")
        pending = []
        for entry in manifest.campaigns:
            status = read_receipt(root, entry)["status"]
            if (
                status == "pending"
                or status in {"interrupted", "blocked"}
                or (status == "error" and retry_errors)
            ):
                pending.append(entry)
        if max_campaigns is not None:
            if max_campaigns < 1:
                raise ValueError("max_campaigns must be positive")
            pending = pending[:max_campaigns]
        running = {}
        try:
            while pending or running:
                while (
                    pending
                    and len(running) < manifest.workers
                    and time.time() < metadata["deadline"]
                ):
                    entry = pending.pop(0)
                    run_dir = root / "campaigns" / entry.id
                    run_dir.mkdir(parents=True, exist_ok=True)
                    log = (run_dir / "worker.log").open("a")
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "scientisttwo.batch",
                            "worker",
                            str(frozen),
                            str(root),
                            entry.id,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=clean_environment(),
                        start_new_session=True,
                    )
                    running[entry.id] = (process, log)
                if pending and time.time() >= metadata["deadline"]:
                    for entry in pending:
                        run_dir = root / "campaigns" / entry.id
                        run_dir.mkdir(parents=True, exist_ok=True)
                        atomic_json(
                            run_dir / "batch-result.json",
                            {
                                "status": "not_started_budget_exhausted",
                                "error": "Global batch deadline exhausted",
                            },
                        )
                    pending = []
                for campaign_id, (process, log) in list(running.items()):
                    if process.poll() is None:
                        continue
                    log.close()
                    receipt = root / "campaigns" / campaign_id / "batch-result.json"
                    if not receipt.exists():
                        atomic_json(
                            receipt,
                            {
                                "status": "error",
                                "error": f"Worker exited {process.returncode} without a receipt",
                            },
                        )
                    del running[campaign_id]
                    summary = summarize(root, manifest)
                    print(
                        json.dumps(
                            {
                                "campaign": campaign_id,
                                "completed": summary["completed_campaigns"],
                                "assigned": len(manifest.campaigns),
                                "statuses": summary["status_counts"],
                            }
                        ),
                        flush=True,
                    )
                if running:
                    time.sleep(0.25)
        except KeyboardInterrupt:
            for process, _ in running.values():
                os.killpg(process.pid, signal.SIGINT)
            for process, log in running.values():
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                log.close()
            summarize(root, manifest)
            raise
        return summarize(root, manifest)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "worker":
        raise SystemExit(campaign_worker(*sys.argv[2:]))
    raise SystemExit("Internal worker entry point; use scientisttwo batch-run")
