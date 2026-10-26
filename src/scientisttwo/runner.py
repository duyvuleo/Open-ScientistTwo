"""Local container-only execution with recoverable outputs and budget reservations."""

import importlib.metadata
import json
from pathlib import Path
import time


from .models import EvalSpec, ExperimentResult, Idea, Profile
from .process import run_process
from .store import IntegrityError, Store, canonical, digest
from .splits import bind_spec


class PrerequisiteError(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


def evaluator_identity() -> dict:
    directory = Path(__file__).parent
    return {
        "files": {
            name: digest((directory / name).read_bytes())
            for name in (
                "data.py",
                "evaluation.py",
                "worker.py",
                "models.py",
                "splits.py",
                "catalog.py",
            )
        },
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "scipy")
        },
    }


class ContainerRunner:
    def __init__(
        self,
        store: Store,
        profile: Profile,
        runtime="docker",
        image="scientisttwo-worker:0.1",
        *,
        adapter,
        context=None,
        executor=run_process,
    ):
        if runtime not in {"docker", "podman"}:
            raise ValueError("Only docker and podman are supported; no host-code fallback")
        self.store, self.profile = store, profile
        self.runtime, self.image, self.executor = runtime, image, executor
        self.image_id = None
        self.adapter = adapter
        self.prefix = [runtime] + (
            ["--context", context] if context and runtime == "docker" else []
        )
        if context and runtime != "docker":
            raise ValueError("Named contexts are only supported by the Docker backend")

    def preflight(self):
        try:
            info = self.executor(
                self.prefix
                + [
                    "info",
                    "--format",
                    "{{.Host.Arch}}" if self.runtime == "podman" else "{{.Architecture}}",
                ],
                10,
            )
            if info.code:
                raise PrerequisiteError(f"{self.runtime} daemon unavailable; start it explicitly")
            image = self.executor(
                self.prefix + ["image", "inspect", self.image, "--format", "{{.Id}}"], 10
            )
            if image.code or not image.stdout.strip():
                raise PrerequisiteError("Worker image missing; build the pinned worker image first")
            self.image_id = image.stdout.strip()
        except FileNotFoundError as error:
            raise PrerequisiteError(f"{self.runtime} executable unavailable") from error
        previous = self.store.get("worker_image_id")
        if previous and previous != self.image_id:
            raise IntegrityError("Worker image changed since this campaign started")
        self.store.set("worker_image_id", self.image_id)

    def _stop_container(self, name):
        self.executor(self.prefix + ["rm", "--force", name], 10)

    def execute(
        self, key: str, idea: Idea, source: str, spec: EvalSpec, *, audit=False
    ) -> ExperimentResult:
        if self.image_id is None:
            self.preflight()
        manifest = self.adapter.split_manifest(spec.task)
        split_ref = self.store.put_json(manifest, "split-manifest")
        expected_split = self.store.get("split_manifest_ref")
        if expected_split and expected_split != split_ref:
            raise IntegrityError("Task partitions or dataset contents changed during the campaign")
        if expected_split is None:
            self.store.set("split_manifest_ref", split_ref)
        spec = bind_spec(spec, manifest)
        code_ref = self.store.put(source.encode(), "py", "candidate-code")
        spec_ref = self.store.put_json(spec, "evaluation-spec")
        request = {
            "code": code_ref,
            "spec": spec_ref,
            "image": self.image_id,
            "evaluator": evaluator_identity(),
            "adapter": self.adapter.identity(),
            "idea": idea.model_dump(mode="json"),
        }
        request_hash = digest(canonical(request))
        job = self.store.db.execute("SELECT * FROM jobs WHERE key=?", (key,)).fetchone()
        if job and job["request_hash"] != request_hash:
            raise IntegrityError(f"Job inputs changed: {key}")
        # Artifact hashes, evaluator identity, and source are checked before every execution/recovery.
        errors = self.store.verify()
        if errors:
            raise IntegrityError(errors[0])
        expected = self.store.get("evaluator")
        if expected and expected != evaluator_identity():
            raise IntegrityError("Trusted evaluator or dependency versions changed")
        if job and job["result_ref"]:
            return ExperimentResult.model_validate(self.store.json(job["result_ref"]))
        directory = self.store.root / "work" / digest(key.encode())[:20]
        inputs, candidate, outputs = (directory / p for p in ("input", "candidate", "output"))
        for path in (inputs, candidate, outputs):
            path.mkdir(parents=True, exist_ok=True)
        outputs.chmod(0o777)
        marker = outputs / "completed.json"
        name = "scientisttwo-" + digest((str(self.store.root) + key).encode())[:20]
        if job:
            # A controller crash may leave its own container running. Do not start a duplicate.
            self._stop_container(name)
            elapsed = min(time.time() - job["started"], job["reserved"])
            if marker.exists():
                return self._collect(
                    key, idea, spec, code_ref, spec_ref, outputs, request_hash, elapsed
                )
            return self._finish(
                key,
                ExperimentResult(
                    id=key,
                    idea_id=idea.id,
                    status="interrupted",
                    code_ref=code_ref,
                    spec_ref=spec_ref,
                    protocol_hash=digest(canonical(spec)),
                    reason="Interrupted execution; no complete prediction artifact",
                    duration_seconds=max(0, elapsed),
                ),
            )
        deadline = self.store.get("deadline")
        remaining = (
            self.profile.experiment_seconds
            if audit
            else (deadline - time.time() if deadline else self.profile.campaign_seconds)
        )
        if remaining <= 0:
            raise BudgetExhausted("Campaign wall-clock budget exhausted")
        if audit and self.store.get("batch_deadline") is not None:
            remaining = min(remaining, self.store.get("batch_deadline") - time.time())
            if remaining <= 0:
                raise BudgetExhausted("Batch deadline exhausted before audit execution")
        reservation = min(remaining, self.profile.experiment_seconds)
        (candidate / "candidate.py").write_text(source)
        manifest = self.adapter.prepare_inputs(spec, inputs)
        (inputs / "manifest.json").write_bytes(
            canonical({**manifest, "request_hash": request_hash})
        )
        worker = self.adapter.worker_path.resolve()
        argv = self.prefix + [
            "run",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit=128",
            "--memory=1g",
            "--cpus=1",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=128m",
            "-e",
            "OMP_NUM_THREADS=1",
            "-e",
            "OPENBLAS_NUM_THREADS=1",
            "-e",
            "MKL_NUM_THREADS=1",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "HOME=/tmp",
            "-v",
            f"{inputs}:/input:ro",
            "-v",
            f"{candidate}:/candidate:ro",
            "-v",
            f"{outputs}:/output:rw",
            "-v",
            f"{worker}:/worker.py:ro",
            self.image_id,
            "python",
            "/worker.py",
        ]
        with self.store.db:
            self.store.db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?,NULL)",
                (key, request_hash, "running", time.time(), reservation, 0),
            )
        result = None
        try:
            result = self.executor(argv, reservation)
        finally:
            if result is None or result.timed_out:
                self._stop_container(name)
        log_ref = self.store.put(
            ((result.stdout or "") + "\n" + (result.stderr or "")).encode(), "log", "execution-log"
        )
        if result.code or result.timed_out:
            return self._finish(
                key,
                ExperimentResult(
                    id=key,
                    idea_id=idea.id,
                    status="timeout" if result.timed_out else "execution_failed",
                    protocol_hash=digest(canonical(spec)),
                    code_ref=code_ref,
                    spec_ref=spec_ref,
                    logs_ref=log_ref,
                    duration_seconds=result.seconds,
                    reason="Experiment deadline exceeded"
                    if result.timed_out
                    else f"Container exited with status {result.code}",
                ),
            )
        return self._collect(
            key, idea, spec, code_ref, spec_ref, outputs, request_hash, result.seconds, log_ref
        )

    def _collect(
        self, key, idea, spec, code_ref, spec_ref, outputs, request_hash, seconds, log_ref=None
    ):
        base = dict(
            id=key,
            idea_id=idea.id,
            code_ref=code_ref,
            spec_ref=spec_ref,
            protocol_hash=digest(canonical(spec)),
            duration_seconds=max(0, seconds),
            logs_ref=log_ref,
        )
        try:
            if json.loads((outputs / "completed.json").read_text()) != {
                "request_hash": request_hash
            }:
                raise ValueError("Wrong completion receipt")
            path = outputs / "observations.json"
            if path.stat().st_size > 10_000_000:
                raise ValueError("Prediction output exceeds limit")
            predictions = json.loads(path.read_text())
            rows = self.adapter.score(spec, predictions)
            result = ExperimentResult(
                **base,
                status="success",
                rows=rows,
                observations_ref=self.store.put_json(predictions, "predictions"),
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            result = ExperimentResult(**base, status="invalid", reason=str(error))
        return self._finish(key, result)

    def _finish(self, key, result):
        result = result.model_copy(
            update={"split_manifest_ref": self.store.get("split_manifest_ref")}
        )
        ref = self.store.put_json(result, "experiment-result")
        with self.store.db:
            self.store.db.execute(
                "UPDATE jobs SET status=?,charged=?,result_ref=? WHERE key=?",
                (result.status, result.duration_seconds, ref, key),
            )
        return result
