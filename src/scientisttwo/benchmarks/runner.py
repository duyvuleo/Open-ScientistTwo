"""Native workspace execution with separately mounted official graders and final-only scores."""

import json
from pathlib import Path
import re
import time
import yaml

from ..process import run_process, OutputLimits
from ..store import Store, IntegrityError, canonical, digest
from .files import workspace, artifact_manifest
from .manifest import preflight
from .models import NativeResult, relative_path
from .registry import get_task
from ..runtime_contracts import (
    asset_mounts, effective_execution_timeout, execution_identity, seal_execution_receipt,
)


class NativeRunner:
    def __init__(self, spec, output, *, executor=run_process):
        self.spec = spec
        output = Path(output).resolve()
        for path in (spec.repository, spec.public_dir, spec.private_dir, spec.candidate_dir):
            if path is not None and (
                output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output)
            ):
                raise ValueError(
                    "Run output must not overlap an upstream, data, or candidate input tree"
                )
        self.store = Store(output)
        self.executor = executor
        self.prefix = [spec.runtime] + (["--context", spec.context] if spec.context else [])
        self.task = get_task(spec.benchmark, spec.repository, spec.task_id)
        self.started = time.monotonic()

    def close(self):
        self.store.close()

    def _image(self, tag):
        result = self.executor(self.prefix + ["image", "inspect", tag, "--format", "{{.Id}}"], 15)
        if result.code or result.timed_out or not result.stdout.strip():
            raise ValueError(
                f"Local image unavailable: {tag}; build or supply a compatible image first"
            )
        image_id = result.stdout.strip()
        # Podman 4.9 emits the config digest without Docker's sha256 prefix.
        # Canonicalize only complete IDs; other existing inspection behavior
        # remains unchanged and readiness applies its strict identity contract.
        return "sha256:" + image_id if re.fullmatch(r"[0-9a-f]{64}", image_id) else image_id

    def _launch(self, stage, image, job, mounts, timeout, *, gpus=None, bridge=None, resources=None):
        enforcement = execution_identity(self.spec.runtime)
        enforcement_started = time.time()
        resources = resources or self.spec.resources
        control = self.store.root / "control" / stage
        control.mkdir(parents=True, exist_ok=True)
        (control / "job.json").write_bytes(canonical(job))
        name = "scientisttwo-native-" + digest((str(self.store.root) + stage).encode())[:18]
        argv = self.prefix + [
            "run",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--label",
            f"scientisttwo.timeout-seconds={timeout}",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges",
            f"--memory={resources.memory_mb}m",
            f"--cpus={resources.cpus}",
            f"--pids-limit={resources.pids_limit}",
            f"--shm-size={resources.shm_mb}m",
            "--user",
            "65534:65534",
            "--tmpfs",
            f"/tmp:rw,nosuid,size={resources.tmp_mb}m",
            "-w",
            "/tmp",
            "-e",
            "HOME=/tmp",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            f"OMP_NUM_THREADS={resources.threads}",
            "-e",
            f"OPENBLAS_NUM_THREADS={resources.threads}",
            "-e",
            "HF_HUB_OFFLINE=1",
            "-e",
            "TRANSFORMERS_OFFLINE=1",
            "-e",
            "HF_DATASETS_OFFLINE=1",
            "-v",
            f"{control}:/control:ro",
            "-v",
            f"{bridge or Path(__file__).with_name('bridge.py')}:/bridge/bridge.py:ro",
        ]
        if gpus:
            from ..runtime_profiles import enforce_gpu_allocation

            gpus = enforce_gpu_allocation(gpus)
            if self.spec.runtime == "docker":
                argv += ["--gpus", gpus]
            else:
                devices = ["all"] if gpus == "all" else gpus.removeprefix("device=").split(",")
                for device in devices:
                    argv += ["--device", "nvidia.com/gpu=" + device]
        for host, target, mode in mounts:
            argv += ["-v", f"{host}:{target}:{mode}"]
        argv += [image, "python", "/bridge/bridge.py"]
        timeout = effective_execution_timeout(timeout, enforcement)
        argv[argv.index("--label") + 1] = f"scientisttwo.timeout-seconds={timeout}"
        result = None
        try:
            writable = tuple(Path(host) for host, _, mode in mounts if mode == "rw")
            limits = OutputLimits(writable, resources.output_max_bytes, resources.output_max_files) if writable else None
            result = self.executor(argv, timeout, output_limits=limits, max_output_bytes=resources.log_max_bytes)
            if result.limit_exceeded:
                result.code = result.code or 125
        finally:
            if result is None or result.timed_out or result.limit_exceeded:
                self.executor(self.prefix + ["rm", "--force", name], 15)
        enforcement_receipt = enforcement_error = None
        if enforcement is not None:
            try:
                enforcement_receipt = seal_execution_receipt(
                    self.store, name, enforcement, started_at=enforcement_started,
                    expected_gpu_uuid=gpus.removeprefix("device=") if gpus else None,
                )
            except (ValueError, OSError) as error:
                enforcement_error = str(error)
                result.code = result.code or 125
        ref = self.store.put_json(
            {
                "stage": stage,
                "image_id": image,
                "argv": argv,
                "returncode": result.code,
                "timed_out": result.timed_out,
                "limit_exceeded": result.limit_exceeded,
                "stdout_truncated": result.stdout_truncated,
                "stderr_truncated": result.stderr_truncated,
                "stdout_bytes": result.stdout_bytes,
                "stderr_bytes": result.stderr_bytes,
                "seconds": result.seconds,
                "resources": resources.model_dump(mode="json"),
                "execution_enforcement": enforcement_receipt,
                "execution_enforcement_error": enforcement_error,
                "job_ref": self.store.put_json(job, "execution-job"),
                "log_ref": self.store.put(
                    (result.stdout + "\n" + result.stderr).encode(), "log", "native-log"
                ),
            },
            "native-execution",
        )
        return result, ref

    def _finish(self, status, reason="", **kwargs):
        result = NativeResult(
            benchmark=self.spec.benchmark,
            task_id=self.spec.task_id,
            status=status,
            mode=self.spec.mode,
            data_provenance=self.spec.data_provenance,
            reason=reason,
            metric=self.task.metric,
            direction=self.task.direction,
            warnings=self.task.warnings,
            elapsed_seconds=time.monotonic() - self.started,
            specification_ref=self.store.get("native_spec_ref"),
            **kwargs,
        )
        ref = self.store.put_json(result, "native-benchmark-result")
        self.store.set("native_result_ref", ref)
        if status in {"completed", "invalid_submission", "grading_failed"}:
            self.store.set("native_grading_pending", False)
        (self.store.root / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
        text = [
            f"# {self.spec.benchmark}: {self.spec.task_id}",
            "",
            f"Status: **{result.status}**",
            "",
            f"Mode: `{result.mode}`. Data provenance: `{result.data_provenance}`.",
            "No live agent was invoked. This is a baseline or prepared-candidate evaluation, not an autonomous ScientistTwo score.",
            "",
            f"Native metric: {result.metric}; score: {result.score}; direction: {result.direction}.",
            f"Benchmark data verified: {result.benchmark_data_verified}. Leaderboard eligible: {result.leaderboard_eligible}.",
            "",
            reason,
            "",
            *[f"- {warning}" for warning in result.warnings],
            "",
            f"[Sealed result]({ref})",
        ]
        if result.official_report_ref:
            text.append(f"[Unchanged official grading result]({result.official_report_ref})")
        if result.data_provenance == "contract_fixture":
            text += [
                "",
                "CONTRACT TEST DATA ONLY: numerical scores and raw medal flags are not competition results and must not enter benchmark aggregates.",
            ]
        (self.store.root / "report.md").write_text("\n".join(text) + "\n")
        return result

    def run(self):
        spec = self.spec
        spec_ref = self.store.put_json(spec, "native-specification")
        previous = self.store.get("native_spec_ref")
        if previous and previous != spec_ref:
            raise IntegrityError("Run directory belongs to a different benchmark specification")
        self.store.set("native_spec_ref", spec_ref)
        for path in Path(__file__).parent.glob("*.py"):
            self.store.put_file(path, "native-integration-source")
        checks = preflight(spec)
        self.store.set("native_preflight_ref", self.store.put_json(checks, "native-preflight"))
        if not checks["ready"]:
            return self._finish("blocked", "; ".join(checks["errors"]))
        errors = self.store.verify()
        if errors:
            raise IntegrityError(errors[0])
        saved = self.store.get("native_result_ref")
        if saved:
            result = NativeResult.model_validate(self.store.json(saved))
            if result.status == "completed":
                return result
        try:
            execution_image = self._image(spec.image)
            grading_image = self._image(spec.grader_image)
        except (ValueError, FileNotFoundError) as error:
            return self._finish("blocked", str(error))
        image_ids = {"execution": execution_image, "grading": grading_image}
        if self.store.get("native_images") and self.store.get("native_images") != image_ids:
            raise IntegrityError("Native benchmark image changed during the run")
        self.store.set("native_images", image_ids)
        work = self.store.root / "workspace"
        execution = self.store.get("native_execution_ref")
        if execution is None:
            if self.store.get("native_execution_started"):
                return self._finish(
                    "interrupted",
                    "Execution outcome was not committed. Inspect the preserved attempt and use a new run directory; no implicit restart.",
                )
            links = workspace(spec.public_dir, work, spec.candidate_dir)
            work.chmod(0o777)
            for p in work.rglob("*"):
                if p.is_dir() and not p.is_symlink():
                    p.chmod(0o777)
            self.store.set("native_input_links", sorted(links))
            self.store.set(
                "native_initial_code",
                self.store.put_json(artifact_manifest(work, links), "native-initial-code"),
            )
            self.store.set("native_execution_started", True)
            if spec.mode == "grade_only":
                execution = self.store.put_json(
                    {
                        "stage": "grade_only",
                        "seconds": 0,
                        "source": "externally prepared artifacts",
                    },
                    "native-execution",
                )
            else:
                job = {
                    "stage": "execute",
                    "command": spec.command,
                    "submission": spec.submission,
                    "seed": spec.final_seed,
                }
                if spec.benchmark == "mle-bench" and spec.mode == "baseline":
                    cfg = yaml.safe_load(
                        (
                            spec.repository / "mlebench/competitions" / spec.task_id / "config.yaml"
                        ).read_text()
                    )
                    sample = Path(cfg["dataset"]["sample_submission"]).relative_to(
                        Path(spec.task_id) / "prepared/public"
                    )
                    job["sample_submission"] = relative_path(sample.as_posix())
                process, execution = self._launch(
                    "execute",
                    execution_image,
                    job,
                    [(work, "/workspace", "rw"), (spec.public_dir, "/data/public", "ro")]
                    + asset_mounts(spec, "execute", protected=[spec.private_dir]),
                    spec.resources.seconds,
                    gpus=spec.resources.gpus,
                )
                if process.code or process.timed_out:
                    return self._finish(
                        "execution_failed",
                        "Native program failed or exceeded its budget; no benchmark score was inferred.",
                        execution_ref=execution,
                    )
            files = artifact_manifest(work, set(self.store.get("native_input_links")))
            references = {
                relative: self.store.put_file(work / relative, "native-candidate-artifact")
                for relative in files
            }
            self.store.set(
                "native_artifacts_ref",
                self.store.put_json(
                    {"files": files, "references": references}, "native-artifact-index"
                ),
            )
            self.store.set("native_execution_ref", execution)
        frozen = self.store.json(self.store.get("native_artifacts_ref"))
        if artifact_manifest(work, set(self.store.get("native_input_links"))) != frozen["files"]:
            raise IntegrityError("Candidate artifacts changed after execution was frozen")
        checks = preflight(spec)
        if not checks["ready"]:
            return self._finish("blocked", "Prepared data changed before grading")
        if self.store.get("native_grading_pending"):
            return self._finish(
                "interrupted",
                "Final grading outcome was not committed; no automatic repeat",
                execution_ref=execution,
                artifacts_ref=self.store.get("native_artifacts_ref"),
            )
        attempt = self.store.get("native_grading_attempts", 0)
        output = self.store.root / "grade-output" / str(attempt)
        output.mkdir(parents=True, exist_ok=False)
        output.chmod(0o777)
        job = {
            "stage": "grade",
            "benchmark": spec.benchmark,
            "task_id": spec.task_id,
            "submission": spec.submission,
            "direction": self.task.direction,
            "seed": spec.final_seed,
            "required_artifacts": spec.required_artifacts or self.task.required_artifacts or [spec.submission],
        }
        mounts = [
            (spec.repository, "/upstream", "ro"),
            (work, "/submission", "ro"),
            (output, "/grade-output", "rw"),
            (spec.public_dir, "/data/public", "ro"),
        ]
        if spec.benchmark == "mle-bench":
            mounts += [
                (spec.public_dir, f"/data/{spec.task_id}/prepared/public", "ro"),
                (spec.private_dir, f"/data/{spec.task_id}/prepared/private", "ro"),
            ]
        else:
            mounts += [
                (spec.public_dir, "/task/env", "ro"),
                (spec.private_dir, "/task/scripts", "ro"),
            ]
        self.store.set("native_grading_pending", True)
        process, grading_ref = self._launch(
            "grade",
            grading_image,
            job,
            mounts,
            spec.resources.grading_seconds,
            gpus=spec.resources.gpus,
        )
        self.store.set("native_grading_execution_ref", grading_ref)
        self.store.set("native_grading_attempts", attempt + 1)
        if process.code or process.timed_out or not (output / "grade.json").is_file():
            return self._finish(
                "grading_failed",
                "Official grader could not complete in the declared image; inspect its log and task-specific dependencies.",
                execution_ref=execution,
                artifacts_ref=self.store.get("native_artifacts_ref"),
            )
        grade = json.loads((output / "grade.json").read_text())
        official = self.store.put_json(
            {
                "data_provenance": spec.data_provenance,
                "benchmark": spec.benchmark,
                "task_id": spec.task_id,
                "upstream_revision": spec.revision,
                "report": grade["official_report"],
            },
            "official-grading-report",
        )
        verified = (
            checks["checksums"]["status"] == "pass"
            if spec.benchmark == "mle-bench"
            else spec.data_provenance == "upstream_no_data"
        )
        # No autonomous-agent trajectory is produced by this runner.
        eligible = False
        return self._finish(
            "completed" if grade["valid_submission"] else "invalid_submission",
            "Official grader executed; this run contains no live-agent evaluation.",
            score=grade["score"],
            official_report_ref=official,
            public_data_checksums=checks["checksums"]["status"],
            benchmark_data_verified=verified,
            leaderboard_eligible=eligible,
            execution_ref=execution,
            artifacts_ref=self.store.get("native_artifacts_ref"),
        )
