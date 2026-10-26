"""Persistent container transports for the generic trusted paired-timing controller."""

import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from ..process import BoundedOutputCapture, OutputLimits, clean_environment, run_process
from ..runtime_contracts import effective_execution_timeout, execution_identity, seal_execution_receipt
from ..store import canonical, digest
from .timing import JsonLineTransport


class ContainerTimingTransport:
    """One isolated resident worker; both startup and cleanup have finite deadlines."""

    def __init__(
        self,
        *,
        prefix: list[str],
        image: str,
        source: Path,
        public: Path,
        control: Path,
        outputs: Path,
        resources,
        handler_source: str,
        parameters: dict,
        device="cpu",
        startup_timeout=60,
        execution_timeout=None,
        extra_mounts=(),
    ):
        if not prefix or prefix[0] not in {"docker", "podman"}:
            raise ValueError("Timing candidates require Docker or Podman; no host fallback")
        if device == "cuda" and not resources.gpus:
            raise ValueError("CUDA timing requires explicit GPU allocation")
        assets = []
        for host, target, mode in extra_mounts:
            if mode != "ro" or not target.startswith("/assets/") or ".." in Path(target).parts:
                raise ValueError("Timing extra mounts must be read-only asset paths")
            host = Path(host)
            if not host.is_absolute() or host.is_symlink() or not host.is_dir():
                raise ValueError("Timing assets require real absolute directories")
            assets.append((host, target, mode))
        if len({target for _, target, _ in assets}) != len(assets):
            raise ValueError("Timing asset mount targets must be distinct")
        self.prefix = prefix
        self.transport = self.process = None
        self.log_capture = self.log_thread = self.monitor_thread = None
        self.stop_monitor = threading.Event()
        self.limit_exceeded = None
        self.stderr_bytes, self.stderr_truncated = 0, False
        self.started = time.monotonic()
        self.enforcement_started = time.time()
        self.enforcement = execution_identity(prefix[0])
        requested_seconds = getattr(resources, "seconds", 1800)
        if execution_timeout is not None:
            requested_seconds = min(requested_seconds, execution_timeout)
        self.execution_timeout = effective_execution_timeout(requested_seconds, self.enforcement)
        self.closed = False
        self.name = "scientisttwo-timing-" + digest(str(control.absolute()).encode())[:18]
        self.argv = []
        for path in (source, public):
            if path.is_symlink() or not path.is_dir():
                raise ValueError("Timing source and public inputs require real directories")
        control.mkdir(parents=True, exist_ok=False)
        outputs.mkdir(parents=True, exist_ok=False)
        outputs.chmod(0o777)
        (control / "timing_handler.py").write_text(handler_source)
        (control / "timing_job.json").write_bytes(
            canonical({"parameters": parameters, "device": device})
        )
        self.log_path = control / "worker.log"
        self.output_limits = OutputLimits(
            (outputs,),
            getattr(resources, "output_max_bytes", 64_000_000_000),
            getattr(resources, "output_max_files", 100_000),
        )
        argv = prefix + [
            "run",
            "--rm",
            "-i",
            "--pull=never",
            "--name",
            self.name,
            "--label",
            f"scientisttwo.timeout-seconds={self.execution_timeout}",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            f"--memory={resources.memory_mb}m",
            f"--cpus={resources.cpus}",
            f"--pids-limit={getattr(resources, 'pids_limit', 256)}",
            f"--shm-size={getattr(resources, 'shm_mb', 64)}m",
            "--tmpfs",
            f"/tmp:rw,nosuid,size={getattr(resources, 'tmp_mb', 512)}m",
            "-w",
            "/workspace",
        ]
        from ..runtime_profiles import enforce_gpu_allocation

        gpus = enforce_gpu_allocation(resources.gpus)
        self.expected_gpu_uuid = gpus.removeprefix("device=") if gpus else None
        if gpus:
            if prefix[0] == "docker":
                argv.extend(["--gpus", gpus])
            else:
                devices = ["all"] if gpus == "all" else gpus.removeprefix("device=").split(",")
                for device_id in devices:
                    argv.extend(["--device", "nvidia.com/gpu=" + device_id])
        for key, value in {
            "HOME": "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(getattr(resources, "threads", 1)),
            "OPENBLAS_NUM_THREADS": str(getattr(resources, "threads", 1)),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }.items():
            argv += ["-e", f"{key}={value}"]
        for host, target, mode in [
            (source, "/workspace", "ro"),
            (public, "/data/public", "ro"),
            (control, "/control", "ro"),
            (outputs, "/outputs", "rw"),
            (Path(__file__).with_name("timing_worker.py"), "/bridge/timing_worker.py", "ro"),
        ] + assets:
            if ":" in str(host):
                raise ValueError("Container mount paths cannot contain colon separators")
            argv += ["-v", f"{host.absolute()}:{target}:{mode}"]
        self.argv = argv + [image, "python", "/bridge/timing_worker.py"]
        try:
            self.log_capture = BoundedOutputCapture(getattr(resources, "log_max_bytes", 16_777_216))
            self.process = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=clean_environment(),
                start_new_session=True,
            )
            self.log_thread = threading.Thread(
                target=self.log_capture.drain,
                args=(self.process.stderr,),
                daemon=True,
            )
            self.log_thread.start()
            self.monitor_thread = threading.Thread(target=self._monitor_outputs, daemon=True)
            self.monitor_thread.start()
            self.transport = JsonLineTransport(
                self.process, startup_timeout=min(startup_timeout, self.execution_timeout))
            if self.transport.ready.get("device") != device:
                raise ValueError("Timing worker device differs from its declared protocol")
        except BaseException:
            self.close()
            raise

    @property
    def startup_seconds(self):
        return self.transport.startup_seconds if self.transport is not None else None

    def request(self, payload, *, timeout):
        try:
            if self.limit_exceeded:
                raise RuntimeError(self.limit_exceeded)
            response = self.transport.request(payload, timeout=timeout)
            reason = self.output_limits.check()
            if reason:
                self.limit_exceeded = reason
                raise RuntimeError(reason)
            return response
        except BaseException:
            self.close()
            raise

    def _monitor_outputs(self):
        while not self.stop_monitor.wait(0.2):
            try:
                reason = self.output_limits.check()
                if self.log_capture.error:
                    reason = "Timing worker log capture failed"
            except Exception as error:
                reason = "Timing output monitor failed: " + type(error).__name__
            if reason:
                self.limit_exceeded = reason
                if self.process is not None and self.process.poll() is None:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                # Stopping the client alone does not stop a daemon-owned container.
                run_process(self.prefix + ["rm", "--force", self.name], 15)
                return

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop_monitor.set()
        try:
            if self.process is not None and self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=10)
        finally:
            run_process(self.prefix + ["rm", "--force", self.name], 15)
            if self.monitor_thread is not None:
                self.monitor_thread.join(timeout=16)
            if self.log_thread is not None:
                self.log_thread.join(timeout=5)
            if self.transport is not None:
                self.transport.close()
            if self.process is not None:
                for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                    if pipe is not None and not pipe.closed:
                        pipe.close()
            if self.log_capture is not None:
                self.stderr_bytes = self.log_capture.total
                self.stderr_truncated = self.stderr_bytes > self.log_capture.limit
                self.log_path.write_text(self.log_capture.text())
                self.log_capture.stream.close()
            self.process_seconds = time.monotonic() - self.started

    def receipt(self, store=None):
        result = {
            "argv": self.argv,
            "startup_seconds": self.startup_seconds,
            "process_seconds": time.monotonic() - self.started,
            "returncode": self.process.poll() if self.process is not None else None,
            "log_path": str(self.log_path),
            "closed": self.closed,
            "stderr_bytes": self.stderr_bytes,
            "stderr_truncated": self.stderr_truncated,
            "limit_exceeded": self.limit_exceeded,
        }
        if self.enforcement is not None:
            if store is None or not self.closed:
                raise ValueError("Enforced timing requires a closed worker and evidence store")
            result["execution_enforcement"] = seal_execution_receipt(
                store, self.name, self.enforcement, started_at=self.enforcement_started,
                expected_gpu_uuid=self.expected_gpu_uuid,
            )
        return result
