"""Bounded SSH transport executing a pinned file-queue program on one Slurm node.

Both reads and publication use the controller node's filesystem client. This
removes cross-client visibility from that path; it does not prove an earlier
timeout's cause or reconcile an already failed controller operation. The caller
injects an existing relay service, so claims, authorization and budgets persist.
"""

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re
import shlex
import time

from .backends import strict_json_loads
from .process import run_process
from .studies.media_relay import REQUEST_LIMIT, RESPONSE_LIMIT, canonical


@dataclass(frozen=True)
class NodeQueueConfig:
    host: str
    queue: str
    job_id: str
    node: str
    python: str
    srun: str
    cpus: int
    memory_mib: int
    step_seconds: int
    connect_timeout_seconds: int
    exchange_timeout_seconds: float
    deadline: float
    return_reserve_seconds: float
    remote_program_sha256: str

    def __post_init__(self):
        if (not isinstance(self.host, str)
                or not re.fullmatch(r"[a-z_][a-z0-9_-]*@[a-zA-Z0-9.-]+", self.host)
                or not isinstance(self.job_id, str) or not re.fullmatch(r"[1-9][0-9]{0,15}", self.job_id)
                or not isinstance(self.node, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,127}", self.node)
                or type(self.cpus) is not int or self.cpus != 1
                or type(self.memory_mib) is not int or not 1 <= self.memory_mib <= 1024
                or type(self.step_seconds) is not int or not 1 <= self.step_seconds <= 60
                or type(self.connect_timeout_seconds) is not int or not 1 <= self.connect_timeout_seconds <= 10
                or not isinstance(self.remote_program_sha256, str)
                or not re.fullmatch(r"[a-f0-9]{64}", self.remote_program_sha256)):
            raise ValueError("Invalid explicit node queue allocation or program identity")
        for path in (self.queue, self.python, self.srun):
            if (not isinstance(path, str) or not Path(path).is_absolute()
                    or ".." in Path(path).parts or any(c in path for c in "\x00\n\r")):
                raise ValueError("Node queue paths must be explicit normalized absolute paths")
        if not self.queue.startswith("/home/"):
            raise ValueError("Use the declared existing remote home queue")
        for value in (self.exchange_timeout_seconds, self.deadline, self.return_reserve_seconds):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Node queue time limits must be explicit and finite")
        if (not 0 < self.exchange_timeout_seconds <= 45
                or self.exchange_timeout_seconds > self.step_seconds
                or not self.exchange_timeout_seconds + 2 <= self.return_reserve_seconds <= 120):
            raise ValueError("Node queue exchange or return reserve exceeds its bound")


class NodeQueueTransport:
    """No provisioning or model construction; two fixed queue operations only."""

    def __init__(self, config, remote_program, *, executor=run_process, clock=time.time,
                 receipt_sink=None):
        if not isinstance(config, NodeQueueConfig):
            raise ValueError("A validated explicit node queue configuration is required")
        if (not isinstance(remote_program, str) or not 0 < len(remote_program.encode()) <= 65536
                or hashlib.sha256(remote_program.encode()).hexdigest() != config.remote_program_sha256):
            raise ValueError("Remote queue program differs from its pinned hash")
        self.config, self.remote_program = config, remote_program
        self.executor, self.clock, self.receipt_sink = executor, clock, receipt_sink
        self.identity = hashlib.sha256(canonical({
            "config": asdict(config), "transport_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "remote_program_sha256": config.remote_program_sha256,
        })).hexdigest()

    def command(self, operation):
        if operation not in {"next", "complete"}:
            raise ValueError("Unsupported node queue operation")
        config = self.config
        minutes, seconds = divmod(config.step_seconds, 60)
        remote = [config.srun, "--jobid=" + config.job_id, "--overlap", "--exact",
            "--nodelist=" + config.node, "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
            "--mem=" + str(config.memory_mib) + "M", "--gres=none",
            f"--time=00:{minutes:02d}:{seconds:02d}", "--", config.python, "-I", "-c",
            self.remote_program, operation, config.queue]
        return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o",
                "ConnectTimeout=" + str(config.connect_timeout_seconds), config.host, shlex.join(remote)]

    def exchange(self, operation, payload=None):
        if operation not in {"next", "complete"}:
            raise ValueError("Unsupported node queue operation")
        if operation == "next" and payload is not None:
            raise ValueError("Queue reads cannot carry input")
        if operation == "complete" and (not isinstance(payload, bytes) or len(payload) > RESPONSE_LIMIT):
            raise ValueError("Queue publication requires a bounded serialized response")
        remaining = self.config.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError("Node queue authorization deadline expired")
        started = self.clock()
        response = self.executor(self.command(operation), min(self.config.exchange_timeout_seconds, remaining),
            stdin=payload, max_output_bytes=REQUEST_LIMIT if operation == "next" else 4096)
        receipt = {"schema_version": "scientisttwo-node-queue-transfer-v1", "transport_identity": self.identity,
            "operation": operation, "node": self.config.node, "job_id": self.config.job_id,
            "started_at": started, "finished_at": self.clock(), "returncode": response.code,
            "timed_out": response.timed_out,
            "capture_limited": bool(response.limit_exceeded or response.stdout_truncated or response.stderr_truncated),
            "stdout_sha256": hashlib.sha256(response.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(response.stderr.encode()).hexdigest()}
        if self.receipt_sink is not None:
            self.receipt_sink(receipt)
        if response.code or response.timed_out or receipt["capture_limited"]:
            # Retain structural diagnostics only, never SSH account text or prompts.
            raise RuntimeError("Same-node queue transfer failed: " + canonical(receipt).decode())
        return strict_json_loads(response.stdout) if response.stdout.strip() else None

    def serve_once(self, service):
        """Reserve return time without changing the signed request or its ledger."""
        request = self.exchange("next")
        if request is None:
            return False
        if (not isinstance(request, dict) or type(request.get("deadline")) not in (int, float)
                or not math.isfinite(request["deadline"])):
            raise ValueError("Queue request has no finite deadline")
        original_executor = service.executor
        deadline = min(request["deadline"], self.config.deadline)

        def bounded_executor(argv, timeout, **kwargs):
            available = deadline - self.clock() - self.config.return_reserve_seconds
            if available <= 0:
                # A service claim may already exist; its uncertain state remains
                # fail-closed. This path never calls the model executable.
                raise TimeoutError("No time remains for a model call and response publication")
            return original_executor(argv, min(timeout, available), **kwargs)

        service.executor = bounded_executor
        try:
            response = service.serve(request)
        finally:
            service.executor = original_executor
        payload = canonical(response)
        if len(payload) > RESPONSE_LIMIT:
            raise ValueError("Model response exceeds the queue transfer bound")
        receipt = self.exchange("complete", payload)
        if receipt != {"status": "stored", "request_id": request["request_id"]}:
            raise ValueError("Same-node completion acknowledgement changed")
        return True
