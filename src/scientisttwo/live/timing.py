"""Trusted paired timing of resident isolated workers; never accept worker timings.

Transport startup and oracle verification are outside each measured interval. The
interval includes request/response IPC and output serialization, so it is not a
claim about bare kernel time. GPU synchronization must be enforced by the trusted
worker wrapper before it acknowledges a response.
"""

import json
import os
import selectors
import signal
import statistics
import subprocess
import time
from typing import Callable, Literal, Protocol

from pydantic import Field

from ..models import Record


class TimingTransport(Protocol):
    def request(self, payload: dict, *, timeout: float) -> dict: ...


class TimingSample(Record):
    phase: Literal["warmup", "measurement"]
    round: int = Field(ge=0)
    position: int = Field(ge=0, le=1)
    arm: Literal["baseline", "candidate"]
    payload_index: int = Field(ge=0)
    seconds: float = Field(ge=0)
    response: dict
    correctness: dict


class TimingSummary(Record):
    seconds: list[float]
    median_seconds: float = Field(ge=0)
    q1_seconds: float = Field(ge=0)
    q3_seconds: float = Field(ge=0)
    iqr_seconds: float = Field(ge=0)


class PairedTimingReport(Record):
    schema_version: Literal[1] = 1
    metric: Literal["paired_request_seconds"] = "paired_request_seconds"
    warmup_rounds: int
    measurement_pairs: int
    payload_count: int
    device: Literal["cpu", "cuda"]
    gpu_synchronization: Literal["not_applicable", "enforced"]
    samples: list[TimingSample]
    baseline: TimingSummary
    candidate: TimingSummary
    wall_seconds: float = Field(ge=0)
    measurement_scope: str = (
        "Resident-worker request/response latency including IPC and output serialization; "
        "process and module startup and independent correctness checks are excluded."
    )


def run_paired_timing(
    transports: dict[str, TimingTransport],
    payloads: list[dict],
    correctness: Callable[[str, dict, dict], dict],
    *,
    warmups: int = 5,
    pairs: int = 10,
    request_timeout: float = 120,
    total_timeout: float = 600,
    device: Literal["cpu", "cuda"] = "cpu",
    gpu_synchronization: bool = False,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> PairedTimingReport:
    """Measure matched workloads in alternating order with independent output checks."""
    if set(transports) != {"baseline", "candidate"} or not payloads:
        raise ValueError(
            "Exactly baseline and candidate transports and nonempty workloads required"
        )
    if warmups < 1 or pairs < 2 or min(request_timeout, total_timeout) <= 0:
        raise ValueError("Positive warmups/timeouts and at least two pairs are required")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not gpu_synchronization):
        raise ValueError("CUDA timing requires enforced synchronization in the trusted wrapper")
    started = time.monotonic()
    samples = []
    values = {"baseline": [], "candidate": []}
    for phase, rounds in (("warmup", warmups), ("measurement", pairs)):
        for index in range(rounds):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            for position, arm in enumerate(order):
                seconds = 0.0
                for payload_index, payload in enumerate(payloads):
                    remaining = total_timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("Paired timing exhausted its total wall-clock budget")
                    begin = clock_ns()
                    response = transports[arm].request(
                        payload, timeout=min(request_timeout, remaining)
                    )
                    elapsed = (clock_ns() - begin) / 1_000_000_000
                    if elapsed < 0 or not isinstance(response, dict):
                        raise ValueError("Invalid transport response or monotonic timer")
                    evidence = correctness(arm, payload, response)
                    if not isinstance(evidence, dict) or evidence.get("valid") is not True:
                        raise ValueError(
                            "Independent timing-workload correctness validation failed"
                        )
                    seconds += elapsed
                    samples.append(
                        TimingSample(
                            phase=phase,
                            round=index,
                            position=position,
                            arm=arm,
                            payload_index=payload_index,
                            seconds=elapsed,
                            response=response,
                            correctness=evidence,
                        )
                    )
                if phase == "measurement":
                    values[arm].append(seconds)

    def summary(arm):
        measured = values[arm]
        q1, _, q3 = statistics.quantiles(measured, n=4, method="inclusive")
        return TimingSummary(
            seconds=measured,
            median_seconds=statistics.median(measured),
            q1_seconds=q1,
            q3_seconds=q3,
            iqr_seconds=q3 - q1,
        )

    return PairedTimingReport(
        warmup_rounds=warmups,
        measurement_pairs=pairs,
        payload_count=len(payloads),
        device=device,
        gpu_synchronization="enforced" if device == "cuda" else "not_applicable",
        samples=samples,
        baseline=summary("baseline"),
        candidate=summary("candidate"),
        wall_seconds=time.monotonic() - started,
    )


class JsonLineTransport:
    """Bounded JSONL RPC around a caller-launched isolated process, with no shell.

    The caller owns container isolation and cleanup (killing a Docker client alone
    does not terminate its container). Host execution is appropriate only for fixed
    test fixtures, never generated candidate code.
    """

    def __init__(self, process: subprocess.Popen, *, startup_timeout=60, max_message_bytes=65_536):
        if process.stdin is None or process.stdout is None:
            raise ValueError("RPC process requires binary stdin and stdout pipes")
        self.process = process
        self.max_message_bytes = max_message_bytes
        self.sequence = 0
        self.buffer = bytearray()
        started = time.monotonic()
        self.ready = self._read(startup_timeout)
        self.startup_seconds = time.monotonic() - started
        if self.ready.get("type") != "ready" or self.ready.get("protocol") != 1:
            raise ValueError("Timing worker did not send a valid readiness handshake")

    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self.buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("Resident timing worker did not respond before its deadline")
                block = os.read(self.process.stdout.fileno(), 8192)
                if not block:
                    raise RuntimeError("Resident timing worker exited before responding")
                self.buffer.extend(block)
                if len(self.buffer) > self.max_message_bytes:
                    raise ValueError("Timing worker response exceeded its message limit")
        line, _, remainder = self.buffer.partition(b"\n")
        self.buffer = bytearray(remainder)
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("Timing worker response must be an object")
        return value

    def request(self, payload, *, timeout):
        self.sequence += 1
        message = (
            json.dumps({"request_id": self.sequence, "payload": payload}, allow_nan=False).encode()
            + b"\n"
        )
        if len(message) > self.max_message_bytes:
            raise ValueError("Timing request exceeded its message limit")
        deadline = time.monotonic() + timeout
        # Nonblocking writes prevent a worker that stopped reading from hanging the controller.
        descriptor = self.process.stdin.fileno()
        previous = os.get_blocking(descriptor)
        os.set_blocking(descriptor, False)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(descriptor, selectors.EVENT_WRITE)
                written = 0
                while written < len(message):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError("Resident timing worker stopped accepting requests")
                    try:
                        written += os.write(descriptor, message[written:])
                    except BlockingIOError:
                        continue
        finally:
            os.set_blocking(descriptor, previous)
        response = self._read(max(0, deadline - time.monotonic()))
        if response.get("request_id") != self.sequence or response.get("status") != "ok":
            raise ValueError("Timing worker returned a failed or mismatched response")
        value = response.get("result")
        if not isinstance(value, dict):
            raise ValueError("Timing worker result must be an object")
        return value

    def close(self):
        """Close the local process group; caller must also remove its container."""
        if self.process.poll() is None:
            try:
                if os.getpgid(self.process.pid) == self.process.pid:
                    os.killpg(self.process.pid, signal.SIGKILL)
                else:
                    self.process.kill()
            except ProcessLookupError:
                pass
            self.process.wait(timeout=10)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()
