"""Bounded subprocesses; no shell interpolation or inherited credentials.

Output readers drain pipes into bounded temporary head/tail spools. A noisy child
cannot grow controller RAM or temporary capture storage without bound. Generated
research code belongs in the isolated runners, never in this host helper.
"""

import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ProcessResult:
    code: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    limit_exceeded: str | None = None


@dataclass(frozen=True)
class OutputLimits:
    """Cumulative logical quotas, without following symlinks or counting roots twice.

    Missing directories are allowed before a process creates them. Symlinks count
    as entries, but their targets' data never enter the quota walk.
    """

    directories: tuple[Path, ...]
    max_bytes: int
    max_files: int

    def __post_init__(self):
        if (
            isinstance(self.max_bytes, bool)
            or not isinstance(self.max_bytes, int)
            or self.max_bytes < 0
        ):
            raise ValueError("Output byte limit must be a nonnegative integer")
        if (
            isinstance(self.max_files, bool)
            or not isinstance(self.max_files, int)
            or self.max_files < 0
        ):
            raise ValueError("Output file limit must be a nonnegative integer")
        roots = {Path(os.path.abspath(path)) for path in self.directories}
        if not roots:
            raise ValueError("Output quotas require at least one directory")
        roots = {
            path
            for path in roots
            if not any(path != other and path.is_relative_to(other) for other in roots)
        }
        object.__setattr__(self, "directories", tuple(sorted(roots)))

    def check(self) -> str | None:
        files, size = 0, 0
        pending = []
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            for root in self.directories:
                try:
                    if not stat.S_ISDIR(root.lstat().st_mode):
                        return "Output quota root is not a regular directory"
                    pending.append(os.open(root, flags))
                except FileNotFoundError:
                    continue
            while pending:
                descriptor = pending.pop()
                try:
                    with os.scandir(descriptor) as entries:
                        for entry in entries:
                            try:
                                info = entry.stat(follow_symlinks=False)
                            except FileNotFoundError:
                                continue
                            if stat.S_ISDIR(info.st_mode):
                                try:
                                    pending.append(os.open(entry.name, flags, dir_fd=descriptor))
                                except FileNotFoundError:
                                    continue
                                continue
                            files += 1
                            if stat.S_ISREG(info.st_mode):
                                size += info.st_size
                            if files > self.max_files:
                                return f"Output file quota exceeded ({files} > {self.max_files})"
                            if size > self.max_bytes:
                                return f"Output byte quota exceeded ({size} > {self.max_bytes})"
                finally:
                    os.close(descriptor)
            return None
        finally:
            for descriptor in pending:
                os.close(descriptor)


class BoundedOutputCapture:
    """One-reader bounded spool preserving the beginning and latest output bytes."""

    def __init__(self, limit):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("Output capture limit must be a nonnegative integer")
        self.limit = limit
        self.head_limit = (limit + 1) // 2
        self.tail_limit = limit - self.head_limit
        self.head_size = self.tail_size = self.tail_cursor = self.total = 0
        self.stream = tempfile.TemporaryFile(mode="w+b")
        self.error = None

    def append(self, block):
        self.total += len(block)
        head = min(len(block), self.head_limit - self.head_size)
        if head:
            self.stream.seek(self.head_size)
            self.stream.write(block[:head])
            self.head_size += head
            block = block[head:]
        if not block or not self.tail_limit:
            return
        if len(block) >= self.tail_limit:
            self.stream.seek(self.head_limit)
            self.stream.write(block[-self.tail_limit :])
            self.tail_size, self.tail_cursor = self.tail_limit, 0
            return
        first = min(len(block), self.tail_limit - self.tail_cursor)
        self.stream.seek(self.head_limit + self.tail_cursor)
        self.stream.write(block[:first])
        if first < len(block):
            self.stream.seek(self.head_limit)
            self.stream.write(block[first:])
        self.tail_cursor = (self.tail_cursor + len(block)) % self.tail_limit
        self.tail_size = min(self.tail_limit, self.tail_size + len(block))

    def drain(self, pipe):
        try:
            while block := os.read(pipe.fileno(), 64 * 1024):
                self.append(block)
        except (OSError, ValueError) as error:
            self.error = type(error).__name__

    def text(self):
        self.stream.seek(0)
        head = self.stream.read(self.head_size)
        if self.tail_size == self.tail_limit and self.tail_cursor:
            self.stream.seek(self.head_limit + self.tail_cursor)
            tail = self.stream.read(self.tail_limit - self.tail_cursor)
            self.stream.seek(self.head_limit)
            tail += self.stream.read(self.tail_cursor)
        else:
            self.stream.seek(self.head_limit)
            tail = self.stream.read(self.tail_size)
        return (head + tail).decode("utf-8", errors="replace")


def clean_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG") if key in os.environ}


def _kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS can return EPERM for a group whose leader has already exited.
        # A live child refusal is still an error; container runners additionally
        # remove their owned daemon-side container after any resource violation.
        if proc.poll() is None:
            raise


def _write_input(pipe, payload):
    try:
        if payload:
            view = memoryview(payload)
            while view:
                written = os.write(pipe.fileno(), view)
                view = view[written:]
    except (BrokenPipeError, OSError, ValueError):
        pass
    finally:
        pipe.close()


def run_process(
    argv: list[str],
    timeout: float,
    *,
    cwd=None,
    stdin=None,
    env=None,
    max_output_bytes: int = 16 * 1024**2,
    output_limits: OutputLimits | None = None,
    monitor: Callable[[], str | None] | None = None,
    poll_interval: float = 0.1,
) -> ProcessResult:
    """Run argv with bounded capture and process-group cancellation.

    max_output_bytes caps each stdout/stderr spool's raw bytes. Truncated output
    retains its head and tail with separate flags; structured consumers must reject
    truncation. Quota/monitor checks run at most every 0.5 seconds and at completion.
    Quota violation is distinct from timeout, even when a fast process exits first.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Process timeout must be positive and finite")
    if (
        not isinstance(max_output_bytes, int)
        or isinstance(max_output_bytes, bool)
        or max_output_bytes < 0
    ):
        raise ValueError("Captured output limit must be a nonnegative integer")
    if not math.isfinite(poll_interval) or not 0 < poll_interval <= 0.5:
        raise ValueError("Process monitor poll interval must be in (0, 0.5]")
    if stdin is not None and not isinstance(stdin, (str, bytes)):
        raise TypeError("Process stdin must be text or bytes")
    payload = stdin.encode("utf-8") if isinstance(stdin, str) else stdin
    started = time.monotonic()

    def check_limits():
        try:
            reason = output_limits.check() if output_limits is not None else None
            if reason is None and monitor is not None:
                reason = monitor()
            if reason is not None and (not isinstance(reason, str) or not reason):
                return "Output monitor returned an invalid decision"
            return reason
        except Exception as error:
            return "Output monitor failed: " + type(error).__name__

    initial = check_limits()
    if initial:
        return ProcessResult(-1, "", "", time.monotonic() - started, limit_exceeded=initial)
    captures = [BoundedOutputCapture(max_output_bytes), BoundedOutputCapture(max_output_bytes)]
    proc, threads = None, []
    timed_out, limit_exceeded = False, None
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env if env is not None else clean_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            bufsize=0,
        )
        for capture, pipe in zip(captures, (proc.stdout, proc.stderr), strict=True):
            thread = threading.Thread(target=capture.drain, args=(pipe,), daemon=True)
            thread.start()
            threads.append(thread)
        writer = threading.Thread(target=_write_input, args=(proc.stdin, payload), daemon=True)
        writer.start()
        threads.append(writer)
        next_monitor = started
        while True:
            now = time.monotonic()
            if any(capture.error for capture in captures):
                limit_exceeded = "Output capture failed"
                _kill_group(proc)
                break
            if now >= next_monitor:
                limit_exceeded = check_limits()
                next_monitor = time.monotonic() + poll_interval
                if limit_exceeded:
                    _kill_group(proc)
                    break
            finished = proc.poll() is not None and not any(thread.is_alive() for thread in threads)
            if finished:
                limit_exceeded = check_limits()
                if limit_exceeded:
                    _kill_group(proc)
                break
            if now - started >= timeout:
                timed_out = True
                _kill_group(proc)
                break
            threading.Event().wait(min(poll_interval, max(0.001, timeout - (now - started))))
        proc.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in threads):
            limit_exceeded = limit_exceeded or "Output streams remained open after cancellation"
        return ProcessResult(
            proc.returncode,
            captures[0].text(),
            captures[1].text(),
            time.monotonic() - started,
            timed_out=timed_out,
            stdout_truncated=captures[0].total > max_output_bytes,
            stderr_truncated=captures[1].total > max_output_bytes,
            stdout_bytes=captures[0].total,
            stderr_bytes=captures[1].total,
            limit_exceeded=limit_exceeded,
        )
    except BaseException:
        if proc is not None:
            _kill_group(proc)
            proc.wait(timeout=10)
        raise
    finally:
        if proc is not None:
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None and not pipe.closed:
                    pipe.close()
        for thread in threads:
            thread.join(timeout=1)
        for capture in captures:
            capture.stream.close()
