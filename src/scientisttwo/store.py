"""Content-addressed artifacts and transactional checkpoints, with input-bound memoization."""

import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

from .models import CampaignState


def canonical(value) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class IntegrityError(RuntimeError):
    pass


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts").mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.root / "campaign.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS artifacts(path TEXT PRIMARY KEY, sha TEXT NOT NULL,
                                                kind TEXT NOT NULL, size INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS calls(key TEXT PRIMARY KEY, input_hash TEXT NOT NULL,
                                            output_ref TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(key TEXT PRIMARY KEY, time REAL NOT NULL,
                                             stage TEXT NOT NULL, reason TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(key TEXT PRIMARY KEY, request_hash TEXT NOT NULL,
                status TEXT NOT NULL, started REAL NOT NULL, reserved REAL NOT NULL,
                charged REAL NOT NULL DEFAULT 0, result_ref TEXT);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def materialize_artifact(self, name: str):
        """Ensure a registered payload is local, for streaming artifact consumers.

        Ordinary stores already own their files. Scratch replay stores may
        override this hook to copy a verified payload lazily; callers still
        verify registration, path, size and hash after materialization.
        """
        return None

    def put(self, data: bytes, suffix: str = "json", kind: str = "evidence") -> str:
        sha = digest(data)
        name = f"artifacts/{sha}.{suffix}"
        path = self.root / name
        if path.exists():
            if digest(path.read_bytes()) != sha:
                raise IntegrityError(f"Modified immutable artifact: {name}")
        else:
            # A crash can leave only the temporary file; no committed ref points to it.
            temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
            with temporary.open("wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)", (name, sha, kind, len(data))
            )
        return name

    def put_json(self, value, kind="evidence") -> str:
        return self.put(canonical(value), kind=kind)

    def put_file(self, source: Path, kind="file") -> str:
        """Stream large native benchmark artifacts without loading model weights into memory."""
        source = Path(source)
        h = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        sha = h.hexdigest()
        suffix = source.suffix.lstrip(".") or "bin"
        name = f"artifacts/{sha}.{suffix}"
        target = self.root / name
        if target.exists() and file_digest(target) != sha:
            raise IntegrityError(f"Modified immutable artifact: {name}")
        if not target.exists():
            temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
            shutil.copyfile(source, temporary)
            check = hashlib.sha256()
            with temporary.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    check.update(chunk)
            if check.hexdigest() != sha:
                temporary.unlink()
                raise IntegrityError("Artifact changed during snapshot")
            os.replace(temporary, target)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)",
                (name, sha, kind, source.stat().st_size),
            )
        return name

    def read(self, name: str) -> bytes:
        row = self.db.execute("SELECT * FROM artifacts WHERE path=?", (name,)).fetchone()
        path = (self.root / name).resolve()
        if row is None or not path.is_relative_to(self.root / "artifacts"):
            raise IntegrityError(f"Unregistered artifact: {name}")
        if not path.is_file() or digest(path.read_bytes()) != row["sha"]:
            raise IntegrityError(f"Missing or modified artifact: {name}")
        return path.read_bytes()

    def json(self, name: str):
        return json.loads(self.read(name))

    def set(self, key: str, value):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES(?,?)", (key, canonical(value).decode())
            )

    def get(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def cached(self, key: str, inputs):
        row = self.db.execute("SELECT * FROM calls WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        if row["input_hash"] != digest(canonical(inputs)):
            raise IntegrityError(f"Checkpoint inputs changed: {key}")
        return self.json(row["output_ref"])

    def cache(self, key: str, inputs, output):
        ref = self.put_json(output, "checkpoint")
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO calls VALUES(?,?,?)", (key, digest(canonical(inputs)), ref)
            )
        return self.cached(key, inputs)

    def event(self, key: str, stage: str, reason: str):
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO events VALUES(?,?,?,?)", (key, time.time(), stage, reason)
            )

    def state(self) -> CampaignState:
        return CampaignState.model_validate(self.get("state"))

    def verify(self) -> list[str]:
        errors = []
        for row in self.db.execute("SELECT path,sha FROM artifacts"):
            path = (self.root / row["path"]).resolve()
            if not path.is_relative_to(self.root / "artifacts") or not path.is_file():
                errors.append(f"Missing or invalid artifact: {row['path']}")
            elif file_digest(path) != row["sha"]:
                errors.append(f"Missing or modified artifact: {row['path']}")
        return errors
