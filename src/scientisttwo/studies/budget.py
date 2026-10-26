"""Separate measured development compute from bounded post-freeze evaluation.

Charged seconds are elapsed experiment-operation wall time (training, inference,
scoring and their overhead), a proxy for allocated worker/GPU time, not GPU cycles.
Model latency and final PDF compilation are governed by wall/call/token ceilings,
and do not spend the development experiment allowance.
"""

from contextlib import contextmanager
import math
import re
import time

from ..live.runtime import AmbiguousCall, Budget, LiveBudgetExhausted
from ..store import IntegrityError, canonical, digest

_FINAL_SEED = re.compile(r"final/(baseline|selected)-seed-[012]\Z")
_FINAL_AGGREGATE = re.compile(r"final/(baseline|selected)\Z")


class StudyBudget(Budget):
    def __init__(self, store, config, shared=None, *, max_experiment_seconds,
                 max_final_seconds, max_final_wall_seconds):
        limits = {"max_experiment_seconds": max_experiment_seconds,
                  "max_final_seconds_per_label": max_final_seconds,
                  "max_final_wall_seconds": max_final_wall_seconds}
        if any(type(value) not in {int, float} or not math.isfinite(value) or value <= 0
               for value in limits.values()):
            raise ValueError("Study time ceilings must be explicitly finite and positive")
        super().__init__(store, config, shared)
        self.policy = {**{key: float(value) for key, value in limits.items()},
                       "accounting": "sequential-experiment-wall-seconds-v1",
                       "scope": "Elapsed experiment wall-time proxy, not GPU cycles",
                       "final_policy": "Separate baseline and selected pools; three seeds each; zero-cost aggregates"}
        previous = store.get("study_compute_budget")
        if previous is not None and previous != self.policy:
            raise IntegrityError("Study compute ceilings cannot change on resume")
        store.db.execute("""CREATE TABLE IF NOT EXISTS study_compute_operations(
            key TEXT PRIMARY KEY, bucket TEXT NOT NULL, aggregation INTEGER NOT NULL)""")
        store.db.commit()
        unmapped = store.db.execute("""SELECT 1 FROM live_operations op
            LEFT JOIN study_compute_operations compute ON compute.key=op.key
            WHERE op.kind='experiment' AND compute.key IS NULL""").fetchone()
        if unmapped:
            raise IntegrityError("Cannot add or reconstruct a compute allowance after untracked experiments")
        store.set("study_compute_budget", self.policy)
        self._clock = time.monotonic
        self._scopes = []
        self._active = {}

    def activate_final_grant(self, grant_ref, deadline):
        """Bind a separately approved final deadline after coordinator verification.

        The caller must verify the immutable both-arms grant first. This method
        additionally requires its committed release marker and rejects deadline
        extensions. The earlier campaign deadline remains unchanged.
        """
        if not grant_ref or self.store.get("study_final_release_ref") != grant_ref:
            raise IntegrityError("Final compute requires the committed coordinator release")
        if type(deadline) not in {int, float} or not math.isfinite(deadline):
            raise ValueError("Final wall deadline must be finite")
        binding = {"grant_ref": grant_ref, "deadline": float(deadline),
                   "max_final_wall_seconds": self.policy["max_final_wall_seconds"]}
        previous = self.store.get("study_final_budget_activation")
        if previous is not None:
            if previous != binding:
                raise IntegrityError("Final grant or wall deadline cannot change or be extended")
            return binding
        now = time.time()
        if deadline <= now or deadline > now + self.policy["max_final_wall_seconds"] + 1:
            raise ValueError("Final deadline exceeds its declared window or has already expired")
        self.store.set("study_final_budget_activation", binding)
        return binding

    def _spent(self, bucket):
        return float(self.store.db.execute("""SELECT coalesce(sum(op.seconds),0)
            FROM live_operations op JOIN study_compute_operations c ON c.key=op.key
            WHERE c.bucket=? AND c.aggregation=0 AND op.result_ref IS NOT NULL""",
                                           (bucket,)).fetchone()[0])

    def _limit(self, bucket):
        return self.policy["max_experiment_seconds" if bucket == "development"
                           else "max_final_seconds_per_label"]

    def _available(self, bucket):
        used = self._spent(bucket)
        now = self._clock()
        # Completed row seconds and the elapsed active scope overlap. Take their
        # maximum rather than adding them, so an operation is never counted twice.
        for active in [*self._scopes, *self._active.values()]:
            if active["bucket"] == bucket:
                used = max(used, active["spent_at_start"] + max(0.0, now - active["started"]))
        return self._limit(bucket) - used

    @contextmanager
    def experiment_scope(self, *, final_label=None):
        if final_label not in {None, "baseline", "selected"}:
            raise ValueError("Unknown study experiment scope")
        bucket = "development" if final_label is None else "final/" + final_label
        if self._scopes and self._scopes[-1]["bucket"] != bucket:
            raise IntegrityError("Development and final execution scopes cannot overlap")
        scope = {"bucket": bucket, "started": self._clock(), "spent_at_start": self._spent(bucket)}
        self._scopes.append(scope)
        try:
            yield
        finally:
            self._scopes.pop()
            # An interrupted operation remains running in the durable ledger;
            # new operations will reject it instead of guessing its outcome.
            if not self._scopes:
                self._active.clear()

    def remaining(self):
        final = self.store.get("study_final_budget_activation")
        if final is None:
            remaining = super().remaining()
        else:
            if self.store.get("study_final_release_ref") != final["grant_ref"]:
                raise IntegrityError("Committed final release changed")
            remaining = final["deadline"] - time.time()
            if remaining <= 0:
                raise LiveBudgetExhausted("Separate final-evaluation wall-clock budget exhausted")
            if self.shared is not None:
                remaining = min(remaining, self.shared.remaining())
        buckets = {scope["bucket"] for scope in self._scopes}
        buckets.update(scope["bucket"] for scope in self._active.values())
        for bucket in buckets:
            available = self._available(bucket)
            if available <= 0:
                raise LiveBudgetExhausted("Study experiment wall-time allowance exhausted: " + bucket)
            remaining = min(remaining, available)
        return remaining

    def _classification(self, key, request):
        aggregate = request.get("execution_type") == "aggregation"
        if key.startswith("final/"):
            match = (_FINAL_AGGREGATE if aggregate else _FINAL_SEED).fullmatch(key)
            if not match:
                raise ValueError("Final compute is limited to baseline/selected and fixed seeds 0,1,2")
            return "final/" + match[1], aggregate
        if aggregate:
            raise ValueError("Only fixed final aggregation records can have zero compute charge")
        return "development", False

    def reserve(self, key, kind, request):
        previous = self.store.db.execute("SELECT * FROM live_operations WHERE key=?", (key,)).fetchone()
        if previous is not None:
            if kind == "experiment":
                bucket, aggregate = self._classification(key, request)
                mapped = self.store.db.execute("SELECT * FROM study_compute_operations WHERE key=?", (key,)).fetchone()
                if mapped is None or mapped["bucket"] != bucket or bool(mapped["aggregation"]) != aggregate:
                    raise IntegrityError("Existing experiment compute classification changed")
            return super().reserve(key, kind, request)
        released = self.store.get("study_final_release_ref")
        if released and (kind == "model" or kind == "experiment" and not key.startswith("final/")):
            raise IntegrityError("No model or non-final experiment is permitted after final release")
        if kind != "experiment":
            return super().reserve(key, kind, request)
        bucket, aggregate = self._classification(key, request)
        if bucket.startswith("final/") and not self.store.get("study_final_budget_activation"):
            raise IntegrityError("Final execution requires a verified independent final-budget activation")
        if not aggregate:
            if self._available(bucket) <= 0:
                raise LiveBudgetExhausted("Study experiment wall-time allowance exhausted: " + bucket)
            running = self.store.db.execute("""SELECT op.key FROM live_operations op
                JOIN study_compute_operations c ON c.key=op.key
                WHERE op.status='running' AND c.aggregation=0""").fetchone()
            if running:
                raise AmbiguousCall("A physical experiment has an unresolved completion: " + running["key"])
        self.remaining()
        if bucket == "development":
            count = self.store.db.execute("SELECT count(*) FROM study_compute_operations WHERE bucket='development'").fetchone()[0]
            if count >= self.config.max_experiments:
                raise LiveBudgetExhausted("Development experiment operation limit exhausted; failures consume budget")
        # Fixed final seeds have their own finite key space; their common scoring
        # cost cannot consume one arm's development operation/compute allowance.
        if self.shared and self.shared.reserve(key, kind, request):
            raise AmbiguousCall("Shared reservation has no local completion: " + key)
        with self.store.db:
            self.store.db.execute("INSERT INTO live_operations(key,kind,request_hash,status) VALUES(?,?,?,'running')",
                                  (key, kind, digest(canonical(request))))
            self.store.db.execute("INSERT INTO study_compute_operations VALUES(?,?,?)", (key, bucket, int(aggregate)))
        if not aggregate:
            self._active[key] = {"bucket": bucket, "started": self._clock(), "spent_at_start": self._spent(bucket)}
        return None

    def finish(self, key, status, result, seconds, tokens=0):
        if type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Operation elapsed time must be finite and nonnegative")
        row = self.store.db.execute("SELECT * FROM study_compute_operations WHERE key=?", (key,)).fetchone()
        if row is not None and row["aggregation"] and seconds != 0:
            raise IntegrityError("Final aggregation must have zero charged seconds")
        super().finish(key, status, result, seconds, tokens)
        self._active.pop(key, None)

    def compute_summary(self):
        buckets = {}
        for bucket in ("development", "final/baseline", "final/selected"):
            used = self._spent(bucket)
            buckets[bucket] = {"charged_seconds": used, "limit_seconds": self._limit(bucket),
                               "remaining_seconds": max(0.0, self._available(bucket))}
        unknown = [row[0] for row in self.store.db.execute("""SELECT op.key FROM live_operations op
            JOIN study_compute_operations c ON c.key=op.key WHERE op.status='running' AND c.aggregation=0""")]
        return {"policy": self.policy, "buckets": buckets, "unknown_operations": unknown,
                "accounting_status": "incomplete" if unknown else "complete",
                "final_activation": self.store.get("study_final_budget_activation")}
