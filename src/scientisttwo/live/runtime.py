"""Persistent reservations and isolated native development execution."""

from pathlib import Path
import time

from ..benchmarks.files import tree_manifest, verify_tree, file_hash
from ..benchmarks.runner import NativeRunner
from ..process import run_process
from ..store import IntegrityError, canonical, digest
from .models import Candidate, Observation
from .evaluation import aggregate_scores, normalize_scores
from .workspace import execution_descriptor, artifact_contracts
from .workspace_task import preflight, WorkspaceRunner
from .workspace_models import WorkspaceSpec
from .source_tree import candidate_snapshot, freeze_source_tree, restore_source_tree
from ..artifacts import freeze_bundle, load_bundle
from ..runtime_contracts import asset_mounts, runtime_identity
from .pipeline import declared_pipeline, evaluate_pipeline, pipeline_identity, PipelineInterrupted


class LiveBudgetExhausted(RuntimeError):
    pass


class AmbiguousCall(RuntimeError):
    pass


def validate_prepared(prepared, destination):
    """An adapter may expose only its newly prepared phase tree to candidates."""
    if (
        not isinstance(prepared, dict)
        or not isinstance(prepared.get("files"), dict)
        or not prepared["files"]
    ):
        raise ValueError("Prepared development data require a nonempty frozen file manifest")
    public = Path(prepared["public"])
    root = Path(destination).resolve()
    if (
        not public.is_absolute()
        or not public.is_dir()
        or public.is_symlink()
        or not public.resolve().is_relative_to(root)
    ):
        raise ValueError(
            "Prepared public data must be inside the owned development phase directory"
        )
    verify_tree(public, prepared["files"])
    return prepared


class Budget:
    def __init__(self, store, config, shared=None):
        self.store, self.config, self.shared = store, config, shared
        store.db.execute("""CREATE TABLE IF NOT EXISTS live_operations(
            key TEXT PRIMARY KEY, kind TEXT NOT NULL, request_hash TEXT NOT NULL,
            status TEXT NOT NULL, seconds REAL NOT NULL DEFAULT 0,
            tokens INTEGER NOT NULL DEFAULT 0, result_ref TEXT)""")
        store.db.commit()

    def remaining(self):
        deadline = self.store.get("live_deadline")
        if deadline is None:
            deadline = time.time() + self.config.campaign_seconds
            self.store.set("live_deadline", deadline)
        remaining = deadline - time.time()
        if remaining <= 0:
            raise LiveBudgetExhausted(
                "Campaign wall-clock budget exhausted (including interruption time)"
            )
        return min(remaining, self.shared.remaining()) if self.shared else remaining

    def reserve(self, key, kind, request):
        if kind not in {"model", "experiment", "probe"}:
            raise ValueError("Unknown campaign operation kind")
        row = self.store.db.execute("SELECT * FROM live_operations WHERE key=?", (key,)).fetchone()
        if row:
            if row["request_hash"] != digest(canonical(request)) or row["kind"] != kind:
                raise IntegrityError("Operation inputs changed on resume")
            if self.shared and row["result_ref"]:
                self.shared.reserve(key, kind, request)
                self.shared.finish(
                    key,
                    row["status"],
                    self.store.json(row["result_ref"]),
                    row["seconds"],
                    row["tokens"],
                )
            return row
        self.remaining()
        count = self.store.db.execute(
            "SELECT count(*) FROM live_operations WHERE kind=?", (kind,)
        ).fetchone()[0]
        limit = self.config.max_calls if kind == "model" else self.config.max_experiments
        if kind != "probe" and count >= limit:
            raise LiveBudgetExhausted(
                f"{kind} operation limit exhausted; failed attempts consume budget"
            )
        tokens = self.store.db.execute(
            "SELECT coalesce(sum(tokens),0) FROM live_operations"
        ).fetchone()[0]
        if (
            kind == "model"
            and self.config.token_stop_threshold
            and tokens >= self.config.token_stop_threshold
        ):
            raise LiveBudgetExhausted("Observed token stop threshold reached")
        if self.shared:
            reserved = self.shared.reserve(key, kind, request)
            if reserved:
                raise AmbiguousCall(
                    "Suite reservation has no local completion; refusing to repeat: " + key
                )
        with self.store.db:
            self.store.db.execute(
                "INSERT INTO live_operations(key,kind,request_hash,status) VALUES(?,?,?,'running')",
                (key, kind, digest(canonical(request))),
            )
        return None

    def finish(self, key, status, result, seconds, tokens=0):
        ref = self.store.put_json(result, "live-operation")
        previous = self.store.db.execute(
            "SELECT * FROM live_operations WHERE key=?", (key,)
        ).fetchone()
        if previous is None:
            raise IntegrityError("Cannot finish an unreserved operation")
        if previous["result_ref"] is not None and (
            previous["result_ref"] != ref
            or previous["status"] != status
            or previous["tokens"] != tokens
            or previous["seconds"] != seconds
        ):
            raise IntegrityError("Duplicate completion differs from the committed operation")
        with self.store.db:
            self.store.db.execute(
                "UPDATE live_operations SET status=?,seconds=?,tokens=?,result_ref=? WHERE key=?",
                (status, seconds, tokens, ref, key),
            )
        if self.shared:
            self.shared.finish(key, status, result, seconds, tokens)

    def summary(self):
        return [
            dict(r) for r in self.store.db.execute("SELECT * FROM live_operations ORDER BY rowid")
        ]


class DevelopmentRunner:
    # Reuse the exact isolation policy of native execution without exposing its
    # private grader mounts or combined execute-and-grade entry point.
    _launch = NativeRunner._launch
    _image = NativeRunner._image

    def __init__(self, store, config, adapter, budget, executor=run_process):
        self.store, self.config, self.adapter, self.budget = store, config, adapter, budget
        self.spec = config.native
        self.raw_executor = executor
        self.prefix = [self.spec.runtime] + (
            ["--context", self.spec.context] if self.spec.context else []
        )
        self.image = None
        self.hardware = None

    def executor(self, argv, timeout, **kwargs):
        if argv[:len(self.prefix)] == self.prefix and argv[len(self.prefix):len(self.prefix) + 2] == ["rm", "--force"]:
            return self.raw_executor(argv, timeout, **kwargs)
        return self.raw_executor(argv, min(timeout, self.budget.remaining()), **kwargs)

    def materialize(self, key, proposal):
        """Record a declared adapter transformation before any code execution."""
        proposal_ref = self.store.put_json(proposal, "candidate-proposal")
        transform = getattr(self.adapter, "materialize", None)
        candidate = transform(proposal) if transform is not None else proposal
        candidate = Candidate.model_validate(candidate.model_dump())
        if candidate.base_snapshot_ref is not None and candidate.base_snapshot_ref != self.store.get("authorized_source_snapshot"):
            raise IntegrityError("Candidate refers to an unauthorized source snapshot")
        self.adapter.validate(candidate)
        tree = candidate_snapshot(self.store, candidate)
        tree_ref = self.store.put_json(tree, "materialized-source-tree")
        code_ref = tree_ref if candidate.base_snapshot_ref else self.store.put_json(candidate, "candidate-source")
        request = {"proposal_ref": proposal_ref, "adapter": self.adapter.identity()}
        evidence = {"proposal_ref": proposal_ref, "code_ref": code_ref, "source_tree_ref": tree_ref}
        saved = self.store.cached("materialization/" + key, request)
        if saved is not None and saved != evidence:
            raise IntegrityError("Candidate materialization changed on resume")
        self.store.cache("materialization/" + key, request, evidence)
        return candidate, code_ref

    def preflight(self):
        checks = preflight(self.spec)
        if not checks["ready"]:
            raise ValueError("Native prerequisites unavailable: " + "; ".join(checks["errors"]))
        images = {
            "execution": self._image(self.spec.image),
            "grading": self._image(self.spec.grader_image),
        }
        if self.store.get("live_images") not in (None, images):
            raise IntegrityError("Runtime images changed; create a new live campaign")
        self.store.set("live_images", images)
        self.image = images["execution"]
        declaration = getattr(self.adapter, "source_tree", lambda: None)()
        if declaration:
            ref = freeze_source_tree(self.store, Path(declaration["path"]), declaration["editable_paths"])
            if self.store.get("authorized_source_snapshot") not in (None, ref):
                raise IntegrityError("Frozen baseline source tree changed")
            self.store.set("authorized_source_snapshot", ref)
            self.adapter.bind_source(ref)
        self.hardware = self.store.get("runtime_hardware", {"status": "unverified"})
        profile = self.config.development.settings.get("runtime_profile")
        if profile:
            from ..environment_probe import probe_runtime

            self.hardware = probe_runtime(self, profile)
        self.adapter.validate(self.adapter.baseline())
        for phase in ("subset", "full"):
            destination = self.store.root / "development" / phase
            ref = self.store.get("prepared/" + phase)
            if ref:
                prepared = self.store.json(ref)
                validate_prepared(prepared, destination)
            else:
                prepared = validate_prepared(self.adapter.prepare(phase, destination), destination)
                self.store.set(
                    "prepared/" + phase, self.store.put_json(prepared, "development-partition")
                )

    def execute(self, key, candidate, phase="full", seeds=None):
        paired = getattr(self.adapter, "evaluate_pair", None)
        if callable(paired):
            result = paired(self, key, candidate, phase, seeds)
            if result is not None:
                return Observation.model_validate(result)
        c = self.config.development
        seeds = seeds or (c.seeds[:1] if phase == "subset" else c.seeds)
        prepared = self.store.json(
            self.store.get("prepared/" + ("subset" if phase == "subset" else "full"))
        )
        validate_prepared(
            prepared, self.store.root / "development" / ("subset" if phase == "subset" else "full")
        )
        protocol_spec = {
            "adapter": self.adapter.identity(),
            "data": prepared["files"],
            "seeds": seeds,
            "evaluation": c.model_dump(mode="json"),
            "phase": phase,
            "runtime": runtime_identity(self.spec, self.store.get("live_images"), self.hardware),
            "execution": [execution_descriptor(self.adapter, prepared, seed) for seed in seeds],
            "artifacts": [spec.model_dump(mode="json") for spec in artifact_contracts(self.adapter, prepared)],
        }
        pipeline = declared_pipeline(self.adapter, prepared)
        if pipeline:
            if pipeline.private_dir and not pipeline.private_dir.resolve().is_relative_to((self.store.root / "development" / ("subset" if phase == "subset" else "full")).resolve()):
                raise ValueError("Development scoring labels must belong to the owned phase directory")
            protocol_spec["pipeline"] = pipeline_identity(self, pipeline)
        protocol = digest(canonical(protocol_spec))
        request = {"candidate": candidate.model_dump(), "protocol": protocol}
        cached = self.store.cached("development/" + key, request)
        if cached:
            return Observation.model_validate(cached)
        code_ref = self.store.put_json(candidate, "candidate-source")
        result = Observation(
            id=key,
            phase=phase,
            status="invalid",
            protocol_hash=protocol,
            protocol_ref=self.store.put_json(protocol_spec, "evaluation-protocol"),
            code_ref=code_ref,
            metric=c.metric,
        )
        prior = self.budget.reserve("experiment/" + key, "experiment", request)
        if prior:
            if prior["result_ref"]:
                return Observation.model_validate(self.store.json(prior["result_ref"]))
            result.status, result.reason = (
                "interrupted",
                "Uncommitted execution outcome; attempt will not be repeated",
            )
            return result
        started = time.monotonic()
        budget_error = None
        try:
            candidate, result.code_ref = self.materialize("development/" + key, candidate)
            materialized = self.store.json(self.store.db.execute("SELECT output_ref FROM calls WHERE key=?", ("materialization/development/" + key,)).fetchone()[0])
            result.source_tree_ref = materialized["source_tree_ref"]
            verify_tree(prepared["public"], prepared["files"])
            scores = []
            for seed in seeds:
                self.budget.remaining()
                stage = "dev-" + digest((key + str(seed)).encode())[:24]
                work = self.store.root / "work" / stage
                work.mkdir(parents=True, exist_ok=False)
                restore_source_tree(self.store, result.source_tree_ref, work)
                for path in [work, *work.rglob("*")]:
                    if path.is_dir():
                        path.chmod(0o777)
                execution = execution_descriptor(self.adapter, prepared, seed)
                if execution["kind"] == "python_driver":
                    control = self.store.root / "control" / stage
                    control.mkdir(parents=True, exist_ok=True)
                    (control / "driver.py").write_text(execution["source"])
                process, receipt = self._launch(
                    stage,
                    self.image,
                    {
                        "seed": seed,
                        "command": candidate.command,
                        "execution": execution,
                    },
                    [(work, "/workspace", "rw"), (Path(prepared["public"]), "/data/public", "ro")]
                    + asset_mounts(self.spec, "execute", protected=[self.spec.private_dir, getattr(self.spec, "evaluator_dir", None), pipeline.private_dir if pipeline else None]),
                    min(self.spec.resources.seconds, self.budget.remaining()),
                    gpus=self.spec.resources.gpus,
                    bridge=Path(__file__).with_name("worker.py"),
                )
                result.log_refs.append(receipt)
                if process.code or process.timed_out:
                    result.status = "timeout" if process.timed_out else "execution_failed"
                    result.reason = "Candidate process failed; consult execution log"
                    break
                source_manifest = load_bundle(self.store, self.store.json(result.source_tree_ref)["bundle_ref"])
                for name, original in source_manifest.files.items():
                    source = work / name
                    if source.is_symlink() or not source.is_file() or file_hash(source) != original.sha256:
                        raise IntegrityError("Candidate modified its frozen source during execution")
                contracts = artifact_contracts(self.adapter, prepared)
                self.budget.remaining()
                bundle_ref = freeze_bundle(self.store, work, contracts)
                if pipeline:
                    frozen_source = self.store.root / "inference-source" / stage
                    restore_source_tree(self.store, result.source_tree_ref, frozen_source)
                    evaluated = evaluate_pipeline(self, key + "/" + str(seed), pipeline, bundle_ref, Path(prepared["public"]), candidate_source=frozen_source, seed=seed)
                    score = normalize_scores(c, evaluated["metrics"])
                    result.evaluation_refs.append(self.store.put_json(evaluated, "development-evaluation"))
                else:
                    score = normalize_scores(c, self.adapter.score(phase, prepared, work, process.seconds))
                self.budget.remaining()
                if freeze_bundle(self.store, work, contracts) != bundle_ref:
                    raise IntegrityError("Artifacts changed during scoring")
                manifest = load_bundle(self.store, bundle_ref)
                artifacts = {name: file.ref for name, file in manifest.files.items()}
                # Checkpoint bytes are training outputs, not scored predictions.
                # A staged evaluator names the complete frozen prediction bundle;
                # file-only host adapters retain their existing primary-file ref.
                if pipeline:
                    result.prediction_refs.append(evaluated["prediction_bundle"])
                else:
                    primary_path = contracts[0].path if contracts[0].kind == "file" else next(name for name in artifacts if name.startswith(contracts[0].path + "/"))
                    result.prediction_refs.append(artifacts[primary_path])
                result.artifact_refs.append(artifacts)
                result.bundle_refs.append(bundle_ref)
                scores.append(score)
            else:
                result.metrics, result.metric_values = aggregate_scores(c, scores)
                result.status, result.values, result.value = (
                    "success",
                    result.metric_values[c.metric],
                    result.metrics[c.metric],
                )
                result.reason = "Trusted development evaluator accepted predictions"
        except PipelineInterrupted as error:
            result.status, result.reason = "interrupted", str(error)
        except LiveBudgetExhausted as error:
            result.status, result.reason = "timeout", str(error)
            budget_error = error
        except (ValueError, OSError, KeyError, SyntaxError) as error:
            result.status, result.reason = "invalid", str(error)
        result.seconds = time.monotonic() - started
        result = Observation.model_validate(result.model_dump())
        self.budget.finish("experiment/" + key, result.status, result, result.seconds)
        self.store.cache("development/" + key, request, result)
        if budget_error is not None:
            raise budget_error
        return result

    def final(self, key, candidate):
        """No model calls occur after the controller enters this method."""
        paired = getattr(self.adapter, "evaluate_final", None)
        if callable(paired):
            result = paired(self, key, candidate)
            if result is not None:
                return result
        request = {"candidate": candidate.model_dump(), "native": self.spec.model_dump(mode="json")}
        prior = self.budget.reserve("final/" + key, "experiment", request)
        if prior and prior["result_ref"]:
            return self.store.json(prior["result_ref"])
        started = time.monotonic()
        try:
            return self._run_final(key, candidate)
        except LiveBudgetExhausted as error:
            result = {"status": "timeout", "reason": str(error),
                      "elapsed_seconds": time.monotonic() - started, "live_model_calls": 0}
            self.budget.finish("final/" + key, "timeout", result, result["elapsed_seconds"])
            raise

    def _run_final(self, key, candidate):
        self.budget.remaining()
        candidate, _ = self.materialize("final/" + key, candidate)
        path = self.store.root / "frozen" / key
        tree = candidate_snapshot(self.store, candidate)
        if not path.exists():
            restore_source_tree(self.store, tree, path)
        else:
            expected = {name: {"sha256": file.sha256, "size": file.size} for name, file in load_bundle(self.store, tree.bundle_ref).files.items()}
            verify_tree(path, expected)
        spec = self.spec.model_copy(
            update={
                "mode": "prepared_candidate",
                "candidate_dir": path,
                "candidate_manifest": tree_manifest(path),
                "command": candidate.command,
                "image": self.store.get("live_images")["execution"],
                "grader_image": self.store.get("live_images")["grading"],
            }
        )
        runner_type = WorkspaceRunner if isinstance(spec, WorkspaceSpec) else NativeRunner
        runner = runner_type(spec, self.store.root / "final" / key, executor=self.executor)
        try:
            result = runner.run().model_dump(mode="json")
        finally:
            runner.close()
        self.budget.finish("final/" + key, result["status"], result, result["elapsed_seconds"])
        return result
