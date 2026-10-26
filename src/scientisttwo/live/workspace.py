"""Transport contracts for trusted development plugins, independent of benchmarks."""

import inspect
import json
from pathlib import Path

from ..benchmarks.files import FORBIDDEN, file_hash, tree_manifest
from ..benchmarks.models import relative_path
from .models import Candidate, SourceFile
from ..artifacts import ArtifactSpec


REQUIRED_METHODS = ("identity", "context", "baseline", "prepare", "validate", "score")
ENVIRONMENT_KEYS = {
    "SCIENTISTTWO_SEED",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
    "CUDA_VISIBLE_DEVICES",
}


def validate_adapter(adapter):
    missing = [name for name in REQUIRED_METHODS if not callable(getattr(adapter, name, None))]
    if missing:
        raise ValueError("Development adapter missing methods: " + ", ".join(missing))
    identity = adapter.identity()
    if (
        not isinstance(identity, dict)
        or not identity
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in identity.items())
    ):
        raise ValueError("Adapter identity must be a nonempty string-to-string mapping")
    if not isinstance(adapter.context(), dict):
        raise ValueError("Adapter context must be a mapping")
    return adapter


def validate_baseline_contract(adapter):
    """Validate preparation without fabricating a store-local source snapshot reference."""
    declaration = getattr(adapter, "source_tree", lambda: None)()
    if declaration is None:
        adapter.validate(adapter.baseline())
        return
    if not callable(getattr(adapter, "bind_source", None)):
        raise ValueError("Source-tree adapters must bind their frozen snapshot before execution")
    root = Path(declaration["path"])
    if not root.is_absolute() or not tree_manifest(root):
        raise ValueError("Source-tree baseline requires a nonempty absolute source directory")
    from ..artifacts import safe_path
    for name in declaration["editable_paths"]:
        safe_path(name)


def live_readiness(adapter, *, require=False):
    """Allow trusted plugins to require measured readiness before live invocation."""
    callback = getattr(adapter, "live_readiness", None)
    result = callback() if callable(callback) else {
        "ready": True, "scope": "No additional adapter readiness gate declared"
    }
    if not isinstance(result, dict) or type(result.get("ready")) is not bool:
        raise ValueError("Adapter live readiness must return a boolean ready field")
    if require and not result["ready"]:
        raise ValueError("Adapter is not ready for live execution: " +
                         "; ".join(str(item) for item in result.get("blockers", result.get("errors", []))))
    return result


def execution_descriptor(adapter, prepared, seed):
    """Describe container-only execution; old plugins default to candidate argv."""
    callback = getattr(adapter, "execution", None)
    descriptor = callback(prepared, seed) if callable(callback) else {"kind": "command"}
    if not isinstance(descriptor, dict) or set(descriptor) - {
        "kind",
        "source",
        "parameters",
        "environment",
    }:
        raise ValueError("Invalid development execution descriptor")
    result = dict(descriptor)
    kind = result.get("kind")
    if kind not in {"command", "python_driver"}:
        raise ValueError("Execution kind must be command or python_driver")
    if kind == "python_driver":
        source = result.get("source")
        if not isinstance(source, str) or not source or len(source.encode()) > 250_000:
            raise ValueError("Trusted driver source is missing or oversized")
        # Compilation validates syntax only; candidate or driver code is never executed here.
        compile(source, "<trusted-container-driver>", "exec")
    elif "source" in result:
        raise ValueError("Command execution cannot supply a driver")
    parameters = result.get("parameters", {})
    if not isinstance(parameters, dict) or len(json.dumps(parameters, allow_nan=False)) > 1_000_000:
        raise ValueError("Driver parameters must be a bounded JSON mapping")
    environment = result.get("environment", {})
    if (
        not isinstance(environment, dict)
        or set(environment) - ENVIRONMENT_KEYS
        or any(
            not isinstance(value, str) or "\x00" in value or len(value) > 256
            for value in environment.values()
        )
    ):
        raise ValueError("Driver environment must use allowed reproducibility settings")
    result["parameters"] = parameters
    result["environment"] = {**environment, "SCIENTISTTWO_SEED": str(seed)}
    return result


def artifact_paths(adapter, prepared):
    """Declare every prediction input needed for trusted scoring and audit replay."""
    callback = getattr(adapter, "prediction_artifacts", None)
    paths = (
        callback(prepared)
        if callable(callback)
        else [getattr(getattr(adapter, "native", None), "submission", "submission.csv")]
    )
    if not isinstance(paths, list) or not 1 <= len(paths) <= 32:
        raise ValueError("Declare between one and 32 prediction artifacts")
    result = []
    for path in paths:
        if not isinstance(path, str):
            raise ValueError("Prediction artifact paths must be strings")
        path = relative_path(path)
        if any(part.startswith(".") or part.lower() in FORBIDDEN for part in Path(path).parts):
            raise ValueError("Hidden or credential-like prediction paths are forbidden")
        result.append(path)
    if len(result) != len(set(result)):
        raise ValueError("Duplicate prediction artifact path")
    return result


def artifact_contracts(adapter, prepared):
    callback = getattr(adapter, "artifacts", None)
    if callable(callback):
        values = callback(prepared)
        if not isinstance(values, list) or not values:
            raise ValueError("Declare a nonempty artifact contract list")
        return [ArtifactSpec.model_validate(value) for value in values]
    return [ArtifactSpec(path=name, max_bytes=64_000_000, max_files=1) for name in artifact_paths(adapter, prepared)]


class RegisteredDevelopment:
    """Bind a trusted plugin's declared identity to installed implementation provenance."""

    def __init__(self, adapter, entry):
        self.adapter = validate_adapter(adapter)
        path = inspect.getsourcefile(type(adapter))
        if not path or not Path(path).is_file():
            raise ValueError("Trusted plugin must expose a readable implementation source file")
        distribution = getattr(entry, "dist", None)
        if distribution is None:
            raise ValueError("Trusted plugin must have installed distribution metadata")
        name = distribution.metadata.get("Name")
        if not name or not distribution.version:
            raise ValueError("Trusted plugin distribution name/version unavailable")
        self.provenance = {
            "plugin_entry_point": str(entry.name),
            "plugin_entry_value": str(entry.value),
            "plugin_distribution": str(name),
            "plugin_version": str(distribution.version),
            "plugin_source": file_hash(Path(path)),
        }

    def identity(self):
        return {**self.adapter.identity(), **self.provenance}

    def __getattr__(self, name):
        return getattr(self.adapter, name)


class BaselineOverride:
    """Read a user-declared frozen baseline as source text, never importing its code."""

    def __init__(self, adapter, directory, command):
        self.adapter = adapter
        root = Path(directory)
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("baseline_dir must name an existing absolute source directory")
        manifest = tree_manifest(root)
        paths = sorted(manifest)
        files = []
        for path in paths:
            full = root / path
            if full.stat().st_size > 250_000:
                raise ValueError("Baseline source file is oversized")
            files.append(SourceFile(path=path, content=full.read_text()))
        self.candidate = Candidate(
            files=files, command=command, description="Explicit frozen reference implementation"
        )
        adapter.validate(self.candidate)
        self.provenance = {"baseline_source/" + name: file_hash(root / name) for name in paths}

    def identity(self):
        return {**self.adapter.identity(), **self.provenance}

    def baseline(self):
        return self.candidate.model_copy(deep=True)

    def __getattr__(self, name):
        return getattr(self.adapter, name)


def with_baseline(adapter, config, native=None):
    source_tree = getattr(adapter, "source_tree", None)
    if callable(source_tree) and source_tree() is not None:
        if config.settings.get("baseline_dir"):
            raise ValueError("A source-tree adapter owns its baseline declaration")
        return adapter
    directory, command = (
        config.settings.get("baseline_dir"),
        config.settings.get("baseline_command"),
    )
    if (directory is None) != (command is None):
        raise ValueError("baseline_dir and baseline_command must be declared together")
    if directory is None and getattr(native, "benchmark", None) == "workspace":
        directory, command = native.candidate_dir, native.command
        if directory is not None and not command:
            raise ValueError("A workspace baseline source directory requires its command")
    return BaselineOverride(adapter, directory, command) if directory is not None else adapter
