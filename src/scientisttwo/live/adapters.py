"""Trusted development adapters. Never import or execute candidate code on the host."""

import ast
import zipfile
from importlib.metadata import entry_points, version
from pathlib import Path
from textwrap import indent
from typing import Protocol

import numpy as np

from ..benchmarks.files import file_hash, tree_manifest
from ..store import digest
from .models import Candidate, SourceFile
from .tabular import CSVDevelopment
from .drivers import VECTORIZATION_DRIVER
from .workspace import RegisteredDevelopment, validate_adapter, with_baseline


class DevelopmentAdapter(Protocol):
    """Plugins are trusted evaluator code, installed separately from agent proposals."""

    def identity(self) -> dict[str, str]: ...
    def context(self) -> dict: ...
    def baseline(self) -> Candidate: ...
    def prepare(self, phase: str, destination: Path) -> dict: ...
    def validate(self, candidate: Candidate) -> None: ...
    def score(
        self, phase: str, prepared: dict, workspace: Path, seconds: float
    ) -> float | dict[str, float]: ...
    def execution(self, prepared: dict, seed: int) -> dict: ...
    def prediction_artifacts(self, prepared: dict) -> list[str]: ...


class VectorizationDevelopment:
    def __init__(self, native, config):
        if config.metric != "process_seconds":
            raise ValueError("Vectorization uses independently measured process_seconds")
        self.native, self.config = native, config
        self.proposal_mode = config.settings.get("proposal_mode", "full_file")
        if self.proposal_mode not in ("full_file", "forward_body"):
            raise ValueError("Unknown vectorization proposal_mode; use full_file or forward_body")
        # Preserve the template's newline bytes when constructing body-only proposals.
        self.source = (native.public_dir / "train.py").read_bytes().decode("utf-8")

    def identity(self):
        return {
            "adapter": file_hash(Path(__file__)),
            "baseline": digest(self.source.encode()),
            "proposal_mode": self.proposal_mode,
            "driver": digest(VECTORIZATION_DRIVER.encode()),
        }

    def execution(self, prepared, seed):
        return {
            "kind": "python_driver",
            "source": VECTORIZATION_DRIVER,
            "parameters": {"cases": prepared["cases"]},
            "environment": {"SCIENTISTTWO_SEED": str(seed)},
        }

    def prediction_artifacts(self, prepared):
        return ["predictions.npz"]

    def context(self):
        context = {
            "contract": "Modify only Conv2DLayer.forward in train.py; preserve its interface and "
            "all other source. Use NumPy only. Development validates output against an independent "
            "convolution reference and measures whole isolated process time, including startup. "
            "Native final runtime is self-reported and is reported separately.",
            "starter_source": self.source,
        }
        if self.proposal_mode == "forward_body":
            context["contract"] += (
                " Proposal mode is forward_body. Return Candidate files=[{path:'train.py', "
                "content:<only unindented forward BODY statements>}] and "
                "command=['python','train.py']. Do not return a def forward signature, class, "
                "whole file, or Markdown fences. Nested helper functions inside the body are "
                "allowed. The host inserts this body into the unchanged starter template; "
                "self and features_batch are already in scope. An exact unchanged starter "
                "source is also accepted when requesting the baseline control."
            )
        context["proposal_mode"] = self.proposal_mode
        return context

    def baseline(self):
        return Candidate(
            files=[SourceFile(path="train.py", content=self.source)],
            command=["python", "train.py"],
            description="Unchanged upstream convolution",
        )

    def materialize(self, candidate):
        """Insert a declared method body without running any supplied Python code."""
        if self.proposal_mode == "full_file":
            return candidate
        self._validate_entry_point(candidate)
        body = candidate.files[0].content
        if body == self.source:
            return candidate
        body = body.replace("\r\n", "\n").replace("\r", "\n")
        try:
            wrapped = "def __scientisttwo_forward(self, features_batch):\n" + indent(body, "    ")
            tree = ast.parse(wrapped)
            method = tree.body[0]
            if len(tree.body) != 1 or not isinstance(method, ast.FunctionDef):
                raise ValueError("Expected only a forward body")
            if method.body[0].col_offset != 4:
                raise ValueError("Forward body statements must be unindented")
            if any(
                isinstance(node, ast.ClassDef)
                or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "forward"
                for statement in method.body
                for node in ast.walk(statement)
            ):
                raise ValueError(
                    "Provide only forward body statements, without class/forward wrappers"
                )
            # Compilation checks return/break/nonlocal scope; it never executes the candidate.
            compile(tree, "<forward-body-proposal>", "exec")
        except (SyntaxError, TypeError) as exc:
            raise ValueError(f"Invalid forward body: {exc}") from exc

        original = ast.parse(self.source)
        methods = [
            method
            for cls in original.body
            if isinstance(cls, ast.ClassDef) and cls.name == "Conv2DLayer"
            for method in cls.body
            if isinstance(method, ast.FunctionDef) and method.name == "forward"
        ]
        if len(methods) != 1:
            raise ValueError("Expected exactly one Conv2DLayer.forward")
        original_method = methods[0]
        first, last = original_method.body[0], original_method.body[-1]
        if first.lineno == original_method.lineno:
            raise ValueError("Forward body template must start on a separate line")
        lines = self.source.splitlines(keepends=True)
        first_line = lines[first.lineno - 1]
        padding = first_line[: len(first_line) - len(first_line.lstrip(" \t"))]
        if first.col_offset != len(padding):
            raise ValueError("Forward body template must start on a separate line")
        newline = "\r\n" if first_line.endswith("\r\n") else "\n"
        replacement = (
            newline.join(padding + line for line in body.rstrip("\n").split("\n")) + newline
        )
        source = (
            "".join(lines[: first.lineno - 1]) + replacement + "".join(lines[last.end_lineno :])
        )
        materialized = candidate.model_copy(
            update={"files": [SourceFile(path="train.py", content=source)]}
        )
        self.validate(materialized)
        return materialized

    @staticmethod
    def skeleton(source):
        tree = ast.parse(source)
        found = 0
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef) and cls.name == "Conv2DLayer":
                for method in cls.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "forward":
                        method.body = [ast.Pass()]
                        found += 1
        if found != 1:
            raise ValueError("Expected exactly one Conv2DLayer.forward")
        return ast.dump(tree, include_attributes=False)

    @staticmethod
    def _validate_entry_point(candidate):
        if [f.path for f in candidate.files] != ["train.py"] or candidate.command != [
            "python",
            "train.py",
        ]:
            raise ValueError("Vectorization requires only train.py with its unchanged entry point")

    def validate(self, candidate):
        self._validate_entry_point(candidate)
        if self.skeleton(candidate.files[0].content) != self.skeleton(self.source):
            raise ValueError("Vectorization may change only the forward method body")

    def prepare(self, phase, destination):
        public = destination / "public"
        public.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(self.config.split_seed)
        configurations = [(2, 9, 11, 2, 3, 3, 1, 0), (3, 12, 10, 3, 4, 3, 2, 2)]
        if phase != "subset":
            configurations += [(32, 64, 64, 3, 8, 3, 2, 2)]
        arrays, cases = {}, []
        for i, (n, h, w, cin, cout, k, stride, padding) in enumerate(configurations):
            arrays[f"x{i}"] = rng.normal(size=(n, h, w, cin))
            arrays[f"k{i}"] = rng.normal(size=(k, k, cin, cout))
            arrays[f"b{i}"] = rng.normal(size=(1, 1, 1, cout))
            cases.append(
                {"stride": stride, "padding": padding, "activation": "relu" if i else None}
            )
        np.savez(public / "inputs.npz", **arrays)
        return {"public": str(public), "files": tree_manifest(public), "cases": cases}

    def score(self, phase, prepared, workspace, seconds):
        with zipfile.ZipFile(workspace / "predictions.npz") as archive:
            if sum(f.file_size for f in archive.infolist()) > 128_000_000:
                raise ValueError("Oversized decompressed convolution predictions")
        inputs = np.load(Path(prepared["public"]) / "inputs.npz", allow_pickle=False)
        with np.load(workspace / "predictions.npz", allow_pickle=False) as predictions:
            if set(predictions.files) != {str(i) for i in range(len(prepared["cases"]))}:
                raise ValueError("Missing convolution outputs")
            for i, case in enumerate(prepared["cases"]):
                x, k, bias = inputs[f"x{i}"], inputs[f"k{i}"], inputs[f"b{i}"]
                p, s = case["padding"], case["stride"]
                x = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
                windows = np.lib.stride_tricks.sliding_window_view(x, k.shape[:2], axis=(1, 2))[
                    :, ::s, ::s
                ]
                expected = np.einsum("nhwcij,ijco->nhwo", windows, k) + bias
                if case["activation"] == "relu":
                    expected = np.maximum(expected, 0)
                pred = predictions[str(i)]
                if (
                    pred.shape != expected.shape
                    or not np.isfinite(pred).all()
                    or not np.allclose(pred, expected, rtol=1e-7, atol=1e-8)
                ):
                    raise ValueError("Convolution correctness check failed")
        return float(seconds)


def load_development(native, config):
    if config.adapter in {"csv", "tabular"}:
        adapter = CSVDevelopment(native, config)
    elif config.adapter == "vectorization":
        adapter = VectorizationDevelopment(native, config)
    else:
        entries = list(entry_points(group="scientisttwo.native_development", name=config.adapter))
        if len(entries) != 1:
            raise ValueError("Install one trusted native development adapter for this task")
        adapter = RegisteredDevelopment(entries[0].load()(native, config), entries[0])
    return validate_adapter(with_baseline(adapter, config, native))


def implementation_identity():
    from .. import backends, process, store, models, artifacts, runtime_contracts, runtime_profiles, environment_probe
    from ..benchmarks.files import native_identity

    result = {"native/" + k: v for k, v in native_identity().items()}
    result.update({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    research = Path(__file__).parents[1] / "research"
    result.update({"research/" + p.relative_to(research).as_posix(): file_hash(p)
                   for p in sorted(research.rglob("*"))
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"})
    integrations = Path(__file__).parents[1] / "integrations"
    result.update({"integrations/" + p.relative_to(integrations).as_posix(): file_hash(p)
                   for p in sorted(integrations.rglob("*.py"))})
    studies = Path(__file__).parents[1] / "studies"
    result.update({"studies/" + p.relative_to(studies).as_posix(): file_hash(p)
                   for p in sorted(studies.rglob("*.py"))})
    providers = Path(__file__).parents[1] / "providers"
    result.update({"providers/" + p.relative_to(providers).as_posix(): file_hash(p)
                   for p in sorted(providers.rglob("*.py"))})
    result.update(
        {"core/" + Path(m.__file__).name: file_hash(m.__file__) for m in (backends, process, store, models, artifacts, runtime_contracts, runtime_profiles, environment_probe)}
    )
    locks = Path(runtime_profiles.__file__).with_name("runtime_locks.json")
    if locks.is_file():
        result["core/runtime_locks.json"] = file_hash(locks)
    result.update(
        {
            "dependency/" + name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "pydantic")
        }
    )
    return result
