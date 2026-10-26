"""Container-only framework families and strict prepared-data contracts.

Importing this package never imports Torch, TensorFlow, JAX or model checkpoints.
"""
import hashlib
import json
from pathlib import Path
import shutil
from typing import Literal

from pydantic import Field

from ..models import Record


class TaskFamily(Record):
    task_id: str
    runtime_profile: str
    metric: str
    direction: Literal["maximize", "minimize"]
    training_files: list[str]
    evaluation_files: list[str]
    private_files: list[str] = ["labels.npz"]
    required_assets: list[str] = []
    output_paths: list[str] = ["checkpoint"]
    reference_kind: str = "adapted_reference"
    max_epochs: int = Field(default=10, ge=1)
    max_steps: int = Field(default=1000000, ge=1)
    prediction_max_bytes: int = Field(default=128 * 1024**3, gt=0)
    prediction_max_files: int = Field(default=100000, ge=1, le=1000000)
    notes: list[str] = []

    @property
    def required_files(self) -> dict[str, list[str]]:
        return {"public": self.training_files + self.evaluation_files,
                "private": self.private_files, "assets": self.required_assets}

    def command(self, stage: Literal["train", "infer", "score"]) -> list[str]:
        if stage == "score":
            return ["python", "/evaluator/evaluate.py", "--task", self.task_id,
                    "--predictions", "/predictions/predictions" if self.task_id not in {"CLRS", "llama-inference"} else "/predictions",
                    "--labels", "/data/private/labels.npz",
                    "--output", "/grade-output/grade.json"]
        script = "clrs_worker.py" if self.task_id == "CLRS" else "family_worker.py"
        if stage == "train":
            return ["python", "train.py"]
        return ["python", "/evaluator/" + script, "--stage", stage, "--task", self.task_id,
                "--data", "/data/public", "--assets", "/assets", "--checkpoint", "/artifacts/checkpoint",
                "--output", "/output/predictions" if self.task_id not in {"CLRS", "llama-inference"} else "/output",
                "--config", "/evaluator/training.json", "--model-source",
                "/candidate" if self.task_id == "CLRS" else "/candidate/model.py"]

    def validate_prepared(self, root: Path, metadata: dict) -> dict:
        from .validation import validate_prepared
        if "resource_limits" in metadata:
            configured = get_family(self.task_id, settings=metadata["resource_limits"])
            self.prediction_max_bytes = configured.prediction_max_bytes
            self.prediction_max_files = configured.prediction_max_files
        return validate_prepared(self, Path(root), metadata)

    def artifacts(self, stage: str = "train") -> list:
        from ..artifacts import ArtifactSpec
        if stage == "score":
            return [ArtifactSpec(path="token-counts.json", kind="file", role="diagnostic",
                                 max_bytes=1024**3, max_files=1)] if self.task_id == "babylm" else []
        if stage == "train" and self.task_id != "llama-inference":
            return [ArtifactSpec(path="checkpoint", kind="directory", role="checkpoint",
                                 max_bytes=32 * 1024**3, max_files=10000)]
        if self.task_id not in {"CLRS", "llama-inference"}:
            return [ArtifactSpec(path="predictions", kind="directory", role="prediction",
                                 max_bytes=self.prediction_max_bytes, max_files=self.prediction_max_files)]
        return [ArtifactSpec(path="predictions.npz", kind="file", role="prediction",
                             max_bytes=16 * 1024**3)]

    def write_sources(self, root: Path, *, upstream: Path | None = None) -> dict:
        """Emit declared templates; this does not claim runtime or data validation."""
        root = Path(root)
        templates = Path(__file__).with_name("templates")
        worker = "clrs_worker.py" if self.task_id == "CLRS" else "family_worker.py"
        default = {
            "seed": 0, "epochs": {"cifar10": 5, "ogbn-arxiv": 4, "babylm": 1}.get(self.task_id, 3),
            "max_epochs": self.max_epochs, "max_steps": self.max_steps,
            "batch_size": {"cifar10": 128, "ogbn-arxiv": 1024, "llama-inference": 4}.get(self.task_id, 32),
            "sequence_length": 128 if self.task_id in {"babylm", "llama-inference"} else 256,
        }
        # Explicit script arguments allow sealed phase/seed configuration in each run.
        wrapper = (
            "import os, subprocess, sys\n"
            "from pathlib import Path\n"
            f"command = [sys.executable, str(Path(__file__).with_name({worker!r})), "
            f"'--stage', {'infer' if self.task_id == 'llama-inference' else 'train'!r}, '--task', {self.task_id!r}, "
            "'--data', os.environ.get('SCIENTISTTWO_DATA', '/data/public'), "
            "'--assets', os.environ.get('SCIENTISTTWO_ASSETS', '/assets'), "
            "'--checkpoint', 'checkpoint', '--output', '.', "
            "'--config', os.environ.get('SCIENTISTTWO_TRAINING_CONFIG', str(Path(__file__).with_name('training.json')))]\n"
            "raise SystemExit(subprocess.call(command + sys.argv[1:]))\n"
        )
        for directory in [root / "reference", root / "control"]:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "train.py").write_text(wrapper)
            (directory / worker).write_text((templates / worker).read_text())
            (directory / "model.py").write_text((templates / "model.py").read_text())
            (directory / "training_contract.py").write_text((templates / "training_contract.py").read_text())
            (directory / "array_data.py").write_text(Path(__file__).with_name("array_data.py").read_text())
            (directory / "clrs_inputs.py").write_text(Path(__file__).with_name("clrs_inputs.py").read_text())
            (directory / "transforms.py").write_text((templates / "transforms.py").read_text())
            (directory / "training_config.json").write_text("{}\n")
            (directory / "llama_model.py").write_text((templates / "llama_model.py").read_text())
            (directory / "optimization_config.json").write_text(json.dumps({"compile": self.task_id == "llama-inference" and directory.name == "control"}) + "\n")
            if directory.name == "control" and self.task_id != "llama-inference":
                (directory / "training_config.json").write_text(json.dumps({"learning_rate": 0.05 if self.task_id == "cifar10" else 0.0005}) + "\n")
            (directory / "training.json").write_text(json.dumps(default, indent=2) + "\n")
            if self.task_id == "CLRS":
                helpers = upstream or root / "source_helpers"
                if not (helpers / "baselines.py").is_file():
                    raise ValueError("CLRS requires the pinned upstream env helper tree in source_helpers")
                for path in sorted(helpers.glob("*.py")):
                    if path.name != "train.py" and not path.name.endswith("_test.py"):
                        shutil.copyfile(path, directory / path.name)
        evaluator = root / "evaluator"
        (evaluator / "score").mkdir(parents=True, exist_ok=True)
        (evaluator / "inference").mkdir(parents=True, exist_ok=True)
        (evaluator / "score/evaluate.py").write_text((templates / "score.py").read_text())
        (evaluator / "score/array_data.py").write_text(Path(__file__).with_name("array_data.py").read_text())
        (evaluator / "inference" / worker).write_text((templates / worker).read_text())
        (evaluator / "inference/training_contract.py").write_text((templates / "training_contract.py").read_text())
        (evaluator / "inference/array_data.py").write_text(Path(__file__).with_name("array_data.py").read_text())
        (evaluator / "inference/clrs_inputs.py").write_text(Path(__file__).with_name("clrs_inputs.py").read_text())
        (evaluator / "inference/llama_model.py").write_text((templates / "llama_model.py").read_text())
        (evaluator / "inference/training.json").write_text(json.dumps(default, indent=2) + "\n")
        identity = {path.relative_to(evaluator).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(evaluator.rglob("*")) if path.is_file()}
        editable = ["model.py", "training_config.json"]
        if self.task_id in {"cifar10", "ogbn-arxiv", "identify-contrails", "fathomnet"}:
            editable.append("transforms.py")
        if self.task_id == "CLRS":
            editable = ["baselines.py", "processors.py", "training_config.json"]
        if self.task_id == "llama-inference":
            editable = ["model.py", "optimization_config.json"]
        descriptor = {"family": self.model_dump(), "commands": {stage: self.command(stage) for stage in ("train", "infer", "score")},
                      "evaluator_identity": identity, "training_defaults": default,
                      "editable_files": editable,
                      "runtime_validation": "unverified", "live_model_calls": 0}
        (root / "family-pipeline.json").write_text(json.dumps(descriptor, indent=2) + "\n")
        return descriptor


_FAMILIES = {
    "cifar10": TaskFamily(task_id="cifar10", runtime_profile="torch-cuda", metric="accuracy",
        direction="maximize", training_files=["train.npz"], evaluation_files=["evaluation.npz"],
        notes=["Upstream five-layer CNN with all examples retained; no final labels visible during training."]),
    "imdb": TaskFamily(task_id="imdb", runtime_profile="torch-cuda", metric="accuracy",
        direction="maximize", training_files=["train.jsonl"], evaluation_files=["evaluation.jsonl"],
        required_assets=["distilbert"], notes=["Incomplete original starter; adapted local DistilBERT fine-tuning reference."]),
    "ogbn-arxiv": TaskFamily(task_id="ogbn-arxiv", runtime_profile="torch-cuda", metric="accuracy",
        direction="maximize", training_files=["train.npz"], evaluation_files=["evaluation.npz"],
        notes=["Adapted GraphSAGE reference; full public graph features/edges and global node IDs remain available to candidates; protected node labels excluded."]),
    "identify-contrails": TaskFamily(task_id="identify-contrails", runtime_profile="torch-cuda", metric="dice",
        direction="maximize", training_files=["train.npz"], evaluation_files=["evaluation.npz"],
        notes=["Explicit adapted fully convolutional reference on prepared record channels, binary masks."]),
    "fathomnet": TaskFamily(task_id="fathomnet", runtime_profile="torch-cuda", metric="map_at_20",
        direction="maximize", training_files=["train.npz"], evaluation_files=["evaluation.npz"],
        notes=["Explicit adapted image classifier; real category labels only; OSD/composite unavailable."]),
    "babylm": TaskFamily(task_id="babylm", runtime_profile="torch-cuda", metric="perplexity",
        direction="minimize", training_files=["train.jsonl"], evaluation_files=["evaluation.npz"],
        required_assets=["tokenizer", "model_config"],
        notes=["Weights initialized from config, never pretrained; frozen tokenizer; logits scored by trusted NumPy scorer."]),
    "CLRS": TaskFamily(task_id="CLRS", runtime_profile="jax-clrs", metric="pointer_accuracy",
        direction="maximize", training_files=["train_manifest.json", "spec.json"],
        evaluation_files=["evaluation.npz", "evaluation_manifest.json"],
        max_steps=1000,
        notes=["Frozen upstream source helpers required; Floyd-Warshall train sizes <=16, final size64; no output/hint labels in inference bundle."]),
    "llama-inference": TaskFamily(task_id="llama-inference", runtime_profile="torch-cuda", metric="latency_ratio",
        direction="minimize", training_files=[], evaluation_files=["evaluation.npz"],
        required_assets=["model", "tokenizer"], output_paths=["predictions.npz"],
        notes=["Explicit fixed 7B model asset required; greedy one-token generation; independent parent timing harness required."]),
}


def get_family(task_id: str, settings: dict | None = None) -> TaskFamily:
    try:
        family = _FAMILIES[task_id].model_copy(deep=True)
        if settings:
            unknown = set(settings) - {"prediction_max_bytes", "prediction_max_files"}
            if unknown:
                raise ValueError("Unknown family resource limit: " + ",".join(sorted(unknown)))
            family = TaskFamily.model_validate({**family.model_dump(), **settings})
        return family
    except KeyError as exc:
        raise ValueError(f"No framework family for {task_id}") from exc


def family_task_ids() -> list[str]:
    return list(_FAMILIES)


def template_sources(task_id: str) -> dict[str, str]:
    family = get_family(task_id)
    root = Path(__file__).with_name("templates")
    worker = "clrs_worker.py" if task_id == "CLRS" else "family_worker.py"
    return {worker: (root / worker).read_text(), "score.py": (root / "score.py").read_text(),
            "training_contract.py": (root / "training_contract.py").read_text(),
            "model.py": (root / "model.py").read_text(), "transforms.py": (root / "transforms.py").read_text(),
            "family.json": family.model_dump_json(indent=2)}
