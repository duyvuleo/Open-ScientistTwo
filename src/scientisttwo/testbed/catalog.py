"""Canonical MLAgentBench recipes, separate from upstream aliases and demos."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

UPSTREAM_REVISION = "5d71205cc20a8e95d43aa7cb7120e89ca3323e31"
RECIPE_VERSION = "audited-v1"
FEEDBACK_TARGETS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]


class OutputContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    kind: Literal["file", "directory"] = "file"
    role: Literal["predictions", "checkpoint", "tokenizer"] = "predictions"
    required: bool = True
    max_bytes: int = Field(default=2_000_000_000, gt=0)
    max_files: int = Field(default=1, gt=0)


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    recipe_version: str = RECIPE_VERSION
    upstream_revision: str = UPSTREAM_REVISION
    task_id: str
    family: str
    adapter_family: str
    original_starter: str
    original_starter_status: str
    reference_baseline: str
    runtime_profile: str
    metric: str
    direction: Literal["maximize", "minimize"]
    id_column: str | None = None
    targets: list[str] = []
    editable_files: list[str] = ["train.py"]
    outputs: list[OutputContract] = [OutputContract(path="submission.csv")]
    split_policy: str
    preparation: str
    assets: list[str] = []
    access_requirements: list[str] = []
    training_constraints: dict = {}
    minimum_disk_gb: int = 5
    evaluator: str
    upstream_grading: Literal["available", "known_defective", "unavailable"] = "available"
    deviations: list[str] = []
    seeds: list[int] = [0, 1, 2]

    @property
    def identity(self):
        value = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()


def _recipe(task_id, family, metric, direction, **kwargs):
    if task_id in {"cifar10", "imdb", "ogbn-arxiv", "identify-contrails", "fathomnet", "babylm", "llama-inference"}:
        editable = ["model.py", "optimization_config.json"] if task_id == "llama-inference" else ["model.py", "training_config.json"]
        if task_id in {"cifar10", "ogbn-arxiv", "identify-contrails", "fathomnet"}:
            editable.append("transforms.py")
        kwargs.setdefault("editable_files", editable)
        kwargs["deviations"] = [*kwargs.get("deviations", []),
                                "Audited v1 permits declared model, preprocessing and bounded hyperparameter edits within a frozen execution harness"]
    return Recipe(
        task_id=task_id, family=family, metric=metric, direction=direction,
        adapter_family=("paired_timing" if task_id in {"vectorization", "llama-inference"}
                        else "cpu_frames" if task_id in {"house-price", "spaceship-titanic", "feedback", "amp-parkinsons-disease-progression-prediction"}
                        else "checkpoint_pipeline"),
        original_starter=f"MLAgentBench/benchmarks/{task_id}/env/{'inference.py' if task_id == 'llama-inference' else 'train.py'}",
        original_starter_status=kwargs.pop("original_starter_status", "unvalidated_upstream"),
        reference_baseline=kwargs.pop("reference_baseline", "upstream-with-audited-wrapper"),
        runtime_profile=kwargs.pop("runtime_profile", "torch-cuda"),
        evaluator=kwargs.pop("evaluator", f"independent-{metric}-v1"),
        **kwargs,
    )


_RECIPES = (
    _recipe("vectorization", "timing", "latency_ratio", "minimize",
        runtime_profile="cpu-tabular", reference_baseline="upstream-loop-convolution",
        split_policy="Fixed disjoint development/final input arrays; screening subset",
        preparation="Generate seeded NumPy convolution inputs; no dataset download",
        training_constraints={"warmup_requests": 5, "measurement_pairs": 10},
        outputs=[OutputContract(path="predictions.npz")],
        deviations=["Independent warmed timing replaces self-reported process timing", "Primary latency ratio normalizes candidate median to its interleaved baseline; raw median seconds and IQR are retained"],
        upstream_grading="known_defective"),
    _recipe("house-price", "tabular-regression", "mae", "minimize",
        runtime_profile="cpu-tabular", id_column="Id", targets=["SalePrice"],
        original_starter_status="incomplete_starter", reference_baseline="ridge-train-only-preprocessing-v1",
        split_policy="Fixed development holdout from permitted training rows; preserve supplied final IDs",
        preparation="Import train.csv plus optional test.csv and separately supplied answer.csv",
        access_requirements=["Kaggle competition terms accepted by the user"],
        deviations=["Ridge reference is separate from the incomplete original starter", "Scoring joins exact IDs"]),
    _recipe("spaceship-titanic", "tabular-classification", "accuracy", "maximize",
        runtime_profile="cpu-tabular", id_column="PassengerId", targets=["Transported"],
        original_starter_status="incomplete_starter", reference_baseline="logistic-train-only-preprocessing-v1",
        split_policy="Stratified development holdout; preserve supplied final IDs",
        preparation="Import train.csv plus optional test.csv and separately supplied answer.csv",
        access_requirements=["Kaggle competition terms accepted by the user"],
        deviations=["Logistic regression reference is separate from the incomplete original starter", "Scoring joins exact IDs"]),
    _recipe("amp-parkinsons-disease-progression-prediction", "grouped-multitable", "smape_plus_one", "minimize",
        runtime_profile="cpu-tabular", id_column="prediction_id", targets=["rating"],
        reference_baseline="random-forest-month-horizon-target-v1",
        split_policy="Disjoint patient groups across every clinical, peptide and protein table",
        preparation="Import raw train_clinical_data.csv, train_proteins.csv and train_peptides.csv; optional supplemental clinical data",
        access_requirements=["Kaggle competition terms accepted by the user"],
        deviations=["Patient-disjoint deterministic holdouts replace upstream two-patient random holdout when no prepared final split is supplied", "Scoring joins exact prediction IDs"]),
    _recipe("feedback", "text-multioutput", "mean_rmse", "minimize",
        runtime_profile="cpu-tabular", id_column="text_id", targets=FEEDBACK_TARGETS,
        original_starter_status="random_prediction_starter", reference_baseline="tfidf-ridge-six-target-v1",
        split_policy="Essay-disjoint deterministic development holdout; preserve supplied final IDs",
        preparation="Import train.csv plus optional test.csv and separately supplied answer.csv",
        access_requirements=["Kaggle competition terms accepted by the user"],
        deviations=["TF-IDF/Ridge reference is separate from the random original starter", "Independent scorer never imports train.py"]),
    _recipe("cifar10", "image-classification", "accuracy", "maximize",
        runtime_profile="torch-cuda", id_column="id", targets=["label"],
        reference_baseline="upstream-lenet-audited-wrapper",
        split_policy="Stratified training-derived development split; original test labels protected",
        preparation="Import official CIFAR-10 assets into prepared train/dev/test arrays",
        minimum_disk_gb=8,
        deviations=["Starter test calls use development inputs; final labels removed from candidate mounts"]),
    _recipe("imdb", "text-classification", "accuracy", "maximize",
        id_column="id", targets=["label"], reference_baseline="adapted-distilbert-v1",
        original_starter_status="incomplete_starter",
        split_policy="Stratified training-derived development split; original test labels protected",
        preparation="Import IMDb text splits and cached DistilBERT model/tokenizer",
        assets=["distilbert"], minimum_disk_gb=12,
        deviations=["Named DistilBERT reference; final labels removed from candidate mounts"]),
    _recipe("ogbn-arxiv", "graph-classification", "accuracy", "maximize",
        id_column="node_id", targets=["label"],
        reference_baseline="adapted-graphsage-v1",
        split_policy="Official OGB train/validation/test node masks; transductive features declared",
        preparation="Import OGBN-Arxiv arrays, edges and official masks",
        minimum_disk_gb=12),
    _recipe("identify-contrails", "segmentation", "dice", "maximize",
        id_column="record_id", targets=["encoded_pixels"],
        reference_baseline="adapted-small-fcn-v1",
        split_policy="Record-disjoint image/mask partitions; private final masks",
        preparation="Import prepared image records with private mask arrays and ID manifests",
        access_requirements=["Kaggle competition terms accepted by the user"],
        minimum_disk_gb=100, deviations=["Strict RLE validation and ID joins", "Audited reference is an explicitly adapted convolutional segmentation model"]),
    _recipe("fathomnet", "ranked-classification", "map_at_20", "maximize",
        id_column="id", targets=["categories"],
        reference_baseline="adapted-scratch-resnet18-v1",
        split_policy="Image-disjoint partitions using real category ground truth",
        preparation="Import prepared image records and real category labels; no synthesized OSD labels",
        access_requirements=["Kaggle competition terms accepted by the user"],
        minimum_disk_gb=100, upstream_grading="known_defective",
        deviations=["Audited v1 reports MAP@20 only", "OSD and original composite unavailable: upstream OSD labels are random", "Audited reference is an explicitly adapted image classifier"]),
    _recipe("CLRS", "algorithm-checkpoint", "pointer_accuracy", "maximize",
        runtime_profile="jax-clrs", split_policy="Training graph lengths 4/7/11/13/16; development at training maximum; official larger final sizes",
        preparation="Import CLRS30 assets and separate feature/label trees preserving size metadata",
        minimum_disk_gb=15, reference_baseline="upstream-model-last-step-reference-v1",
        editable_files=["baselines.py", "processors.py", "training_config.json"],
        outputs=[OutputContract(path="checkpoint", kind="directory", role="checkpoint", max_files=1000)],
        training_constraints={"train_steps": 1000},
        deviations=["Checkpoint-dependent inference isolated from protected scoring labels", "Adapted Floyd-Warshall pointer matrix accuracy is reported separately from the upstream aggregate scorer"]),
    _recipe("babylm", "language-model-checkpoint", "perplexity", "minimize",
        split_policy="Official document-level train/dev/test corpora; frozen tokenizer and tokenization policy",
        preparation="Import BabyLM-10M corpora and frozen tokenizer assets; no pretrained model weights",
        assets=["tokenizer", "model_config"], minimum_disk_gb=30,
        reference_baseline="adapted-scratch-gpt2-block128-v1",
        outputs=[OutputContract(path="checkpoint", kind="directory", role="checkpoint", max_bytes=20_000_000_000, max_files=1000)],
        training_constraints={"from_scratch": True, "tokenizer_frozen": True, "training_words_max": 10_000_000},
        deviations=["Fixed tokenizer for comparable perplexity", "Independent protected token-label scoring"]),
    _recipe("llama-inference", "model-timing", "latency_ratio", "minimize",
        split_policy="Frozen model/tokenizer and prompt workload; development/final workload IDs disjoint",
        preparation="Import original or explicitly declared compatible 7B weights and tokenizer; name every adapted variant",
        assets=["model", "tokenizer"], minimum_disk_gb=50,
        access_requirements=["Model license/access accepted by the user"],
        training_constraints={"warmup_requests": 5, "measurement_pairs": 10, "synchronize_device": True,
                              "resident_models": 2, "minimum_free_vram_mb": 32768},
        outputs=[OutputContract(path="predictions.npz")], upstream_grading="unavailable",
        deviations=["New audited output-equivalence and independent timing evaluator; no upstream score", "Primary latency ratio uses its interleaved baseline median; raw median seconds and IQR remain in evidence"]),
)


def recipes():
    """Return fresh records: alias/debug/demo entries never count as task coverage."""
    return [recipe.model_copy(deep=True) for recipe in _RECIPES]


def get_recipe(task_id):
    for recipe in _RECIPES:
        if recipe.task_id == task_id:
            return recipe.model_copy(deep=True)
    raise ValueError(f"Unknown canonical MLAgentBench task: {task_id}")
