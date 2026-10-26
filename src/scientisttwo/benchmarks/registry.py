"""Read upstream definitions without importing their packages or authenticating services."""

import ast
import csv
import json
from pathlib import Path
import subprocess
import yaml
from .lfs import parse_pointer, pointer_at_head, matches_pointer

from .models import BenchmarkTask, relative_path

PINS = {
    "mlagentbench": {
        "url": "https://github.com/snap-stanford/MLAgentBench",
        "revision": "5d71205cc20a8e95d43aa7cb7120e89ca3323e31",
    },
    "mle-bench": {
        "url": "https://github.com/openai/mle-bench",
        "revision": "507f92e1138bb6e40dac5c6ee7a6758e6424bf97",
    },
}
ML_METRICS = {
    "CLRS": ("algorithm_accuracy", "maximize"),
    "amp-parkinsons-disease-progression-prediction": ("smape", "minimize"),
    "babylm": ("perplexity", "minimize"),
    "cifar10": ("accuracy", "maximize"),
    "fathomnet": ("upstream_score", None),
    "feedback": ("mean_rmse", "minimize"),
    "house-price": ("mae", "minimize"),
    "identify-contrails": ("dice", "maximize"),
    "imdb": ("accuracy", "maximize"),
    "ogbn-arxiv": ("accuracy", "maximize"),
    "spaceship-titanic": ("accuracy", "maximize"),
    "vectorization": ("reported_runtime_seconds", "minimize"),
}


def revision(repo):
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if result.returncode:
        raise ValueError("Benchmark repository must be a readable Git checkout")
    return result.stdout.strip()


def verify_repository(repo, expected):
    repo = Path(repo).resolve()
    if revision(repo) != expected:
        raise ValueError("Upstream revision differs from the manifest pin")
    result = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only", "HEAD", "--"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError("Cannot verify the upstream working tree")
    for name in result.stdout.splitlines():
        pointer = pointer_at_head(repo, name)
        if pointer is None or not matches_pointer(repo / name, pointer):
            raise ValueError(
                "Tracked upstream files were modified; only hash-verified LFS materialization is allowed"
            )


def list_tasks(benchmark, repo, split=None):
    repo = Path(repo).resolve()
    commit = revision(repo)
    if benchmark == "mlagentbench":
        if split is not None:
            raise ValueError("MLAgentBench does not define MLE-bench complexity splits")
        root = repo / "MLAgentBench/benchmarks"
        aliases = json.loads((root / "tasks.json").read_text())
        names = {p.name: p.name for p in root.iterdir() if p.is_dir() and (p / "env").is_dir()}
        names.update({key: value["benchmark_folder_name"] for key, value in aliases.items()})
        result = []
        for task_id, folder in sorted(names.items()):
            relative_path(folder)
            grader = root / folder / "scripts/eval.py"
            available = grader.is_file() and any(
                isinstance(n, ast.FunctionDef) and n.name == "get_score"
                for n in ast.parse(grader.read_text()).body
            )
            problem = root / folder / "scripts/research_problem.txt"
            warnings = []
            if folder == "vectorization":
                warnings.append(
                    "Official grader trusts the runtime written by the submitted program; correctness and independent timing are not certified by this score."
                )
            if folder == "fathomnet":
                warnings.append(
                    "Upstream grader generates unseeded random labels; score nondeterminism requires separate investigation."
                )
            if folder in {"babylm", "CLRS"}:
                warnings.append(
                    "Grading loads submitted model artifacts; model-artifact integrity remains a separate audit."
                )
            if folder in {"cifar10", "imdb", "ogbn-arxiv", "babylm", "CLRS"}:
                warnings.append(
                    "Task-specific model/dataset caches and dependencies must be prepared before network-isolated execution."
                )
            metric, direction = ML_METRICS.get(folder, (None, None))
            result.append(
                BenchmarkTask(
                    benchmark=benchmark,
                    task_id=task_id,
                    folder=folder,
                    name=task_id,
                    repository=str(repo),
                    revision=commit,
                    description=aliases.get(task_id, {}).get("research_problem")
                    or (problem.read_text() if problem.exists() else ""),
                    grader=str(grader.relative_to(repo)) if grader.exists() else None,
                    grader_available=available,
                    baseline_command=["python", "train.py"]
                    if (root / folder / "env/train.py").exists()
                    else [],
                    metric=metric,
                    direction=direction,
                    warnings=warnings,
                    required_artifacts={"CLRS": ["checkpoints", "spec_list.pkl", "model_params.pkl"], "babylm": ["output"]}.get(folder, ["submission.csv"]),
                )
            )
        return result
    if benchmark != "mle-bench":
        raise ValueError("Unknown benchmark")
    splits = {
        p.stem: set(p.read_text().splitlines()) for p in (repo / "experiments/splits").glob("*.txt")
    }
    if split is not None and split not in splits:
        raise ValueError(f"Unknown official split: {split}")
    result = []
    for path in sorted((repo / "mlebench/competitions").glob("*/config.yaml")):
        config = yaml.safe_load(path.read_text())
        task_id = config["id"]
        if split is not None and task_id not in splits[split]:
            continue
        leaderboard = path.parent / "leaderboard.csv"
        direction = None
        lfs_pending = (
            leaderboard.exists() and parse_pointer(leaderboard.open("rb").read(256)) is not None
        )
        if leaderboard.exists() and not lfs_pending:
            rows = list(csv.DictReader(leaderboard.open()))
            scores = [float(row["score"]) for row in rows if row.get("score") not in (None, "")]
            if len(scores) > 1:
                direction = "minimize" if scores[0] < scores[-1] else "maximize"
        desc = repo / relative_path(config["description"])
        result.append(
            BenchmarkTask(
                benchmark=benchmark,
                task_id=task_id,
                folder=path.parent.name,
                name=config["name"],
                repository=str(repo),
                revision=commit,
                description=desc.read_text(),
                grader=config["grader"]["grade_fn"],
                grader_available=True,
                metric=config["grader"]["name"],
                direction=direction,
                splits=sorted(k for k, values in splits.items() if task_id in values),
                warnings=[]
                if task_id in splits.get("split75", set())
                else ["Registered task is outside the canonical split75 benchmark set."],
            )
        )
    return result


def get_task(benchmark, repo, task_id):
    matches = [task for task in list_tasks(benchmark, repo) if task.task_id == task_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown task: {benchmark}/{task_id}")
    return matches[0]
