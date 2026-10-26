from pathlib import Path
import yaml

from .files import tree_manifest, verify_tree, native_identity, file_hash
from .models import NativeSpec, NativeResources, relative_path
from .registry import get_task, verify_repository, PINS
from .lfs import parse_pointer


def create_spec(
    benchmark,
    task_id,
    repo,
    *,
    data_root=None,
    candidate=None,
    command=None,
    mode="baseline",
    provenance=None,
    context=None,
    image="scientisttwo-native:0.1",
    grader_image=None,
    seconds=1800,
    memory_mb=2048,
    cpus=1,
    gpus=None,
    submission="submission.csv",
):
    repo = Path(repo).resolve()
    task = get_task(benchmark, repo, task_id)
    if task.revision != PINS[benchmark]["revision"]:
        raise ValueError(
            "Use the supported pinned upstream revision before creating a benchmark manifest"
        )
    verify_repository(repo, task.revision)
    if benchmark == "mlagentbench":
        root = (
            Path(data_root).resolve()
            if data_root
            else repo / "MLAgentBench/benchmarks" / task.folder
        )
        public, private = root / "env", root / "scripts"
        provenance = provenance or (
            "upstream_no_data"
            if task.folder == "vectorization" and data_root is None
            else "user_prepared"
        )
        argv = command if command is not None else task.baseline_command
    else:
        if data_root is None:
            raise ValueError(
                "MLE-bench requires its prepared data root; the integration never downloads data implicitly"
            )
        root = Path(data_root).resolve() / task_id / "prepared"
        public, private = root / "public", root / "private"
        provenance = provenance or "official_prepared"
        argv = command or []
    candidate = Path(candidate).resolve() if candidate else None
    return NativeSpec(
        name=f"{benchmark}-{task_id}-{mode}",
        benchmark=benchmark,
        task_id=task_id,
        repository=repo,
        revision=task.revision,
        mode=mode,
        data_provenance=provenance,
        public_dir=public,
        private_dir=private,
        candidate_dir=candidate,
        command=argv,
        submission=submission,
        image=image,
        grader_image=grader_image or image,
        context=context,
        resources=NativeResources(seconds=seconds, memory_mb=memory_mb, cpus=cpus, gpus=gpus),
        required_artifacts=task.required_artifacts or [submission],
        public_manifest=tree_manifest(public),
        private_manifest=tree_manifest(private),
        candidate_manifest=tree_manifest(candidate) if candidate else {},
        implementation=native_identity(),
    )


def upstream_checksums(spec):
    if spec.benchmark != "mle-bench":
        return {
            "status": "unavailable",
            "reason": "MLAgentBench does not supply canonical prepared-data checksums",
        }
    path = spec.repository / "mlebench/competitions" / spec.task_id / "checksums.yaml"
    if not path.exists():
        return {"status": "unavailable", "reason": "Upstream checksum file is absent"}
    expected = yaml.safe_load(path.read_text())
    actual = {}
    for label, root in (("public", spec.public_dir), ("private", spec.private_dir)):
        actual[label] = (
            {
                p.name: file_hash(p, "md5")
                for p in root.iterdir()
                if p.is_file()
                and p.suffix.lstrip(".") in {"csv", "json", "jsonl", "parquet", "bson"}
            }
            if root and root.exists()
            else {}
        )
    matched = all(
        expected.get(label) and expected[label] == actual[label] for label in ("public", "private")
    )
    return {
        "status": "pass" if matched else "fail",
        "scope": "Upstream top-level CSV/JSON/JSONL/parquet/BSON checksums; binary image archives are outside this upstream checksum scope",
    }


def preflight(spec):
    errors = []
    verify_repository(spec.repository, spec.revision)
    if spec.revision != PINS[spec.benchmark]["revision"]:
        raise ValueError("Manifest does not reference the supported official upstream revision")
    if spec.implementation != native_identity():
        raise ValueError(
            "Native integration changed after manifest creation; create a new manifest"
        )
    task = get_task(spec.benchmark, spec.repository, spec.task_id)
    if not task.grader_available:
        errors.append("Upstream task has no native get_score/grade entry point")
    for label, path, expected in (
        ("public", spec.public_dir, spec.public_manifest),
        ("private", spec.private_dir, spec.private_manifest),
    ):
        if path is None or not path.is_dir() or not expected:
            errors.append(
                f"Missing prepared {label} tree; prepare data and create a new frozen manifest"
            )
        else:
            verify_tree(path, expected)
    if spec.candidate_dir is not None:
        if not spec.candidate_manifest:
            errors.append("Prepared candidate/submission directory is empty")
        verify_tree(spec.candidate_dir, spec.candidate_manifest)
    if spec.benchmark == "mlagentbench" and task.grader:
        official = spec.repository / task.grader
        actual = spec.private_dir / "eval.py" if spec.private_dir else None
        if actual is None or not actual.is_file() or file_hash(actual) != file_hash(official):
            errors.append("Prepared MLAgentBench scripts must retain the pinned upstream eval.py")
        if spec.mode != "grade_only" and not spec.command:
            errors.append("No execution command declared")
    checksums = upstream_checksums(spec)
    if spec.benchmark == "mle-bench":
        leaderboard = spec.repository / "mlebench/competitions" / spec.task_id / "leaderboard.csv"
        if not leaderboard.exists() or parse_pointer(leaderboard.read_bytes()) is not None:
            errors.append(
                "Official leaderboard is missing or still a Git LFS pointer; materialize and verify it before grading"
            )
        config = yaml.safe_load(
            (spec.repository / "mlebench/competitions" / spec.task_id / "config.yaml").read_text()
        )
        answer = Path(config["dataset"]["answers"]).relative_to(
            Path(spec.task_id) / "prepared/private"
        )
        sample = Path(config["dataset"]["sample_submission"]).relative_to(
            Path(spec.task_id) / "prepared/public"
        )
        relative_path(answer.as_posix())
        relative_path(sample.as_posix())
        if not (spec.private_dir / answer).is_file():
            errors.append("Official private answer artifact is missing")
        if not (spec.public_dir / sample).is_file():
            errors.append("Official public sample-submission artifact is missing")
        if spec.data_provenance == "official_prepared" and checksums["status"] != "pass":
            errors.append(
                "Prepared-data checksums do not match upstream; no official benchmark result may be claimed"
            )
    if spec.data_provenance == "upstream_no_data" and not (
        spec.benchmark == "mlagentbench" and task.folder == "vectorization"
    ):
        errors.append(
            "upstream_no_data is supported only for the self-contained vectorization task"
        )
    return {
        "ready": not errors,
        "errors": errors,
        "task": task.model_dump(),
        "checksums": checksums,
        "live_calls": False,
        "data_provenance": spec.data_provenance,
    }


def preparation_plan(benchmark, task_id, repo, data_root=None):
    task = get_task(benchmark, repo, task_id)
    if benchmark == "mle-bench":
        return {
            "task": task.model_dump(),
            "command": [
                "mlebench",
                "prepare",
                "-c",
                task_id,
                "--data-dir",
                str(data_root or "<prepared-data-root>"),
            ],
            "requirements": [
                "Configure Kaggle credentials outside the repository",
                "Accept the competition's terms personally when required",
                "Use a separate upstream environment with task dependencies",
            ],
            "automatic_download": False,
        }
    return {
        "task": task.model_dump(),
        "command": ["python", "-m", "MLAgentBench.prepare_task", task_id],
        "working_directory": str(Path(repo).resolve()),
        "requirements": [
            "Review the task's scripts/prepare.py and install its dependencies",
            "Keep prepared env and scripts directories separate",
            "Kaggle tasks require account credentials and competition consent",
        ],
        "automatic_download": False,
    }
