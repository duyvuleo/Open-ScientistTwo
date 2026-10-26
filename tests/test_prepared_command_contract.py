"""Synthetic adapter-contract fixtures; no datasets, containers, or model calls."""
import json
from types import SimpleNamespace

import pytest

from scientisttwo.live.adapters import VectorizationDevelopment
from scientisttwo.live.models import Candidate, DevelopmentSpec, SourceFile
from scientisttwo.benchmarks.files import file_hash
from scientisttwo.task_families import family_task_ids, get_family
from scientisttwo.testbed.adapter import PreparedDevelopment
from scientisttwo.testbed.catalog import recipes


TASKS = [recipe.task_id for recipe in recipes()]
SNAPSHOT = "artifacts/" + "a" * 64 + ".json"


def contract_fixture(root, task):
    # Exercise context/validation without pretending these source-only fixtures
    # establish prepared-data validity or actual benchmark readiness.
    adapter = PreparedDevelopment.__new__(PreparedDevelopment)
    adapter.root, adapter.snapshot_ref = root, SNAPSHOT
    adapter.manifest = {"split_metadata": {"normalization": "synthetic input convention"}}
    adapter.recipe = next(recipe for recipe in recipes() if recipe.task_id == task)
    adapter.family = get_family(task) if task in family_task_ids() else None
    adapter.vectorization = None
    reference = root / "reference"
    if adapter.family:
        helpers = None
        if task == "CLRS":
            helpers = root / "synthetic-upstream-helpers"
            helpers.mkdir(parents=True)
            for name in ("baselines.py", "processors.py"):
                (helpers / name).write_text("# Synthetic source-only fixture\n")
        adapter.family.write_sources(root, upstream=helpers)
        candidate = Candidate(files=[], base_snapshot_ref=SNAPSHOT, command=["python", "train.py"],
                              description="Synthetic command contract fixture")
    else:
        reference.mkdir(parents=True)
        source = ("import numpy as np\nclass Conv2DLayer:\n"
                  "    def forward(self, features_batch):\n        return features_batch\n")
        (reference / "train.py").write_text(source)
        if task == "vectorization":
            adapter.vectorization = VectorizationDevelopment(SimpleNamespace(public_dir=reference), DevelopmentSpec())
        candidate = Candidate(files=[SourceFile(path="train.py", content=source)],
                              command=["python", "train.py"], description="Synthetic command contract fixture")
    return adapter, candidate


@pytest.mark.parametrize("task", TASKS)
def test_each_prepared_task_discloses_the_exact_accepted_argv(tmp_path, task):
    adapter, candidate = contract_fixture(tmp_path, task)
    context = adapter.context()
    assert context["command"] == ["python", "train.py"]
    if adapter.family:
        assert context["command"] == adapter.family.command("train")
    candidate.command = context["command"]
    adapter.validate(candidate)
    # Context consumers cannot mutate the validator's accepted argv.
    context["command"].append("--config")
    assert adapter.context()["command"] == ["python", "train.py"]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("command", [
    ["python", "/opt/benchmark/harness.py", "--mode", "train"],
    ["python", "train.py", "--config", "training_config.json"],
    ["checkpoint"], ["python", "different.py"], ["python3", "train.py"],
])
def test_unknown_commands_and_extra_arguments_still_fail_with_actionable_diagnostic(tmp_path, task, command):
    adapter, candidate = contract_fixture(tmp_path, task)
    candidate.command = command
    with pytest.raises(ValueError) as error:
        adapter.validate(candidate)
    assert 'command exactly ["python", "train.py"]' in str(error.value)
    assert "do not add arguments" in str(error.value)
    # Errors expose the expected contract, not arbitrary supplied command text.
    assert "/opt/benchmark/harness.py" not in str(error.value)


def test_cifar_context_and_command_diagnostic_do_not_read_protected_or_prepared_solutions(tmp_path, monkeypatch):
    adapter, candidate = contract_fixture(tmp_path, "cifar10")
    markers = {
        "control/model.py": "PREPARED_CONTROL_MUST_NOT_ENTER_PROMPT",
        "phases/final/private/labels.json": "FINAL_LABELS_MUST_NOT_ENTER_PROMPT",
        "validation/prior-results.json": "PRIOR_SCORES_MUST_NOT_ENTER_PROMPT",
    }
    for name, marker in markers.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker)
    from pathlib import Path
    original = Path.read_text
    original_bytes = Path.read_bytes
    read_paths = []
    def record(path, *args, **kwargs):
        read_paths.append(path)
        return original(path, *args, **kwargs)
    def record_bytes(path, *args, **kwargs):
        read_paths.append(path)
        return original_bytes(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", record)
    monkeypatch.setattr(Path, "read_bytes", record_bytes)
    context = adapter.context()
    allowed = {"training.json", "train.py", "training_contract.py", "array_data.py", "family_worker.py",
               *adapter.recipe.editable_files}
    assert set(read_paths) <= {tmp_path / "family-pipeline.json", *[
        tmp_path / "reference" / name for name in allowed]}
    serialized = json.dumps(context)
    for marker in markers.values():
        assert marker not in serialized
    assert "labels.json" not in serialized and "prior-results" not in serialized
    assert set(context) == {"task_id", "metric", "direction", "targets", "id_column",
                            "editable_files", "reference_baseline", "input_files", "submission",
                            "deviations", "command", "base_snapshot_ref", "editable_source", "execution_contract"}
    candidate.command = ["checkpoint"]
    with pytest.raises(ValueError) as error:
        adapter.validate(candidate)
    assert all(marker not in str(error.value) for marker in markers.values())


def test_command_disclosure_does_not_relax_source_snapshot_or_edit_boundary(tmp_path):
    adapter, candidate = contract_fixture(tmp_path, "cifar10")
    candidate.base_snapshot_ref = "artifacts/" + "b" * 64 + ".json"
    with pytest.raises(ValueError, match="authorized source snapshot"):
        adapter.validate(candidate)
    candidate.base_snapshot_ref = SNAPSHOT
    candidate.files = [SourceFile(path="family_worker.py", content="pass\n")]
    with pytest.raises(ValueError, match="outside the recipe edit boundary"):
        adapter.validate(candidate)


@pytest.mark.parametrize("task", family_task_ids())
def test_training_guidance_is_exact_frozen_reference_source_not_a_second_validator(tmp_path, task):
    adapter, _ = contract_fixture(tmp_path, task)
    # Descriptive context must never import the framework implementation.
    before = set(__import__("sys").modules)
    contract = adapter.context()["execution_contract"]
    added = set(__import__("sys").modules) - before
    assert not any(name.split(".")[0] in {"torch", "tensorflow", "jax"} for name in added)
    reference = tmp_path / "reference"
    assert contract["base_configuration"] == json.loads((reference / "training.json").read_text())
    assert contract["input_normalization"] == "synthetic input convention"
    for name, content in contract["read_only_sources"].items():
        assert content == (reference / name).read_text()
        assert name not in adapter.editable_paths()
    assert 'allowed = {' in contract["read_only_sources"]["training_contract.py"]
    assert '"learning_rate": (1e-8, 10.0)' in contract["read_only_sources"]["training_contract.py"]
    assert set(contract["read_only_sources"]) == {"train.py", "training_contract.py"}
    excerpts = {}
    for item in contract["source_excerpts"]:
        source = reference / item["path"]
        lines = source.read_text().splitlines(keepends=True)
        assert item["sha256"] == file_hash(source)
        for fragment in item["fragments"]:
            assert fragment["text"] == "".join(lines[fragment["start_line"] - 1:fragment["end_line"]])
        excerpts[item["path"]] = "\n".join(fragment["text"] for fragment in item["fragments"])
    assert len(json.dumps(contract, sort_keys=True)) <= 12_000
    if task == "cifar10":
        assert contract["base_configuration"]["epochs"] == 5
        assert contract["base_configuration"]["batch_size"] == 128
        worker = excerpts["family_worker.py"]
        assert '"sgd" if task == "cifar10" else "adam"' in worker
        assert '0.1 if task == "cifar10" else 0.001' in worker
        assert '"momentum", 0.9' in worker
        assert 'directory if directory.is_dir() else packed' in excerpts["array_data.py"]
        assert "def inference(" not in worker and 'elif task == "babylm"' not in worker
        assert len(worker) < len((reference / "family_worker.py").read_text()) // 2


def test_only_normalization_metadata_is_presented_not_entire_prepared_manifest(tmp_path):
    adapter, _ = contract_fixture(tmp_path, "cifar10")
    adapter.manifest["split_metadata"].update(final_labels="FINAL_DATA_SENTINEL", prior_score="PRIOR_SCORE_SENTINEL")
    context = json.dumps(adapter.context())
    assert "synthetic input convention" in context
    assert "FINAL_DATA_SENTINEL" not in context and "PRIOR_SCORE_SENTINEL" not in context


def test_unavailable_frozen_harness_rules_fail_visibly_instead_of_inventing_defaults(tmp_path):
    adapter, _ = contract_fixture(tmp_path, "cifar10")
    (tmp_path / "reference/family_worker.py").write_text("# Missing contract fixture\n")
    with pytest.raises(ValueError, match="unavailable"):
        adapter.context()


def test_execution_context_has_a_fixed_bound_and_does_not_silently_truncate_rules(tmp_path):
    adapter, _ = contract_fixture(tmp_path, "cifar10")
    adapter.manifest["split_metadata"]["normalization"] = "x" * 12001
    with pytest.raises(ValueError, match="12000-character bound"):
        adapter.context()
