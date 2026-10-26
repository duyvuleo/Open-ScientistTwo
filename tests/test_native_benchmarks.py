import json
from pathlib import Path

import pytest

from scientisttwo.benchmarks.files import (
    tree_manifest,
    workspace,
    artifact_manifest,
    native_identity,
)
from scientisttwo.benchmarks.manifest import create_spec, preflight
from scientisttwo.benchmarks.models import NativeSpec, BenchmarkTask
from scientisttwo.benchmarks.registry import list_tasks, PINS, verify_repository
from scientisttwo.benchmarks.lfs import parse_pointer, matches_pointer
from scientisttwo.benchmarks.runner import NativeRunner
from scientisttwo.benchmarks.contract_fixtures import make_fixture
from scientisttwo.process import ProcessResult
from scientisttwo.store import Store, IntegrityError


from upstream_checkouts import upstream_path

MLA = upstream_path("mlagentbench")
MLE = upstream_path("mle-bench")


@pytest.mark.parametrize("name", ["../submission.csv", "/etc/passwd", "a/../../b", "a\\b"])
def test_unsafe_submission_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        NativeSpec(
            name="test",
            benchmark="mle-bench",
            task_id="test",
            repository=tmp_path,
            revision="a" * 40,
            mode="baseline",
            data_provenance="contract_fixture",
            public_dir=tmp_path / "public",
            submission=name,
        )


def test_frozen_inputs_reject_credentials_and_symlinks(tmp_path):
    (tmp_path / "kaggle.json").write_text("{}")
    with pytest.raises(ValueError, match="Credential"):
        tree_manifest(tmp_path)
    (tmp_path / "kaggle.json").unlink()
    (tmp_path / "linked").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlinks"):
        tree_manifest(tmp_path)


def test_workspace_data_readonly_links_and_deleted_link_detection(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    (public / "train.py").write_text('print("fixture")')
    (public / "train.csv").write_text("x,y\n1,2\n")
    target = tmp_path / "work"
    links = workspace(public, target)
    assert (target / "train.csv").readlink() == Path("/data/public/train.csv")
    assert not (target / "train.py").is_symlink()
    assert "train.csv" not in artifact_manifest(target, links)
    (target / "train.csv").unlink()
    with pytest.raises(ValueError, match="removed"):
        artifact_manifest(target, links)


def test_streaming_artifact_corruption_detected(tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"fixture" * 10000)
    store = Store(tmp_path / "store")
    ref = store.put_file(source)
    assert not store.verify()
    (store.root / ref).write_bytes(b"corrupt")
    assert store.verify()
    with pytest.raises(IntegrityError):
        store.put_file(source)
    store.close()


def test_lfs_hash_contract(tmp_path):
    from hashlib import sha256

    data = b"score\n1\n"
    path = tmp_path / "leaderboard.csv"
    path.write_bytes(data)
    pointer = parse_pointer(
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256(data).hexdigest()}\nsize {len(data)}\n".encode()
    )
    assert matches_pointer(path, pointer)
    path.write_bytes(b"changed")
    assert not matches_pointer(path, pointer)


def test_official_catalog_and_split_membership():
    if not MLA.exists() or not MLE.exists():
        pytest.skip("Clone the pinned public benchmark sources for catalog tests")
    verify_repository(MLA, PINS["mlagentbench"]["revision"])
    verify_repository(MLE, PINS["mle-bench"]["revision"])
    assert len(list_tasks("mlagentbench", MLA)) == 16
    assert sum(t.grader_available for t in list_tasks("mlagentbench", MLA)) == 13
    assert len(list_tasks("mle-bench", MLE)) == 82
    assert len(list_tasks("mle-bench", MLE, "split75")) == 75
    assert len(list_tasks("mle-bench", MLE, "low")) == 22
    assert (
        next(t for t in list_tasks("mlagentbench", MLA) if t.task_id == "debug").folder == "cifar10"
    )


def test_contract_data_cannot_masquerade_as_official_prepared_data(tmp_path):
    if not MLE.exists():
        pytest.skip("Pinned source checkout unavailable")
    data = make_fixture(tmp_path / "data", "detecting-insults-in-social-commentary")
    spec = create_spec(
        "mle-bench",
        "detecting-insults-in-social-commentary",
        MLE,
        data_root=data,
        provenance="official_prepared",
    )
    checks = preflight(spec)
    assert not checks["ready"] and checks["checksums"]["status"] == "fail"
    assert any("checksums" in error for error in checks["errors"])
    contract = spec.model_copy(update={"data_provenance": "contract_fixture"})
    assert preflight(contract)["ready"]


class FakeContainer:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, argv, timeout, **kwargs):
        self.calls.append(argv)
        if "inspect" in argv:
            return ProcessResult(0, "sha256:fixture", "", 0.01)
        if "run" not in argv:
            return ProcessResult(0, "", "", 0.01)
        mounts = {
            v.rsplit(":", 2)[1]: Path(v.rsplit(":", 2)[0])
            for i, v in enumerate(argv)
            if i and argv[i - 1] == "-v"
        }
        job = json.loads((mounts["/control"] / "job.json").read_text())
        if job["stage"] == "execute":
            assert "/task/scripts" not in mounts and "/upstream" not in mounts
            if self.fail:
                return ProcessResult(1, "", "failure", 0.1)
            (mounts["/workspace"] / "submission.csv").write_text("runtime\n1\n")
        else:
            assert "/task/scripts" in mounts and "/workspace" not in mounts
            (mounts["/grade-output"] / "grade.json").write_text(
                json.dumps(
                    {
                        "official_report": {"get_score": 1.0},
                        "score": 1.0,
                        "valid_submission": True,
                        "direction": "minimize",
                    }
                )
            )
        return ProcessResult(0, "", "", 0.1)


def test_container_stage_separation_and_completed_run_idempotence(tmp_path, monkeypatch):
    public = tmp_path / "public"
    private = tmp_path / "private"
    repo = tmp_path / "repo"
    for p in (public, private, repo):
        p.mkdir()
    (public / "train.py").write_text("# fixture")
    task = BenchmarkTask(
        benchmark="mlagentbench",
        task_id="vectorization",
        folder="vectorization",
        name="fixture",
        revision="a" * 40,
        repository=str(repo),
        grader_available=True,
        metric="reported_runtime_seconds",
        direction="minimize",
    )
    monkeypatch.setattr("scientisttwo.benchmarks.runner.get_task", lambda *args: task)
    monkeypatch.setattr(
        "scientisttwo.benchmarks.runner.preflight",
        lambda *args: {"ready": True, "checksums": {"status": "unavailable"}},
    )
    spec = NativeSpec(
        name="fixture",
        benchmark="mlagentbench",
        task_id="vectorization",
        repository=repo,
        revision="a" * 40,
        mode="baseline",
        data_provenance="upstream_no_data",
        public_dir=public,
        private_dir=private,
        command=["python", "train.py"],
        public_manifest=tree_manifest(public),
        implementation=native_identity(),
    )
    fake = FakeContainer()
    runner = NativeRunner(spec, tmp_path / "run", executor=fake)
    first = runner.run()
    calls = len(fake.calls)
    second = runner.run()
    assert first == second and len(fake.calls) == calls and first.status == "completed"
    assert not first.agent_evaluation and not first.leaderboard_eligible
    assert first.live_model_calls == 0
    assert all("--network=none" in c for c in fake.calls if "run" in c)
    runner.close()


def test_output_must_not_overlap_inputs(tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.mkdir()
    spec = NativeSpec(
        name="fixture",
        benchmark="mlagentbench",
        task_id="vectorization",
        repository=tmp_path / "repo",
        revision="a" * 40,
        mode="baseline",
        data_provenance="upstream_no_data",
        public_dir=public,
    )
    with pytest.raises(ValueError, match="overlap"):
        NativeRunner(spec, public / "output")
