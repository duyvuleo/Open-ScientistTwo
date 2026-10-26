"""Real filesystem payload tests; fixture bytes are not benchmark outcomes."""

import os
from pathlib import Path

import pytest

from scientisttwo.artifacts import (
    ArtifactSpec,
    BundleManifest,
    freeze_bundle,
    load_bundle,
    restore_bundle,
)
from scientisttwo.store import IntegrityError, Store


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "store")
    yield value
    value.close()


def checkpoint(tmp_path):
    work = tmp_path / "work"
    (work / "checkpoint" / "empty").mkdir(parents=True)
    (work / "checkpoint" / "weights.bin").write_bytes(b"weights\0" * 250_000)
    (work / "checkpoint" / "config.json").write_text('{"layers":2}')
    return work


def test_directory_payloads_stream_without_store_read(store, tmp_path, monkeypatch):
    work = checkpoint(tmp_path)
    ref = freeze_bundle(
        store, work, [ArtifactSpec(path="checkpoint", kind="directory", role="checkpoint")]
    )
    manifest = load_bundle(store, ref)
    original = store.read

    def manifest_only(name):
        assert name == ref, "Payloads must not pass through Store.read"
        return original(name)

    monkeypatch.setattr(store, "read", manifest_only)
    destination = tmp_path / "restored"
    restore_bundle(store, ref, destination)
    assert (destination / "checkpoint" / "empty").is_dir()
    for relative in manifest.files:
        assert (destination / relative).read_bytes() == (work / relative).read_bytes()
    assert store.verify() == []


@pytest.mark.parametrize("limit", [{"max_bytes": 12}, {"max_files": 1}])
def test_bundle_quotas_fail_before_copy(store, tmp_path, limit):
    with pytest.raises(IntegrityError, match="quota"):
        freeze_bundle(
            store,
            checkpoint(tmp_path),
            [ArtifactSpec(path="checkpoint", kind="directory", **limit)],
        )
    assert not list((store.root / "artifacts").iterdir())


def test_optional_artifacts_and_required_missing(store, tmp_path):
    work = checkpoint(tmp_path)
    ref = freeze_bundle(store, work, [ArtifactSpec(path="optional.json", required=False)])
    assert load_bundle(store, ref).missing == ["optional.json"]
    with pytest.raises(IntegrityError, match="Required"):
        freeze_bundle(store, work, [ArtifactSpec(path="missing.json")])


@pytest.mark.parametrize(
    "path",
    ["../escape", "/escape", "foo//bar", "./foo", "x/../y", "a\\b", "credentials", ".secret"],
)
def test_artifact_paths_are_canonical_and_safe(path):
    with pytest.raises(ValueError):
        ArtifactSpec(path=path)


@pytest.mark.parametrize("link_parent", [False, True])
def test_symlink_and_parent_escapes_rejected(store, tmp_path, link_parent):
    work = checkpoint(tmp_path)
    if link_parent:
        (work / "link").symlink_to(work / "checkpoint", target_is_directory=True)
        spec = ArtifactSpec(path="link/config.json")
    else:
        (work / "checkpoint" / "linked.bin").symlink_to(work / "checkpoint" / "weights.bin")
        spec = ArtifactSpec(path="checkpoint", kind="directory")
    with pytest.raises(IntegrityError, match="symlink"):
        freeze_bundle(store, work, [spec])


def test_special_files_and_overlapping_contracts_rejected(store, tmp_path):
    work = checkpoint(tmp_path)
    os.mkfifo(work / "checkpoint" / "fifo")
    with pytest.raises(IntegrityError, match="Special"):
        freeze_bundle(store, work, [ArtifactSpec(path="checkpoint", kind="directory")])
    with pytest.raises(ValueError, match="overlap"):
        freeze_bundle(
            store,
            work,
            [
                ArtifactSpec(path="checkpoint", kind="directory"),
                ArtifactSpec(path="checkpoint/config.json"),
            ],
        )


def test_corruption_detected_before_restore_and_refreeze(store, tmp_path):
    work = checkpoint(tmp_path)
    spec = ArtifactSpec(path="checkpoint", kind="directory")
    ref = freeze_bundle(store, work, [spec])
    manifest = load_bundle(store, ref)
    payload = store.root / manifest.files["checkpoint/weights.bin"].ref
    payload.write_bytes(b"corrupted")
    with pytest.raises(IntegrityError, match="modified"):
        restore_bundle(store, ref, tmp_path / "restored")
    with pytest.raises(IntegrityError, match="Modified immutable"):
        freeze_bundle(store, work, [spec])
    assert not (tmp_path / "restored").exists()


def test_restore_does_not_merge_stale_or_partial_artifacts(store, tmp_path):
    ref = freeze_bundle(
        store, checkpoint(tmp_path), [ArtifactSpec(path="checkpoint", kind="directory")]
    )
    destination = tmp_path / "restored"
    destination.mkdir()
    (destination / "partial.bin").write_bytes(b"old")
    with pytest.raises(IntegrityError, match="absent or empty"):
        restore_bundle(store, ref, destination)
    assert (destination / "partial.bin").read_bytes() == b"old"


def test_modification_during_snapshot_detected(store, tmp_path, monkeypatch):
    work = checkpoint(tmp_path)
    original = store.put_file

    def mutate(path, **kwargs):
        ref = original(path, **kwargs)
        Path(path).write_bytes(b"changed after snapshot")
        return ref

    monkeypatch.setattr(store, "put_file", mutate)
    with pytest.raises(IntegrityError, match="changed"):
        freeze_bundle(store, work, [ArtifactSpec(path="checkpoint", kind="directory")])


def test_forged_manifest_metadata_rejected(store, tmp_path):
    ref = freeze_bundle(
        store, checkpoint(tmp_path), [ArtifactSpec(path="checkpoint", kind="directory")]
    )
    manifest = load_bundle(store, ref).model_dump()
    manifest["files"]["checkpoint/config.json"]["size"] += 1
    forged = store.put_json(manifest)
    with pytest.raises(IntegrityError, match="modified"):
        restore_bundle(store, forged, tmp_path / "restored")
    manifest["files"]["checkpoint/config.json"]["ref"] = "../outside.json"
    with pytest.raises(IntegrityError):
        restore_bundle(store, store.put_json(manifest), tmp_path / "restored")


def test_manifest_rejects_file_directory_conflicts():
    with pytest.raises(ValueError, match="collision"):
        BundleManifest(
            specs=[ArtifactSpec(path="a", kind="directory")],
            directories=["a"],
            files={"a": {"ref": "artifacts/x.bin", "sha256": "a" * 64, "size": 0}},
        )
