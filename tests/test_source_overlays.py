import pytest

from scientisttwo.artifacts import load_bundle
from scientisttwo.live.models import Candidate, SourceFile
from scientisttwo.live.source_tree import (
    build_source,
    candidate_snapshot,
    compare_source_trees,
    freeze_source_tree,
    materialize_source_tree,
    restore_source_tree,
)
from scientisttwo.store import IntegrityError, Store


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "store")
    yield value
    value.close()


def starter(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_bytes(b"print('baseline')\r\n")
    for index in range(40):
        (source / f"helper_{index}.py").write_text(f"VALUE = {index}\n")
    return source


def proposal(ref, content="print('candidate')\n"):
    return Candidate(
        base_snapshot_ref=ref,
        files=[SourceFile(path="train.py", content=content)],
        command=["python", "train.py"],
        description="Explicit fixture edit",
    )


def test_large_tree_uses_bounded_edits_without_regenerating_helpers(store, tmp_path):
    source = starter(tmp_path)
    base = freeze_source_tree(store, source, ["train.py"])
    candidate = proposal(base)
    tree = materialize_source_tree(store, candidate, expected_base_ref=base)
    assert len(candidate.files) == 1
    assert len(load_bundle(store, tree.bundle_ref).files) == 41
    assert tree.changed_paths == ["train.py"]
    assert store.json(tree.proposal_ref) == candidate.model_dump()
    restore_source_tree(store, tree, tmp_path / "restored")
    for index in range(40):
        assert (tmp_path / "restored" / f"helper_{index}.py").read_bytes() == (
            source / f"helper_{index}.py"
        ).read_bytes()
    assert (tmp_path / "restored" / "train.py").read_text() == candidate.files[0].content


def test_baseline_overlay_and_materialized_ablation_comparison(store, tmp_path):
    base = freeze_source_tree(store, starter(tmp_path), ["train.py"])
    baseline = Candidate(
        base_snapshot_ref=base, command=["python", "train.py"], description="Frozen baseline"
    )
    first = store.put_json(candidate_snapshot(store, baseline))
    second = store.put_json(candidate_snapshot(store, proposal(base)))
    assert compare_source_trees(store, first, second) == ["train.py"]
    assert compare_source_trees(store, first, first) == []
    assert candidate_snapshot(store, baseline).changed_paths == []


def test_legacy_candidate_has_same_materialization_interface(store, tmp_path):
    candidate = Candidate(
        files=[SourceFile(path="src/main.py", content="print(1)\n")],
        command=["python", "src/main.py"],
        description="Legacy",
    )
    tree = build_source(store, candidate, tmp_path / "restored")
    assert tree.base_snapshot_ref is None
    assert (tmp_path / "restored" / "src/main.py").read_text() == "print(1)\n"
    with pytest.raises(ValueError, match="require"):
        Candidate(command=["python"], description="No source")


def test_undeclared_source_edits_and_base_substitution_rejected(store, tmp_path):
    source = starter(tmp_path)
    base = freeze_source_tree(store, source, ["train.py"])
    other = freeze_source_tree(store, source, ["helper_0.py"])
    with pytest.raises(IntegrityError, match="authorized"):
        materialize_source_tree(store, proposal(other), expected_base_ref=base)
    candidate = proposal(base)
    candidate.files[0].path = "helper_0.py"
    with pytest.raises(IntegrityError, match="editable"):
        materialize_source_tree(store, candidate)


def test_allowed_new_file_and_total_tree_quota(store, tmp_path):
    source = starter(tmp_path)
    base = freeze_source_tree(store, source, ["new/helper.py"], max_files=42)
    candidate = Candidate(
        base_snapshot_ref=base,
        files=[SourceFile(path="new/helper.py", content="VALUE=7")],
        command=["python", "train.py"],
        description="New helper",
    )
    build_source(store, candidate, tmp_path / "restored")
    assert (tmp_path / "restored" / "new/helper.py").read_text() == "VALUE=7"
    small = freeze_source_tree(store, source, ["new/helper.py"], max_files=41)
    with pytest.raises(IntegrityError, match="limits"):
        candidate_snapshot(store, candidate.model_copy(update={"base_snapshot_ref": small}))


def test_overlays_still_obey_candidate_edit_limits(store, tmp_path):
    base = freeze_source_tree(store, starter(tmp_path), ["train.py"])
    candidate = proposal(base)
    bad = candidate.model_copy(update={"files": [candidate.files[0]] * 25})
    with pytest.raises(ValueError):
        candidate_snapshot(store, bad)


def test_modified_base_and_forged_materialization_rejected(store, tmp_path):
    base = freeze_source_tree(store, starter(tmp_path), ["train.py"])
    tree = candidate_snapshot(store, proposal(base))
    forged = tree.model_copy(update={"changed_paths": []})
    with pytest.raises(IntegrityError, match="does not match"):
        restore_source_tree(store, forged, tmp_path / "restored")
    manifest = load_bundle(store, tree.bundle_ref)
    (store.root / manifest.files["helper_0.py"].ref).write_text("modified")
    with pytest.raises(IntegrityError, match="modified"):
        restore_source_tree(store, tree, tmp_path / "restored")


def test_source_snapshot_rejects_links_and_oversize_before_copy(store, tmp_path):
    source = starter(tmp_path)
    with pytest.raises(IntegrityError, match="limits"):
        freeze_source_tree(store, source, ["train.py"], max_files=1)
    assert not list((store.root / "artifacts").iterdir())
    (source / "linked.py").symlink_to(source / "train.py")
    with pytest.raises(IntegrityError, match="symlinks"):
        freeze_source_tree(store, source, ["train.py"])
