"""Model-free source-fidelity checks; no new empirical experiment results."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scientisttwo.artifacts import load_bundle
from scientisttwo.live.adapters import VectorizationDevelopment
from scientisttwo.live.models import Candidate, DevelopmentSpec, SourceFile
from scientisttwo.live.parent_edits import (
    AnchoredEdit, MethodChange, ParentChangePlan, ParentEditError, ParentPatch,
    apply_parent_patch, parent_context, validate_executable_parent_change,
)
from scientisttwo.live.source_tree import candidate_snapshot, freeze_source_tree
from scientisttwo.store import IntegrityError, Store, canonical, digest


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "evidence")
    yield value
    value.close()


def candidate(files=None, **values):
    return Candidate(files=[SourceFile(path=path, content=content) for path, content in
                            (files or {"model.py": "VALUE = 1\r\n# preserve exact trailing bytes\r\n"}).items()],
                     command=["python", "train.py"], description="Synthetic parent", **values)


def source_ref(store, parent, materialize=None):
    return store.put_json(candidate_snapshot(store, materialize(parent) if materialize else parent))


def patch(ref, path="model.py", before="VALUE = 1", after="VALUE = 2"):
    return ParentPatch(parent_source_ref=ref, edits=[AnchoredEdit(
        path=path, before=before, after=after, reason="Synthetic localized edit")],
        description="Synthetic changed parent")


def test_full_parent_context_and_exact_nonedited_bytes(store):
    original = candidate({"model.py": "VALUE = 1\r\n# keep\r\n", "helper.py": "SENTINEL = '研究'\n"})
    before = deepcopy(original)
    ref = source_ref(store, original)
    context = parent_context(store, original, ref)
    assert {file["path"]: file["content"] for file in context["files"]} == {
        file.path: file.content for file in original.files}
    assert context["parent_source_ref"] == ref
    child, proof = apply_parent_patch(store, original, ref, patch(ref), declared_paths=["model.py"])
    assert child.files[0].content == "VALUE = 2\r\n# keep\r\n"
    assert child.files[1] == original.files[1]
    assert child.command == original.command and child.base_snapshot_ref == original.base_snapshot_ref
    assert original == before
    assert proof["changed_paths"] == ["model.py"]
    assert proof["source_hashes"]["helper.py"]["before"] == proof["source_hashes"]["helper.py"]["after"]
    assert proof["semantic_intent"] == "unverified"
    assert store.json(proof["patch_ref"]) == patch(ref).model_dump()


def test_overlay_exposes_current_parent_not_base_and_preserves_untouched_source(store, tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "model.py").write_text("BASELINE = True\n")
    (root / "transforms.py").write_bytes(b"def transform(x):\r\n    return x\r\n")
    (root / "harness.py").write_text("READ_ONLY = True\n")
    base = freeze_source_tree(store, root, ["model.py", "transforms.py", "new.py"])
    # The exact initializer expressions removed in the recorded v3 evolution.
    # This is a synthetic source fixture, not model execution or a measured result.
    initializer = (
        'for layer in model.modules():\n'
        '    if isinstance(layer, nn.Conv2d):\n'
        '        nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")\n'
        '    elif isinstance(layer, nn.BatchNorm2d):\n'
        '        nn.init.ones_(layer.weight)\n'
        '        nn.init.zeros_(layer.bias)\n'
    )
    parent = candidate({"model.py": "# active residual architecture\n" + "# retain context\n" * 200 + initializer},
                       base_snapshot_ref=base)
    ref = source_ref(store, parent)
    context = parent_context(store, parent, ref)
    files = {file["path"]: file for file in context["files"]}
    assert files["model.py"]["content"] == parent.files[0].content
    assert initializer in files["model.py"]["content"] and "BASELINE" not in files["model.py"]["content"]
    assert files["transforms.py"]["content"].endswith("\r\n")
    assert context["new_files"] == ["new.py"]
    assert "harness.py" not in files
    child, proof = apply_parent_patch(store, parent, ref,
        patch(ref, "transforms.py", "return x", "return x.flip(-1)"), declared_paths=["transforms.py"])
    assert child.base_snapshot_ref == base
    assert child.files[0] == parent.files[0]  # entire initializer retained byte for byte
    assert child.files[1].content == "def transform(x):\r\n    return x.flip(-1)\r\n"
    assert proof["source_hashes"]["model.py"]["before"] == proof["source_hashes"]["model.py"]["after"]
    assert proof["source_hashes"]["harness.py"]["before"] == proof["source_hashes"]["harness.py"]["after"]
    assert proof["changed_paths"] == ["transforms.py"]


def test_no_silent_context_truncation_or_binary_decoding(store, tmp_path):
    parent = candidate()
    ref = source_ref(store, parent)
    context = parent_context(store, parent, ref)
    size = len(canonical(context))
    assert parent_context(store, parent, ref, max_context_bytes=size) == context
    with pytest.raises(ParentEditError, match="not truncated"):
        parent_context(store, parent, ref, max_context_bytes=size - 1)
    root = tmp_path / "binary"
    root.mkdir()
    (root / "model.py").write_bytes(b"\xff\xfe")
    base = freeze_source_tree(store, root, ["model.py"])
    binary = Candidate(files=[], base_snapshot_ref=base, command=["python", "model.py"], description="binary")
    with pytest.raises(ParentEditError, match="UTF-8"):
        parent_context(store, binary, source_ref(store, binary))


def test_materialized_snapshot_and_parent_binding_are_verified(store):
    parent = candidate()
    other = candidate({"model.py": "VALUE = 3\n"})
    ref, other_ref = source_ref(store, parent), source_ref(store, other)
    with pytest.raises(ParentEditError, match="stale parent"):
        apply_parent_patch(store, parent, ref, patch(other_ref), declared_paths=["model.py"])
    with pytest.raises(IntegrityError, match="exact active parent"):
        parent_context(store, parent, other_ref)
    tree = candidate_snapshot(store, parent)
    forged_ref = store.put_json(tree.model_copy(update={"changed_paths": []}))
    with pytest.raises(IntegrityError, match="exact active parent"):
        apply_parent_patch(store, parent, forged_ref, patch(forged_ref), declared_paths=["model.py"])


def test_corrupt_source_is_fatal_not_an_agent_correction(store):
    parent = candidate()
    ref = source_ref(store, parent)
    tree = candidate_snapshot(store, parent)
    entry = load_bundle(store, tree.bundle_ref).files["model.py"]
    (store.root / entry.ref).write_text("tampered")
    with pytest.raises(IntegrityError, match="Modified immutable artifact"):
        apply_parent_patch(store, parent, ref, patch(ref), declared_paths=["model.py"])


@pytest.mark.parametrize("before", ["absent", "VALUE", "aba"])
def test_missing_ambiguous_and_overlapping_occurrence_anchors_rejected(store, before):
    parent = candidate({"model.py": "VALUE = 1\nVALUE = 2\nababa\n"})
    ref = source_ref(store, parent)
    with pytest.raises(ParentEditError, match="exactly once"):
        apply_parent_patch(store, parent, ref, patch(ref, before=before), declared_paths=["model.py"])


def test_hunks_resolve_against_original_parent_and_must_not_overlap(store):
    parent = candidate({"model.py": "alpha beta gamma\n"})
    ref = source_ref(store, parent)
    proposed = ParentPatch(parent_source_ref=ref, description="Two independent hunks", edits=[
        AnchoredEdit(path="model.py", before="alpha", after="omega", reason="first"),
        AnchoredEdit(path="model.py", before="gamma", after="delta", reason="second"),
    ])
    child, _ = apply_parent_patch(store, parent, ref, proposed, declared_paths=["model.py"])
    assert child.files[0].content == "omega beta delta\n"
    proposed.edits[1] = AnchoredEdit(path="model.py", before="alpha beta", after="new", reason="overlap")
    with pytest.raises(ParentEditError, match="overlap"):
        apply_parent_patch(store, parent, ref, proposed, declared_paths=["model.py"])
    proposed.edits[1] = AnchoredEdit(path="model.py", before="omega", after="new", reason="sequential")
    with pytest.raises(ParentEditError, match="exactly once"):
        apply_parent_patch(store, parent, ref, proposed, declared_paths=["model.py"])


def test_declared_scope_blocks_unplanned_model_changes_and_extra_paths(store):
    parent = candidate({"model.py": "VALUE = 1\n", "transforms.py": "return_inputs = True\n"})
    ref = source_ref(store, parent)
    with pytest.raises(ParentEditError, match="independently declared"):
        apply_parent_patch(store, parent, ref, patch(ref), declared_paths=["transforms.py"])
    with pytest.raises(ParentEditError, match="editable file scope"):
        apply_parent_patch(store, parent, ref, patch(ref), declared_paths=["private.py"])
    with pytest.raises(ParentEditError, match="independently declared"):
        apply_parent_patch(store, parent, ref, patch(ref), declared_paths=["model.py", "transforms.py"])


def test_new_file_requires_explicit_null_and_unchanged_command(store):
    parent = candidate()
    ref = source_ref(store, parent)
    scope = ["model.py", "helper.py"]
    child, proof = apply_parent_patch(store, parent, ref, patch(ref, "helper.py", None, "NEW = True\n"),
        declared_paths=["helper.py"], editable_paths=scope)
    assert child.command == parent.command
    assert child.files[0] == parent.files[0]
    assert child.files[1].path == "helper.py"
    assert proof["source_hashes"]["helper.py"]["before"] is None
    with pytest.raises(ParentEditError, match="before=null"):
        apply_parent_patch(store, parent, ref, patch(ref, "helper.py", "missing", "NEW"),
                           declared_paths=["helper.py"], editable_paths=scope)
    with pytest.raises(ParentEditError, match="existing file"):
        apply_parent_patch(store, parent, ref, patch(ref, before=None), declared_paths=["model.py"])
    with pytest.raises(ValueError):
        ParentPatch.model_validate({**patch(ref).model_dump(), "command": ["sh", "other.sh"]})


def test_forward_body_parent_preserves_proposal_form_and_exact_template(store, tmp_path):
    root = tmp_path / "vector"
    root.mkdir()
    template = "import numpy as np\n\nclass Conv2DLayer:\n    def forward(self, features_batch):\n        return features_batch\n\nUNCHANGED = 3\n"
    (root / "train.py").write_text(template)
    adapter = VectorizationDevelopment(SimpleNamespace(public_dir=root),
        DevelopmentSpec(settings={"proposal_mode": "forward_body"}))
    parent = candidate({"train.py": "values = features_batch * 2\nreturn values\n"})
    ref = source_ref(store, parent, adapter.materialize)
    context = parent_context(store, parent, ref, materialize=adapter.materialize)
    file = context["files"][0]
    assert file["content"] == parent.files[0].content
    assert file["representation"] == "proposal"
    assert "UNCHANGED = 3" in file["materialized_content"]
    assert file["materialized_content"] == adapter.materialize(parent).files[0].content
    child, proof = apply_parent_patch(store, parent, ref,
        patch(ref, "train.py", "* 2", "* 3"), declared_paths=["train.py"], materialize=adapter.materialize)
    assert child.files[0].content == "values = features_batch * 3\nreturn values\n"
    assert not child.files[0].content.startswith("import")
    assert "UNCHANGED = 3\n" in adapter.materialize(child).files[0].content
    assert proof["changed_paths"] == ["train.py"]
    with pytest.raises(ParentEditError, match="materialization contract"):
        apply_parent_patch(store, parent, ref,
            patch(ref, "train.py", "return values", "return (("),
            declared_paths=["train.py"], materialize=adapter.materialize)


def test_path_and_patch_quotas_are_enforced(store):
    parent = candidate()
    ref = source_ref(store, parent)
    for unsafe in ("../model.py", "/tmp/model.py", ".ssh/id_rsa", "model.bin"):
        with pytest.raises(ValueError):
            patch(ref, unsafe)
    for before, after in (("", "x"), ("same", "same"), (None, "")):
        with pytest.raises(ValueError):
            patch(ref, before=before, after=after)
    with pytest.raises(ValueError):
        patch(ref, after="a" * 32_001)
    large = candidate({"model.py": "A" * 249_999 + "Z"})
    large_ref = source_ref(store, large)
    with pytest.raises(ParentEditError, match="bounded source"):
        apply_parent_patch(store, large, large_ref, patch(large_ref, before="Z", after="ZZ"), declared_paths=["model.py"])
    many = candidate({f"file{i}.py": "X=1\n" for i in range(24)})
    many_ref = source_ref(store, many)
    with pytest.raises(ParentEditError, match="bounded source"):
        apply_parent_patch(store, many, many_ref, patch(many_ref, "new.py", None, "X=2\n"),
                           declared_paths=["new.py"], editable_paths=[f.path for f in many.files] + ["new.py"])


def test_materialization_cannot_silently_change_other_files_or_command(store):
    parent = candidate({"model.py": "VALUE = 1\n", "helper.py": "SAFE = True\n"})
    ref = source_ref(store, parent)
    def unstable(candidate):
        if candidate.files[0].content.startswith("VALUE = 2"):
            candidate.files[1].content = "SAFE = False\n"
        return candidate
    with pytest.raises(ParentEditError, match="Materialized changes"):
        apply_parent_patch(store, parent, ref, patch(ref), declared_paths=["model.py"], materialize=unstable)
    def command_change(candidate):
        candidate.command = ["python", "other.py"]
        return candidate
    with pytest.raises(IntegrityError, match="command"):
        parent_context(store, parent, ref, materialize=command_change)


def test_method_change_plan_binds_a_bounded_distinct_scope():
    idea = {"title": "Synthetic", "hypothesis": "Improve fixture", "novelty_score": 1,
            "novelty_scope": "Synthetic test", "source_ids": ["task"]}
    change = MethodChange(idea=idea, change_plan=ParentChangePlan(changed_files=["transforms.py"], reason="Add flips only"))
    assert change.change_plan.changed_files == ["transforms.py"]
    assert digest(canonical(change))
    with pytest.raises(ValueError):
        ParentChangePlan(changed_files=["model.py", "model.py"], reason="Duplicate")


def test_source_overlay_edit_scope_cannot_expand_or_create_over_directories(store, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src/model.py").write_text("VALUE = 1\n")
    base = freeze_source_tree(store, root, ["src/model.py", "src"])
    parent = Candidate(files=[], base_snapshot_ref=base, command=["python", "src/model.py"], description="base")
    ref = source_ref(store, parent)
    with pytest.raises(ParentEditError, match="exceeds the frozen"):
        parent_context(store, parent, ref, editable_paths=["outside.py"])
    with pytest.raises(ParentEditError, match="conflicts"):
        parent_context(store, parent, ref)
    context = parent_context(store, parent, ref, editable_paths=["src/model.py"])
    assert context["editable_paths"] == ["src/model.py"]


def test_comment_only_method_revision_is_rejected_but_historical_patch_still_replays(store):
    # Reduced source-shaped fixture from v4's editorial patch, without importing
    # archived results or presenting synthetic observations as empirical scores.
    anchor = "                return self.relu(self.residual(inputs) + self.shortcut(inputs))"
    body = "class Block:\r\n    def forward(self, inputs):\r\n" + anchor + "\r\n"
    parent = candidate({"model.py": body, "training_config.json": "{}\r\n",
                        "transforms.py": "def transform(inputs):\r\n    return inputs\r\n"})
    original = deepcopy(parent)
    ref = source_ref(store, parent)
    editorial = patch(ref, before=anchor, after=(
        "                # Disabling this branch also skips projection convolutions.\r\n"
        "                # The ablation tests the branch package, not addition alone.\r\n"
        "                # Semantic isolation remains unverified. Retain the incumbent.\r\n"
        + anchor))
    child, proof = apply_parent_patch(store, parent, ref, editorial, declared_paths=["model.py"])
    with pytest.raises(ParentEditError, match="route manuscript clarification"):
        validate_executable_parent_change(store, ref, proof["candidate_source_ref"])
    assert parent == original
    assert child.files[0].content == body.replace(anchor, editorial.edits[0].after)
    assert child.files[1:] == original.files[1:]
    # Admission is deliberately separate: the old exact-source audit contract
    # must reproduce this patch even though new campaigns reject it as a method.
    replay, replay_proof = apply_parent_patch(store, parent, ref, editorial,
                                             declared_paths=["model.py"])
    assert replay == child and replay_proof == proof


def test_formatting_only_python_revision_does_not_admit_retraining(store):
    parent = candidate({"model.py": "VALUE = (1 + 2)\n"})
    ref = source_ref(store, parent)
    _, proof = apply_parent_patch(store, parent, ref,
        patch(ref, before="VALUE = (1 + 2)", after="\nVALUE=(1+2)  # clarify\n"),
        declared_paths=["model.py"])
    with pytest.raises(ParentEditError, match="identical parsed ASTs"):
        validate_executable_parent_change(store, ref, proof["candidate_source_ref"])


def test_real_python_method_change_is_admitted_without_semantic_claim(store):
    parent = candidate()
    ref = source_ref(store, parent)
    child, proof = apply_parent_patch(store, parent, ref, patch(ref), declared_paths=["model.py"])
    before = deepcopy(child)
    check = validate_executable_parent_change(store, ref, proof["candidate_source_ref"])
    assert check["status"] == "python_ast_changed"
    assert check["file_checks"] == {"model.py": "python_ast_changed"}
    assert "semantic equivalence remain unverified" in check["scope"]
    assert child == before and child.files[0].content.endswith("trailing bytes\r\n")


def test_invalid_python_is_unverified_not_misclassified_as_editorial_or_valid_method(store):
    parent = candidate()
    ref = source_ref(store, parent)
    _, proof = apply_parent_patch(store, parent, ref,
        patch(ref, after="VALUE = ("), declared_paths=["model.py"])
    check = validate_executable_parent_change(store, ref, proof["candidate_source_ref"])
    assert check["status"] == "unverified"
    assert check["file_checks"] == {"model.py": "python_parse_unverified"}


def test_config_change_with_python_comment_does_not_get_rejected_as_editorial(store):
    parent = candidate({"model.py": "VALUE = 1\n", "training_config.json": '{"epochs": 5}\n'})
    ref = source_ref(store, parent)
    proposed = ParentPatch(parent_source_ref=ref, description="Configuration change with commentary",
        edits=[AnchoredEdit(path="model.py", before="VALUE = 1", after="VALUE = 1  # documented",
                            reason="Explain method"),
               AnchoredEdit(path="training_config.json", before='"epochs": 5', after='"epochs": 6',
                            reason="Change declared training budget")])
    _, proof = apply_parent_patch(store, parent, ref, proposed,
                                 declared_paths=["model.py", "training_config.json"])
    check = validate_executable_parent_change(store, ref, proof["candidate_source_ref"])
    assert check["status"] == "unverified"  # Protocol/budget checks still apply elsewhere.
    assert check["file_checks"] == {"model.py": "python_ast_unchanged",
                                    "training_config.json": "non_python_unverified"}


def test_new_python_file_presence_is_not_treated_as_comment_equivalence(store):
    parent = candidate()
    ref = source_ref(store, parent)
    _, proof = apply_parent_patch(store, parent, ref,
        patch(ref, path="new.py", before=None, after="# module intentionally empty\n"),
        declared_paths=["new.py"], editable_paths=["model.py", "new.py"])
    check = validate_executable_parent_change(store, ref, proof["candidate_source_ref"])
    assert check["status"] == "unverified"
    assert check["file_checks"] == {"new.py": "file_presence_changed"}


def test_body_only_adapter_admission_checks_materialized_python(store, tmp_path):
    root = tmp_path / "vector"
    root.mkdir()
    (root / "train.py").write_text(
        "class Conv2DLayer:\n    def forward(self, features_batch):\n        return features_batch\n")
    adapter = VectorizationDevelopment(SimpleNamespace(public_dir=root),
        DevelopmentSpec(settings={"proposal_mode": "forward_body"}))
    parent = candidate({"train.py": "return features_batch\n"})
    ref = source_ref(store, parent, adapter.materialize)
    child, proof = apply_parent_patch(store, parent, ref,
        patch(ref, "train.py", "return features_batch", "# Editorial clarification\nreturn features_batch"),
        declared_paths=["train.py"], materialize=adapter.materialize)
    assert child.files[0].content.startswith("# Editorial clarification\nreturn")
    with pytest.raises(ParentEditError, match="identical parsed ASTs"):
        validate_executable_parent_change(store, ref, proof["candidate_source_ref"])


def test_admission_check_rejects_unchanged_source_and_detects_tampering(store):
    parent = candidate()
    ref = source_ref(store, parent)
    with pytest.raises(ParentEditError, match="no changed source"):
        validate_executable_parent_change(store, ref, ref)
    _, proof = apply_parent_patch(store, parent, ref, patch(ref), declared_paths=["model.py"])
    tree = store.json(proof["candidate_source_ref"])
    entry = load_bundle(store, tree["bundle_ref"]).files["model.py"]
    (store.root / entry.ref).write_text("VALUE = 999\n")
    with pytest.raises(IntegrityError, match="modified bundle artifact"):
        validate_executable_parent_change(store, ref, proof["candidate_source_ref"])
