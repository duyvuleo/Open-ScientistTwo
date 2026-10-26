"""Synthetic source contracts only: no external data, live models or experiments.

Prepared checkpoint baselines legitimately have an empty overlay. A scripted
control must resolve their frozen bytes instead of returning an incumbent no-op.
"""

from copy import deepcopy

import pytest

from scientisttwo.artifacts import load_bundle
from scientisttwo.live.models import Candidate, SourceFile
from scientisttwo.live.parent_edits import ParentPatch, apply_parent_patch, parent_context
from scientisttwo.live.source_tree import candidate_snapshot, freeze_source_tree
from scientisttwo.research.fixtures import ResearchFixtures
from scientisttwo.store import IntegrityError, Store


CODER_ROLES = ("ablation_coder", "rebuttal_coder")
BASE_MODEL = "WIDTH = 8\r\n# frozen reference bytes\r\n"
ACTIVE_MODEL = "WIDTH = 16\r\n# active candidate bytes\r\n"
ACTIVE_TRANSFORMS = "def transform(x):\n    return x.flip(-1)\n"


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "evidence")
    yield value
    value.close()


def source_candidate(files=None, *, base=None):
    return Candidate(
        files=[SourceFile(path=name, content=text) for name, text in (files or {}).items()],
        base_snapshot_ref=base,
        command=["python", "train.py"],
        description="Explicit synthetic source fixture",
    )


def overlay_pair(tmp_path, store):
    source = tmp_path / "frozen-source"
    source.mkdir()
    files = {
        "model.py": BASE_MODEL,
        "transforms.py": "def transform(x):\n    return x\n",
        "train.py": "# Read-only synthetic harness; never executed\n",
        "helper.py": "# Preserve UTF-8 and CRLF: 研究\r\n",
    }
    for name, text in files.items():
        (source / name).write_bytes(text.encode())
    ref = freeze_source_tree(store, source, ["model.py", "transforms.py"])
    baseline = source_candidate(base=ref)
    incumbent = source_candidate(
        {"model.py": ACTIVE_MODEL, "transforms.py": ACTIVE_TRANSFORMS}, base=ref
    )
    return baseline, incumbent


def coder_inputs(store, incumbent, *, changed="model.py", materialize=None):
    effective = materialize(incumbent) if materialize else incumbent
    ref = store.put_json(candidate_snapshot(store, effective))
    return {
        "parent": parent_context(store, incumbent, ref, materialize=materialize),
        "idea": {"title": "Explicit synthetic component control"},
        "change_plan": {"changed_files": [changed]},
    }


def legacy_target(target):
    def respond(role, inputs):
        assert role == "ablations"
        assert "incumbent" in inputs
        return {"plans": [{"candidate": target.model_dump()}]}
    return respond


def materialized_bytes(store, candidate):
    tree = candidate_snapshot(store, candidate)
    bundle = load_bundle(store, tree.bundle_ref, verify=True)
    return {name: store.read(file.ref) for name, file in bundle.files.items()}


@pytest.mark.parametrize("role", CODER_ROLES)
def test_empty_baseline_overlay_restores_target_and_preserves_exact_parent(store, tmp_path, role):
    target, incumbent = overlay_pair(tmp_path, store)
    assert target.files == []
    assert incumbent.files
    inputs = coder_inputs(store, incumbent)
    before_inputs, before_candidate = deepcopy(inputs), deepcopy(incumbent)
    before = materialized_bytes(store, incumbent)
    backend = ResearchFixtures(legacy_target(target), store=store)

    patch = ParentPatch.model_validate(backend.respond(role, inputs))
    assert patch.edits[0].before == ACTIVE_MODEL
    assert patch.edits[0].after == BASE_MODEL
    revised, proof = apply_parent_patch(
        store, incumbent, inputs["parent"]["parent_source_ref"], patch,
        declared_paths=["model.py"],
    )
    after = materialized_bytes(store, revised)
    assert after["model.py"] == BASE_MODEL.encode()
    assert {name: raw for name, raw in before.items() if name != "model.py"} == {
        name: raw for name, raw in after.items() if name != "model.py"
    }
    assert after["transforms.py"] == ACTIVE_TRANSFORMS.encode()
    assert revised.command == incumbent.command
    assert revised.base_snapshot_ref == incumbent.base_snapshot_ref
    assert proof["changed_paths"] == ["model.py"]
    assert proof["command_inherited"] and proof["base_snapshot_inherited"]
    assert inputs == before_inputs and incumbent == before_candidate
    assert not store.verify()


@pytest.mark.parametrize("role", CODER_ROLES)
def test_target_overlay_takes_precedence_over_its_frozen_base(store, tmp_path, role):
    baseline, incumbent = overlay_pair(tmp_path, store)
    target_text = "WIDTH = 4\n# explicit target overlay\n"
    target = source_candidate({"model.py": target_text}, base=baseline.base_snapshot_ref)
    patch = ParentPatch.model_validate(ResearchFixtures(
        legacy_target(target), store=store
    ).respond(role, coder_inputs(store, incumbent)))
    assert patch.edits[0].after == target_text
    assert patch.edits[0].after not in (BASE_MODEL, ACTIVE_MODEL)


@pytest.mark.parametrize("role", CODER_ROLES)
def test_frozen_target_uses_supplied_materializer(store, tmp_path, role):
    target, incumbent = overlay_pair(tmp_path, store)
    inputs = coder_inputs(store, incumbent)
    supplied = []
    expected = "WIDTH = 2\n# materialized target contract\n"

    def materialize(candidate):
        supplied.append(deepcopy(candidate))
        return candidate.model_copy(update={"files": [SourceFile(path="model.py", content=expected)]})

    patch = ParentPatch.model_validate(ResearchFixtures(
        legacy_target(target), store=store, materialize=materialize
    ).respond(role, inputs))
    assert supplied == [target]
    assert target.files == []
    assert patch.edits[0].after == expected


@pytest.mark.parametrize("role", CODER_ROLES)
@pytest.mark.parametrize("binding", ("command", "base_snapshot_ref"))
def test_materializer_cannot_rebind_target_command_or_frozen_base(store, tmp_path, role, binding):
    target, incumbent = overlay_pair(tmp_path, store)
    inputs = coder_inputs(store, incumbent)
    original_target = deepcopy(target)

    def materialize(candidate):
        # Mutating the callback argument must not evade comparison with the
        # original target or mutate the fixture recipe object.
        if binding == "command":
            candidate.command.append("--different-budget")
        else:
            candidate.base_snapshot_ref = "artifacts/" + "f" * 64 + ".json"
        return candidate

    with pytest.raises(IntegrityError, match="command or frozen base"):
        ResearchFixtures(
            legacy_target(target), store=store, materialize=materialize
        ).respond(role, inputs)
    assert target == original_target


@pytest.mark.parametrize("role", CODER_ROLES)
@pytest.mark.parametrize("frozen", (False, True))
def test_control_target_cannot_change_parent_argv(store, tmp_path, role, frozen):
    baseline, incumbent = overlay_pair(tmp_path, store)
    target = baseline if frozen else source_candidate({"model.py": BASE_MODEL})
    target.command = ["python", "train.py", "--different-budget"]
    with pytest.raises(ValueError, match="parent's command"):
        ResearchFixtures(legacy_target(target), store=store).respond(
            role, coder_inputs(store, incumbent)
        )


@pytest.mark.parametrize("role", CODER_ROLES)
def test_frozen_target_requires_explicit_store_even_when_overlay_has_changed_file(store, tmp_path, role):
    baseline, incumbent = overlay_pair(tmp_path, store)
    target = source_candidate({"model.py": "WIDTH = 4\n"}, base=baseline.base_snapshot_ref)
    with pytest.raises(ValueError, match="[Ss]tore"):
        ResearchFixtures(legacy_target(target)).respond(role, coder_inputs(store, incumbent))


@pytest.mark.parametrize("role", CODER_ROLES)
@pytest.mark.parametrize("damage", ("missing", "modified"))
def test_frozen_target_integrity_is_verified_without_incumbent_fallback(store, tmp_path, role, damage):
    target, incumbent = overlay_pair(tmp_path, store)
    inputs = coder_inputs(store, incumbent)
    tree = candidate_snapshot(store, target)
    file = load_bundle(store, tree.bundle_ref).files["model.py"]
    path = store.root / file.ref
    if damage == "missing":
        path.unlink()
    else:
        path.write_text("WIDTH = 999\n# corrupt frozen bytes\n")
    with pytest.raises(IntegrityError):
        ResearchFixtures(legacy_target(target), store=store).respond(role, inputs)


@pytest.mark.parametrize("role", CODER_ROLES)
def test_undeclared_target_dependency_corruption_is_not_ignored(store, tmp_path, role):
    target, incumbent = overlay_pair(tmp_path, store)
    inputs = coder_inputs(store, incumbent)
    bundle = load_bundle(store, candidate_snapshot(store, target).bundle_ref)
    (store.root / bundle.files["helper.py"].ref).write_text("# corrupted helper\n")
    with pytest.raises(IntegrityError):
        ResearchFixtures(legacy_target(target), store=store).respond(role, inputs)


@pytest.mark.parametrize("role", CODER_ROLES)
def test_absent_changed_file_in_frozen_target_fails_explicitly(store, tmp_path, role):
    source = tmp_path / "incomplete-frozen-source"
    source.mkdir()
    (source / "train.py").write_text("# Synthetic read-only harness\n")
    base = freeze_source_tree(store, source, ["model.py"])
    incumbent = source_candidate({"model.py": ACTIVE_MODEL}, base=base)
    target = source_candidate(base=base)
    with pytest.raises(ValueError, match="model.py"):
        ResearchFixtures(legacy_target(target), store=store).respond(
            role, coder_inputs(store, incumbent)
        )


@pytest.mark.parametrize("role", CODER_ROLES)
@pytest.mark.parametrize("raw", (b"\xff\xfe", b"NUL = '\x00'\n", b"#" * 250_001))
def test_invalid_frozen_target_source_never_becomes_a_parent_patch(store, tmp_path, role, raw):
    target, incumbent = overlay_pair(tmp_path, store)
    source = tmp_path / "frozen-source"
    (source / "model.py").write_bytes(raw)
    ref = freeze_source_tree(store, source, ["model.py", "transforms.py"])
    target.base_snapshot_ref = incumbent.base_snapshot_ref = ref
    with pytest.raises(ValueError, match="UTF-8|source text limit|bounded"):
        ResearchFixtures(legacy_target(target), store=store).respond(
            role, coder_inputs(store, incumbent)
        )


@pytest.mark.parametrize("role", CODER_ROLES)
def test_missing_existing_parent_anchor_is_an_explicit_error(store, tmp_path, role):
    target, incumbent = overlay_pair(tmp_path, store)
    inputs = coder_inputs(store, incumbent)
    inputs["parent"]["files"] = [
        entry for entry in inputs["parent"]["files"] if entry["path"] != "model.py"
    ]
    with pytest.raises(ValueError, match="missing a declared existing edit target"):
        ResearchFixtures(legacy_target(target), store=store).respond(role, inputs)


@pytest.mark.parametrize("role", CODER_ROLES)
def test_legacy_full_file_control_needs_no_store_and_preserves_other_files(store, role):
    incumbent = source_candidate({"model.py": ACTIVE_MODEL, "helper.py": "KEEP = True\n"})
    target = source_candidate({"model.py": BASE_MODEL})
    inputs = coder_inputs(store, incumbent)
    patch = ParentPatch.model_validate(ResearchFixtures(legacy_target(target)).respond(role, inputs))
    revised, proof = apply_parent_patch(
        store, incumbent, inputs["parent"]["parent_source_ref"], patch,
        declared_paths=["model.py"],
    )
    assert {file.path: file.content for file in revised.files} == {
        "model.py": BASE_MODEL, "helper.py": "KEEP = True\n"
    }
    assert revised.command == incumbent.command and revised.base_snapshot_ref is None
    assert proof["changed_paths"] == ["model.py"]


@pytest.mark.parametrize("role", CODER_ROLES)
def test_absent_changed_file_in_full_file_target_does_not_copy_incumbent(store, role):
    incumbent = source_candidate({"model.py": ACTIVE_MODEL})
    target = source_candidate({"train.py": "# unrelated complete-file target\n"})
    with pytest.raises(ValueError, match="model.py"):
        ResearchFixtures(legacy_target(target)).respond(role, coder_inputs(store, incumbent))


@pytest.mark.parametrize("role", CODER_ROLES)
def test_real_prepared_cifar_recipe_empty_overlay_resolves_frozen_reference(tmp_path, store, role):
    from scientisttwo.testbed.adapter import PreparedDevelopment
    from scientisttwo.testbed.preparation import prepare
    from scientisttwo.testbed.validation import ScriptedResearch, create_validation_spec
    from test_testbed_families import family_bundle

    # Tiny declared synthetic arrays exercise preparation, never training or scoring.
    source = family_bundle(tmp_path / "synthetic-cifar")
    prepared = tmp_path / "prepared"
    prepare("cifar10", source, prepared, provenance="synthetic_fixture", source_format="bundle")
    config = create_validation_spec(
        prepared, image="test:torch", grader_image="test:numpy", level="full", gpus="device=0"
    )
    adapter = PreparedDevelopment(config.native, config.development)
    descriptor = adapter.source_tree()
    adapter.bind_source(freeze_source_tree(store, descriptor["path"], descriptor["editable_paths"]))
    target, incumbent = adapter.baseline(), adapter.control()
    assert target.files == [] and incumbent.files
    changed = adapter.control_component()
    assert changed is not None
    inputs = coder_inputs(store, incumbent, changed=changed)
    patch = ParentPatch.model_validate(ResearchFixtures(
        ScriptedResearch(adapter).respond, store=store
    ).respond(role, inputs))
    revised, proof = apply_parent_patch(
        store, incumbent, inputs["parent"]["parent_source_ref"], patch, declared_paths=[changed]
    )
    adapter.validate(revised)
    assert patch.edits[0].after.encode() == (prepared / "reference" / changed).read_bytes()
    assert materialized_bytes(store, revised) == materialized_bytes(store, target)
    assert proof["changed_paths"] == [changed]
    assert not store.verify()
