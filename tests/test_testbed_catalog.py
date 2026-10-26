import pytest

from scientisttwo.testbed.catalog import get_recipe, recipes, UPSTREAM_REVISION


def test_exact_canonical_coverage_no_aliases_or_demos():
    records = recipes()
    assert len(records) == 13
    assert len({item.task_id for item in records}) == 13
    assert not {"debug", "literature-review-tool", "bibtex-generation"} & {r.task_id for r in records}
    assert all(item.upstream_revision == UPSTREAM_REVISION for item in records)
    assert all(len(item.identity) == 64 for item in records)
    with pytest.raises(ValueError, match="canonical"):
        get_recipe("debug")


def test_recipe_boundaries_and_baseline_provenance():
    assert get_recipe("house-price").original_starter_status == "incomplete_starter"
    assert get_recipe("feedback").original_starter_status == "random_prediction_starter"
    assert get_recipe("fathomnet").metric == "map_at_20"
    assert get_recipe("fathomnet").upstream_grading == "known_defective"
    assert get_recipe("llama-inference").upstream_grading == "unavailable"
    assert get_recipe("babylm").training_constraints["from_scratch"]
    assert get_recipe("babylm").training_constraints["tokenizer_frozen"]
    first = get_recipe("feedback")
    first.targets.append("accidental-caller-mutation")
    assert "accidental-caller-mutation" not in get_recipe("feedback").targets


def test_timing_profile_and_checkpoint_contracts():
    for task in ("vectorization", "llama-inference"):
        recipe = get_recipe(task)
        assert recipe.training_constraints["warmup_requests"] == 5
        assert recipe.training_constraints["measurement_pairs"] == 10
    for task in ("CLRS", "babylm"):
        outputs = get_recipe(task).outputs
        assert outputs[0].kind == "directory" and outputs[0].role == "checkpoint"
        assert outputs[0].max_files > 1
