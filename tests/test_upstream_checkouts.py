"""Test discovery policy only: no downloads, services, datasets or model calls."""

import pytest

import upstream_checkouts as sources


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(sources, "PROJECT_ROOT", root)
    for _, variable, legacy in sources.CHECKOUTS.values():
        monkeypatch.delenv(variable, raising=False)
        if legacy:
            monkeypatch.delenv(legacy, raising=False)
    monkeypatch.delenv("SCIENTISTTWO_REQUIRE_UPSTREAMS", raising=False)
    return root


def test_explicit_checkout_wins_and_bad_explicit_path_does_not_skip(isolated, monkeypatch):
    custom = isolated / "custom"
    custom.mkdir()
    monkeypatch.setenv("SCIENTISTTWO_PAPERORCHESTRA_ROOT", str(custom))
    assert sources.require_upstream("paperorchestra") == custom
    monkeypatch.setenv("SCIENTISTTWO_PAPERORCHESTRA_ROOT", str(custom / "absent"))
    with pytest.raises(pytest.fail.Exception, match="not a directory"):
        sources.require_upstream("paperorchestra")


def test_portable_checkout_locations(isolated):
    sibling = isolated.parent / "MLAgentBench"
    sibling.mkdir()
    assert sources.upstream_path("mlagentbench") == sibling
    local = isolated / ".upstream" / "MLAgentBench"
    local.mkdir(parents=True)
    assert sources.upstream_path("mlagentbench") == local


def test_bundled_paperorchestra_is_available_in_required_mode(isolated, monkeypatch):
    bundled = isolated / "src" / "scientisttwo" / "_vendor" / "paperorchestra"
    bundled.mkdir(parents=True)
    monkeypatch.setenv("SCIENTISTTWO_REQUIRE_UPSTREAMS", "all")
    assert sources.require_upstream("paperorchestra") == bundled


def test_optional_missing_is_explicit_skip(isolated):
    with pytest.raises(pytest.skip.Exception, match="Optional pinned"):
        sources.require_upstream("paperorchestra")


@pytest.mark.parametrize("selection", ["all", "1", "true", "mlagentbench,paperorchestra"])
def test_required_missing_fails_instead_of_skipping(isolated, monkeypatch, selection):
    monkeypatch.setenv("SCIENTISTTWO_REQUIRE_UPSTREAMS", selection)
    with pytest.raises(pytest.fail.Exception, match="Required upstream"):
        sources.require_upstream("mlagentbench")


def test_typo_in_required_mode_fails(isolated, monkeypatch):
    monkeypatch.setenv("SCIENTISTTWO_REQUIRE_UPSTREAMS", "misspelled")
    with pytest.raises(pytest.fail.Exception, match="Unknown required upstream"):
        sources.upstream_path("paperorchestra")
