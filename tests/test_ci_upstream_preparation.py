"""Bounded metadata preparation never admits other LFS content or corrupt bytes."""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

from scientisttwo.benchmarks import lfs


SPEC = importlib.util.spec_from_file_location(
    "prepare_ci_upstreams", Path(__file__).resolve().parents[1] / "scripts" / "prepare_ci_upstreams.py"
)
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)


@pytest.fixture
def metadata_checkout(tmp_path, monkeypatch):
    payload = b"team,score\nfixture,1.0\n"
    pointer = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    tracked = ("version https://git-lfs.github.com/spec/v1\n"
               f"oid sha256:{pointer['sha256']}\nsize {pointer['size']}\n").encode()
    target = tmp_path / preparation.RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(tracked)
    checked, requests = [], []

    def verify(root, revision):
        assert root == tmp_path
        assert revision == preparation.PINS["mle-bench"]["revision"]
        checked.append(revision)

    def pointer_at_head(root, relative):
        assert root == tmp_path and relative == preparation.RELATIVE_PATH
        return pointer

    def download(url, timeout):
        assert url == ("https://media.githubusercontent.com/media/openai/mle-bench/"
                       + preparation.PINS["mle-bench"]["revision"] + "/" + preparation.RELATIVE_PATH)
        assert timeout == 60
        requests.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(preparation, "verify_repository", verify)
    monkeypatch.setattr(preparation, "pointer_at_head", pointer_at_head)
    monkeypatch.setattr(lfs, "pointer_at_head", pointer_at_head)
    monkeypatch.setattr(lfs.urllib.request, "urlopen", download)
    return tmp_path, target, tracked, payload, pointer, checked, requests


def test_fetches_only_pinned_metadata_and_verifies_size_hash(metadata_checkout):
    root, target, _, payload, pointer, checked, requests = metadata_checkout
    result = preparation.prepare(root)
    assert target.read_bytes() == payload
    assert result["status"] == "pass" and result["cached"] is False
    assert result["bytes"] == pointer["size"] and result["sha256"] == pointer["sha256"]
    assert len(requests) == 1 and len(checked) == 2
    assert [path for path in root.rglob("*") if path.is_file()] == [target]


def test_existing_verified_metadata_requires_no_download(metadata_checkout):
    root, target, _, payload, _, checked, requests = metadata_checkout
    target.write_bytes(payload)
    assert preparation.prepare(root)["cached"] is True
    assert not requests and len(checked) == 2


def test_oversized_declared_metadata_is_rejected_before_download(metadata_checkout):
    root, target, tracked, _, pointer, _, requests = metadata_checkout
    pointer["size"] = preparation.MAX_METADATA_BYTES + 1
    with pytest.raises(ValueError, match="bounded tracked"):
        preparation.prepare(root)
    assert not requests and target.read_bytes() == tracked


@pytest.mark.parametrize("change", ["oversized", "truncated", "wrong_hash"])
def test_corrupt_response_preserves_original_pointer(metadata_checkout, monkeypatch, change):
    root, target, tracked, payload, _, _, _ = metadata_checkout
    broken = {"oversized": payload + b"extra", "truncated": payload[:-1],
              "wrong_hash": b"x" * len(payload)}[change]
    monkeypatch.setattr(lfs.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(broken))
    with pytest.raises(ValueError, match="exceeds its declared size|hash verification failed"):
        preparation.prepare(root)
    assert target.read_bytes() == tracked
    assert not target.with_suffix(".download").exists()


def test_revision_failure_prevents_download(metadata_checkout, monkeypatch):
    root, target, tracked, _, _, _, requests = metadata_checkout

    def changed_revision(*args):
        raise ValueError("Upstream revision differs from the manifest pin")

    monkeypatch.setattr(preparation, "verify_repository", changed_revision)
    with pytest.raises(ValueError, match="revision differs"):
        preparation.prepare(root)
    assert not requests and target.read_bytes() == tracked
