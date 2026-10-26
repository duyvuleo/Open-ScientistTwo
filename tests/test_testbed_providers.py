import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile

import pytest

from scientisttwo.benchmarks.files import file_hash
from scientisttwo.testbed import providers
from scientisttwo.testbed.source_identity import dataset_identity
from scientisttwo.testbed.preparation import prepare_plan


def test_acquisition_requires_explicit_permission_and_terms(tmp_path):
    with pytest.raises(ValueError, match="allow-downloads"):
        providers.acquire("cifar10", tmp_path / "out")
    with pytest.raises(ValueError, match="terms"):
        providers.acquire("house-price", tmp_path / "out", allow_downloads=True)
    assert not (tmp_path / "out").exists()


def test_archive_extraction_rejects_traversal_links_and_expansion(tmp_path):
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escaped", "payload")
    with pytest.raises(ValueError, match="Unsafe"):
        providers._extract(unsafe, tmp_path / "out", max_bytes=100)
    assert not (tmp_path / "escaped").exists()
    linked = tmp_path / "linked.tar"
    with tarfile.open(linked, "w") as archive:
        item = tarfile.TarInfo("link")
        item.type, item.linkname = tarfile.SYMTYPE, "/etc/passwd"
        archive.addfile(item)
    with pytest.raises(ValueError, match="links"):
        providers._extract(linked, tmp_path / "links", max_bytes=100)
    large = tmp_path / "large.zip"
    with zipfile.ZipFile(large, "w") as archive:
        archive.writestr("data.csv", "0123456789")
    with pytest.raises(ValueError, match="size"):
        providers._extract(large, tmp_path / "large", max_bytes=5)


def test_archive_member_hashes_are_recomputed_from_bytes(tmp_path):
    path = tmp_path / "data.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        item = tarfile.TarInfo("folder/data.csv")
        item.size = 5
        archive.addfile(item, io.BytesIO(b"a,b\n1"))
    members = providers._extract(path, tmp_path / "out", max_bytes=100)
    assert providers.archive_members(path) == members
    assert members["folder/data.csv"]["sha256"] == file_hash(tmp_path / "out/folder/data.csv")


def test_kaggle_receipt_and_modified_data_detection_without_network(tmp_path, monkeypatch):
    # Explicit provider contract fixture. This invokes no Kaggle or network call
    # and does not establish canonical task readiness.
    executable = tmp_path / "fake-kaggle"
    executable.write_text("synthetic CLI executable")
    monkeypatch.setattr(shutil, "which", lambda value: str(executable))

    def fake_run(argv, **kwargs):
        destination = Path(argv[argv.index("-p") + 1])
        competition = argv[argv.index("-c") + 1]
        with zipfile.ZipFile(destination / f"{competition}.zip", "w") as archive:
            archive.writestr("train.csv", "Id,SalePrice\n1,100\n")
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    output = tmp_path / "source"
    providers.acquire("house-price", output, allow_downloads=True, terms_accepted=True)
    receipt = json.loads((output / "source_receipt.json").read_text())
    assert receipt["provider"]["locator"] == "home-data-for-ml-course"
    assert "stderr" not in receipt and "stdout" not in receipt
    identity = dataset_identity("house-price", output, {}, provenance="actual_data")
    assert identity["kind"] == "recorded-provider-acquisition-and-member-hashes"
    (output / "train.csv").write_text("Id,SalePrice\n1,9999\n")
    with pytest.raises(ValueError, match="Extracted raw data"):
        dataset_identity("house-price", output, {}, provenance="actual_data")


def test_download_receipts_drop_query_credentials():
    assert providers._safe_url("https://example.org/file?token=private#fragment") == "https://example.org/file"
    with pytest.raises(ValueError):
        providers._safe_url("https://user:password@example.org/file")


def test_prepare_plans_only_link_real_provider_families():
    for task in ("cifar10", "imdb", "ogbn-arxiv", "babylm"):
        plan = prepare_plan(task)
        assert not any("kaggle.com" in value for value in plan["source_urls"])
        assert plan["acquisition_command"]
    assert any("kaggle.com/competitions/home-data-for-ml-course" in value
               for value in prepare_plan("house-price")["source_urls"])
