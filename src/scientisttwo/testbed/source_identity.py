"""Independent provider identity checks, distinct from caller provenance labels."""

import hashlib
import json
from pathlib import Path
import tarfile
import sqlite3
from urllib.parse import quote

from ..benchmarks.files import file_hash, tree_manifest
from ..benchmarks.models import relative_path

CIFAR_BINARY_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
CIFAR_BINARY_SIZE = 170052171
CIFAR_BINARY_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"


def _generated(task_id, source, split_metadata, receipt, receipt_path):
    if task_id not in {"CLRS", "llama-inference"} or receipt.get("status") != "completed":
        raise ValueError("Unexpected generated-task provider receipt")
    evidence = source / relative_path(receipt["generation_store"])
    database = evidence / "campaign.sqlite"
    if not database.is_file() or database.is_symlink():
        raise ValueError("Generated dataset has no immutable execution evidence store")
    connection = sqlite3.connect("file:" + quote(str(database)) + "?mode=ro", uri=True)
    try:
        artifacts = {}
        for path, sha, size in connection.execute("SELECT path,sha,size FROM artifacts"):
            local = evidence / relative_path(path)
            if not local.is_file() or local.is_symlink() or local.stat().st_size != size or file_hash(local) != sha:
                raise ValueError("Generated data source execution artifacts changed")
            artifacts[path] = sha

        def read(ref):
            if ref not in artifacts:
                raise ValueError("Generated data receipt refers to an unrecorded artifact")
            return json.loads((evidence / ref).read_text())

        row = connection.execute("SELECT value FROM metadata WHERE key='generation_complete'").fetchone()
        result_ref = json.loads(row[0]) if row else None
        if result_ref != receipt.get("generation_result_ref"):
            raise ValueError("Generated source result differs from the completed transaction")
        result = read(result_ref)
        request, execution = read(result["request_ref"]), read(result["execution_ref"])
        if request["task_id"] != task_id or execution.get("returncode") != 0 or execution.get("timed_out"):
            raise ValueError("Generated dataset lacks a successful task-specific isolated execution")
        if execution.get("image_id") != result["image_id"] or not result["image_id"].startswith("sha256:"):
            raise ValueError("Generated dataset runtime image identity differs")
        if "--network=none" not in execution.get("argv", []) or "--read-only" not in execution.get("argv", []):
            raise ValueError("Dataset generator was not executed under the required isolation contract")
        package = Path(__file__).parents[1]
        expected_sources = {
            "generator": package / "task_families/templates" / ("prepare_clrs.py" if task_id == "CLRS" else "prepare_llama.py"),
            "orchestrator": Path(__file__).with_name("generation.py"),
            "runner": package / "benchmarks/runner.py", "bridge": package / "live/workspace_worker.py",
        }
        for name, path in expected_sources.items():
            if request["implementation"].get(name) != file_hash(path) or artifacts.get(result["source_refs"].get(name)) != file_hash(path):
                raise ValueError("Generated data implementation differs from the trusted installed generator")
        actual = tree_manifest(source)
        for path in list(actual):
            if path.startswith("provider-control/") or path == "source_receipt.json":
                actual.pop(path)
        if actual != result["output_manifest"]:
            raise ValueError("Generated source bytes differ from the successful execution receipt")
        if task_id == "llama-inference" and (
            not request["variant"].startswith("adapted-") or split_metadata.get("requests_per_partition") != 400
            or split_metadata.get("context_length") != 128 or split_metadata.get("batch_size") != 4
        ):
            raise ValueError("Llama generated source differs from its explicitly adapted full workload")
        return {"status": "verified", "kind": "isolated-generated-benchmark-variant",
                "variant": request["variant"], "image_id": result["image_id"],
                "generator_sha256": request["implementation"]["generator"],
                "receipt_sha256": file_hash(receipt_path),
                "scope": "Verified generated inputs for the declared adapted protocol; original published dataset/model equivalence is not asserted"}
    finally:
        connection.close()


def _receipted_provider(task_id, source, split_metadata, receipt, receipt_path):
    from .providers import PUBLIC_ARCHIVES, KAGGLE_COMPETITIONS, archive_members
    implementation = Path(__file__).with_name("providers.py")
    if receipt.get("schema_version") != 1 or receipt.get("status") != "completed" or receipt.get("live_model_calls") != 0:
        raise ValueError("Provider receipt is not a completed supported acquisition")
    snapshot = source / "provider-control/providers.py"
    if not snapshot.is_file() or file_hash(snapshot) != receipt.get("acquisition_implementation_sha256"):
        raise ValueError("Provider acquisition source provenance is missing or modified")
    if file_hash(snapshot) != file_hash(implementation):
        raise ValueError("Provider acquisition implementation differs; reverify using the declared supported acquisition version")
    provider = receipt.get("provider", {})
    if task_id in PUBLIC_ARCHIVES:
        expected = PUBLIC_ARCHIVES[task_id]
        transport = provider.get("transport", {})
        if provider.get("kind") != "https" or provider.get("locator") != expected or transport.get("requested_url") != expected or transport.get("tls_validation") != "system-certificate-verification":
            raise ValueError("Public provider identity or TLS acquisition evidence differs from the task recipe")
    elif task_id in KAGGLE_COMPETITIONS:
        if (provider.get("kind") != "kaggle-cli" or provider.get("locator") != KAGGLE_COMPETITIONS[task_id]
                or provider.get("returncode") != 0 or provider.get("terms_accepted") is not True
                or not provider.get("cli_sha256")):
            raise ValueError("Kaggle acquisition receipt does not match the authorized competition")
    else:
        raise ValueError("This task requires its generated-workload or licensed-model provider receipt")
    archive = receipt.get("archive", {})
    name = archive.get("path")
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Provider archive path must remain inside the source bundle")
    archive_path = source / name
    if archive_path.is_symlink() or not archive_path.is_file() or archive_path.stat().st_size != archive.get("size") or file_hash(archive_path) != archive.get("sha256"):
        raise ValueError("Provider archive differs from its acquisition receipt")
    if task_id in PUBLIC_ARCHIVES:
        transport = provider["transport"]
        if transport.get("sha256") != archive["sha256"] or transport.get("size") != archive["size"]:
            raise ValueError("Transport and archive content identities differ")
    members = archive_members(archive_path)
    if not members or members != receipt.get("members"):
        raise ValueError("Provider member manifest differs from the actual source archive")
    for name, expected in members.items():
        path = source / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != expected["size"] or file_hash(path) != expected["sha256"]:
            raise ValueError("Extracted raw data differ from the recorded provider archive")
    if task_id == "imdb" and (split_metadata.get("original_training_rows") != 25000 or split_metadata.get("original_final_rows") != 25000):
        raise ValueError("IMDb canonical identity requires all 25000 training and 25000 final labeled reviews")
    if task_id == "ogbn-arxiv" and any(split_metadata.get(key) != expected for key, expected in
                                       {"nodes": 169343, "edges": 1166243, "features": 128, "classes": 40}.items()):
        raise ValueError("OGBN-Arxiv canonical graph dimensions differ from the official dataset")
    if task_id == "fathomnet":
        # Image files are fetched separately from the competition archive. Until
        # their provider receipts are supplied, label identity alone is not full
        # dataset identity.
        if receipt.get("media_complete") is not True:
            return {"status": "unverified", "kind": "verified-label-archive-missing-image-provider-receipts",
                    "archive_sha256": archive["sha256"], "reason": "FathomNet external images need content-bound provider acquisition evidence; acquire with include_images"}
        import csv
        from urllib.parse import urlparse
        with (source / "multilabel_classification/train.csv").open() as stream:
            required = {row["id"] + ".png" for row in csv.DictReader(stream)}
        media = receipt.get("media", {})
        if set(media) != required:
            raise ValueError("FathomNet image receipts do not cover every category-labeled image")
        for name, item in media.items():
            if Path(name).name != name:
                raise ValueError("Unsafe image provider receipt path")
            path = source / "images" / name
            if not path.is_file() or path.is_symlink() or file_hash(path) != item.get("sha256") or path.stat().st_size != item.get("size"):
                raise ValueError("FathomNet image bytes differ from their provider receipt")
            if item["source"] == "declared-image-provider" and (
                item.get("tls_validation") != "system-certificate-verification"
                or urlparse(item.get("requested_url", "")).scheme != "https"
            ):
                raise ValueError("FathomNet image receipt lacks a valid TLS download")
    return {"status": "verified", "kind": "recorded-provider-acquisition-and-member-hashes",
            "provider": provider, "archive_sha256": archive["sha256"], "archive_size": archive["size"],
            "member_count": len(members), "receipt_sha256": file_hash(receipt_path),
            "dataset_version": receipt["dataset_version"],
            "verification_scope": "Trusted local acquisition receipt and byte consistency, not an independently signed provider attestation"}


def _stream_hash(stream):
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return {"sha256": digest.hexdigest(), "size": size}


def dataset_identity(task_id, source, split_metadata, *, provenance, source_receipt=None):
    """Only independently checked data can satisfy the actual-benchmark data gate."""
    source = Path(source)
    if provenance == "synthetic_fixture":
        return {"status": "synthetic_fixture", "reason": "Explicitly synthetic contract data"}
    if task_id == "vectorization":
        # preparation verified the immutable Git pin before generating these
        # workloads and retained the baseline and deterministic generation seeds.
        return {"status": "verified", "kind": "pinned-generated-benchmark-workload",
                "source_sha256": file_hash(source / "MLAgentBench/benchmarks/vectorization/env/train.py"),
                "split_seed": split_metadata["split_seed"], "final_seed": split_metadata["final_seed"]}
    receipt_path = Path(source_receipt).resolve() if source_receipt else source / "source_receipt.json"
    if not receipt_path.is_file():
        return {"status": "unverified", "kind": "user_asserted_actual_data",
                "reason": "No independently verifiable provider receipt; declared actual_data is not proof of canonical dataset identity"}
    if receipt_path.is_symlink() or receipt_path.stat().st_size > 128_000_000:
        raise ValueError("Source provider receipt must be a bounded regular JSON file")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("task_id") != task_id:
        raise ValueError("Provider receipt belongs to a different task")
    if receipt.get("provider", {}).get("kind") == "isolated-generator":
        return _generated(task_id, source, split_metadata, receipt, receipt_path)
    if task_id != "cifar10":
        return _receipted_provider(task_id, source, split_metadata, receipt, receipt_path)
    archive_value = receipt.get("archive", {})
    archive_name = archive_value.get("path") if isinstance(archive_value, dict) else None
    if not archive_name:
        raise ValueError("CIFAR provider receipt requires archive.path")
    archive_path = Path(archive_name)
    if not archive_path.is_absolute():
        archive_path = receipt_path.parent / archive_path
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("CIFAR archive must be a regular locally available file")
    if archive_path.stat().st_size != CIFAR_BINARY_SIZE or file_hash(archive_path, "md5") != CIFAR_BINARY_MD5:
        raise ValueError("CIFAR archive differs from the published official binary archive")
    if split_metadata.get("raw_format") != "official-cifar10-binary" or split_metadata.get("original_training_rows") != 50000 or split_metadata.get("original_final_rows") != 10000:
        raise ValueError("Canonical CIFAR readiness requires the complete 50000/10000 official binary partitions")
    root = source / "cifar-10-batches-bin" if (source / "cifar-10-batches-bin").is_dir() else source
    required = {f"data_batch_{number}.bin" for number in range(1, 6)} | {"test_batch.bin"}
    found = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = Path(member.name)
            if relative.as_posix() not in {"cifar-10-batches-bin/" + name for name in required}:
                continue
            if not member.isfile() or member.name in found:
                raise ValueError("CIFAR archive has invalid or duplicate binary members")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("CIFAR archive member cannot be read")
            with stream:
                value = _stream_hash(stream)
            local = root / relative.name
            if not local.is_file() or local.is_symlink() or local.stat().st_size != value["size"] or file_hash(local) != value["sha256"]:
                raise ValueError("Prepared raw CIFAR batch does not match its verified official archive member")
            found[member.name] = value
    if set(Path(name).name for name in found) != required:
        raise ValueError("CIFAR source archive does not contain the full required binary dataset")
    return {"status": "verified", "kind": "official-published-archive-and-member-hashes",
            "provider_url": CIFAR_BINARY_URL, "published_md5": CIFAR_BINARY_MD5,
            "archive_sha256": file_hash(archive_path), "archive_size": CIFAR_BINARY_SIZE,
            "raw_members": found, "receipt_sha256": file_hash(receipt_path),
            "training_rows": 50000, "final_rows": 10000}
