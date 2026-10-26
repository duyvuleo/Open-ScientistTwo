"""Explicit provider acquisition with bounded downloads and content-bound receipts.

Provider receipts record this trusted acquisition operation. They are not provider
signatures. Kaggle credentials remain in the user's existing CLI configuration;
stdout/stderr and signed redirect URLs are never persisted or displayed.
"""

from datetime import datetime, timezone
import csv
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import ipaddress
import socket
import subprocess
import tarfile
import tempfile
import time
from urllib.parse import urlparse, urlunparse
import urllib.request
import zipfile

from ..benchmarks.files import FORBIDDEN, file_hash, tree_manifest
from .catalog import get_recipe

PUBLIC_ARCHIVES = {
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
    "imdb": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    "ogbn-arxiv": "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
    "babylm": "https://github.com/babylm/babylm.github.io/raw/main/babylm_data.zip",
}
KAGGLE_COMPETITIONS = {
    "house-price": "home-data-for-ml-course", "spaceship-titanic": "spaceship-titanic",
    "feedback": "feedback-prize-english-language-learning",
    "amp-parkinsons-disease-progression-prediction": "amp-parkinsons-disease-progression-prediction",
    "identify-contrails": "google-research-identify-contrails-reduce-global-warming",
    "fathomnet": "fathomnet-out-of-sample-detection",
}


def _safe_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or not parsed.hostname:
        raise ValueError("Provider downloads require HTTPS without URL credentials")
    return urlunparse(parsed._replace(query="", fragment=""))


def _public_image_url(url):
    parsed = urlparse(url)
    if parsed.scheme == "http":
        parsed = parsed._replace(scheme="https")
        url = urlunparse(parsed)
    _safe_url(url)
    if parsed.port not in (None, 443):
        raise ValueError("Image providers must use the public HTTPS port")
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Image provider addresses must be public")
    return url


class _HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, public_images=False):
        super().__init__()
        self.public_images = public_images

    def redirect_request(self, request, fp, code, message, headers, newurl):
        if self.public_images:
            newurl = _public_image_url(newurl)
        else:
            _safe_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def _download(url, output, *, max_bytes, seconds, public_images=False):
    if public_images:
        url = _public_image_url(url)
    else:
        _safe_url(url)
    started = time.monotonic()
    digest, size = hashlib.sha256(), 0
    try:
        opener = urllib.request.build_opener(_HTTPSRedirects(public_images=public_images))
        with opener.open(url, timeout=min(seconds, 30)) as response, Path(output).open("xb") as stream:
            final_url = _safe_url(response.geturl())
            if getattr(response, "status", 200) != 200:
                raise ValueError("Provider returned a non-success response")
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > max_bytes or time.monotonic() - started > seconds:
                    raise ValueError("Provider download exceeded the explicit size/time budget")
                digest.update(block)
                stream.write(block)
    except Exception:
        # Network exception messages may include signed redirect query strings.
        raise ValueError("Provider download failed or exceeded budget; inspect access/network configuration outside the task bundle") from None
    if size == 0:
        raise ValueError("Provider returned an empty archive")
    return {"requested_url": _safe_url(url), "final_url": final_url,
            "tls_validation": "system-certificate-verification", "sha256": digest.hexdigest(), "size": size}


def _member_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or "\x00" in name:
        raise ValueError("Unsafe archive member path")
    if any(part.lower() in FORBIDDEN or part.startswith(".env") for part in path.parts):
        raise ValueError("Credential-like files cannot be extracted into provider data")
    return path.as_posix()


def _extract(archive, destination, *, max_bytes, max_files=250_000):
    """Extract only ordinary files/directories; never invoke external unpackers."""
    destination = Path(destination)
    destination.mkdir()
    records, total = {}, 0

    def write(name, size, stream):
        nonlocal total
        name = _member_name(name)
        if name in records or len(records) >= max_files:
            raise ValueError("Duplicate archive member or file-count budget exceeded")
        total += size
        if size < 0 or total > max_bytes:
            raise ValueError("Archive uncompressed size exceeds the explicit preparation budget")
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        digest, written = hashlib.sha256(), 0
        with path.open("xb") as target:
            while block := stream.read(1024 * 1024):
                written += len(block)
                if written > size:
                    raise ValueError("Archive member size differs from its header")
                digest.update(block)
                target.write(block)
        if written != size:
            raise ValueError("Truncated archive member")
        records[name] = {"sha256": digest.hexdigest(), "size": written}

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _member_name(member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError("Archive symlinks are forbidden")
                if member.is_dir():
                    continue
                if member.flag_bits & 1:
                    raise ValueError("Encrypted data archives are unsupported")
                with bundle.open(member) as stream:
                    write(member.filename, member.file_size, stream)
    else:
        with tarfile.open(archive, "r:*") as bundle:
            for member in bundle:
                _member_name(member.name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("Archive links and special files are forbidden")
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError("Archive member cannot be read")
                with stream:
                    write(member.name, member.size, stream)
    if not records:
        raise ValueError("Provider archive contains no ordinary files")
    return records


def archive_members(archive, *, max_bytes=200 * 1024**3, max_files=250_000):
    """Recompute archive members without extraction or reliance on a caller manifest."""
    records, total = {}, 0

    def record(name, expected_size, stream):
        nonlocal total
        name = _member_name(name)
        total += expected_size
        if name in records or len(records) >= max_files or total > max_bytes:
            raise ValueError("Archive exceeds identity verification bounds")
        digest, size = hashlib.sha256(), 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            if size > expected_size:
                raise ValueError("Archive member exceeded its header size")
            digest.update(block)
        if size != expected_size:
            raise ValueError("Truncated provider archive")
        records[name] = {"sha256": digest.hexdigest(), "size": size}

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _member_name(member.filename)
                if member.is_dir():
                    continue
                if stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1:
                    raise ValueError("Unsafe provider archive member")
                with bundle.open(member) as stream:
                    record(member.filename, member.file_size, stream)
    else:
        with tarfile.open(archive, "r:*") as bundle:
            for member in bundle:
                _member_name(member.name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("Unsafe provider archive member")
                with bundle.extractfile(member) as stream:
                    record(member.name, member.size, stream)
    return records


def acquire(task_id, destination, *, allow_downloads=False, terms_accepted=False,
            download_seconds=3600, max_download_bytes=100 * 1024**3,
            max_extracted_bytes=200 * 1024**3, kaggle_executable="kaggle", include_images=False):
    """Download a declared provider archive and retain exact archive/member receipts."""
    recipe = get_recipe(task_id)
    if not allow_downloads:
        raise ValueError("Downloads require explicit --allow-downloads")
    if recipe.access_requirements and not terms_accepted:
        raise ValueError("Accept applicable provider terms externally, then explicitly set --terms-accepted")
    if min(download_seconds, max_download_bytes, max_extracted_bytes) <= 0:
        raise ValueError("Positive acquisition time and byte budgets are required")
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Acquisition destination must be new; verified existing data can be reused by preparation")
    if task_id not in PUBLIC_ARCHIVES and task_id not in KAGGLE_COMPETITIONS:
        raise ValueError("This recipe uses generated inputs or explicitly licensed model assets; use its dedicated preparation procedure")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=".testbed-acquire-", dir=destination.parent) as temporary:
        work = Path(temporary)
        if task_id in PUBLIC_ARCHIVES:
            url = PUBLIC_ARCHIVES[task_id]
            name = Path(urlparse(url).path).name
            archive = work / name
            transport = _download(url, archive, max_bytes=max_download_bytes, seconds=download_seconds)
            provider = {"kind": "https", "locator": url, "transport": transport}
        else:
            executable = shutil.which(kaggle_executable)
            if not executable:
                raise ValueError("Install the official Kaggle CLI and authenticate outside the task bundle")
            try:
                installed_version = version("kaggle")
            except PackageNotFoundError:
                installed_version = "external-cli-version-unverified"
            competition = KAGGLE_COMPETITIONS[task_id]
            try:
                result = subprocess.run([executable, "competitions", "download", "-c", competition,
                                         "-p", str(work), "-q"], capture_output=True,
                                        timeout=download_seconds, check=False)
            except (OSError, subprocess.TimeoutExpired):
                raise ValueError("Kaggle acquisition failed or timed out; credentials and CLI logs remain outside the bundle") from None
            if result.returncode:
                raise ValueError("Kaggle rejected the download; check account access and competition terms outside the bundle")
            archive = work / f"{competition}.zip"
            if not archive.is_file() or archive.stat().st_size > max_download_bytes:
                raise ValueError("Kaggle did not produce a bounded expected competition archive")
            provider = {"kind": "kaggle-cli", "locator": competition,
                        "cli_sha256": file_hash(executable), "cli_version": installed_version,
                        "terms_accepted": True, "returncode": 0}
        extracted = work / "extracted"
        members = _extract(archive, extracted, max_bytes=max_extracted_bytes)
        archive_record = {"path": f"provider-archive/{archive.name}", "sha256": file_hash(archive),
                          "size": archive.stat().st_size}
        (extracted / "provider-archive").mkdir()
        shutil.move(str(archive), extracted / archive_record["path"])
        receipt = {"schema_version": 1, "task_id": task_id, "status": "completed",
                   "acquisition_implementation_sha256": file_hash(Path(__file__)),
                   "provider": provider, "archive": archive_record, "members": members,
                   "dataset_version": "archive-sha256:" + archive_record["sha256"],
                   "acquired_at": datetime.now(timezone.utc).isoformat(),
                   "live_model_calls": 0,
                   "verification_scope": "Recorded provider acquisition and exact archive/member bytes; not a provider digital signature"}
        if task_id == "fathomnet" and include_images:
            with (extracted / "multilabel_classification/train.csv").open() as stream:
                required = {row["id"] + ".png" for row in csv.DictReader(stream)}
            source_images = json.loads((extracted / "object_detection/train.json").read_text())["images"]
            indexed = {item["file_name"]: item for item in source_images}
            if not required <= set(indexed):
                raise ValueError("FathomNet category IDs do not match official image metadata")
            media, downloaded = {}, archive_record["size"]
            image_root = extracted / "images"
            image_root.mkdir(exist_ok=True)
            for name in sorted(required):
                if PurePosixPath(name).name != name:
                    raise ValueError("Image names must not contain directories")
                path = image_root / name
                if path.exists():
                    member = members.get("images/" + name)
                    if member is None:
                        raise ValueError("Unexpected preexisting image outside the provider archive")
                    media[name] = {"source": "competition-archive", **member}
                    continue
                remaining = download_seconds - (time.monotonic() - started)
                if remaining <= 0 or downloaded >= max_download_bytes:
                    raise ValueError("FathomNet media acquisition exhausted its explicit time/byte budget")
                item = _download(indexed[name]["coco_url"], path, max_bytes=max_download_bytes - downloaded,
                                 seconds=remaining, public_images=True)
                downloaded += item["size"]
                media[name] = {"source": "declared-image-provider", **item}
            receipt["media"] = media
            receipt["media_complete"] = True
        elif task_id == "fathomnet":
            receipt["media_complete"] = False
        (extracted / "provider-control").mkdir()
        shutil.copyfile(Path(__file__), extracted / "provider-control/providers.py")
        (extracted / "source_receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        tree_manifest(extracted)
        extracted.rename(destination)
    return {"source": str(destination), "receipt": str(destination / "source_receipt.json"),
            "archive_sha256": archive_record["sha256"], "files": len(members), "live_model_calls": 0}
