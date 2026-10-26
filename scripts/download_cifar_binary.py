"""Explicit, resumable download of official CIFAR-10 binary data with published checksum."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

URL = "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz"
SIZE = 170052171
MD5 = "c32a1d4ab5d03f1284b67883e8d87530"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--connections", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    target = args.destination.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    parts = target.with_suffix(target.suffix + ".parts")
    parts.mkdir(exist_ok=True)
    stride = (SIZE + args.connections - 1) // args.connections

    def fetch(index):
        start, end = index * stride, min((index + 1) * stride, SIZE) - 1
        path = parts / f"{start}-{end}"
        length = end - start + 1
        for attempt in range(3):
            offset = path.stat().st_size if path.exists() else 0
            if offset == length:
                return path
            if offset > length:
                raise ValueError("Existing range exceeds its declared size")
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start + offset}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f"bytes {start + offset}-{end}/{SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor the exact byte range")
                    with path.open("ab") as stream:
                        remaining = length - offset
                        while remaining:
                            data = response.read(min(1024 * 1024, remaining))
                            if not data:
                                raise OSError("Incomplete range response")
                            stream.write(data)
                            remaining -= len(data)
            except (TimeoutError, OSError):
                if attempt == 2:
                    raise
        return path

    print(json.dumps({"source": URL, "expected_bytes": SIZE, "connections": args.connections}), flush=True)
    with ThreadPoolExecutor(max_workers=args.connections) as pool:
        paths = list(pool.map(fetch, range(args.connections)))
    temporary = target.with_suffix(target.suffix + ".verified-tmp")
    with temporary.open("wb") as stream:
        for path in paths:
            with path.open("rb") as reader:
                shutil.copyfileobj(reader, stream)
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with temporary.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    if temporary.stat().st_size != SIZE or md5.hexdigest() != MD5:
        raise ValueError("Official CIFAR-10 archive checksum mismatch")
    temporary.replace(target)
    receipt = {"status": "pass", "url": URL, "source_page": "https://cave.cs.toronto.edu/kriz/cifar.html", "size": SIZE, "published_md5": MD5, "sha256": sha256.hexdigest()}
    target.with_suffix(target.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
