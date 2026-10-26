"""Prepared NumPy archives with a memory-mapped directory format for large data."""
from contextlib import contextmanager
from pathlib import Path
import zipfile

import numpy as np


class ArrayDirectory(dict):
    @property
    def files(self):
        return list(self)


def archive_path(root, name):
    root = Path(root)
    packed, directory = root / (name + ".npz"), root / name
    if packed.exists() and directory.exists():
        raise ValueError("Ambiguous packed and directory array representations")
    return directory if directory.is_dir() else packed


@contextmanager
def open_arrays(path):
    path = Path(path)
    if path.suffix == ".npz" and not path.exists() and path.with_suffix("").is_dir():
        path = path.with_suffix("")
    if path.is_dir():
        files = sorted(path.iterdir())
        if not files or any(item.is_symlink() or not item.is_file() or item.suffix != ".npy" for item in files):
            raise ValueError("Array directories contain only regular NumPy files")
        arrays = ArrayDirectory({item.stem: np.load(item, mmap_mode="r", allow_pickle=False) for item in files})
        yield arrays
    else:
        with zipfile.ZipFile(path) as archive:
            if len(archive.infolist()) > 1000 or sum(item.file_size for item in archive.infolist()) > 512 * 1024**2:
                raise ValueError("Packed array archive exceeds the bounded-memory limit; prepare memory-mapped .npy directories instead")
        with np.load(path, allow_pickle=False) as arrays:
            yield arrays


def all_finite(values, *, rows=32):
    for offset in range(0, len(values), rows):
        if not np.isfinite(values[offset:offset + rows]).all():
            return False
    return True
