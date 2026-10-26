"""Deterministic conversion of official raw formats into protected task bundles.

These functions do not download, unpickle, import ML frameworks, or run submitted
code. Model caches remain explicit inputs. The BabyLM converter uses only the
optional Rust tokenizer binding against a frozen local tokenizer.json.
"""

import hashlib
import ast
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..benchmarks.files import tree_manifest

RAW_TASKS = {"cifar10", "imdb", "ogbn-arxiv", "babylm", "identify-contrails", "fathomnet"}


def _roots(destination):
    result = {}
    for phase in ("screen", "full", "final"):
        public, private = destination / "phases" / phase / "public", destination / "phases" / phase / "private"
        public.mkdir(parents=True)
        private.mkdir(parents=True)
        result[phase] = (public, private)
    return result


def _copy_assets(source, destination):
    if (source / "assets").is_dir():
        tree_manifest(source / "assets")
        shutil.copytree(source / "assets", destination / "assets")


def _holdouts(labels, seed):
    train, development = train_test_split(np.arange(len(labels)), test_size=0.2,
                                         stratify=labels, random_state=seed)
    screen, _ = train_test_split(train, test_size=0.5, stratify=np.asarray(labels)[train], random_state=seed + 1)
    return np.sort(train), np.sort(development), np.sort(screen)


def _cifar(source, destination, seed):
    root = source / "cifar-10-batches-bin" if (source / "cifar-10-batches-bin").is_dir() else source
    paths = [root / f"data_batch_{i}.bin" for i in range(1, 6)]
    test_path = root / "test_batch.bin"
    if not all(path.is_file() for path in [*paths, test_path]):
        raise ValueError("CIFAR raw conversion requires official binary-format batches; Python pickle batches are not loaded on the host")

    def read(paths):
        batches = []
        for path in paths:
            values = np.fromfile(path, dtype=np.uint8)
            if not values.size or values.size % 3073:
                raise ValueError("CIFAR binary batch byte length is invalid")
            batches.append(values.reshape(-1, 3073))
        rows = np.concatenate(batches)
        labels = rows[:, 0].astype(np.int64)
        if np.any(labels > 9):
            raise ValueError("CIFAR class IDs must be in [0,9]")
        features = rows[:, 1:].reshape(-1, 3, 32, 32).astype(np.float32) / 127.5 - 1
        return features, labels

    features, labels = read(paths)
    final_x, final_y = read([test_path])
    train, development, screen = _holdouts(labels, seed)
    for phase, (public, private) in _roots(destination).items():
        rows = screen if phase == "screen" else np.arange(len(labels)) if phase == "final" else train
        ids = np.asarray([f"cifar-train-{i}" for i in rows])
        evaluation = final_x if phase == "final" else features[development]
        targets = final_y if phase == "final" else labels[development]
        evaluation_ids = np.asarray([f"cifar-test-{i}" for i in range(len(final_y))] if phase == "final"
                                    else [f"cifar-train-{i}" for i in development])
        for directory, values in ((public / "train", {"ids": ids, "x": features[rows], "y": labels[rows]}),
                                  (public / "evaluation", {"ids": evaluation_ids, "x": evaluation}),
                                  (private / "labels", {"ids": evaluation_ids, "y": targets})):
            directory.mkdir()
            for name, value in values.items():
                np.save(directory / f"{name}.npy", value, allow_pickle=False)
    return {"dataset_origin": "https://www.cs.toronto.edu/~kriz/cifar.html",
            "raw_format": "official-cifar10-binary", "split_policy": "stratified-training-development; original-test",
            "normalization": "float32 NCHW: uint8 / 127.5 - 1", "split_seed": seed,
            "final_training_policy": "Restore all original training examples only after method freeze; original test labels remain protected",
            "original_training_rows": len(labels), "original_final_rows": len(final_y)}


def _write_jsonl(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _imdb(source, destination, seed):
    root = source / "aclImdb" if (source / "aclImdb").is_dir() else source

    def read(split):
        rows = []
        for label, name in enumerate(("neg", "pos")):
            directory = root / split / name
            if not directory.is_dir():
                raise ValueError("IMDb raw format requires aclImdb/{train,test}/{neg,pos} text directories")
            for path in sorted(directory.glob("*.txt")):
                rows.append({"id": f"imdb-{split}-{name}-{path.stem}", "text": path.read_text(), "label": label})
        if not rows:
            raise ValueError("IMDb split is empty")
        return rows

    training, final = read("train"), read("test")
    train, development, screen = _holdouts([row["label"] for row in training], seed)
    for phase, (public, private) in _roots(destination).items():
        rows = screen if phase == "screen" else train
        evaluation = final if phase == "final" else [training[i] for i in development]
        _write_jsonl(public / "train.jsonl", [training[i] for i in rows])
        _write_jsonl(public / "evaluation.jsonl", [{"id": row["id"], "text": row["text"]} for row in evaluation])
        np.savez(private / "labels.npz", ids=np.asarray([row["id"] for row in evaluation]),
                 y=np.asarray([row["label"] for row in evaluation], dtype=np.int64))
    return {"dataset_origin": "https://ai.stanford.edu/~amaas/data/sentiment/",
            "raw_format": "official-aclImdb-text", "split_policy": "stratified-training-development; original-test",
            "split_seed": seed, "unsupervised_training_reviews": "excluded", "model_asset": "explicit-local-distilbert",
            "original_training_rows": len(training), "original_final_rows": len(final)}


def _ogb(source, destination, seed):
    root = next((source / name for name in ("ogbn_arxiv", "arxiv") if (source / name).is_dir()), source)
    raw, split = root / "raw", root / "split/time"
    names = [raw / name for name in ("node-feat.csv.gz", "node-label.csv.gz", "edge.csv.gz")]
    names += [split / f"{name}.csv.gz" for name in ("train", "valid", "test")]
    if not all(path.is_file() for path in names):
        raise ValueError("OGB raw format requires raw/node-feat,node-label,edge gzip CSVs and official split/time/{train,valid,test} masks")
    x = pd.read_csv(names[0], header=None).to_numpy(dtype=np.float32)
    y = pd.read_csv(names[1], header=None).to_numpy(dtype=np.int64).ravel()
    edges = pd.read_csv(names[2], header=None).to_numpy(dtype=np.int64).T
    if len(x) != len(y) or edges.shape[0] != 2 or np.any(edges < 0) or np.any(edges >= len(x)):
        raise ValueError("OGB raw graph dimensions or edge IDs are invalid")
    masks = {name: pd.read_csv(split / f"{name}.csv.gz", header=None).to_numpy(dtype=np.int64).ravel()
             for name in ("train", "valid", "test")}
    for name, values in masks.items():
        if not len(values) or len(set(values)) != len(values) or np.any(values < 0) or np.any(values >= len(x)):
            raise ValueError(f"Invalid official OGB {name} mask")
    if any(set(masks[a]) & set(masks[b]) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("OGB official masks overlap")
    screen, _ = train_test_split(masks["train"], test_size=0.5, random_state=seed, stratify=y[masks["train"]])
    ids = np.arange(len(x)).astype(str)
    for phase, (public, private) in _roots(destination).items():
        rows = np.sort(screen) if phase == "screen" else masks["train"]
        evaluation = masks["test" if phase == "final" else "valid"]
        # Public all-node features/edges are allowed transductive data. Only
        # permitted training-node labels appear in the public archive.
        np.savez(public / "train.npz", ids=ids, x=x, y=y[rows], edge_index=edges, train_indices=rows)
        np.savez(public / "evaluation.npz", ids=ids[evaluation], x=x[evaluation])
        np.savez(private / "labels.npz", ids=ids[evaluation], y=y[evaluation])
    return {"dataset_origin": "https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv",
            "raw_format": "official-ogb-gzip-csv", "split_policy": "official_ogb_masks",
            "split_seed": seed, "transductive_public_features": True,
            "nodes": len(x), "edges": edges.shape[1], "features": x.shape[1], "classes": len(np.unique(y)),
            "mask_files": {name: f"split/time/{name}.csv.gz" for name in masks}}


def _babylm(source, destination, seed):
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ValueError("BabyLM raw preparation requires the optional tokenizers package; no ML framework or network is used") from error
    root = source / "babylm_data" if (source / "babylm_data").is_dir() else source
    tokenizer_path = source / "assets/tokenizer/tokenizer.json"
    if not tokenizer_path.is_file():
        raise ValueError("BabyLM raw preparation requires a frozen local assets/tokenizer/tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    config_path = source / "assets/model_config/config.json"
    if not config_path.is_file():
        raise ValueError("BabyLM raw preparation requires a scratch GPT2 model config")
    config = json.loads(config_path.read_text())
    pad_id = config.get("eos_token_id")
    if not isinstance(pad_id, int) or not 0 <= pad_id < tokenizer.get_vocab_size():
        raise ValueError("Model config must declare a valid tokenizer EOS ID for padding")

    def read(directory, split):
        files = sorted(directory.glob(f"*.{split}"))
        if not files:
            raise ValueError(f"BabyLM raw corpus requires {directory.name}/*.{split}")
        rows = []
        for path in files:
            with path.open() as stream:
                for index, text in enumerate(stream):
                    if text.strip():
                        rows.append({"id": f"babylm-{split}-{path.stem}-{index}", "text": text.rstrip("\n")})
        return rows

    training = read(root / "babylm_10M", "train")
    development, final = read(root / "babylm_dev", "dev"), read(root / "babylm_test", "test")
    if sum(len(row["text"].split()) for row in training) > 10_000_000:
        raise ValueError("BabyLM-10M raw training corpus exceeds the declared word budget")
    indices = np.arange(len(training))
    rng = np.random.default_rng(seed)
    screen = np.sort(rng.choice(indices, size=max(1, len(indices) // 2), replace=False))
    documents = {"train": [row["id"] for row in training], "dev": [row["id"] for row in development],
                 "test": [row["id"] for row in final]}
    evaluation_counts = {}
    for phase, (public, private) in _roots(destination).items():
        _write_jsonl(public / "train.jsonl", [training[i] for i in screen] if phase == "screen" else training)
        evaluation = final if phase == "final" else development
        ids, doc_ids, inputs, targets, masks = [], [], [], [], []
        unscorable = 0
        for row in evaluation:
            tokens = tokenizer.encode(row["text"], add_special_tokens=False).ids
            unscorable += len(tokens) < 2
            for start in range(0, max(0, len(tokens) - 1), 128):
                block = tokens[start:start + 129]
                count = len(block) - 1
                if count <= 0:
                    continue
                ids.append(f"{row['id']}:token-{start}")
                doc_ids.append(row["id"])
                inputs.append(block[:-1] + [pad_id] * (128 - count))
                targets.append(block[1:] + [pad_id] * (128 - count))
                masks.append([1] * count + [0] * (128 - count))
        if not ids:
            raise ValueError("BabyLM evaluation corpus has no next-token examples")
        np.savez(public / "evaluation.npz", ids=np.asarray(ids), input_ids=np.asarray(inputs, dtype=np.int32),
                 attention_mask=np.asarray(masks, dtype=np.uint8))
        np.savez(private / "labels.npz", ids=np.asarray(ids), target_ids=np.asarray(targets, dtype=np.int32),
                 attention_mask=np.asarray(masks, dtype=np.uint8), doc_ids=np.asarray(doc_ids))
        evaluation_counts[phase] = {"source_documents": len(evaluation), "evaluated_documents": len(set(doc_ids)),
                                    "unscorable_documents": unscorable, "chunks": len(ids),
                                    "scored_next_tokens": sum(sum(mask) for mask in masks)}
    return {"dataset_origin": "https://babylm.github.io/", "raw_format": "official-babylm-line-corpus",
            "split_policy": "official-document-splits", "split_seed": seed,
            "document_split_manifest": {name: {"count": len(ids), "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
                                        for name, ids in documents.items()},
            "evaluation_counts": evaluation_counts,
            "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            "tokenization": {"sequence_length": 128, "cross_document_chunks": False, "padding": "masked",
                             "target_alignment": "next-token", "truncation": False}}


def _record_arrays(path, records, loader, *, include_labels):
    """Write record-sized arrays through memmaps; never accumulate full image data."""
    if not records:
        raise ValueError("An image partition cannot be empty")
    path.mkdir(parents=True)
    np.save(path / "ids.npy", np.asarray([item[0] for item in records]), allow_pickle=False)
    arrays = {}
    for index, record in enumerate(records):
        features, target = loader(record)
        values = {"x": np.asarray(features, dtype=np.float32)}
        if include_labels:
            values["y"] = np.asarray(target, dtype=np.uint8)
        for key, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite image or mask data")
            if index == 0:
                arrays[key] = np.lib.format.open_memmap(path / f"{key}.npy", mode="w+",
                                                       dtype=value.dtype, shape=(len(records), *value.shape))
            if value.shape != arrays[key].shape[1:]:
                raise ValueError("Image/mask shapes differ inside a fixed task partition")
            arrays[key][index] = value
    for value in arrays.values():
        value.flush()


def _label_arrays(path, records, loader):
    path.mkdir(parents=True)
    np.save(path / "ids.npy", np.asarray([item[0] for item in records]), allow_pickle=False)
    output = None
    for index, record in enumerate(records):
        _, target = loader(record)
        target = np.asarray(target, dtype=np.uint8)
        if output is None:
            output = np.lib.format.open_memmap(path / "y.npy", mode="w+", dtype=target.dtype,
                                               shape=(len(records), *target.shape))
        if target.shape != output.shape[1:]:
            raise ValueError("Private target shapes differ inside a fixed task partition")
        output[index] = target
    if output is None:
        raise ValueError("Protected target partition cannot be empty")
    output.flush()


def _image_partitions(destination, training, final, loader, seed):
    if set(item[0] for item in training) & set(item[0] for item in final):
        raise ValueError("Image IDs overlap official training and final records")
    train_indices, development_indices = train_test_split(np.arange(len(training)), test_size=0.2, random_state=seed)
    screen_indices, _ = train_test_split(train_indices, test_size=0.5, random_state=seed + 1)
    for phase, (public, private) in _roots(destination).items():
        indices = screen_indices if phase == "screen" else train_indices
        train = [training[i] for i in sorted(indices)]
        evaluation = final if phase == "final" else [training[i] for i in sorted(development_indices)]
        _record_arrays(public / "train", train, loader, include_labels=True)
        _record_arrays(public / "evaluation", evaluation, loader, include_labels=False)
        _label_arrays(private / "labels", evaluation, loader)


def _contrails(source, destination, seed):
    def records(name):
        root = source / name
        if not root.is_dir():
            raise ValueError("Contrails raw conversion requires labeled train/ and validation/ record directories")
        found = [(path.name, path) for path in sorted(root.iterdir()) if path.is_dir()]
        if not found:
            raise ValueError("Contrails record partition is empty")
        return found

    def load(record):
        _, path = record
        bands = []
        for number in (11, 14, 15):
            values = np.load(path / f"band_{number}.npy", allow_pickle=False, mmap_mode="r")
            if values.ndim != 3 or values.shape[-1] <= 4:
                raise ValueError("Contrails thermal bands must contain time index four")
            bands.append(np.asarray(values[..., 4], dtype=np.float32))
        band11, band14, band15 = bands
        # The exact upstream false-color transformation at the annotated frame.
        features = np.clip(np.stack([(band15 - band14 + 4) / 6,
                                     (band14 - band11 + 4) / 9, (band14 - 243) / 60]), 0, 1)
        mask = np.load(path / "human_pixel_masks.npy", allow_pickle=False)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.shape != features.shape[1:] or not np.isin(mask, [0, 1]).all():
            raise ValueError("Contrails masks must be binary and match the thermal record shape")
        return features, mask[None]

    training, final = records("train"), records("validation")
    _image_partitions(destination, training, final, load, seed)
    return {"dataset_origin": "https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming",
            "raw_format": "official-contrails-npy-records", "split_policy": "training-derived-development; original-validation-as-final",
            "split_seed": seed, "image_storage": "mmap-directory-arrays", "frame": 4,
            "channels": "upstream false-color band15-band14, band14-band11, band14"}


def _fathomnet(source, destination, seed):
    try:
        from PIL import Image
    except ImportError as error:
        raise ValueError("FathomNet raw preparation requires the optional Pillow image decoder") from error
    path = source / "multilabel_classification/train.csv"
    if not path.is_file():
        raise ValueError("FathomNet raw format requires multilabel_classification/train.csv and images/")
    frame = pd.read_csv(path, dtype={"id": str})
    if not {"id", "categories"} <= set(frame) or frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("FathomNet needs unique image IDs and actual category ground truth")
    parsed = []
    for value in frame.categories:
        try:
            labels = ast.literal_eval(value) if isinstance(value, str) else value
        except (ValueError, SyntaxError) as error:
            raise ValueError("FathomNet categories must be a literal list of category IDs") from error
        if not isinstance(labels, list) or not labels or any(not isinstance(label, (str, int)) for label in labels):
            raise ValueError("Every FathomNet image requires genuine category IDs")
        parsed.append([int(label) for label in labels])
    category_key = source / "category_key.csv"
    if not category_key.is_file():
        raise ValueError("FathomNet requires the full category_key.csv to freeze category ordering")
    classes = len(pd.read_csv(category_key))
    if classes < 1 or any(min(labels) < 1 or max(labels) > classes for labels in parsed):
        raise ValueError("FathomNet labels fall outside the declared category key")
    records = []
    for identifier, labels in zip(frame.id, parsed):
        if Path(identifier).name != identifier or identifier in {".", ".."}:
            raise ValueError("FathomNet image IDs cannot contain path components")
        records.append((identifier, source / "images" / f"{identifier}.png", labels))
    # Match the pinned upstream prepare.py exactly; only development is new.
    order = pd.Series(np.arange(len(records))).sample(frac=1, random_state=42).to_numpy()
    boundary = int(len(order) * 0.98)
    train, final = order[:boundary], order[boundary:]

    def load(record):
        _, path, labels = record
        with Image.open(path) as image:
            features = np.asarray(image.convert("RGB").resize((256, 256), resample=Image.Resampling.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255
        target = np.zeros(classes, dtype=np.uint8)
        target[np.asarray(labels) - 1] = 1
        return features, target

    _image_partitions(destination, [records[i] for i in sorted(train)], [records[i] for i in sorted(final)], load, seed + 1)
    return {"dataset_origin": "https://www.kaggle.com/competitions/fathomnet-out-of-sample-detection",
            "raw_format": "official-fathomnet-images-and-categories", "split_policy": "upstream-seed42-shuffle-98/2-final; disjoint-image-development",
            "split_seed": seed, "classes": classes, "category_ids": "one-based category_key.csv row order",
            "image_storage": "mmap-directory-arrays", "image_transform": "RGB bilinear resize256 float32 /255",
            "osd": "unavailable; no random or synthetic OSD labels"}


def convert_raw(task_id, source, destination, *, split_seed=1729):
    source, destination = Path(source), Path(destination)
    if task_id not in RAW_TASKS:
        raise ValueError("This task requires --source-format bundle with its declared raw preparation procedure")
    if destination.exists():
        raise ValueError("Raw conversion destination must be new")
    tree_manifest(source)
    destination.mkdir(parents=True)
    converters = {"cifar10": _cifar, "imdb": _imdb, "ogbn-arxiv": _ogb, "babylm": _babylm,
                  "identify-contrails": _contrails, "fathomnet": _fathomnet}
    metadata = converters[task_id](source, destination, split_seed)
    _copy_assets(source, destination)
    (destination / "split_metadata.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    return metadata
