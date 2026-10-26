"""Host-safe prepared-input checks; never import a candidate or ML framework."""
import json
from pathlib import Path

import numpy as np

from .array_data import all_finite, archive_path, open_arrays
from .clrs_inputs import FLOYD_SPEC, validate_trajectory


def _ids(values):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values):
        raise ValueError("Example IDs must be a nonempty vector")
    output = values.astype(str).tolist()
    if len(set(output)) != len(output):
        raise ValueError("Duplicate example IDs")
    return output


def _jsonl(path, keys):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(set(row) != keys for row in rows):
        raise ValueError("Unexpected text fields or missing examples")
    ids = _ids([row["id"] for row in rows])
    if any(not isinstance(row["text"], str) for row in rows):
        raise ValueError("Text must be a string")
    return ids


def _tokenizer_vocabulary_size(directory):
    path = directory / "tokenizer.json"
    if path.is_file():
        tokenizer = json.loads(path.read_text())
        vocabulary = tokenizer.get("model", {}).get("vocab")
        ids = list(vocabulary.values()) if isinstance(vocabulary, dict) else list(range(len(vocabulary or [])))
        ids += [token["id"] for token in tokenizer.get("added_tokens", [])]
    elif (directory / "vocab.json").is_file():
        ids = list(json.loads((directory / "vocab.json").read_text()).values())
        added = directory / "added_tokens.json"
        if added.is_file():
            ids += list(json.loads(added.read_text()).values())
    else:
        raise ValueError("A locally inspectable frozen tokenizer vocabulary is required")
    if not ids or any(not isinstance(value, int) or value < 0 for value in ids):
        raise ValueError("Invalid frozen tokenizer vocabulary IDs")
    return max(ids) + 1


def _validate_cifar_final_refit(root):
    """Verify refit content, not just IDs, using bounded final-data batches."""
    with open_arrays(archive_path(root / "phases/full/public", "train")) as archive:
        train = {key: archive[key] for key in archive.files}
    with open_arrays(archive_path(root / "phases/full/public", "evaluation")) as archive:
        development = {key: archive[key] for key in archive.files}
    with open_arrays(archive_path(root / "phases/full/private", "labels")) as archive:
        targets = {key: archive[key] for key in archive.files}
    train_lookup = {str(value): index for index, value in enumerate(train["ids"])}
    dev_lookup = {str(value): index for index, value in enumerate(development["ids"])}
    label_lookup = {str(value): index for index, value in enumerate(targets["ids"])}
    with open_arrays(archive_path(root / "phases/final/public", "train")) as final:
        final_ids, final_x, final_y = final["ids"].astype(str), final["x"], final["y"]
        for offset in range(0, len(final_ids), 32):
            for row, identifier in enumerate(final_ids[offset:offset + 32], start=offset):
                if identifier in train_lookup:
                    expected_x, expected_y = train["x"][train_lookup[identifier]], train["y"][train_lookup[identifier]]
                else:
                    expected_x, expected_y = development["x"][dev_lookup[identifier]], targets["y"][label_lookup[identifier]]
                if not np.array_equal(final_x[row], expected_x) or not np.array_equal(final_y[row], expected_y):
                    raise ValueError("CIFAR final refit changed the original training/development row content")


def validate_prepared(family, root, metadata):
    if not metadata.get("dataset_origin"):
        raise ValueError("Prepared framework data requires dataset_origin provenance")
    if family.task_id == "llama-inference":
        identity = metadata.get("model_identity", "")
        if identity != "original-llama-7b" and not identity.startswith("adapted-"):
            raise ValueError("Llama requires the original or an explicitly named adapted 7B model")
        model_config = root / "assets/model/config.json"
        if not model_config.is_file():
            raise ValueError("Model architecture metadata missing")
        config = json.loads(model_config.read_text())
        # The original 7B Llama shape; alternate architectures need a separately reviewed variant.
        if config.get("hidden_size") != 4096 or config.get("num_hidden_layers") != 32:
            raise ValueError("Expected declared Llama-compatible 7B architecture")
    for asset in family.required_assets:
        location = root / "assets" / asset
        if not location.is_dir() or not any(path.is_file() for path in location.rglob("*")):
            raise ValueError(f"Missing required prepared asset directory: {asset}")
    if family.task_id == "babylm":
        if not (root / "assets/tokenizer/tokenizer.json").is_file():
            raise ValueError("BabyLM requires frozen tokenizer.json shared by preparation and training")
        forbidden = {".pt", ".bin", ".safetensors", ".h5", ".ckpt", ".onnx"}
        if any(path.suffix in forbidden for path in (root / "assets").rglob("*")):
            raise ValueError("BabyLM training assets must not contain pretrained weights")
        config = json.loads((root / "assets/model_config/config.json").read_text())
        if config.get("model_type") != "gpt2" or config.get("auto_map"):
            raise ValueError("Audited BabyLM v1 uses trusted GPT2 architecture only")
        if _tokenizer_vocabulary_size(root / "assets/tokenizer") != config.get("vocab_size"):
            raise ValueError("BabyLM model vocabulary differs from the frozen tokenizer")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Prepared bundles must not contain symlinks")

    train_ids, evaluation_ids, prediction_bytes, word_counts = {}, {}, {}, {}
    document_ids = {}
    for phase in ["screen", "full", "final"]:
        public = root / "phases" / phase / "public"
        private = root / "phases" / phase / "private"
        for name in family.training_files + family.evaluation_files:
            location = archive_path(public, name.removesuffix(".npz")) if name.endswith(".npz") else public / name
            if not location.exists():
                raise ValueError(f"Missing {phase} public input: {name}")
        for name in family.private_files:
            location = archive_path(private, name.removesuffix(".npz")) if name.endswith(".npz") else private / name
            if not location.exists():
                raise ValueError(f"Missing {phase} protected labels: {name}")
        permitted = set(family.training_files + family.evaluation_files)
        for name in list(permitted):
            if name.endswith(".npz") and (public / name.removesuffix(".npz")).is_dir():
                permitted.remove(name)
                permitted |= {path.relative_to(public).as_posix() for path in (public / name.removesuffix(".npz")).iterdir()}
        if family.task_id == "CLRS":
            permitted |= {item["path"] for item in json.loads((public / "train_manifest.json").read_text()).get("shards", [])}
        observed = {path.relative_to(public).as_posix() for path in public.rglob("*") if path.is_file()}
        if observed != permitted:
            raise ValueError("Public bundle contains undeclared files; label isolation requires an exact input allowlist")
        with open_arrays(archive_path(private, "labels")) as labels:
            label_ids = _ids(labels["ids"])
            if family.task_id == "imdb":
                training = _jsonl(public / "train.jsonl", {"id", "text", "label"})
                evaluation = _jsonl(public / "evaluation.jsonl", {"id", "text"})
            elif family.task_id == "CLRS":
                if json.loads((public / "spec.json").read_text()) != FLOYD_SPEC:
                    raise ValueError("CLRS spec differs from the pinned Floyd-Warshall contract")
                manifest = json.loads((public / "train_manifest.json").read_text())
                if set(manifest) != {"algorithm", "train_lengths", "shards"}:
                    raise ValueError("CLRS training manifest contains undeclared fields")
                if manifest.get("algorithm") != "floyd_warshall" or set(manifest.get("train_lengths", [])) != {4, 7, 11, 13, 16}:
                    raise ValueError("CLRS must preserve all declared training lengths")
                training = []
                for shard in manifest.get("shards", []):
                    if set(shard) != {"path", "length"}:
                        raise ValueError("CLRS shard metadata contains undeclared fields")
                    relative = Path(shard["path"])
                    if relative.is_absolute() or ".." in relative.parts or shard["length"] not in manifest["train_lengths"]:
                        raise ValueError("Invalid CLRS training shard")
                    with np.load(public / relative, allow_pickle=False) as values:
                        validate_trajectory(values, nodes=shard["length"], supervised=True)
                        training += _ids(values["ids"])
                if {shard["length"] for shard in manifest.get("shards", [])} != {4, 7, 11, 13, 16}:
                    raise ValueError("CLRS training shards omit a declared graph size")
                _ids(training)
                evaluation_manifest = json.loads((public / "evaluation_manifest.json").read_text())
                if set(evaluation_manifest) - {"phase", "graph_length", "zero_hint_shapes"}:
                    raise ValueError("CLRS public evaluation metadata contains undeclared fields")
                if evaluation_manifest.get("phase") != phase or evaluation_manifest.get("graph_length") != (64 if phase == "final" else 16):
                    raise ValueError("CLRS evaluation length/phase mismatch")
                with np.load(public / "evaluation.npz", allow_pickle=False) as values:
                    validate_trajectory(values, nodes=64 if phase == "final" else 16, supervised=False)
                    if any(key not in {"ids", "lengths"} and not key.startswith("input:") for key in values.files):
                        raise ValueError("Protected CLRS hints/outputs must not appear in public inputs")
                    evaluation = _ids(values["ids"])
                    if "zero_hint_shapes" in evaluation_manifest:
                        nodes = evaluation_manifest["graph_length"]
                        expected_shapes = {name: {"name": name, "location": location, "type": kind,
                                                   "shape": [nodes, len(evaluation), nodes, nodes] if location == "edge" else [nodes, len(evaluation), nodes]}
                                           for name, (stage, location, kind) in FLOYD_SPEC.items() if stage == "hint"}
                        observed_shapes = {item["name"]: item for item in evaluation_manifest["zero_hint_shapes"]}
                        if len(evaluation_manifest["zero_hint_shapes"]) != 4 or observed_shapes != expected_shapes:
                            raise ValueError("CLRS hint shape metadata differs from the input-computable rollout")
                    expected = (len(evaluation), 64 if phase == "final" else 16, 64 if phase == "final" else 16)
                    if labels["y"].shape != expected or np.any(labels["y"] != labels["y"].astype(int)) or np.any(labels["y"] < 0) or np.any(labels["y"] >= expected[-1]):
                        raise ValueError("CLRS protected targets do not match the actual OOD graph size")
            else:
                if family.task_id == "babylm":
                    training = _jsonl(public / "train.jsonl", {"id", "text"})
                    with (public / "train.jsonl").open() as handle:
                        word_counts[phase] = sum(len(json.loads(line)["text"].split()) for line in handle if line.strip())
                    if word_counts[phase] > 10000000:
                        raise ValueError("BabyLM-10M training word budget exceeded")
                elif family.task_id == "llama-inference":
                    training = []
                else:
                    with open_arrays(archive_path(public, "train")) as values:
                        allowed = {"ids", "x", "y"}
                        if family.task_id == "ogbn-arxiv":
                            allowed |= {"edge_index", "train_indices"}
                        if set(values.files) != allowed:
                            raise ValueError("Training archive differs from the declared contract")
                        training = _ids(values["ids"])
                        if len(values["x"]) != len(training):
                            raise ValueError("Training rows/IDs differ")
                        if family.task_id == "ogbn-arxiv":
                            indices = values["train_indices"]
                            if indices.ndim != 1 or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(training)) or len(set(indices)) != len(indices):
                                raise ValueError("Invalid official training node indices")
                            edges = values["edge_index"]
                            if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu" or np.any(edges < 0) or np.any(edges >= len(training)):
                                raise ValueError("Graph edges must reference valid global node IDs")
                            all_node_ids = training
                            all_graph_features = values["x"]
                            training = [training[index] for index in indices]
                        if len(values["y"]) != len(training):
                            raise ValueError("Training labels must cover only training nodes/rows")
                        if not all_finite(values["x"]) or not all_finite(values["y"]):
                            raise ValueError("Non-finite training data")
                        if family.task_id == "identify-contrails":
                            if values["y"].ndim not in {3, 4} or (values["y"].ndim == 4 and values["y"].shape[1] != 1):
                                raise ValueError("Contrails training masks require one binary channel")
                with open_arrays(archive_path(public, "evaluation")) as values:
                    allowed = {"ids", "input_ids", "attention_mask"} if family.task_id in {"babylm", "llama-inference"} else {"ids", "x"}
                    if set(values.files) != allowed:
                        raise ValueError("Public evaluation inputs contain undeclared or protected label fields")
                    evaluation = _ids(values["ids"])
                    if any(len(values[key]) != len(evaluation) for key in allowed - {"ids"}):
                        raise ValueError("Evaluation row/ID mismatch")
                    if family.task_id == "ogbn-arxiv":
                        lookup = {node_id: index for index, node_id in enumerate(all_node_ids)}
                        if any(node_id not in lookup for node_id in evaluation):
                            raise ValueError("Evaluation nodes are absent from the public transductive graph")
                        for offset in range(0, len(evaluation), 32):
                            selected = [lookup[node_id] for node_id in evaluation[offset:offset + 32]]
                            if not np.array_equal(values["x"][offset:offset + 32], all_graph_features[selected]):
                                raise ValueError("Evaluation features differ from their global graph node features")
                    if family.task_id == "babylm":
                        shape = values["input_ids"].shape
                        if len(shape) != 2 or labels["target_ids"].shape != shape or labels["attention_mask"].shape != shape:
                            raise ValueError("Language-model input/target/mask shapes differ")
                        vocabulary = int(config["vocab_size"])
                        if "doc_ids" not in labels.files or len(labels["doc_ids"]) != len(evaluation):
                            raise ValueError("BabyLM protected labels require document IDs for complete token accounting")
                        document_ids[phase] = set(labels["doc_ids"].astype(str))
                        if document_ids[phase] & set(training):
                            raise ValueError("BabyLM training and evaluation documents overlap")
                        if values["input_ids"].dtype.kind not in "iu" or labels["target_ids"].dtype.kind not in "iu":
                            raise ValueError("BabyLM token IDs must be integers")
                        if not np.isin(labels["attention_mask"], [0, 1]).all() or not np.array_equal(values["attention_mask"], labels["attention_mask"]):
                            raise ValueError("BabyLM input and protected masks differ")
                        target_mask = labels["attention_mask"].astype(bool)
                        if np.any(values["input_ids"] < 0) or np.any(values["input_ids"] >= vocabulary) or np.any(labels["target_ids"][target_mask] < 0) or np.any(labels["target_ids"][target_mask] >= vocabulary):
                            raise ValueError("BabyLM token ID lies outside the frozen vocabulary")
                        forecast = int(np.prod(shape)) * vocabulary * 4
                        prediction_bytes[phase] = forecast
                        if forecast > family.artifacts("infer")[0].max_bytes:
                            raise ValueError("Full language-model logits exceed the declared output quota; allocate an explicit larger protocol or change the trusted streaming interface, never truncate evaluation")
            if set(evaluation) != set(label_ids):
                raise ValueError("Protected labels do not exactly match evaluation IDs")
            if set(training) & set(evaluation):
                raise ValueError("Training/evaluation IDs overlap")
            train_ids[phase], evaluation_ids[phase] = set(training), set(evaluation)
    if not train_ids["screen"] <= train_ids["full"]:
        raise ValueError("Screening training rows must be a subset of full training rows")
    expected_final_training = train_ids["full"] | evaluation_ids["full"] if family.task_id == "cifar10" else train_ids["full"]
    if train_ids["final"] != expected_final_training:
        raise ValueError("Final training partition differs from the declared task policy (CIFAR refits on all original training rows only after freezing)")
    if family.task_id == "cifar10":
        _validate_cifar_final_refit(root)
    if evaluation_ids["screen"] != evaluation_ids["full"]:
        raise ValueError("Screening and full development must use the same frozen evaluation partition")
    if evaluation_ids["final"] & (evaluation_ids["screen"] | evaluation_ids["full"]):
        raise ValueError("Final evaluation IDs overlap development IDs")
    if evaluation_ids["final"] & set.union(*train_ids.values()):
        raise ValueError("Final evaluation IDs appear in training")
    if family.task_id == "babylm" and not metadata.get("document_split_manifest"):
        raise ValueError("BabyLM requires a documented train/development/final document split manifest")
    if family.task_id == "babylm" and (document_ids["final"] & document_ids["full"] or document_ids["final"] & set.union(*train_ids.values())):
        raise ValueError("BabyLM final documents leaked into training or development")
    if family.task_id == "ogbn-arxiv" and metadata.get("split_policy") != "official_ogb_masks":
        raise ValueError("OGBN-Arxiv requires the official mask provenance")
    return {"status": "pass", "scope": "prepared_input_contract_only",
            "runtime_validation": "unverified", "actual_task_execution": "unverified",
            "predicted_output_bytes": prediction_bytes,
            "training_word_counts": word_counts,
            "counts": {phase: {"training": len(train_ids[phase]), "evaluation": len(evaluation_ids[phase])}
                       for phase in train_ids}}
