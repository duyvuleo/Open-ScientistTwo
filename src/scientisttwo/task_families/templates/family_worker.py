"""Frozen training/inference entry point, executed only in an isolated container.

Training receives training labels only. Inference receives inputs and checkpoint,
never protected labels. Scoring is performed in a separate NumPy-only container.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time


def read_jsonl(path, keys):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(set(row) != set(keys) for row in rows):
        raise ValueError("Unexpected JSONL fields; evaluation data must not contain labels")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate example IDs")
    return rows


def configure(config):
    import numpy as np
    import torch
    seed = int(config["seed"])
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("Invalid seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("GPU profile requires CUDA; no silent CPU fallback")
    if not 1 <= int(config["epochs"]) <= int(config["max_epochs"]):
        raise ValueError("Training epoch budget exceeded")
    if not 1 <= int(config["max_steps"]) <= 1000000:
        raise ValueError("Invalid frozen step budget")
    if not 1 <= int(config.get("batch_size", 32)) <= 4096:
        raise ValueError("Invalid batch size")
    return torch.device("cuda:0")


def load_model(task, config, assets, source=None):
    source = source or Path(__file__).with_name("model.py")
    spec = importlib.util.spec_from_file_location("candidate_model", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build_model(task, copy.deepcopy(config), assets)
    model._scientisttwo_predict_batch = getattr(module, "predict_batch", None)
    return model


def predict_batch(model, task, inputs, config, context=None):
    hook = getattr(model, "_scientisttwo_predict_batch", None)
    return hook(model, task, inputs, config=copy.deepcopy(config), context=context or {}) if hook else model(inputs)


def graph_context(args, device):
    import numpy as np
    import torch
    from array_data import archive_path, open_arrays
    with open_arrays(archive_path(args.data, "train")) as archive:
        return {"graph_x": torch.as_tensor(np.asarray(archive["x"], dtype=np.float32), device=device),
                "edge_index": torch.as_tensor(np.asarray(archive["edge_index"], dtype=np.int64), device=device),
                "node_ids": archive["ids"].astype(str)}


def transform_batch(args, config, inputs, training):
    source = (args.model_source.parent if args.model_source else Path(__file__).parent) / "transforms.py"
    if not source.is_file():
        return inputs
    module_spec = importlib.util.spec_from_file_location("candidate_transforms", source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    transformed = module.transform_batch(args.task, inputs, training=training, config=copy.deepcopy(config))
    if transformed.shape != inputs.shape:
        raise ValueError("Batch transform must preserve input shape and row order")
    return transformed


def tokenizer_digest(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            if path.is_symlink():
                raise ValueError("Tokenizer symlinks are not allowed")
            digest.update(path.relative_to(directory).as_posix().encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def train(args, config, device):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    from torch import nn
    from array_data import archive_path, open_arrays
    task = args.task
    batch_size = int(config.get("batch_size", 32))
    if task in {"cifar10", "ogbn-arxiv", "identify-contrails", "fathomnet"}:
        with open_arrays(archive_path(args.data, "train")) as data:
            x, y = data["x"], data["y"]
            indices = data["train_indices"] if task == "ogbn-arxiv" else np.arange(len(x))
            if len(indices) != len(y):
                raise ValueError("Training inputs and labels differ in length")
            if task == "ogbn-arxiv":
                config["input_features"] = x.shape[-1]
                config["classes"] = int(config.get("classes", 40))
            elif task == "identify-contrails":
                config["channels"] = x.shape[1]
            elif task == "fathomnet":
                config["classes"] = y.shape[1]
            class LazyArrayRows(Dataset):
                def __len__(self):
                    return len(y)

                def __getitem__(self, index):
                    inputs = torch.from_numpy(np.array(x[indices[index]], dtype=np.float32, copy=True))
                    target = torch.as_tensor(np.array(y[index], copy=True))
                    if task == "identify-contrails" and target.ndim == 2:
                        target = target[None]
                    return (inputs, target, int(indices[index])) if task == "ogbn-arxiv" else (inputs, target)

            loader = DataLoader(LazyArrayRows(), batch_size=batch_size, shuffle=True,
                                num_workers=0, drop_last=False,
                                generator=torch.Generator().manual_seed(config["seed"]))
        criterion = nn.BCEWithLogitsLoss() if task in {"identify-contrails", "fathomnet"} else nn.CrossEntropyLoss()
    elif task == "imdb":
        from transformers import AutoTokenizer
        rows = read_jsonl(args.data / "train.jsonl", ["id", "text", "label"])
        tokenizer = AutoTokenizer.from_pretrained(args.assets / "distilbert", local_files_only=True,
                                                  trust_remote_code=False)
        encoded = tokenizer([row["text"] for row in rows], truncation=True, padding="max_length",
                            max_length=int(config.get("sequence_length", 256)), return_tensors="pt")
        loader = DataLoader(TensorDataset(encoded["input_ids"], encoded["attention_mask"],
                                          torch.tensor([row["label"] for row in rows])),
                            batch_size=batch_size, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(config["seed"]))
    elif task == "babylm":
        from transformers import PreTrainedTokenizerFast
        rows = read_jsonl(args.data / "train.jsonl", ["id", "text"])
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.assets / "tokenizer/tokenizer.json"))
        tokenizer.backend_tokenizer.no_padding()
        tokenizer.backend_tokenizer.no_truncation()
        architecture = json.loads((args.assets / "model_config/config.json").read_text())
        eos_id = architecture["eos_token_id"]
        if not isinstance(eos_id, int) or not 0 <= eos_id < len(tokenizer):
            raise ValueError("Frozen model EOS token is outside the tokenizer vocabulary")
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(eos_id)
        tokenizer.pad_token = tokenizer.eos_token
        config["tokenizer_sha256"] = tokenizer_digest(args.assets / "tokenizer")
        length = int(config.get("sequence_length", 128))
        sequences, masks, targets = [], [], []
        for row in rows:
            tokens = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
            # Document boundaries are preserved; never concatenate train and held-out documents.
            for start in range(0, len(tokens) - 1, length):
                window = tokens[start:start + length + 1]
                count = len(window) - 1
                if not count:
                    continue
                sequences.append(window[:-1] + [tokenizer.pad_token_id] * (length - count))
                targets.append(window[1:] + [-100] * (length - count))
                masks.append([1] * count + [0] * (length - count))
        if not sequences:
            raise ValueError("No trainable language-model sequences")
        loader = DataLoader(TensorDataset(torch.tensor(sequences), torch.tensor(masks),
                                          torch.tensor(targets)), batch_size=batch_size,
                            shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(config["seed"]))
    else:
        raise ValueError("Llama inference does not have a training stage")

    if task in {"cifar10", "ogbn-arxiv"} and config["max_epochs"] > 10:
        raise ValueError("Upstream maximum is ten epochs")
    model = load_model(task, config, args.assets, args.model_source).to(device)
    graph = graph_context(args, device) if task == "ogbn-arxiv" else {}
    if task == "babylm":
        from transformers import GPT2LMHeadModel
        if type(model) is not GPT2LMHeadModel:
            raise ValueError("Audited BabyLM v1 requires the trusted GPT2 causal architecture")
    optimizer_name = config.get("optimizer", "sgd" if task == "cifar10" else "adam")
    optimizer_type = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer_name]
    optimizer_args = {"lr": float(config.get("learning_rate", 0.1 if task == "cifar10" else 0.001)),
                      "weight_decay": float(config.get("weight_decay", 0.0))}
    if optimizer_name == "sgd":
        optimizer_args["momentum"] = float(config.get("momentum", 0.9))
    optimizer = optimizer_type(model.parameters(), **optimizer_args)
    steps = 0
    start = time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        for batch in loader:
            if steps >= config["max_steps"]:
                break
            optimizer.zero_grad()
            if task in {"imdb", "babylm"}:
                input_ids, attention_mask, labels = (value.to(device) for value in batch)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                if task == "imdb":
                    loss = nn.functional.cross_entropy(output.logits, labels.long())
                else:
                    loss = nn.functional.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]),
                                                       labels.reshape(-1), ignore_index=-100)
            else:
                inputs, labels = (value.to(device) for value in batch[:2])
                inputs = transform_batch(args, config, inputs, training=True)
                # BatchNorm cannot train on a final singleton; use its running statistics.
                if len(inputs) == 1:
                    for layer in model.modules():
                        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                            layer.eval()
                context = {**graph, "node_indices": batch[2].to(device)} if task == "ogbn-arxiv" else {}
                output = predict_batch(model, task, inputs, config, context)
                labels = labels.float() if task in {"identify-contrails", "fathomnet"} else labels.long()
                loss = criterion(output, labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            if config.get("gradient_clip_norm", 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            steps += 1
        if steps >= config["max_steps"]:
            break
    torch.cuda.synchronize()
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint / "weights.pt")
    (args.checkpoint / "config.json").write_text(json.dumps(config, sort_keys=True))
    if task == "babylm":
        model.config.to_json_file(args.checkpoint / "architecture.json")
    receipt = {"task": task, "seed": config["seed"], "steps": steps,
               "epochs_started": epoch + 1, "max_steps": config["max_steps"],
               "training_seconds": time.perf_counter() - start,
               "initialization": "scratch" if task == "babylm" else "declared_reference"}
    (args.checkpoint / "training.json").write_text(json.dumps(receipt, sort_keys=True))


def inference(args, config, device, server=False):
    import numpy as np
    import torch
    from array_data import archive_path, open_arrays
    task = args.task
    if task != "llama-inference":
        trained_config = json.loads((args.checkpoint / "config.json").read_text())
        if trained_config["seed"] != config["seed"]:
            raise ValueError("Checkpoint seed mismatch")
        config = trained_config
    if task == "babylm" and tokenizer_digest(args.assets / "tokenizer") != config["tokenizer_sha256"]:
        raise ValueError("Frozen tokenizer changed")
    if task == "babylm":
        # Never import candidate Python during protected perplexity inference.
        # A trusted causal implementation prevents future-token access by a custom forward.
        from transformers import GPT2Config, GPT2LMHeadModel
        architecture = json.loads((args.checkpoint / "architecture.json").read_text())
        if architecture.get("model_type") != "gpt2" or architecture.get("auto_map"):
            raise ValueError("Only the trusted GPT2 causal architecture is supported")
        model = GPT2LMHeadModel(GPT2Config.from_dict(architecture)).to(device).eval()
    else:
        model = load_model(task, config, args.assets, args.model_source).to(device).eval()
    if task != "llama-inference":
        model.load_state_dict(torch.load(args.checkpoint / "weights.pt", map_location=device, weights_only=True))
    args.output.mkdir(parents=True, exist_ok=True)
    batch_size = int(config.get("inference_batch_size", 1 if task == "babylm" else config.get("batch_size", 32)))
    if task == "imdb":
        from transformers import AutoTokenizer
        rows = read_jsonl(args.data / "evaluation.jsonl", ["id", "text"])
        ids = np.array([str(row["id"]) for row in rows])
        tokenizer = AutoTokenizer.from_pretrained(args.assets / "distilbert", local_files_only=True,
                                                  trust_remote_code=False)
        encoded = tokenizer([row["text"] for row in rows], truncation=True, padding="max_length",
                            max_length=int(config.get("sequence_length", 256)), return_tensors="pt")
        arrays = {key: value for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
    else:
        with open_arrays(archive_path(args.data, "evaluation")) as data:
            allowed = {"ids", "input_ids", "attention_mask"} if task in {"babylm", "llama-inference"} else {"ids", "x"}
            if set(data.files) != allowed:
                raise ValueError("Inference data contains undeclared or label fields")
            ids = data["ids"].astype(str)
            arrays = {key: data[key] for key in allowed - {"ids"}}
    graph = graph_context(args, device) if task == "ogbn-arxiv" else {}
    if task == "ogbn-arxiv":
        lookup = {node_id: index for index, node_id in enumerate(graph["node_ids"])}
        graph_indices = np.array([lookup[node_id] for node_id in ids])

    def execute():
        predictions = []
        with torch.inference_mode():
            for offset in range(0, len(ids), batch_size):
                batch = {key: torch.as_tensor(np.array(value[offset:offset + batch_size], copy=True), device=device)
                         for key, value in arrays.items()}
                if task == "llama-inference":
                    result = model.generate(**batch, max_new_tokens=1, do_sample=False,
                                            pad_token_id=config.get("pad_token_id", 0))[:, -1:]
                elif task in {"imdb", "babylm"}:
                    result = model(**batch).logits
                else:
                    context = {**graph, "node_indices": torch.as_tensor(graph_indices[offset:offset + batch_size], device=device)} if task == "ogbn-arxiv" else {}
                    result = predict_batch(model, task, transform_batch(args, config, batch["x"].float(), training=False), config, context)
                    if task in {"identify-contrails", "fathomnet"}:
                        result = torch.sigmoid(result)
                torch.cuda.synchronize()
                output = result.detach().float().cpu().numpy()
                if task == "babylm":
                    np.savez(args.output / f"logits-{offset:08d}.npz", ids=ids[offset:offset + batch_size], logits=output)
                elif task == "llama-inference":
                    predictions.append(output)
                else:
                    np.savez(args.output / f"predictions-{offset:08d}.npz", ids=ids[offset:offset + batch_size], predictions=output)
        if task == "llama-inference":
            np.savez(args.output / "predictions.npz", ids=ids, predictions=np.concatenate(predictions))

    if server:
        # Parent measures request/response time after warm-up; candidate cannot submit a score.
        import sys
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            if set(request) != {"request_id"}:
                raise ValueError("Invalid timed request")
            execute()
            torch.cuda.synchronize()
            print(json.dumps({"request_id": request["request_id"], "completed": True}), flush=True)
    else:
        execute()


def main():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "infer", "serve"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-source", type=Path)
    for name in ["data", "assets", "checkpoint", "output", "config"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    from training_contract import load_configuration
    source_directory = args.model_source.parent if args.model_source else Path(__file__).parent
    config = load_configuration(args.config, source_directory, args.task)
    device = configure(config)
    if args.stage == "train":
        train(args, config, device)
    else:
        inference(args, config, device, server=args.stage == "serve")


if __name__ == "__main__":
    main()
