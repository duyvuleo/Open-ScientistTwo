"""Trusted hyperparameter boundary; candidates cannot change data paths or caps."""
import json
import math
import os


def load_configuration(base_path, candidate_directory, task):
    config = json.loads(base_path.read_text())
    path = candidate_directory / "training_config.json"
    proposed = json.loads(path.read_text()) if path.is_file() else {}
    allowed = {"epochs", "batch_size", "learning_rate", "optimizer", "weight_decay", "momentum",
               "dropout", "hidden_size", "sequence_length", "inference_batch_size", "gradient_clip_norm"}
    if task == "CLRS":
        allowed = {"learning_rate", "hidden_size", "dropout", "gradient_clip_norm"}
    if set(proposed) - allowed:
        raise ValueError("Candidate config changes a protected field or undeclared hyperparameter")
    for key, value in proposed.items():
        if key == "optimizer":
            if value not in {"sgd", "adam", "adamw"}:
                raise ValueError("Unsupported declared optimizer")
        elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("Hyperparameters must be finite numbers")
    config.update(proposed)
    config["seed"] = int(os.environ.get("SCIENTISTTWO_SEED", config["seed"]))
    integers = {"epochs": (1, config["max_epochs"]), "batch_size": (1, 4096),
                "hidden_size": (1, 8192), "sequence_length": (2, 1024),
                "inference_batch_size": (1, 128)}
    for name, (minimum, maximum) in integers.items():
        if name in config and (not isinstance(config[name], int) or not minimum <= config[name] <= maximum):
            raise ValueError("Candidate exceeds the declared integer hyperparameter range: " + name)
    ranges = {"learning_rate": (1e-8, 10.0), "weight_decay": (0.0, 1.0),
              "momentum": (0.0, 0.9999), "dropout": (0.0, 0.99), "gradient_clip_norm": (0.0, 1000.0)}
    for name, (minimum, maximum) in ranges.items():
        if name in config and not minimum <= config[name] <= maximum:
            raise ValueError("Candidate exceeds the declared hyperparameter range: " + name)
    if task in {"cifar10", "ogbn-arxiv"} and config["epochs"] > 10:
        raise ValueError("Upstream task limits training to ten epochs")
    if task == "imdb" and config.get("sequence_length", 256) > 512:
        raise ValueError("DistilBERT maximum context length is 512")
    return config
