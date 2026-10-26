"""Study-specific frozen configuration boundary; no framework imports."""

import json
import math
import os

FIXED_CONFIGURATION = {"epochs": 10, "max_epochs": 10, "max_steps": 3910, "batch_size": 128}
ALLOWED_CONFIGURATION = {
    "optimizer", "learning_rate", "weight_decay", "momentum", "gradient_clip_norm",
    "crop_padding", "horizontal_flip_probability",
}


def validate_overrides(proposed):
    if not isinstance(proposed, dict) or set(proposed) - ALLOWED_CONFIGURATION:
        raise ValueError("Candidate changes a protected or unsupported study parameter")
    for key, value in proposed.items():
        if key == "optimizer":
            if not isinstance(value, str) or value not in {"sgd", "adam", "adamw"}:
                raise ValueError("Unsupported declared optimizer")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Study parameters must be finite numeric values")
    ranges = {"learning_rate": (1e-5, 1.0), "weight_decay": (0.0, 0.1),
              "momentum": (0.0, 0.999), "gradient_clip_norm": (0.0, 100.0),
              "horizontal_flip_probability": (0.0, 1.0)}
    for key, (minimum, maximum) in ranges.items():
        if key in proposed and not minimum <= proposed[key] <= maximum:
            raise ValueError("Study parameter out of bounds: " + key)
    if "crop_padding" in proposed and (not isinstance(proposed["crop_padding"], int)
                                       or not 0 <= proposed["crop_padding"] <= 4):
        raise ValueError("crop_padding must be an integer in [0, 4]")
    return dict(proposed)


def load_configuration(base_path, candidate_directory, task):
    if task != "cifar10":
        raise ValueError("The fixed study configuration is for CIFAR-10 only")
    config = json.loads(base_path.read_text())
    if any(type(config.get(key)) is not int or config[key] != value
           for key, value in FIXED_CONFIGURATION.items()):
        raise ValueError("Frozen training epochs, batch size, or step budget changed")
    path = candidate_directory / "training_config.json"
    proposed = json.loads(path.read_text()) if path.is_file() else {}
    config.update(validate_overrides(proposed))
    if not ALLOWED_CONFIGURATION <= set(config):
        raise ValueError("Frozen study configuration omits declared optimization or augmentation values")
    validate_overrides({key: config[key] for key in ALLOWED_CONFIGURATION})
    config["seed"] = int(os.environ.get("SCIENTISTTWO_SEED", config["seed"]))
    if not 0 <= config["seed"] < 2**32:
        raise ValueError("Invalid controller-selected seed")
    return config
