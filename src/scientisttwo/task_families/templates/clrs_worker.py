"""CLRS container bridge using frozen prepared trajectories and upstream helpers.

Only training trajectories contain supervision. Evaluation features carry input
DataPoints and lengths; hint arrays are generated as zeros from declared shapes.
Checkpoint pickle loading happens only in the untrusted inference container.
"""
import argparse
import json
from pathlib import Path
import shutil
import time


def feedback(path, spec, inference=False):
    import clrs
    from clrs._src.samplers import Features, Feedback
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        inputs, hints, outputs = [], [], []
        lengths = archive["lengths"]
        allowed = {"ids", "lengths"}
        for name, (stage, location, kind) in spec.items():
            key = stage + ":" + name
            if inference and stage != "input":
                continue
            allowed.add(key)
            point = clrs.DataPoint(name=name, location=location, type_=kind, data=archive[key])
            {"input": inputs, "hint": hints, "output": outputs}[stage].append(point)
        if set(archive.files) != allowed:
            raise ValueError("CLRS archive contains undeclared fields or evaluation supervision")
        ids = archive["ids"].copy()
    return Feedback(Features(tuple(inputs), tuple(hints), lengths), tuple(outputs)), ids


def make_model(config, checkpoint, spec, dummy):
    # These helpers come from the frozen source overlay; candidate changes stay
    # inside training/inference containers and never enter the trusted scorer.
    from baselines import BaselineModel
    from processors import get_processor_factory
    return BaselineModel(
        spec=[spec], dummy_trajectory=[dummy],
        processor_factory=get_processor_factory("triplet_gmpnn", use_ln=True, nb_triplet_fts=8, nb_heads=1),
        hidden_dim=int(config.get("hidden_size", 128)), encode_hints=True, decode_hints=True,
        encoder_init="xavier_on_scalars", learning_rate=float(config.get("learning_rate", 0.001)),
        checkpoint_path=str(checkpoint), hint_teacher_forcing=0.0, hint_repred_mode="soft",
        dropout_prob=float(config.get("dropout", 0.0)), grad_clip_max_norm=float(config.get("gradient_clip_norm", 1.0)),
    )


def run(args):
    import sys
    if args.model_source:
        sys.path.insert(0, str(args.model_source))
    import jax
    import numpy as np
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("CLRS runtime requires CUDA; no silent CPU fallback")
    from training_contract import load_configuration
    config = load_configuration(args.config, args.model_source or Path(__file__).parent, "CLRS")
    steps = int(config["max_steps"])
    if not 1 <= steps <= 1000000:
        raise ValueError("Invalid frozen CLRS step budget")
    seed = int(config["seed"])
    key = jax.random.PRNGKey(seed)
    if args.stage == "train":
        spec = json.loads((args.data / "spec.json").read_text())
        manifest = json.loads((args.data / "train_manifest.json").read_text())
        if manifest["algorithm"] != "floyd_warshall" or set(manifest["train_lengths"]) != {4, 7, 11, 13, 16}:
            raise ValueError("Declared CLRS training-size protocol changed")
        shards = []
        for shard in manifest["shards"]:
            relative = Path(shard["path"])
            if relative.is_absolute() or ".." in relative.parts or shard["length"] not in manifest["train_lengths"]:
                raise ValueError("Invalid CLRS training shard")
            shards.append(args.data / relative)
        if not shards:
            raise ValueError("No CLRS training trajectories")
        dummy, _ = feedback(shards[0], spec)
        args.checkpoint.mkdir(parents=True, exist_ok=True)
        model = make_model(config, args.checkpoint, spec, dummy)
        model.init([dummy.features], seed + 1)
        start = time.perf_counter()
        for step in range(steps):
            batch, _ = feedback(shards[step % len(shards)], spec)
            key, subkey = jax.random.split(key)
            loss = model.feedback(subkey, batch, algorithm_index=0)
            if not np.isfinite(np.asarray(loss)).all():
                raise ValueError("Non-finite CLRS training loss")
        model.save_model("best.pkl")
        # The final trained checkpoint is explicit: no held-out-score model selection here.
        shutil.copyfile(shards[0], args.checkpoint / "dummy.npz")
        (args.checkpoint / "spec.json").write_text(json.dumps(spec, sort_keys=True))
        (args.checkpoint / "config.json").write_text(json.dumps(config, sort_keys=True))
        (args.checkpoint / "training.json").write_text(json.dumps({
            "seed": seed, "steps": steps, "checkpoint_selection": "last_training_step",
            "training_seconds": time.perf_counter() - start,
        }))
    else:
        spec = json.loads((args.checkpoint / "spec.json").read_text())
        config = json.loads((args.checkpoint / "config.json").read_text())
        dummy, _ = feedback(args.checkpoint / "dummy.npz", spec)
        model = make_model(config, args.checkpoint, spec, dummy)
        model.init([dummy.features], seed + 1)
        model.restore_model("best.pkl", only_load_processor=False)
        evaluation, ids = feedback(args.data / "evaluation.npz", spec, inference=True)
        manifest = json.loads((args.data / "evaluation_manifest.json").read_text())
        expected_length = 64 if manifest["phase"] == "final" else 16
        if manifest["graph_length"] != expected_length:
            raise ValueError("CLRS size-generalization protocol changed")
        # Preserve legitimate input-computable initial state; no future answer hints.
        import clrs
        from clrs_inputs import FLOYD_SPEC, inference_hints
        inputs = {point.name: point.data for point in evaluation.features.inputs}
        initial = inference_hints(np.asarray(inputs["A"]), np.asarray(inputs["adj"]))
        hints = tuple(clrs.DataPoint(name=name, location=FLOYD_SPEC[name][1],
                                    type_=FLOYD_SPEC[name][2], data=values)
                      for name, values in initial.items())
        features = evaluation.features._replace(hints=hints)
        outputs, _ = model.predict(key, features, algorithm_index=0)
        predicted = np.asarray(outputs["Pi"].data)
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez(args.output / "predictions.npz", ids=ids, predictions=predicted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "infer"], required=True)
    parser.add_argument("--task", choices=["CLRS"], required=True)
    parser.add_argument("--model-source", type=Path)
    for name in ["data", "assets", "checkpoint", "output", "config"]:
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
