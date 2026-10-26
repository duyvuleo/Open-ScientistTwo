"""Trusted, offline CLRS bundle generation inside the pinned JAX container.

This is a declared generated-sampler variant, not a claim of byte-identical TFDS
datasets. The host launcher seals image/source/input/output receipts separately.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def _archive(feedback, start, stop, ids, *, labels=True):
    values = {"ids": np.asarray(ids[start:stop]), "lengths": np.asarray(feedback.features.lengths)[start:stop]}
    for point in feedback.features.inputs:
        values["input:" + point.name] = np.asarray(point.data)[start:stop]
    if labels:
        for point in feedback.features.hints:
            values["hint:" + point.name] = np.asarray(point.data)[:, start:stop]
        for point in feedback.outputs:
            values["output:" + point.name] = np.asarray(point.data)[start:stop]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, default=Path("/helpers"))
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--source", type=Path, default=Path("/source"))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--variant", default="audited-v1")
    parser.add_argument("--helper-manifest", type=Path)
    args = parser.parse_args()
    if importlib.metadata.version("dm-clrs") != "1.0.0":
        raise ValueError("CLRS preparation requires the locked dm-clrs 1.0.0 sampler")
    import clrs
    if (args.output / "phases").exists():
        raise ValueError("Refusing to overwrite a generated bundle")
    lengths = [4, 7, 11, 13, 16]
    roots = {}
    manifests = {}
    for phase in ["screen", "full", "final"]:
        public, private = args.output / "phases" / phase / "public", args.output / "phases" / phase / "private"
        (public / "train").mkdir(parents=True)
        private.mkdir(parents=True)
        roots[phase] = public, private
        manifests[phase] = {"algorithm": "floyd_warshall", "train_lengths": lengths, "shards": []}
    for nodes in lengths:
        sampler, spec = clrs.build_sampler("floyd_warshall", num_samples=1000,
                                           length=nodes, seed=args.seed + nodes)
        feedback = sampler.next()
        ids = [f"clrs-train-n{nodes}-{index}" for index in range(1000)]
        for phase, (public, _) in roots.items():
            count = 500 if phase == "screen" else 1000
            for start in range(0, count, 32):
                relative = f"train/n{nodes}-{start:04d}.npz"
                np.savez(public / relative, **_archive(feedback, start, min(start + 32, count), ids))
                manifests[phase]["shards"].append({"path": relative, "length": nodes})
            (public / "spec.json").write_text(json.dumps(spec, sort_keys=True))
    for partition, nodes, seed in [("development", 16, args.seed + 10000), ("final", 64, args.seed + 20000)]:
        sampler, _ = clrs.build_sampler("floyd_warshall", num_samples=32, length=nodes, seed=seed)
        feedback = sampler.next()
        ids = [f"clrs-{partition}-n{nodes}-{index}" for index in range(32)]
        target = next(point.data for point in feedback.outputs if point.name == "Pi")
        for phase in (["screen", "full"] if partition == "development" else ["final"]):
            public, private = roots[phase]
            np.savez(public / "evaluation.npz", **_archive(feedback, 0, 32, ids, labels=False))
            np.savez(private / "labels.npz", ids=np.asarray(ids), y=np.asarray(target))
            (public / "evaluation_manifest.json").write_text(json.dumps({"phase": phase, "graph_length": nodes}))
    for phase, (public, _) in roots.items():
        (public / "train_manifest.json").write_text(json.dumps(manifests[phase], sort_keys=True))
    metadata = {
        "dataset_origin": "https://pypi.org/project/dm-clrs/1.0.0/",
        "source_variant": "generated-dm-clrs-1.0.0-floyd-warshall",
        "variant": args.variant,
        "split_policy": "fixed-generated-sampler-train4-7-11-13-16-dev16-final64",
        "generator_seed": args.seed, "training_samples_per_size": 1000,
        "screen_samples_per_size": 500, "development_samples": 32, "final_samples": 32,
        "dm_clrs_version": "1.0.0", "upstream_tfds_byte_reproduction": False,
        "deviations": ["Finite deterministic sampler shards replace upstream streaming/TFDS loading; training sizes and held-out size64 are preserved"],
    }
    (args.output / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "generated", "task": "CLRS", "variant": metadata["source_variant"], "live_model_calls": 0}))


if __name__ == "__main__":
    main()
