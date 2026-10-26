"""Floyd-Warshall input state from the pinned CLRS protocol, without frameworks.

Checked against dm_clrs-1.0.0 wheel SHA256
eba1a7692ab4de88c37906d1ca66ee56acbaee66a5ca0cde55906ac9cf6906e8,
clrs/_src/algorithms/graphs.py lines 1448-1499 and the pinned upstream spec.
Only the initial hints are derivable from inputs; future supervision stays private.
"""
import numpy as np


FLOYD_SPEC = {
    "pos": ["input", "node", "scalar"], "A": ["input", "edge", "scalar"],
    "adj": ["input", "edge", "mask"], "Pi": ["output", "edge", "pointer"],
    "Pi_h": ["hint", "edge", "pointer"], "D": ["hint", "edge", "scalar"],
    "msk": ["hint", "edge", "mask"], "k": ["hint", "node", "mask_one"],
}


def initial_hint_frames(weights, adjacency):
    weights, adjacency = np.asarray(weights), np.asarray(adjacency)
    if weights.ndim != 3 or weights.shape[-1] != weights.shape[-2] or adjacency.shape != weights.shape:
        raise ValueError("Floyd-Warshall needs batch-aligned square adjacency matrices")
    batch, nodes, _ = weights.shape
    expected_mask = ((weights + np.eye(nodes)[None]) != 0).astype(np.float32)
    if not np.array_equal(adjacency, expected_mask):
        raise ValueError("Adjacency mask differs from the input weights and self edges")
    pointers = np.broadcast_to(np.arange(nodes)[None, :, None], (batch, nodes, nodes)).copy()
    iteration = np.zeros((batch, nodes), dtype=np.float32)
    iteration[:, 0] = 1
    return {"Pi_h": pointers, "D": weights.copy(), "msk": adjacency.copy(), "k": iteration}


def inference_hints(weights, adjacency):
    frames = initial_hint_frames(weights, adjacency)
    nodes = weights.shape[-1]
    result = {}
    for name, first in frames.items():
        values = np.zeros((nodes, *first.shape), dtype=first.dtype)
        values[0] = first
        result[name] = values
    return result


def validate_trajectory(archive, *, nodes, supervised):
    ids, lengths = archive["ids"], archive["lengths"]
    batch = len(ids)
    expected = {"ids", "lengths"} | {stage + ":" + name for name, (stage, _, _) in FLOYD_SPEC.items()
                                      if supervised or stage == "input"}
    if set(archive.files) != expected:
        raise ValueError("CLRS arrays differ from the canonical Floyd-Warshall fields")
    if lengths.shape != (batch,) or not np.all(lengths == nodes):
        raise ValueError("CLRS rollout lengths differ from the actual graph sizes")
    weights, adjacency, positions = archive["input:A"], archive["input:adj"], archive["input:pos"]
    if weights.shape != (batch, nodes, nodes) or positions.shape != (batch, nodes):
        raise ValueError("CLRS input arrays do not have the declared graph size")
    if not np.isfinite(weights).all() or not np.isfinite(positions).all():
        raise ValueError("Non-finite CLRS inputs")
    first = initial_hint_frames(weights, adjacency)
    if supervised:
        output = archive["output:Pi"]
        if output.shape != (batch, nodes, nodes) or not np.isfinite(output).all() or np.any(output != output.astype(int)) or np.any(output < 0) or np.any(output >= nodes):
            raise ValueError("Invalid Floyd-Warshall pointer targets")
        for name, initial in first.items():
            values = archive["hint:" + name]
            if values.shape != (nodes, *initial.shape) or not np.isfinite(values).all():
                raise ValueError("CLRS hint arrays do not match rollout/graph dimensions")
            if not np.array_equal(values[0], initial):
                raise ValueError("CLRS initial hints differ from the input-computable algorithm state")
