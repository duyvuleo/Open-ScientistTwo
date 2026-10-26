"""Synthetic token fixtures verify timing contracts, not GPU benchmark performance."""
import itertools
import json

import numpy as np
import pytest

from scientisttwo.benchmarks.files import tree_manifest
from scientisttwo.live.timing import run_paired_timing
from scientisttwo.live.timing_llama import audit_llama_report, expected_tokens, llama_oracle
from scientisttwo.store import Store


def test_llama_pair_seals_every_prediction_and_replays_exact_token_checks(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    ids, tokens = np.array(["a", "b"]), np.array([[7], [21]])
    np.savez(private / "labels.npz", ids=ids, y=tokens)
    prepared = {"private": str(private), "private_files": tree_manifest(private)}
    expected = expected_tokens(prepared)
    outputs = {arm: tmp_path / arm for arm in ("baseline", "candidate")}
    for directory in outputs.values():
        directory.mkdir()
        (directory / "model-identity.json").write_text(json.dumps({"before": {"parameters_sha256": "synthetic"}, "after": {"parameters_sha256": "synthetic"}, "status": "identical_declared_state"}))
    store = Store(tmp_path / "store")

    class SyntheticTokens:
        def __init__(self, arm):
            self.arm, self.count = arm, 0

        def request(self, payload, *, timeout):
            self.count += 1
            name = f"predictions-{self.count:04d}.npz"
            np.savez(outputs[self.arm] / name, ids=ids[::-1], predictions=tokens[::-1])
            return {"predictions": name}

    try:
        report = run_paired_timing(
            {arm: SyntheticTokens(arm) for arm in outputs},
            [{"workload": "fixed_one_token_generation"}], llama_oracle(store, outputs, expected),
            device="cuda", gpu_synchronization=True,
            clock_ns=itertools.count(0, 1000000).__next__,
        )
        assert len(report.samples) == 30
        assert audit_llama_report(store, prepared, report) == {"predictions": "pass", "sample_count": 30}
        report.samples[0].correctness["exact_token_equivalence"] = False
        with pytest.raises(ValueError, match="claims"):
            audit_llama_report(store, prepared, report)
    finally:
        store.close()


def test_wrong_generated_token_and_reused_output_are_rejected(tmp_path):
    outputs = {arm: tmp_path / arm for arm in ("baseline", "candidate")}
    for directory in outputs.values():
        directory.mkdir()
        (directory / "model-identity.json").write_text(json.dumps({"before": {"parameters_sha256": "synthetic"}, "after": {"parameters_sha256": "synthetic"}, "status": "identical_declared_state"}))
    store = Store(tmp_path / "store")
    ids, truth = np.array(["a"]), np.array([[7]])
    try:
        oracle = llama_oracle(store, outputs, (ids, truth))
        np.savez(outputs["candidate"] / "wrong.npz", ids=ids, predictions=np.array([[8]]))
        with pytest.raises(ValueError, match="differ"):
            oracle("candidate", {"workload": "fixed_one_token_generation"}, {"predictions": "wrong.npz"})
        np.savez(outputs["candidate"] / "correct.npz", ids=ids, predictions=truth)
        assert oracle("candidate", {"workload": "fixed_one_token_generation"}, {"predictions": "correct.npz"})["valid"]
        with pytest.raises(ValueError, match="fresh"):
            oracle("candidate", {"workload": "fixed_one_token_generation"}, {"predictions": "correct.npz"})
    finally:
        store.close()
