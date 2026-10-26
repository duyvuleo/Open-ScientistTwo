from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from scientisttwo.live import timing_llama, timing_vectorization
from scientisttwo.store import Store, digest


@pytest.mark.parametrize("module", [timing_vectorization, timing_llama])
@pytest.mark.parametrize("original_failure", [False, True])
@pytest.mark.parametrize("retention_failure", ["receipt", "close"])
def test_both_arm_evidence_survives_retention_failure(
        tmp_path, monkeypatch, module, original_failure, retention_failure):
    monkeypatch.setattr(module, "verify_tree", lambda *args: None)
    monkeypatch.setattr(module, "expected_outputs" if module is timing_vectorization
                        else "expected_tokens", lambda *args: (["one"], {}))
    if module is timing_llama:
        @contextmanager
        def arrays(*args):
            yield {"ids": np.array(["one"])}
        monkeypatch.setattr(module, "open_arrays", arrays)
        monkeypatch.setattr(module, "asset_mounts", lambda *args, **kwargs: [
            (tmp_path, "/assets/model", "ro"), (tmp_path, "/assets/tokenizer", "ro")])

    class Transport:
        def __init__(self, **options):
            assert 0 < options["execution_timeout"] <= 60
            assert options["resources"].seconds == 900
            self.arm = options["control"].parent.name
            self.closed = False
            options["control"].mkdir(parents=True)
            (options["control"] / "worker.log").write_text(self.arm + " diagnostic log")

        def close(self):
            self.closed = True
            if self.arm == "baseline" and retention_failure == "close":
                raise ValueError("original cleanup failure")

        def receipt(self, store):
            if self.arm == "baseline" and retention_failure == "receipt":
                raise ValueError("missing trusted enforcement receipt")
            return {"closed": self.closed, "returncode": 0}

    monkeypatch.setattr(module, "ContainerTimingTransport", Transport)
    failure = TimeoutError("original measurement timeout")

    def measured(*args, **kwargs):
        if original_failure:
            raise failure
        return {"measured": True}

    monkeypatch.setattr(module, "run_paired_timing", measured)
    spec = SimpleNamespace(runtime="podman", context=None,
                           resources=SimpleNamespace(seconds=900, gpus=(
                               "device=GPU-test" if module is timing_llama else None)))
    prepared = {"public": str(tmp_path / "public"), "private": str(tmp_path / "private"),
                "files": {}, "private_files": {}, "cases": []}
    output = tmp_path / "execution"
    store = Store(tmp_path / "campaign")
    try:
        function = module.run_vectorization_pair if module is timing_vectorization else module.run_llama_pair
        with pytest.raises(TimeoutError if original_failure else RuntimeError) as caught:
            function(store, spec, "sha256:" + "a" * 64, tmp_path / "base", tmp_path / "candidate",
                     prepared, output, timeout=60)
        if original_failure:
            assert caught.value is failure
        attempt = store.json(store.get("paired_timing/" + digest(str(output).encode())))
        assert attempt["status"] == "failed"
        assert set(attempt["receipts"]) == {"baseline", "candidate"}
        assert "baseline/" + retention_failure in attempt["retention_errors"]
        for arm, reference in attempt["receipts"].items():
            receipt = store.json(reference)
            assert store.read(receipt["log_ref"]) == (arm + " diagnostic log").encode()
        assert store.verify() == []
    finally:
        store.close()
