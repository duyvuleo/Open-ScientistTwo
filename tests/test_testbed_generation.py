import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scientisttwo.testbed import generation


def setup_generation(tmp_path, monkeypatch):
    assets, source = tmp_path / "assets", tmp_path / "corpus"
    assets.mkdir()
    source.mkdir()
    (assets / "data.bin").write_bytes(b"SYNTHETIC generation-contract asset")
    (source / "wiki.test.tokens").write_text("Synthetic contract fixture only")
    calls = []
    monkeypatch.setattr(generation._GeneratorRuntime, "_image", lambda self, tag: "sha256:" + "a" * 64)

    def execute(self, stage, image, job, mounts, timeout, **kwargs):
        calls.append(job)
        output = next(host for host, target, mode in mounts if target == "/output")
        (output / "split_metadata.json").write_text(json.dumps({"dataset_origin": "SYNTHETIC CONTRACT FIXTURE"}))
        (output / "phases").mkdir()
        (output / "phases/payload.txt").write_text("SYNTHETIC, never actual benchmark data")
        ref = self.store.put_json({"returncode": 0, "timed_out": False, "image_id": image,
                                   "argv": ["docker", "--network=none", "--read-only"]})
        return SimpleNamespace(code=0, timed_out=False), ref

    monkeypatch.setattr(generation._GeneratorRuntime, "_launch", execute)
    kwargs = {"image": "fixture:no-runtime", "assets": assets, "source": source,
              "gpus": "device=0", "variant": "adapted-synthetic-generation-test", "terms_accepted": True}
    return kwargs, calls


def test_generation_export_failure_resumes_without_reexecuting(tmp_path, monkeypatch):
    kwargs, calls = setup_generation(tmp_path, monkeypatch)
    original = generation._export_receipt

    def interrupted(data, evidence, receipt):
        target = data / "provider-control/evidence"
        target.mkdir(parents=True)
        (target / "partial-export.txt").write_text("partial")
        raise OSError("injected export interruption")

    monkeypatch.setattr(generation, "_export_receipt", interrupted)
    with pytest.raises(OSError, match="injected"):
        generation.generate_bundle("llama-inference", tmp_path / "run", **kwargs)
    assert len(calls) == 1
    monkeypatch.setattr(generation, "_export_receipt", original)
    # Partial unregistered export files are not valid evidence. Use a realistic
    # interrupted copy: a registered SQLite file truncated before receipt commit.
    target = tmp_path / "run/source/provider-control/evidence"
    (target / "partial-export.txt").unlink()
    (target / "campaign.sqlite").write_bytes(b"partial copy")
    result = generation.generate_bundle("llama-inference", tmp_path / "run", **kwargs)
    assert result["cached"] and Path(result["receipt"]).is_file()
    assert len(calls) == 1
    again = generation.generate_bundle("llama-inference", tmp_path / "run", **kwargs)
    assert again["cached"] and len(calls) == 1


def test_known_completed_execution_resumes_partial_input_copy(tmp_path, monkeypatch):
    kwargs, calls = setup_generation(tmp_path, monkeypatch)
    original = generation._copy_inputs

    def interrupted(inputs, request, data, store):
        path = data / "assets/.data.bin.copy-pending"
        path.parent.mkdir()
        store.set("generation_pending_copy", "assets/.data.bin.copy-pending")
        path.write_bytes(b"partial")
        raise OSError("injected copy interruption")

    monkeypatch.setattr(generation, "_copy_inputs", interrupted)
    with pytest.raises(OSError, match="injected"):
        generation.generate_bundle("llama-inference", tmp_path / "run", **kwargs)
    monkeypatch.setattr(generation, "_copy_inputs", original)
    result = generation.generate_bundle("llama-inference", tmp_path / "run", **kwargs)
    assert result["cached"] and len(calls) == 1
    assert (Path(result["source"]) / "assets/data.bin").read_bytes() == (kwargs["assets"] / "data.bin").read_bytes()


def test_unknown_generation_is_never_automatically_repeated(tmp_path, monkeypatch):
    kwargs, calls = setup_generation(tmp_path, monkeypatch)

    def interrupted(*args, **kwargs):
        calls.append("uncommitted")
        raise KeyboardInterrupt

    monkeypatch.setattr(generation._GeneratorRuntime, "_launch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        generation.generate_bundle("llama-inference", tmp_path / "run", **kwargs)
    with pytest.raises(ValueError, match="refusing an automatic duplicate"):
        generation.generate_bundle("llama-inference", tmp_path / "run", **kwargs)
    assert calls == ["uncommitted"]


def test_missing_model_terms_or_allocation_blocks_before_runtime(tmp_path):
    with pytest.raises(ValueError, match="accepted model terms"):
        generation.generate_bundle("llama-inference", tmp_path / "run", image="fixture")
    assert not (tmp_path / "run").exists()
