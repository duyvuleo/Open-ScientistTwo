import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from upstream_checkouts import require_upstream

from scientisttwo.benchmarks.models import NativeSpec
from scientisttwo.live.adapters import (
    CSVDevelopment,
    VectorizationDevelopment,
    implementation_identity,
)
from scientisttwo.live.models import Candidate, DevelopmentSpec, LiveSpec, SourceFile
from scientisttwo.live.runtime import Budget, DevelopmentRunner
from scientisttwo.process import ProcessResult
from scientisttwo.store import Store


def csv_setup(tmp_path):
    public = tmp_path / "public"
    public.mkdir(parents=True)
    pd.DataFrame(
        {
            "Insult": [0, 1] * 30,
            "Date": [""] * 60,
            "Comment": [f"{'moon' if i % 2 == 0 else 'sun'} row {i}" for i in range(60)],
        }
    ).to_csv(public / "train.csv", index=False)
    pd.DataFrame({"Date": [""], "Comment": ["FINAL TEST CANARY"]}).to_csv(
        public / "test.csv", index=False
    )
    pd.DataFrame({"Insult": [0], "Date": [""], "Comment": ["FINAL TEST CANARY"]}).to_csv(
        public / "sample_submission_null.csv", index=False
    )
    native = NativeSpec(
        name="csv-test",
        benchmark="mle-bench",
        task_id="detecting-insults-in-social-commentary",
        repository=tmp_path / "repo",
        revision="a" * 40,
        mode="baseline",
        data_provenance="contract_fixture",
        public_dir=public,
        private_dir=tmp_path / "private",
    )
    config = DevelopmentSpec(adapter="csv", metric="roc_auc")
    return native, config, CSVDevelopment(native, config)


def test_csv_partition_no_final_or_validation_target_leak(tmp_path):
    native, config, adapter = csv_setup(tmp_path)
    subset = adapter.prepare("subset", tmp_path / "subset")
    full = adapter.prepare("full", tmp_path / "full")
    assert set(subset["train_indices"]) <= set(full["train_indices"])
    assert not set(full["train_indices"]) & set(full["validation_indices"])
    assert subset["validation_indices"] == full["validation_indices"]
    assert "FINAL TEST CANARY" not in json.dumps(adapter.context())
    for prepared in (subset, full):
        public = Path(prepared["public"])
        assert "Insult" not in pd.read_csv(public / "test.csv")
        assert not pd.read_csv(public / "sample_submission_null.csv")["Insult"].any()
        assert "FINAL TEST CANARY" not in (public / "test.csv").read_text()
    allowed = set(adapter.frame.iloc[adapter.subset]["Comment"])
    assert all(r["Comment"] in allowed for r in adapter.context()["sample_training_rows"])


@pytest.mark.parametrize("values", [[float("nan")], [2.0], []])
def test_csv_invalid_prediction_detection(tmp_path, values):
    _, _, adapter = csv_setup(tmp_path)
    prepared = adapter.prepare("full", tmp_path / "full")
    work = tmp_path / "work"
    work.mkdir()
    values = values * len(prepared["labels"])
    pd.DataFrame({"Insult": values}).to_csv(work / "submission.csv", index=False)
    with pytest.raises(ValueError):
        adapter.score("full", prepared, work, 1)


class ContainerDouble:
    def __init__(self):
        self.calls = []
        self.timeout = False

    def __call__(self, argv, timeout, **kwargs):
        self.calls.append((argv, timeout))
        if "inspect" in argv:
            return ProcessResult(0, "sha256:test-image", "", 0.01)
        if "run" not in argv:
            return ProcessResult(0, "", "", 0.01)
        mounts = {
            v.rsplit(":", 2)[1]: Path(v.rsplit(":", 2)[0])
            for i, v in enumerate(argv)
            if i and argv[i - 1] == "-v"
        }
        assert set(mounts) == {"/control", "/bridge/bridge.py", "/workspace", "/data/public"}
        assert "--network=none" in argv
        if self.timeout:
            return ProcessResult(-9, "", "", timeout, timed_out=True)
        rows = pd.read_csv(mounts["/data/public"] / "test.csv")
        rows["Insult"] = [0.8 if "sun" in t else 0.2 for t in rows["Comment"]]
        rows.to_csv(mounts["/workspace"] / "submission.csv", index=False)
        return ProcessResult(0, "synthetic container transport", "", 0.1)


def runtime_setup(tmp_path, monkeypatch):
    native, development, adapter = csv_setup(tmp_path)
    config = LiveSpec(
        native=native,
        development=development,
        model="fake",
        max_calls=10,
        campaign_seconds=300,
        implementation=implementation_identity(),
    )
    store = Store(tmp_path / "campaign")
    budget = Budget(store, config)
    fake = ContainerDouble()
    monkeypatch.setattr("scientisttwo.live.runtime.preflight", lambda s: {"ready": True})
    runner = DevelopmentRunner(store, config, adapter, budget, executor=fake)
    runner.preflight()
    store.set("live_spec_ref", store.put_json(config))
    store.set("live_adapter_identity", adapter.identity())
    return store, adapter, runner, fake


def test_development_transport_scores_predictions_caches_and_audits(tmp_path, monkeypatch):
    from scientisttwo.live.audit import audit_live

    store, adapter, runner, fake = runtime_setup(tmp_path, monkeypatch)
    result = runner.execute("first", adapter.baseline())
    assert result.status == "success" and result.value == pytest.approx(1)
    assert len(result.prediction_refs) == 3
    count = len(fake.calls)
    assert runner.execute("first", adapter.baseline()) == result
    assert len(fake.calls) == count
    assert audit_live(store.root)["status"] == "pass"
    (store.root / result.prediction_refs[0]).write_text("fabricated")
    assert audit_live(store.root)["status"] == "fail"
    store.close()


def test_timeout_cancels_container_and_failed_attempts_cached(tmp_path, monkeypatch):
    store, adapter, runner, fake = runtime_setup(tmp_path, monkeypatch)
    fake.timeout = True
    result = runner.execute("timeout", adapter.baseline())
    assert result.status == "timeout" and result.value is None
    assert any("rm" in argv and "--force" in argv for argv, _ in fake.calls)
    count = len(fake.calls)
    assert runner.execute("timeout", adapter.baseline()) == result
    assert len(fake.calls) == count
    store.close()


def test_interrupted_development_is_not_reexecuted(tmp_path, monkeypatch):
    store, adapter, runner, fake = runtime_setup(tmp_path, monkeypatch)
    actual = fake.__call__

    def interrupted(argv, timeout, **kwargs):
        if "run" in argv:
            raise KeyboardInterrupt
        return actual(argv, timeout, **kwargs)

    runner.raw_executor = interrupted
    with pytest.raises(KeyboardInterrupt):
        runner.execute("interrupted", adapter.baseline())
    runner.raw_executor = fake
    before = len(fake.calls)
    result = runner.execute("interrupted", adapter.baseline())
    assert result.status == "interrupted" and len(fake.calls) == before
    store.close()


def test_vector_reference_checks_wrong_convolution(tmp_path):
    repo = require_upstream("mlagentbench")
    if not repo.exists():
        pytest.skip("Pinned benchmark checkout unavailable")
    native = NativeSpec(
        name="vector",
        benchmark="mlagentbench",
        task_id="vectorization",
        repository=repo,
        revision="a" * 40,
        mode="baseline",
        data_provenance="upstream_no_data",
        public_dir=repo / "MLAgentBench/benchmarks/vectorization/env",
    )
    adapter = VectorizationDevelopment(native, DevelopmentSpec())
    adapter.validate(adapter.baseline())
    prepared = adapter.prepare("subset", tmp_path / "prepared")
    np.savez(tmp_path / "predictions.npz", **{"0": np.zeros((1,)), "1": np.zeros((1,))})
    with pytest.raises(ValueError, match="correctness"):
        adapter.score("subset", prepared, tmp_path, 1)


VECTOR_TEMPLATE = '''import numpy as np

class Conv2DLayer:
    """Original class documentation: café."""
    def __init__(self):
        """Keep the original spelling: Kernal Matrix."""
        self.stride = 1

    def forward(
        self, features_batch: np.ndarray
    ) -> np.ndarray:
        """Original forward documentation."""
        return features_batch  # Original body comment

# Preserve the timing block byte for byte.
if __name__ == "__main__":
    print("unchanged timing placeholder")
'''


def vector_body_setup(tmp_path, *, mode="forward_body", source=VECTOR_TEMPLATE):
    public = tmp_path / "public"
    public.mkdir(parents=True)
    (public / "train.py").write_bytes(source.encode())
    native = NativeSpec(
        name="vector-body",
        benchmark="mlagentbench",
        task_id="vectorization",
        repository=tmp_path / "repo",
        revision="a" * 40,
        mode="baseline",
        data_provenance="upstream_no_data",
        public_dir=public,
    )
    config = DevelopmentSpec(settings={} if mode is None else {"proposal_mode": mode})
    return VectorizationDevelopment(native, config)


def body_proposal(content, *, command=None, extra_files=None):
    return Candidate(
        files=[SourceFile(path="train.py", content=content)] + (extra_files or []),
        command=["python", "train.py"] if command is None else command,
        description="Declared NumPy optimization; preserve this explanation",
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_vector_body_materialization_preserves_template_and_proposal(tmp_path, newline):
    source = VECTOR_TEMPLATE.replace("\n", newline)
    adapter = vector_body_setup(tmp_path, source=source)
    body = (
        '"""Original forward documentation."""\n'
        "def local_helper(values):\n"
        "    return values.copy()\n"
        "return local_helper(features_batch)"
    )
    proposal = body_proposal(body)
    declared = proposal.model_dump()
    rendered = adapter.materialize(proposal)
    original_body = (
        '        """Original forward documentation."""'
        + newline
        + "        return features_batch  # Original body comment"
        + newline
    )
    prefix, suffix = source.split(original_body)
    expected = (
        prefix + newline.join("        " + line for line in body.split("\n")) + newline + suffix
    )
    assert rendered.files[0].content == expected
    assert rendered.files[0].content.startswith(prefix)
    assert rendered.files[0].content.endswith(suffix)
    assert rendered.description == proposal.description
    assert proposal.model_dump() == declared
    assert rendered is not proposal
    adapter.validate(rendered)
    assert adapter.identity()["proposal_mode"] == "forward_body"
    assert "only unindented forward BODY" in adapter.context()["contract"]
    assert adapter.context()["starter_source"] == source
    assert adapter.materialize(adapter.baseline()) == adapter.baseline()


def test_vector_body_is_parsed_without_executing_and_accepts_crlf(tmp_path):
    adapter = vector_body_setup(tmp_path)
    proposal = body_proposal('raise RuntimeError("Candidate executed on host")\r\n')
    rendered = adapter.materialize(proposal)
    assert '        raise RuntimeError("Candidate executed on host")\n' in rendered.files[0].content
    assert "\r\r" not in rendered.files[0].content
    assert proposal.files[0].content.endswith("\r\n")


@pytest.mark.parametrize(
    "body",
    [
        "",
        "# no statements",
        "return (",
        "    return features_batch",
        "```python\nreturn features_batch\n```",
        "def forward(self, features_batch):\n    return features_batch",
        "async def forward(self, features_batch):\n    return features_batch",
        "class Conv2DLayer:\n    pass",
        VECTOR_TEMPLATE.replace("Kernal Matrix", "Kernel Matrix"),
        "break",
        "nonlocal features_batch\nreturn features_batch",
        "return features_batch\n\x00",
    ],
)
def test_vector_body_rejects_malformed_scope_or_wrappers(tmp_path, body):
    adapter = vector_body_setup(tmp_path)
    with pytest.raises(ValueError):
        adapter.materialize(body_proposal(body))


@pytest.mark.parametrize(
    "command,extra_files",
    [
        (["python", "train.py", "--extra"], []),
        (["python", "other.py"], [SourceFile(path="other.py", content="pass")]),
        (["python", "train.py"], [SourceFile(path="other.py", content="pass")]),
    ],
)
def test_vector_body_rejects_changed_file_scope_or_argv(tmp_path, command, extra_files):
    adapter = vector_body_setup(tmp_path)
    with pytest.raises(ValueError, match="unchanged entry point"):
        adapter.materialize(
            body_proposal("return features_batch", command=command, extra_files=extra_files)
        )


@pytest.mark.parametrize("mode", ["full_file", None])
def test_vector_legacy_mode_remains_full_file(tmp_path, mode):
    adapter = vector_body_setup(tmp_path, mode=mode)
    proposal = body_proposal(
        VECTOR_TEMPLATE.replace("return features_batch", "return features_batch.copy()")
    )
    assert adapter.materialize(proposal) is proposal
    adapter.validate(proposal)
    assert adapter.identity()["proposal_mode"] == "full_file"
    bad = body_proposal(VECTOR_TEMPLATE.replace("Kernal Matrix", "Kernel Matrix"))
    assert adapter.materialize(bad) is bad
    with pytest.raises(ValueError, match="only the forward method body"):
        adapter.validate(bad)


@pytest.mark.parametrize("mode", ["unknown", [], None])
def test_vector_rejects_unknown_explicit_proposal_mode(tmp_path, mode):
    # None here is an explicit invalid setting, whereas omitted mode is legacy-compatible.
    adapter = vector_body_setup(tmp_path)
    config = adapter.config.model_copy(update={"settings": {"proposal_mode": mode}})
    with pytest.raises(ValueError, match="proposal_mode"):
        VectorizationDevelopment(adapter.native, config)
