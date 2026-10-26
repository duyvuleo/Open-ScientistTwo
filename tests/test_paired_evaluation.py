"""Explicitly synthetic traces for paired timing cache, audit, and final boundaries."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from scientisttwo.artifacts import ArtifactSpec, freeze_bundle, load_bundle
from scientisttwo.live.paired_evaluation import (
    audit_paired_observation,
    execute_paired,
    execute_paired_final,
)
from scientisttwo.live.timing import run_paired_timing
from scientisttwo.live.controller import LiveController
from test_live_development import runtime_setup


def synthetic_pair(store, spec, image, baseline, candidate, prepared, output, *, timeout, seed):
    """Fake timing is limited to unit tests and is labeled in every output artifact."""
    output.mkdir(parents=True)
    ticks = [0]
    calls = {"baseline": 0, "candidate": 0}

    class Transport:
        def __init__(self, arm):
            self.arm = arm

        def request(self, payload, *, timeout):
            ticks[0] += 2_000_000 if self.arm == "baseline" else 1_000_000
            calls[self.arm] += 1
            path = output / self.arm
            path.mkdir(exist_ok=True)
            name = f"fixture-{calls[self.arm]}.json"
            (path / name).write_text(json.dumps({"fixture": "synthetic", "seed": seed}))
            return {"predictions": name}

    def oracle(arm, payload, response):
        ref = freeze_bundle(store, output / arm, [ArtifactSpec(path=response["predictions"])])
        record = load_bundle(store, ref).files[response["predictions"]]
        return {
            "valid": True,
            "bundle_ref": ref,
            "prediction_ref": record.ref,
            "fixture": "synthetic",
        }

    report = run_paired_timing(
        {arm: Transport(arm) for arm in calls},
        [{"fixture": "synthetic"}],
        oracle,
        clock_ns=lambda: ticks[0],
    )
    return report, {
        arm: store.put_json(
            {
                "fixture": "synthetic",
                "returncode": 0,
                "log_ref": store.put(
                    f"SYNTHETIC {arm} timing worker output\n".encode(), "log", "fixture-log"
                ),
            }
        )
        for arm in calls
    }


def setup(tmp_path, monkeypatch):
    store, adapter, runner, _ = runtime_setup(tmp_path, monkeypatch)
    runner.config.development.metric = "latency_ratio"
    runner.config.development.direction = "minimize"

    def audit(store, prepared, report):
        for sample in report.samples:
            assert store.json(sample.correctness["prediction_ref"])["fixture"] == "synthetic"
            load_bundle(store, sample.correctness["bundle_ref"])

    adapter.audit_timing = audit
    return store, adapter, runner


def test_paired_baseline_identity_ratio_candidate_ratio_and_noop_resume(tmp_path, monkeypatch):
    store, adapter, runner = setup(tmp_path, monkeypatch)
    try:
        baseline = execute_paired(runner, "base", adapter.baseline(), run_pair=synthetic_pair)
        assert baseline.status == "success" and baseline.value == 1.0
        candidate = adapter.baseline()
        candidate.files[0].content += "\n# explicitly synthetic fixture candidate\n"
        result = execute_paired(runner, "candidate", candidate, run_pair=synthetic_pair)
        assert result.status == "success" and result.value == 0.5
        assert result.protocol_hash == baseline.protocol_hash
        count = len(runner.budget.summary())
        assert execute_paired(runner, "candidate", candidate, run_pair=synthetic_pair) == result
        assert len(runner.budget.summary()) == count
        prepared = store.json(store.get("prepared/full"))
        assert (
            audit_paired_observation(store, result, runner.config, adapter, prepared)["metric"]
            == "pass"
        )
    finally:
        store.close()


def test_controller_feedback_reads_paired_worker_logs_with_standard_receipt(tmp_path, monkeypatch):
    store, adapter, runner = setup(tmp_path, monkeypatch)
    try:
        result = execute_paired(
            runner, "feedback", adapter.baseline(), seeds=[0], run_pair=synthetic_pair
        )
        assert result.status == "success"
        feedback = LiveController.feedback(SimpleNamespace(store=store), result)
        assert len(feedback["execution_logs"]) == 1
        assert "SYNTHETIC baseline timing worker output" in feedback["execution_logs"][0]
        assert "SYNTHETIC candidate timing worker output" in feedback["execution_logs"][0]
        receipt = store.json(result.log_refs[0])
        assert set(receipt["worker_log_refs"]) == {"baseline", "candidate"}
        assert not receipt["log_truncated"]
        assert store.read(receipt["log_ref"]).decode() == feedback["execution_logs"][0]
    finally:
        store.close()


def test_paired_feedback_log_retains_bounded_head_tail_from_large_real_shaped_logs(
    tmp_path, monkeypatch
):
    store, adapter, runner = setup(tmp_path, monkeypatch)
    runner.spec.resources.log_max_bytes = 1024

    def noisy_pair(*args, **kwargs):
        report, receipts = synthetic_pair(*args, **kwargs)
        for arm, ref in list(receipts.items()):
            value = store.json(ref)
            value["log_ref"] = store.put(
                (arm + "-HEAD\n" + "x" * 5000 + "\n" + arm + "-TAIL\n").encode(),
                "log",
                "fixture-log",
            )
            receipts[arm] = store.put_json(value)
        return report, receipts

    try:
        result = execute_paired(
            runner, "bounded-feedback", adapter.baseline(), seeds=[0], run_pair=noisy_pair
        )
        assert result.status == "success"
        feedback = LiveController.feedback(SimpleNamespace(store=store), result)
        receipt = store.json(result.log_refs[0])
        assert len(store.read(receipt["log_ref"])) <= 1024
        assert receipt["log_truncated"] and receipt["log_bytes"] > 10000
        assert "baseline-HEAD" in feedback["execution_logs"][0]
        assert "candidate-TAIL" in feedback["execution_logs"][0]
    finally:
        store.close()


@pytest.mark.native_container
def test_real_paired_subset_feedback():
    """Real CPU subset measurements and feedback; no role/model invocation."""
    from contextlib import nullcontext
    from scientisttwo.live.adapters import load_development
    from scientisttwo.live.runtime import Budget, DevelopmentRunner
    from scientisttwo.store import Store
    from scientisttwo.testbed.validation import create_validation_spec

    context = os.environ.get("SCIENTISTTWO_NATIVE_CONTEXT")
    if not context:
        pytest.skip("Set SCIENTISTTWO_NATIVE_CONTEXT for isolated paired feedback validation")
    project = Path(__file__).resolve().parents[1]
    prepared = project / ".testbed/vectorization/prepared"
    if not prepared.is_dir():
        pytest.skip("Prepared pinned vectorization task is required")
    output = os.environ.get("SCIENTISTTWO_PAIRED_FEEDBACK_OUTPUT")
    cache = project / ".cache"
    cache.mkdir(exist_ok=True)
    directory = (
        nullcontext(output)
        if output
        else tempfile.TemporaryDirectory(prefix="paired-feedback-", dir=cache)
    )
    with directory as value:
        root = Path(value).resolve()
        root.mkdir(parents=True, exist_ok=True)
        config = create_validation_spec(
            prepared,
            image=os.environ.get("SCIENTISTTWO_TESTBED_IMAGE", "scientisttwo-cpu-tabular:audited-v1-arm"),
            context=context,
            level="smoke",
            seconds=120,
            campaign_seconds=300,
        )
        adapter = load_development(config.native, config.development)
        store = Store(root / "campaign")
        try:
            budget = Budget(store, config)
            runner = DevelopmentRunner(store, config, adapter, budget)
            runner.preflight()
            result = runner.execute("feedback-control", adapter.control(), "subset")
            assert result.status == "success", result
            feedback = LiveController.feedback(SimpleNamespace(store=store), result)
            assert len(feedback["execution_logs"]) == 1
            receipt = store.json(result.log_refs[0])
            assert receipt["log_ref"] and set(receipt["worker_log_refs"]) == {
                "baseline",
                "candidate",
            }
            assert (
                store.read(receipt["log_ref"]).decode(errors="replace")
                == feedback["execution_logs"][0]
            )
            assert all(row["kind"] != "model" for row in budget.summary())
            (root / "paired-feedback-verification.json").write_text(
                json.dumps(
                    {
                        "scope": "Actual isolated CPU vectorization subset and controller feedback; no live model calls",
                        "observation": result.model_dump(),
                        "receipt": receipt,
                        "live_model_calls": 0,
                        "feedback_log_count": len(feedback["execution_logs"]),
                        "integrity_errors": store.verify(),
                    },
                    indent=2,
                )
                + "\n"
            )
        finally:
            store.close()


def test_modified_timing_aggregate_is_detected(tmp_path, monkeypatch):
    store, adapter, runner = setup(tmp_path, monkeypatch)
    try:
        result = execute_paired(
            runner, "base", adapter.baseline(), seeds=[0], run_pair=synthetic_pair
        )
        evidence = store.json(result.timing_refs[0])
        report = store.json(evidence["report_ref"])
        report["candidate"]["median_seconds"] *= 10
        evidence["report_ref"] = store.put_json(report)
        result.timing_refs[0] = store.put_json(evidence)
        with pytest.raises(ValueError, match="summaries"):
            audit_paired_observation(
                store, result, runner.config, adapter, store.json(store.get("prepared/full"))
            )
    finally:
        store.close()


def test_final_evidence_requires_freeze_and_uses_separate_namespace(tmp_path, monkeypatch):
    store, adapter, runner = setup(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError, match="frozen"):
            execute_paired_final(runner, "selected", adapter.baseline(), run_pair=synthetic_pair)
        assert store.get("prepared/final") is None
        store.set("live_frozen", {"fixture": "synthetic frozen selection"})
        result = execute_paired_final(
            runner, "selected", adapter.baseline(), run_pair=synthetic_pair
        )
        assert result["status"] == "completed"
        observation = store.json(result["observation_ref"])
        assert observation["phase"] == "final"
        assert (
            store.db.execute(
                "SELECT count(*) FROM calls WHERE key LIKE 'development/%'"
            ).fetchone()[0]
            == 0
        )
        assert (
            store.db.execute(
                "SELECT count(*) FROM calls WHERE key LIKE 'final-paired/%'"
            ).fetchone()[0]
            == 1
        )
        count = len(runner.budget.summary())
        assert (
            execute_paired_final(runner, "selected", adapter.baseline(), run_pair=synthetic_pair)
            == result
        )
        assert len(runner.budget.summary()) == count
    finally:
        store.close()


def test_interrupted_paired_execution_never_repeats_attempt(tmp_path, monkeypatch):
    store, adapter, runner = setup(tmp_path, monkeypatch)
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            execute_paired(runner, "interrupted", adapter.baseline(), run_pair=interrupted)
        result = execute_paired(runner, "interrupted", adapter.baseline(), run_pair=interrupted)
        assert result.status == "interrupted" and len(calls) == 1
    finally:
        store.close()
