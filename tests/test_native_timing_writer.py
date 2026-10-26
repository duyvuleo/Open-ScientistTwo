"""Synthetic transport traces exercise native writer contracts, never performance claims."""
from copy import deepcopy

import pytest

from scientisttwo.integrations.paperorchestra.writer_context import (
    build_writer_context, context_method_fact, verify_context_method_fact,
)
from scientisttwo.live.audit import audit_live
from scientisttwo.live.paired_evaluation import execute_paired, execute_paired_final
from scientisttwo.research.audit import _manuscripts
from scientisttwo.research.manuscript import (
    build_evidence_ledger, final_results_appendix, fixture_manuscript,
    render_manuscript, verify_evidence_ledger, legacy_draft_view, validate_manuscript,
)
from scientisttwo.research.timing_evidence import timing_measurements, verify_final_result_binding
from scientisttwo.store import IntegrityError, canonical, digest
from test_paired_evaluation import setup, synthetic_pair


@pytest.fixture
def paired(tmp_path, monkeypatch):
    store, adapter, runner = setup(tmp_path, monkeypatch)
    monkeypatch.setattr("scientisttwo.live.audit.load_development", lambda *args: adapter)
    store.set("live_spec_ref", store.put_json(runner.config))
    yield store, adapter, runner
    store.close()


def measured(store, adapter, runner):
    candidate = adapter.baseline()
    candidate.files[0].content += "\n# SYNTHETIC writer timing candidate\n"
    return execute_paired(runner, "candidate", candidate, run_pair=synthetic_pair)


def test_timing_stats_are_empirical_and_bound_to_each_arm_seed_and_native_audit(paired):
    store, adapter, runner = paired
    observed = measured(store, adapter, runner)
    ledger = build_evidence_ledger(store, [observed])
    stats = [entry for entry in ledger["entries"].values() if entry.get("arm")]
    assert len(stats) == 4
    assert {entry["statistic"] for entry in stats} == {"median_seconds", "iqr_seconds"}
    for entry in stats:
        assert entry["kind"] == "measured" and entry["unit"] == "seconds"
        assert [point["seed"] for point in entry["per_seed"]] == [0, 1, 2]
        for index, point in enumerate(entry["per_seed"]):
            envelope = store.json(observed.timing_refs[index])
            assert point["source_ref"] == envelope[entry["arm"] + "_source_tree_ref"]
            assert point["report_ref"] == envelope["report_ref"]
        expected = .002 if entry["arm"] == "baseline" else .001
        assert entry["value"] == pytest.approx(0 if entry["statistic"] == "iqr_seconds" else expected)
    verify_evidence_ledger(store, ledger)
    assert audit_live(store.root)["status"] == "pass"
    context = build_writer_context(store, ledger)
    assert all(field["value"] not in {.001, .002} for field in context["numeric_metadata"])
    pair = context["protocols"][observed.id]["paired_timing"]
    assert pair["warmups"] == 5 and pair["pairs"] == 10
    assert context["task"]["status"] == "recorded_native_task_declaration"


def test_identity_ratio_does_not_relabel_either_raw_arm_as_one_second(paired):
    store, adapter, runner = paired
    observed = execute_paired(runner, "baseline", adapter.baseline(), run_pair=synthetic_pair)
    ledger = build_evidence_ledger(store, [observed])
    assert observed.value == 1
    raw = {item["metric"]: item["value"] for item in ledger["entries"].values()}
    assert raw["paired.baseline.median_seconds"] == .002
    assert raw["paired.candidate.median_seconds"] == .001


@pytest.mark.parametrize("change", ["seed", "source", "median", "sample", "ratio", "receipt"])
def test_sealed_timing_binding_and_arithmetic_reject_tampering(paired, change):
    store, adapter, runner = paired
    observed = measured(store, adapter, runner)
    original = canonical(observed)
    envelope = store.json(observed.timing_refs[0])
    if change == "seed":
        envelope["seed"] = 99
    elif change == "source":
        envelope["baseline_source_tree_ref"] = observed.source_tree_ref
    elif change in {"median", "sample"}:
        report = store.json(envelope["report_ref"])
        if change == "median":
            report["candidate"]["median_seconds"] *= 3
        else:
            report["samples"][-1]["seconds"] *= 3
        envelope["report_ref"] = store.put_json(report)
    elif change == "ratio":
        envelope["value"] = 1.0
    else:
        envelope["receipts"]["candidate"] = envelope["receipts"]["baseline"]
    changed = observed.model_copy(deep=True)
    changed.timing_refs[0] = store.put_json(envelope)
    changed_ref = store.put_json(changed)
    # Even a coordinated rewritten cache/operation must not forge report math.
    with store.db:
        store.db.execute("UPDATE calls SET output_ref=? WHERE key='development/candidate'", (changed_ref,))
        store.db.execute("UPDATE live_operations SET result_ref=? WHERE key='experiment/candidate'", (changed_ref,))
    with pytest.raises((IntegrityError, ValueError)):
        timing_measurements(store, changed, changed_ref, store.json(changed.protocol_ref))
    assert canonical(observed) == original


def test_legacy_ledger_retains_its_original_shape_and_new_stats_cannot_be_forged(paired):
    store, adapter, runner = paired
    observed = measured(store, adapter, runner)
    legacy = build_evidence_ledger(store, [observed], timing_evidence=False)
    assert "timing_evidence_contract" not in legacy
    verify_evidence_ledger(store, legacy)
    current = build_evidence_ledger(store, [observed])
    historical = build_evidence_ledger(store, [observed], timing_evidence="paired-timing-measurements-v1")
    assert len([entry for entry in historical["entries"].values() if entry.get("arm")]) == 8
    verify_evidence_ledger(store, historical)
    next(entry for entry in current["entries"].values() if entry.get("arm"))["value"] = 99
    with pytest.raises(IntegrityError, match="ledger differs"):
        verify_evidence_ledger(store, current)


def test_native_workload_declarations_distinguish_generation_and_timing_seeds(paired, tmp_path):
    store, adapter, runner = paired
    manifest = {"task_id": "vectorization", "split_metadata": {
        "split_seed": 1729, "final_seed": 2729, "private_secret_canary": "DO_NOT_EXPORT"}}
    root = tmp_path / "prepared"
    root.mkdir()
    (root / "manifest.json").write_bytes(canonical(manifest))
    runner.config.development.adapter = "testbed"
    runner.config.development.settings["prepared_root"] = str(root)
    runner.config.native.task_id = "vectorization"
    store.set("live_spec_ref", store.put_json(runner.config))
    identity = adapter.identity()
    adapter.identity = lambda: {**identity, "prepared_manifest": digest(canonical(manifest))}
    prepared = store.json(store.get("prepared/full"))
    prepared["cases"] = [{"stride": 1, "padding": 0, "activation": None},
                         {"stride": 2, "padding": 2, "activation": "relu"}]
    store.set("prepared/full", store.put_json(prepared))
    observed = measured(store, adapter, runner)
    ledger = build_evidence_ledger(store, [observed])
    context = build_writer_context(store, ledger)
    assert context["task"]["fields"]["final_seed"] == 0
    assert context["task"]["workload_generation"]["held_out_final_seed"] == 2729
    assert context["task"]["workload_generation"]["development_seed"] == 1729
    assert context["protocols"][observed.id]["seeds"] == [0, 1, 2]
    assert context["protocols"][observed.id]["paired_timing"]["public_workloads"]["case_count"] == 2
    text = canonical(context).decode()
    assert "DO_NOT_EXPORT" not in text and str(root) not in text
    assert "train_indices" not in text and "test_indices" not in text
    fact = context_method_fact(store, ledger)
    # Verification uses the sealed declaration, not mutable external data paths.
    (root / "manifest.json").unlink()
    verify_context_method_fact(store, fact, ledger["observations"])
    fields = fact["numeric_metadata"]
    assert any(item["source_path"] == ["split_metadata", "final_seed"] and item["value"] == 2729 for item in fields)
    assert any(item["source_path"] == ["native", "final_seed"] and item["value"] == 0 for item in fields)


def test_native_final_appendix_and_audit_preserve_observation_operation(paired):
    store, adapter, runner = paired
    observed = measured(store, adapter, runner)
    ledger = build_evidence_ledger(store, [observed])
    store.set("live_backend", {"synthetic": True})
    rendered = render_manuscript(store, fixture_manuscript(ledger), ledger)
    compatibility = legacy_draft_view(fixture_manuscript(ledger), ledger)
    assert {claim.metric for claim in compatibility.claims} == {"latency_ratio"}
    store.set("live_draft", store.put_json(compatibility))
    original = store.read(rendered.latex_ref)
    store.set("live_frozen", {"fixture": "synthetic frozen method"})
    result = execute_paired_final(runner, "selected", adapter.baseline(), run_pair=synthetic_pair)
    store.set("live_final", {"selected": result})
    before = runner.budget.summary()
    operation_ref = store.db.execute("SELECT result_ref FROM live_operations WHERE key='final/selected'").fetchone()[0]
    final = final_results_appendix(store, rendered, {"selected": result})
    manifest = store.json(final.manifest_ref)
    summary_ref = manifest["final_result_refs"]["selected"]
    assert summary_ref != operation_ref
    binding = verify_final_result_binding(store, "selected", result, summary_ref=summary_ref)
    assert binding["kind"] == "paired_observation_summary"
    assert runner.budget.summary() == before
    assert store.json(operation_ref)["phase"] == "final"
    assert store.read(rendered.latex_ref) == original
    assert _manuscripts(store, store, runner.config)[-1]["status"] == "pass"
    assert audit_live(store.root)["status"] == "pass"
    altered = deepcopy(result)
    altered["score"] = 42
    with pytest.raises(IntegrityError, match="summary fields"):
        verify_final_result_binding(store, "selected", altered)
    altered = deepcopy(result)
    altered["observation_ref"] = ledger["observations"][observed.id]["observation_ref"]
    with pytest.raises(IntegrityError, match="different committed"):
        verify_final_result_binding(store, "selected", altered)


def test_expanded_timing_fixture_cites_bounded_representative_evidence_without_mutating_ledger(paired):
    from scientisttwo.research.writer_comparisons import register_writer_comparisons
    store, adapter, runner = paired
    baseline = execute_paired(runner, "baseline/full", adapter.baseline(), seeds=[0], run_pair=synthetic_pair)
    observations = [baseline]
    for index in range(10):
        candidate = adapter.baseline()
        candidate.files[0].content += f"\n# SYNTHETIC timing proposal {index}\n"
        observations.append(execute_paired(runner, f"candidate-{index}", candidate, seeds=[0], run_pair=synthetic_pair))
    ledger = build_evidence_ledger(store, observations, timing_evidence="paired-timing-measurements-v1")
    ledger, _ = register_writer_comparisons(store, ledger, {"full": baseline})
    before = canonical(ledger)
    manuscript = fixture_manuscript(ledger)
    assert len(ledger["entries"]) > 256
    assert len(manuscript.claims) == 128
    assert validate_manuscript(manuscript, ledger).status == "pass"
    primary_ids = {key for key, entry in ledger["entries"].items()
                   if entry["kind"] == "measured" and entry["metric"] == "latency_ratio"}
    assert primary_ids <= {key for claim in manuscript.claims for key in claim.evidence_ids}
    assert "unchanged complete evidence ledger" in manuscript.limitations.text
    reversed_ledger = {**ledger, "entries": dict(reversed(list(ledger["entries"].items())))}
    assert fixture_manuscript(reversed_ledger) == manuscript
    assert canonical(ledger) == before
    # Presentation projection shrinks; original reports still contain quartiles.
    current = build_evidence_ledger(store, observations)
    current, _ = register_writer_comparisons(store, current, {"full": baseline})
    assert len(canonical(current)) < len(before) * .7
    report = store.json(store.json(baseline.timing_refs[0])["report_ref"])
    assert "q1_seconds" in report["baseline"] and "q3_seconds" in report["candidate"]


def test_fixture_table_and_figure_selection_is_bounded_and_prefers_cited_items():
    ledger = {"entries": {"measurement:primary": {"kind": "measured", "value": .5,
        "metric": "accuracy", "observation_id": "primary"}}, "tables": {}, "figures": {}}
    for index in range(200):
        key = f"table:{index:03}"
        ledger["tables"][key] = {"evidence_ids": ["measurement:primary"] if index == 199 else []}
        ledger["figures"][f"figure:{index:03}"] = {"table_id": key}
    before = canonical(ledger)
    manuscript = fixture_manuscript(ledger)
    assert len(manuscript.table_ids) == len(manuscript.figure_ids) == 128
    assert "table:199" in manuscript.table_ids and "figure:199" in manuscript.figure_ids
    assert canonical(ledger) == before
