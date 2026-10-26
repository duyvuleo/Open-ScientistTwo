"""Real isolated portable tasks and scripted agents; no model endpoint is invoked."""

import json
import os
from pathlib import Path
import re
import tempfile
import pytest
from scientisttwo.live.adapters import implementation_identity, load_development
from scientisttwo.live.audit import audit_live
from scientisttwo.live.controller import LiveController
from scientisttwo.live.evaluation import compare_observations
from scientisttwo.live.models import LiveSpec
from scientisttwo.live.runtime import Budget, DevelopmentRunner
from scientisttwo.live.workspace_task import WorkspaceRunner, freeze_workspace
from scientisttwo.store import Store
from test_workspace_task import (
    FINAL_CANARY,
    ResearchFixtures,
    classification_candidate,
    development,
    install_plugin,
    make_workspace,
)


@pytest.fixture
def isolated_workspace(request):
    context = os.environ.get("SCIENTISTTWO_NATIVE_CONTEXT")
    if not context:
        pytest.skip("Set SCIENTISTTWO_NATIVE_CONTEXT to opt into isolated execution")
    image = os.environ.get("SCIENTISTTWO_WORKSPACE_IMAGE", "scientisttwo-native:0.1")
    output = os.environ.get("SCIENTISTTWO_GENERALIZATION_OUTPUT")
    if output:
        root = Path(output).resolve() / re.sub(r"[^a-zA-Z0-9_-]", "-", request.node.name)
        root.mkdir(parents=True, exist_ok=False)
        yield root, context, image
    else:
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="workspace-validation-", dir=cache) as temp:
            yield Path(temp), context, image


@pytest.mark.native_container
@pytest.mark.parametrize(
    "kind,meta_improves",
    [
        ("regression", True),
        ("regression", False),
        ("classification", False),
        ("text", False),
    ],
)
def test_real_workspace_full_research(isolated_workspace, monkeypatch, kind, meta_improves):
    root, context, image = isolated_workspace
    native = make_workspace(root / "inputs", kind=kind, context=context, image=image)
    install_plugin(monkeypatch)
    config = LiveSpec(
        native=native,
        development=development(kind),
        model="scripted-fixture-v1",
        mode="research",
        max_calls=100,
        campaign_seconds=600,
        max_experiments=100,
        model_retries=0,
        search_rounds=2,
        engineering_retries=0,
        ablation_refinements=1,
        review_rounds=1,
        meta_refinements=1,
        implementation=implementation_identity(),
    )
    adapter = load_development(native, config.development)
    backend = ResearchFixtures(kind=kind, meta_improves=meta_improves)
    store = Store(root / "campaign")
    budget = Budget(store, config)
    runner = DevelopmentRunner(store, config, adapter, budget)
    controller = LiveController(store, config, adapter, backend, runner, budget=budget)
    try:
        result = controller.run()
        assert result["status"] == "complete", result
        assert result["outcome"] == "development_improvement", result
        roles = {role for role, _ in backend.seen}
        assert {
            "evolve",
            "ablations",
            "ablation_critic",
            "refine",
            "draft",
            "review",
            "rebuttal",
            "meta",
        } <= roles
        assert FINAL_CANARY not in json.dumps(backend.seen)
        trials = [
            store.json(row[0])
            for row in store.db.execute(
                "SELECT output_ref FROM calls WHERE key LIKE 'live-trial/%'"
            )
        ]
        assert any(t["decision"]["decision"] == "Bad" for t in trials)
        assert all(t["result"]["status"] == "success" for t in trials)
        best = store.get("live_best")
        meta_trial = next(t for t in trials if t["id"] == "meta/0")
        if kind == "regression":
            assert any(
                "ablation/0/revision" in t["id"] and t["decision"]["decision"] == "Good"
                for t in trials
            )
            assert len(best["result"]["artifact_refs"][0]) == 2
            if meta_improves:
                assert meta_trial["decision"]["decision"] == "Good"
                assert best["id"] == "meta/0"
                assert best["result"]["value"] < 1e-8
                assert backend.meta_count == 2
            else:
                assert meta_trial["decision"]["decision"] == "Bad"
                assert "ablation/0/revision" in best["id"]
                assert best["result"]["value"] < meta_trial["result"]["value"]
        else:
            assert meta_trial["decision"]["decision"] == "Bad"
            assert best["id"] != "meta/0"
            assert {"accuracy", "macro_f1"} == set(best["result"]["metrics"])
        final = json.loads((store.root / "final" / "selected" / "result.json").read_text())
        assert final["status"] == "completed" and final["live_model_calls"] == 0
        audit = audit_live(store.root)
        assert audit["status"] == "pass", audit
        operations, calls = len(budget.summary()), len(backend.seen)
        assert controller.run() == result
        assert len(budget.summary()) == operations and len(backend.seen) == calls
        summary = json.loads((store.root / "summary.json").read_text())
        assert summary["synthetic_agents"] and summary["live_model_calls"] == 0
        evidence = {
            "scope": "Synthetic fixture agents and task data; actual isolated CPU predictions and metrics",
            "kind": kind,
            "meta_improves": meta_improves,
            "state": result,
            "roles": sorted(roles),
            "model_calls": 0,
            "fixture_role_invocations": calls,
            "best": best["result"],
            "final": final,
            "audit": audit,
            "operation_count": operations,
            "resume_added_operations": 0,
        }
        (root / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    finally:
        store.close()


@pytest.mark.native_container
def test_real_classification_guardrail_veto(isolated_workspace):
    root, context, image = isolated_workspace
    native = make_workspace(root / "inputs", kind="classification", context=context, image=image)
    config = LiveSpec(
        native=native,
        development=development("classification"),
        model="scripted-fixture-v1",
        max_calls=1,
        campaign_seconds=120,
        implementation=implementation_identity(),
    )
    adapter = load_development(native, config.development)
    store = Store(root / "campaign")
    budget = Budget(store, config)
    runner = DevelopmentRunner(store, config, adapter, budget)
    try:
        runner.preflight()
        baseline = runner.execute("baseline", adapter.baseline(), "full")
        candidate = runner.execute("tradeoff", classification_candidate("guardrail"), "full")
        assert baseline.status == candidate.status == "success"
        result = compare_observations(config, candidate, baseline)
        assert result["delta"] > 0, result
        assert not result["improved"] and not result["valid"]
        assert "accuracy exceeds max_degradation" in result["constraint_failures"]
        (root / "verification.json").write_text(
            json.dumps(
                {
                    "scope": "Actual isolated classifier predictions; primary gain vetoed by measured guardrail",
                    "baseline": baseline.model_dump(),
                    "candidate": candidate.model_dump(),
                    "comparison": result,
                    "model_calls": 0,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        store.close()


@pytest.mark.native_container
def test_real_non_python_shell_candidate(isolated_workspace):
    root, context, image = isolated_workspace
    spec = make_workspace(root / "inputs", context=context, image=image)
    (spec.candidate_dir / "fit.py").unlink()  # Only this test's newly created fixture source.
    source = """set -eu
test ! -e /data/private
test ! -e /evaluator
awk -F, 'NR==FNR {if(FNR>1) {n++; sx+=$1; sy+=$2; sxx+=$1*$1; sxy+=$1*$2} next}
FNR==1 {print "target"; next}
{b=(n*sxy-sx*sy)/(n*sxx-sx*sx); a=(sy-b*sx)/n; printf "%.12g\\n", a+b*$1}' /data/public/train.csv /data/public/test.csv > forecast.csv
"""
    (spec.candidate_dir / "fit.sh").write_text(source)
    spec = freeze_workspace({**spec.model_dump(mode="json"), "command": ["sh", "fit.sh"]})
    runner = WorkspaceRunner(spec, root / "campaign")
    try:
        result = runner.run()
        assert result.status == "completed", result
        assert result.score < 1e-8
        assert result.live_model_calls == 0
        (root / "verification.json").write_text(
            json.dumps(
                {
                    "scope": "Real shell/awk train-fitted regression in isolated container; no host candidate execution",
                    "result": result.model_dump(mode="json"),
                    "model_calls": 0,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        runner.close()
