"""Fake Codex executable plus real isolated CPU experiments; no inference requests."""

import ast
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd
import pytest

from upstream_checkouts import require_upstream

from scientisttwo.backends import CodexBackend
from scientisttwo.benchmarks.contract_fixtures import make_fixture
from scientisttwo.benchmarks.manifest import create_spec
from scientisttwo.live.adapters import implementation_identity, load_development
from scientisttwo.live.audit import audit_live
from scientisttwo.live.controller import LiveController
from scientisttwo.live.models import Candidate, DevelopmentSpec, LiveSpec, SourceFile
from scientisttwo.live.runtime import Budget, DevelopmentRunner
from scientisttwo.store import Store


class RecordedCodex(CodexBackend):
    synthetic = True


def vector_candidate(source):
    tree = ast.parse(source)
    body = ast.parse("""
pad = self.padding if isinstance(self.padding, int) else 0
x = np.pad(features_batch, ((0,0),(pad,pad),(pad,pad),(0,0)))
k = self.kernel_matrices
windows = np.lib.stride_tricks.sliding_window_view(x, k.shape[:2], axis=(1,2))[:, ::self.stride, ::self.stride]
out = np.einsum('nhwcij,ijco->nhwo', windows, k, optimize=True) + self.biases
return np.maximum(out, 0) if self.activation == 'relu' else out
""").body
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "Conv2DLayer":
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and method.name == "forward":
                    method.body = body
    return Candidate(
        files=[SourceFile(path="train.py", content=ast.unparse(tree))],
        command=["python", "train.py"],
        description="Known NumPy vectorization fixture; not model generated",
    )


def fake_codex(path, proposal):
    replies = {
        "limitations": {
            "limitations": ["Known fixture limitation"],
            "relevant": True,
            "reason": "Fixture",
        },
        "verify_limitations": {
            "limitations": ["Known fixture limitation"],
            "relevant": True,
            "reason": "Fixture",
        },
        "seeds": {
            "ideas": [
                {
                    "title": "Known fixture method",
                    "hypothesis": "Exercise live transport",
                    "novelty_score": 0,
                    "novelty_scope": "No novelty claim",
                    "source_ids": ["task"],
                }
            ]
        },
        "implement": proposal.model_dump(),
        "critic": {"decision": "Good", "reason": "Controller must enforce measured comparison"},
    }
    path.write_text(f"""#!{sys.executable}
import json, pathlib, sys
argv = sys.argv
assert '--ignore-user-config' in argv and '--strict-config' in argv
assert '--disable' in argv and 'shell_tool' in argv
assert argv[argv.index('--sandbox')+1] == 'read-only'
request = json.load(sys.stdin)
assert 'private' not in json.dumps(request['inputs']).lower()
replies = {replies!r}
pathlib.Path(argv[argv.index('--output-last-message')+1]).write_text(json.dumps(replies[request['role']]))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':10,'output_tokens':10}}}}))
""")
    path.chmod(0o755)
    return str(path)


@pytest.mark.native_container
@pytest.mark.parametrize("proposal_mode", ["full_file", "forward_body"])
def test_fake_codex_real_vectorization_campaign(proposal_mode):
    context = os.environ.get("SCIENTISTTWO_NATIVE_CONTEXT")
    if not context:
        pytest.skip("Set SCIENTISTTWO_NATIVE_CONTEXT to opt in")
    project = Path(__file__).resolve().parents[1]
    repo = require_upstream("mlagentbench")
    cache = project / ".cache"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temp:
        root = Path(temp)
        native = create_spec("mlagentbench", "vectorization", repo, context=context, seconds=120)
        config = LiveSpec(
            native=native,
            development=DevelopmentSpec(
                seeds=[0], epsilon=0, settings={"proposal_mode": proposal_mode}
            ),
            model="recorded-test-cli",
            mode="smoke",
            max_calls=10,
            campaign_seconds=300,
            search_rounds=1,
            implementation=implementation_identity(),
        )
        adapter = load_development(native, config.development)
        proposal = vector_candidate(adapter.source)
        if proposal_mode == "forward_body":
            cls = next(
                n
                for n in ast.parse(proposal.files[0].content).body
                if isinstance(n, ast.ClassDef) and n.name == "Conv2DLayer"
            )
            method = next(
                n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
            )
            body = ast.unparse(ast.Module(body=method.body, type_ignores=[]))
            proposal = proposal.model_copy(
                update={"files": [SourceFile(path="train.py", content=body)]}
            )
        materialized = adapter.materialize(proposal)
        adapter.validate(materialized)
        backend = RecordedCodex(
            root / "agent",
            config.model,
            executable=fake_codex(root / "fake-codex", proposal),
            allow_live=True,
        )
        store = Store(root / "run")
        budget = Budget(store, config)
        runner = DevelopmentRunner(store, config, adapter, budget)
        controller = LiveController(store, config, adapter, backend, runner, budget=budget)
        try:
            result = controller.run()
            assert result["status"] == "complete", result
            assert result["outcome"] in {"no_improvement", "development_improvement"}
            # Timing noise may reject a method; correctness must still be measured.
            trials = [
                store.json(r[0])
                for r in store.db.execute(
                    "SELECT output_ref FROM calls WHERE key LIKE 'live-trial/%'"
                )
            ]
            assert trials and all(t["result"]["status"] == "success" for t in trials)
            summary = json.loads((store.root / "summary.json").read_text())
            assert summary["synthetic_agents"] and summary["live_model_calls"] == 0
            assert audit_live(store.root)["status"] == "pass"
            if proposal_mode == "forward_body":
                evidence = store.json(
                    store.db.execute(
                        "SELECT output_ref FROM calls WHERE key='materialization/development/search/0/0/subset/0'"
                    ).fetchone()[0]
                )
                assert store.json(evidence["proposal_ref"]) == proposal.model_dump()
                assert store.json(evidence["code_ref"]) == materialized.model_dump()
                assert evidence["proposal_ref"] != evidence["code_ref"]
            counts = len(budget.summary())
            assert controller.run() == result
            assert len(budget.summary()) == counts
        finally:
            store.close()


@pytest.mark.native_container
@pytest.mark.parametrize(
    "task_id", ["detecting-insults-in-social-commentary", "spooky-author-identification"]
)
def test_real_csv_development_and_final_native_grader(task_id):
    context = os.environ.get("SCIENTISTTWO_NATIVE_CONTEXT")
    if not context:
        pytest.skip("Set SCIENTISTTWO_NATIVE_CONTEXT to opt in")
    project = Path(__file__).resolve().parents[1]
    repo = require_upstream("mle-bench")
    with tempfile.TemporaryDirectory(dir=project / ".cache") as temp:
        root = Path(temp)
        data = make_fixture(root / "data", task_id)
        train = data / task_id / "prepared/public/train.csv"
        if task_id.startswith("detecting"):
            pd.DataFrame(
                {
                    "Insult": [0, 1] * 40,
                    "Date": [""] * 80,
                    "Comment": [
                        f"{'moon' if i % 2 == 0 else 'sun'} training {i}" for i in range(80)
                    ],
                }
            ).to_csv(train, index=False)
            development = DevelopmentSpec(adapter="csv", metric="roc_auc", seeds=[0, 1])
        else:
            names = ["EAP", "HPL", "MWS"]
            pd.DataFrame(
                {
                    "id": [f"train-{i}" for i in range(90)],
                    "text": [f"fixture {names[i % 3]} training {i}" for i in range(90)],
                    "author": [names[i % 3] for i in range(90)],
                }
            ).to_csv(train, index=False)
            development = DevelopmentSpec(
                adapter="csv",
                metric="log_loss",
                target="author",
                text_column="text",
                classes=names,
                prediction_columns=names,
                seeds=[0, 1],
            )
        native = create_spec(
            "mle-bench",
            task_id,
            repo,
            data_root=data,
            provenance="contract_fixture",
            context=context,
            seconds=120,
        )
        config = LiveSpec(
            native=native,
            development=development,
            model="not-invoked",
            max_calls=1,
            campaign_seconds=300,
            implementation=implementation_identity(),
        )
        adapter = load_development(native, development)
        store = Store(root / "run")
        budget = Budget(store, config)
        runner = DevelopmentRunner(store, config, adapter, budget)
        try:
            runner.preflight()
            store.set("live_spec_ref", store.put_json(config))
            store.set("live_adapter_identity", adapter.identity())
            for phase in ("subset", "full"):
                result = runner.execute(phase, adapter.baseline(), phase)
                assert result.status == "success", result
            final = runner.final("baseline", adapter.baseline())
            assert final["status"] == "completed", final
            assert final["data_provenance"] == "contract_fixture"
            assert audit_live(store.root)["status"] == "pass"
        finally:
            store.close()
