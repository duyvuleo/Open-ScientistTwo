import os
from pathlib import Path
import tempfile

import pytest

from upstream_checkouts import require_upstream

from scientisttwo.benchmarks.contract_fixtures import make_fixture, make_candidate, write_csv
from scientisttwo.benchmarks.manifest import create_spec
from scientisttwo.benchmarks.runner import NativeRunner


@pytest.mark.native_container
def test_actual_official_graders_multifile_candidate_and_invalid_submission():
    context = os.environ.get("SCIENTISTTWO_NATIVE_CONTEXT")
    if not context:
        pytest.skip("Set SCIENTISTTWO_NATIVE_CONTEXT to opt in to native benchmark container tests")
    project = Path(__file__).resolve().parents[1]
    mla = require_upstream("mlagentbench")
    mle = require_upstream("mle-bench")
    cache = project / ".cache"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        root = Path(temporary)
        data = make_fixture(root / "data", "detecting-insults-in-social-commentary")
        make_fixture(root / "data", "spooky-author-identification")
        candidate = make_candidate(root / "candidate")
        bad = root / "bad"
        bad.mkdir()
        write_csv(bad / "submission.csv", ["Wrong"], [["invalid"]])
        cases = [
            (
                create_spec("mlagentbench", "vectorization", mla, context=context, seconds=120),
                "completed",
                None,
            ),
            (
                create_spec(
                    "mle-bench",
                    "detecting-insults-in-social-commentary",
                    mle,
                    data_root=data,
                    provenance="contract_fixture",
                    context=context,
                ),
                "completed",
                0.5,
            ),
            (
                create_spec(
                    "mle-bench",
                    "detecting-insults-in-social-commentary",
                    mle,
                    data_root=data,
                    candidate=candidate,
                    command=["python", "train.py"],
                    mode="prepared_candidate",
                    provenance="contract_fixture",
                    context=context,
                ),
                "completed",
                1.0,
            ),
            (
                create_spec(
                    "mle-bench",
                    "spooky-author-identification",
                    mle,
                    data_root=data,
                    provenance="contract_fixture",
                    context=context,
                ),
                "completed",
                1.09861,
            ),
            (
                create_spec(
                    "mle-bench",
                    "detecting-insults-in-social-commentary",
                    mle,
                    data_root=data,
                    candidate=bad,
                    mode="grade_only",
                    provenance="contract_fixture",
                    context=context,
                ),
                "invalid_submission",
                None,
            ),
        ]
        for index, (spec, status, score) in enumerate(cases):
            runner = NativeRunner(spec, root / f"run-{index}")
            try:
                result = runner.run()
                assert result.status == status
                if score is not None:
                    assert result.score == pytest.approx(score)
                elif status == "completed":
                    assert result.score > 0
                assert result.live_model_calls == 0 and not result.agent_evaluation
                assert not result.leaderboard_eligible
                if spec.benchmark == "mle-bench":
                    assert not result.benchmark_data_verified
                assert not runner.store.verify()
                before = runner.store.db.execute("SELECT count(*) FROM artifacts").fetchone()[0]
                if status == "completed":
                    assert runner.run() == result
                    assert (
                        runner.store.db.execute("SELECT count(*) FROM artifacts").fetchone()[0]
                        == before
                    )
            finally:
                runner.close()
