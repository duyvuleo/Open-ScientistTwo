import os
from pathlib import Path
import tempfile
import time

import pytest

from scientisttwo.candidates import baseline_idea, source_for
from scientisttwo.evaluation import quality_matches
from scientisttwo.models import EvalSpec, Profile, TaskSpec
from scientisttwo.process import run_process
from scientisttwo.runner import ContainerRunner, evaluator_identity
from scientisttwo.store import Store
from scientisttwo.tasks import ClassificationTask


@pytest.mark.container
def test_real_container_reproduction_and_label_isolation():
    context = os.environ.get("SCIENTISTTWO_CONTAINER_CONTEXT")
    if not context:
        pytest.skip("Set SCIENTISTTWO_CONTAINER_CONTEXT to opt in to real container tests")
    cache = Path.cwd() / ".cache"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        store = Store(Path(temporary))
        store.set("evaluator", evaluator_identity())
        store.set("deadline", time.time() + 120)
        runner = ContainerRunner(store, Profile(), adapter=ClassificationTask(), context=context)
        spec = EvalSpec.for_phase(
            TaskSpec(datasets=["iris"], subset_datasets=["iris"], seeds=[0]), "full"
        )
        idea = baseline_idea()
        a = runner.execute("first", idea, source_for(idea), spec)
        b = runner.execute("second", idea, source_for(idea), spec)
        assert a.status == b.status == "success"
        assert quality_matches(a, b, 1e-8)
        assert a.rows[0].training_seconds > 0
        check = run_process(
            [
                "docker",
                "--context",
                context,
                "run",
                "--rm",
                "--network=none",
                runner.image_id,
                "python",
                "-c",
                "from sklearn.datasets import load_iris; load_iris()",
            ],
            30,
        )
        assert check.code != 0 and any(
            error in check.stderr for error in ("FileNotFoundError", "ModuleNotFoundError")
        )
        store.close()
