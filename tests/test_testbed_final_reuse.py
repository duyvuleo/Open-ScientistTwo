from types import SimpleNamespace

from scientisttwo.live.models import Candidate, SourceFile
from scientisttwo.store import Store
from scientisttwo.testbed.validation import _final_once


def candidate(value, *, description="Synthetic source-only contract fixture", command=None):
    return Candidate(files=[SourceFile(path="train.py", content=f"# source fixture {value}\n")],
                     command=command or ["python", "train.py"], description=description)


def test_completed_final_reused_for_exact_source_even_description_differs(tmp_path):
    store = Store(tmp_path)
    baseline, selected = candidate(1), candidate(2)
    baseline_result, selected_result = {"status": "completed", "synthetic_test": "baseline"}, {"status": "completed", "synthetic_test": "selected"}
    store.set("live_frozen", {"baseline": baseline.model_dump(), "best": {"candidate": selected.model_dump()}})
    store.set("live_final", {"baseline": baseline_result, "selected": selected_result})
    calls = []
    runner = SimpleNamespace(store=store, final=lambda *args: calls.append(args))
    try:
        assert _final_once(runner, "reference", candidate(1, description="Different prose")) == (baseline_result, "final/baseline")
        assert _final_once(runner, "control", candidate(2)) == (selected_result, "final/selected")
        assert calls == []
    finally:
        store.close()


def test_different_final_source_or_argv_is_not_aliased(tmp_path):
    store = Store(tmp_path)
    baseline = candidate(1)
    store.set("live_frozen", {"baseline": baseline.model_dump(), "best": {"candidate": baseline.model_dump()}})
    store.set("live_final", {"baseline": {"status": "completed", "synthetic_test": True}})
    calls = []
    def final(key, code):
        calls.append(key)
        return {"status": "synthetic_contract_only"}
    runner = SimpleNamespace(store=store, final=final)
    try:
        _final_once(runner, "control", candidate(2))
        _final_once(runner, "control", candidate(1, command=["python", "train.py", "changed"]))
        assert calls == ["validation-control", "validation-control"]
    finally:
        store.close()
