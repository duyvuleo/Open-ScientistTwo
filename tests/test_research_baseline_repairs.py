"""Synthetic reconstructed-baseline failure and bounded-repair workflow traces."""

from copy import deepcopy
from pathlib import Path

import pytest

from scientisttwo.live.models import Candidate, SourceFile
from scientisttwo.process import ProcessResult
from scientisttwo.research.controller import ResearchController

from test_research_workflow import ResearchStagedTransport, campaign, with_offset

BROKEN = "SYNTHETIC_BROKEN_REFERENCE = True"
REPAIRED = "# Synthetic baseline engineering repair"


def change_reference(value, *, broken):
    candidate = Candidate.model_validate(value)
    files = []
    for item in candidate.files:
        if item.path == "custom.py":
            content = "\n".join(line for line in item.content.splitlines() if line != BROKEN)
            content += "\n" + (BROKEN if broken else REPAIRED) + "\n"
            item = item.model_copy(update={"content": content})
        files.append(item)
    return candidate.model_copy(update={"files": files})


class FailingReferenceTransport(ResearchStagedTransport):
    def __call__(self, argv, timeout, **kwargs):
        if "run" in argv:
            mounts = {value.rsplit(":", 2)[1]: Path(value.rsplit(":", 2)[0])
                      for index, value in enumerate(argv) if index and argv[index - 1] == "-v"}
            if "/workspace" in mounts and BROKEN in (mounts["/workspace"] / "custom.py").read_text():
                self.stages.append(("synthetic-failed-reference", mounts))
                return ProcessResult(1, "", "Explicit synthetic baseline failure; fixture marker present", .1)
        return super().__call__(argv, timeout, **kwargs)


def repair_campaign(tmp_path, *, mode="reconstruct_from_sources", disposition="repair"):
    controller, store, runner, agents, _ = campaign(tmp_path, baseline_mode=mode)
    transport = FailingReferenceTransport()
    runner.raw_executor = transport
    original = agents.responder
    if mode == "prepared_reference":
        baseline = controller.adapter.baseline
        controller.adapter.baseline = lambda: change_reference(baseline(), broken=True)

    def respond(role, inputs):
        if role == "baseline_coder":
            agents.seen.append((role, deepcopy(inputs)))
            return change_reference(inputs["reference"], broken=True).model_dump()
        if role == "baseline_engineer":
            agents.seen.append((role, deepcopy(inputs)))
            assert inputs["feedback"]["observation"]["status"] == "execution_failed"
            assert "synthetic baseline failure" in str(inputs["feedback"]["execution_logs"])
            candidate = change_reference(inputs["candidate"], broken=disposition == "still-broken")
            if disposition == "command":
                candidate = candidate.model_copy(update={"command": ["python", "alternate.py"]})
            if disposition == "scope":
                candidate = candidate.model_copy(update={"files": [*candidate.files, SourceFile(path="undeclared.py", content="x = 1\n")]})
            return candidate.model_dump()
        return original(role, inputs)

    agents.responder = respond
    if disposition == "command":
        # Let the controller's command-invariance check own this fixture branch.
        controller.adapter.validate = lambda candidate: None
    return controller, store, runner, agents, transport


def test_reconstructed_baseline_repair_preserves_failure_and_replaces_reference(tmp_path):
    controller, store, _, agents, _ = repair_campaign(tmp_path)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        failed = store.json(store.get("baseline-attempt/baseline/subset"))
        repaired = store.json(store.get("baseline-attempt/baseline/subset/repair/1"))
        assert failed["status"] == "execution_failed" and failed["value"] is None
        assert repaired["status"] == "success"
        repaired_code = store.json(repaired["code_ref"])
        assert REPAIRED in str(repaired_code) and BROKEN not in str(repaired_code)
        assert controller.baselines["subset"].id == "baseline/subset/repair/1"
        assert REPAIRED in str(store.json(controller.baselines["full"].code_ref))
        assert REPAIRED in str(store.get("live_frozen")["baseline"])
        critic_inputs = [inputs for role, inputs in agents.seen if role == "critic"]
        assert critic_inputs
        assert all("baseline/subset/repair/1" in str(inputs["evidence_package"])
                   for inputs in critic_inputs if inputs["evidence"]["observation"]["phase"] == "subset")
        assert any(role == "full_set_coder" for role, _ in agents.seen)
        assert sum(role == "baseline_engineer" for role, _ in agents.seen) == 1
        novelty = [inputs["idea"]["title"] for role, inputs in agents.seen if role == "novelty_checker"]
        assert novelty == ["fixture-seed-0", "fixture-seed-1"]
        operations = controller.budget.summary()
        baseline_attempts = [row for row in operations if row["kind"] == "experiment" and row["key"].startswith("experiment/baseline/subset")]
        assert len(baseline_attempts) == 2 and all(row["seconds"] > 0 for row in baseline_attempts)
        assert any(row["key"] == "model/baseline/subset/repair-code/0" for row in operations)
    finally:
        store.close()


def test_exhausted_baseline_repairs_are_explicit_and_do_not_repeat_on_resume(tmp_path):
    controller, store, _, agents, transport = repair_campaign(tmp_path, disposition="still-broken")
    try:
        state = controller.run()
        assert state["outcome"] == "baseline_failed", state
        assert controller.config.engineering_retries == 1
        operations = controller.budget.summary()
        attempts = [row for row in operations if row["kind"] == "experiment"]
        assert len(attempts) == 2
        assert all(row["status"] == "execution_failed" for row in attempts)
        assert sum(role == "baseline_engineer" for role, _ in agents.seen) == 1
        assert not any(role == "implement" for role, _ in agents.seen)
        calls, stages = len(agents.seen), len(transport.stages)
        assert controller.run() == state
        assert controller.budget.summary() == operations
        assert len(agents.seen) == calls and len(transport.stages) == stages
    finally:
        store.close()


@pytest.mark.parametrize("disposition", ["scope", "command"])
def test_baseline_engineering_cannot_change_command_or_undeclared_sources(tmp_path, disposition):
    controller, store, _, agents, _ = repair_campaign(tmp_path, disposition=disposition)
    try:
        state = controller.run()
        assert state["status"] == "blocked" and state["outcome"] == "invalid_agent_output", state
        assert "scope" in state["reason"]
        operations = controller.budget.summary()
        assert len([row for row in operations if row["kind"] == "experiment"]) == 1
        assert any(role == "baseline_engineer" for role, _ in agents.seen)
        assert store.get("live_frozen") is None
    finally:
        store.close()


def test_prepared_reference_failure_never_enters_baseline_coding_or_repair(tmp_path):
    controller, store, _, agents, _ = repair_campaign(tmp_path, mode="prepared_reference")
    try:
        state = controller.run()
        assert state["outcome"] == "baseline_failed", state
        roles = [role for role, _ in agents.seen]
        assert "baseline_plan" in roles
        assert "baseline_coder" not in roles and "baseline_engineer" not in roles
        operations = controller.budget.summary()
        assert len([row for row in operations if row["kind"] == "experiment"]) == 1
        failure = store.json(store.get("baseline-attempt/baseline/subset"))
        assert BROKEN in str(store.json(failure["code_ref"]))
        assert store.get("live_frozen") is None
    finally:
        store.close()


class FullOnlyFailureTransport(ResearchStagedTransport):
    """Synthetic transient full failure, optionally followed by alignment failure."""
    def __init__(self, *, fail_alignment=False):
        super().__init__()
        self.full_failed = False
        self.fail_alignment = fail_alignment

    def __call__(self, argv, timeout, **kwargs):
        mounts = {value.rsplit(":", 2)[1]: Path(value.rsplit(":", 2)[0])
                  for index, value in enumerate(argv) if index and argv[index - 1] == "-v"}
        if "/workspace" in mounts and "/data/public" in mounts:
            phase = mounts["/data/public"].parent.name
            if phase == "full" and not self.full_failed:
                self.full_failed = True
                self.stages.append(("synthetic-full-failure", mounts))
                return ProcessResult(1, "", "Synthetic full-only reference failure", .1)
            if phase == "subset" and self.full_failed and self.fail_alignment:
                self.stages.append(("synthetic-alignment-failure", mounts))
                return ProcessResult(1, "", "Synthetic aligned-subset failure", .1)
        return super().__call__(argv, timeout, **kwargs)


def full_repair_campaign(tmp_path, *, fail_alignment=False, description_only=False):
    controller, store, runner, agents, _ = campaign(tmp_path, baseline_mode="reconstruct_from_sources")
    transport = FullOnlyFailureTransport(fail_alignment=fail_alignment)
    runner.raw_executor = transport
    original = agents.responder

    def respond(role, inputs):
        if role == "baseline_engineer":
            agents.seen.append((role, deepcopy(inputs)))
            candidate = Candidate.model_validate(inputs["candidate"])
            if not description_only:
                candidate = with_offset(candidate, .8)
            return candidate.model_copy(update={"description": "Synthetic repaired full-stage reference"}).model_dump()
        return original(role, inputs)

    agents.responder = respond
    return controller, store, runner, agents, transport


def test_full_repair_realigns_subset_before_any_search(tmp_path):
    controller, store, _, agents, _ = full_repair_campaign(tmp_path)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        aligned = controller.baselines["subset"]
        full = controller.baselines["full"]
        assert aligned.id == "baseline/subset/aligned-with-full"
        assert aligned.value == pytest.approx(.64) and full.value == pytest.approx(.64)
        assert store.json(aligned.source_tree_ref)["bundle_ref"] == store.json(full.source_tree_ref)["bundle_ref"]
        proof = store.json(store.get("baseline_alignment"))
        assert proof["status"] == "pass"
        assert proof["original_subset_id"] == "baseline/subset"
        assert proof["full_id"] == "baseline/full/repair/1"
        assert proof["aligned_subset_id"] == aligned.id
        assert proof["changed_paths"] == ["custom.py"]
        old = store.json(store.get("baseline-attempt/baseline/subset"))
        assert old["value"] == 1.
        assert store.json(store.get("baseline-attempt/baseline/full"))["status"] == "execution_failed"
        for role, inputs in agents.seen:
            if role == "critic" and inputs["evidence"]["observation"]["phase"] == "subset":
                assert aligned.id in str(inputs["evidence_package"])
        keys = [row["key"] for row in controller.budget.summary()]
        assert keys.index("experiment/" + aligned.id) < keys.index("model/search/0/0/implement/0")
        assert sum(role == "baseline_engineer" for role, _ in agents.seen) == 1
    finally:
        store.close()


def test_alignment_failure_is_blocked_without_repairing_or_replacing_full(tmp_path):
    controller, store, _, agents, transport = full_repair_campaign(tmp_path, fail_alignment=True)
    try:
        state = controller.run()
        assert state["status"] == "blocked" and state["outcome"] == "baseline_alignment_failed", state
        assert store.json(store.get("baseline_alignment"))["status"] == "failed"
        assert controller.baselines["full"].value == pytest.approx(.64)
        assert sum(role == "baseline_engineer" for role, _ in agents.seen) == 1
        assert not any(role == "implement" for role, _ in agents.seen)
        assert store.get("live_frozen") is None
        operations, calls, stages = controller.budget.summary(), len(agents.seen), len(transport.stages)
        assert controller.run() == state
        assert controller.budget.summary() == operations
        assert len(agents.seen) == calls and len(transport.stages) == stages
    finally:
        store.close()


def test_reference_description_change_does_not_reexecute_subset(tmp_path):
    controller, store, _, _, _ = full_repair_campaign(tmp_path, description_only=True)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        assert controller.baselines["subset"].id == "baseline/subset"
        assert store.json(store.get("baseline_alignment"))["status"] == "not_required"
        assert not any("aligned-with-full" in row["key"] for row in controller.budget.summary())
    finally:
        store.close()


def test_committed_alignment_replays_without_duplicate_execution(tmp_path):
    controller, store, runner, agents, transport = full_repair_campaign(tmp_path)
    original = runner.execute
    interrupted = False

    def interrupt(key, *args, **kwargs):
        nonlocal interrupted
        result = original(key, *args, **kwargs)
        if key == "baseline/subset/aligned-with-full" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return result

    runner.execute = interrupt
    try:
        with pytest.raises(KeyboardInterrupt):
            controller.run()
        assert store.json(store.get("baseline_alignment"))["status"] == "pending"
        before = controller.budget.summary()
        resumed = ResearchController(store, controller.config, controller.adapter, agents,
                                     runner, budget=controller.budget)
        state = resumed.run()
        assert state["status"] == "complete", state
        assert store.json(store.get("baseline_alignment"))["status"] == "pass"
        assert resumed.budget.summary()[:len(before)] == before
        assert sum(row["key"] == "experiment/baseline/subset/aligned-with-full"
                   for row in resumed.budget.summary()) == 1
        assert sum(role == "baseline_engineer" for role, _ in agents.seen) == 1
    finally:
        store.close()
