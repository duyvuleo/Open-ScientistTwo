"""Complete controller contracts on synthetic API and experiment transports.

No HTTP, model or container is run. Fixture predictions and provider responses
exercise workflow/accounting contracts, not live research quality or benchmarks.
"""
from copy import deepcopy
import json

import pytest

from scientisttwo.backends import BackendError, StructuredOutputCodec
from scientisttwo.live.models import Hypothesis, LiveSpec
from scientisttwo.live.runtime import Budget, DevelopmentRunner
from scientisttwo.process import ProcessResult
from scientisttwo.providers.audit import inspect_api_ledger, verify_api_accounting
from scientisttwo.providers.backend import APIBackend
from scientisttwo.providers.config import ModelProvider
from scientisttwo.research.controller import ResearchController
from scientisttwo.research.manuscript import ManuscriptRecord, verify_evidence_ledger
from scientisttwo.research.minimal_live_acceptance import capture_resume_snapshot
from scientisttwo.store import canonical, digest

from test_provider_audit import Answer, api_campaign as api_campaign
from test_research_workflow import MARKER, campaign
from test_structured_map_codec import ROLE_SCHEMAS


def _assert_supplied_values_preserved(supplied, parsed):
    """Schema defaults may be added; supplied fixture values may not transform."""
    if isinstance(supplied, dict):
        assert isinstance(parsed, dict)
        for key, value in supplied.items():
            _assert_supplied_values_preserved(value, parsed[key])
    elif isinstance(supplied, list):
        assert isinstance(parsed, list) and len(parsed) == len(supplied)
        for value, normalized in zip(supplied, parsed, strict=True):
            _assert_supplied_values_preserved(value, normalized)
    else:
        assert supplied == parsed


def _wire_value(codec, node, value):
    """Encode fixture domain values into the same required wire-map format."""
    if "$ref" in node:
        return _wire_value(codec, codec._resolve(node["$ref"]), value)
    if codec._mapping(node):
        return canonical(value).decode()
    if "anyOf" in node:
        for alternative in node["anyOf"]:
            try:
                encoded = _wire_value(codec, alternative, value)
                codec._decode(alternative, encoded)
                return encoded
            except (ValueError, KeyError, TypeError):
                continue
        raise AssertionError("Fixture value matches no declared wire alternative")
    if node.get("type") == "object":
        return {key: _wire_value(codec, child, value[key])
                for key, child in node["properties"].items()}
    if node.get("type") == "array":
        return [_wire_value(codec, node["items"], item) for item in value]
    return value


class ScriptedProviderTransport:
    """Supply fixture JSON to the genuine backend's decode/validation boundary."""
    def __init__(self, store, agents):
        self.store, self.agents, self.calls = store, agents, []

    def __call__(self, argv, timeout, **kwargs):
        try:
            return self._respond(argv, timeout, **kwargs)
        except Exception as error:
            pytest.fail(f"Synthetic provider fixture failed: {error!r}")

    def _respond(self, argv, timeout, **kwargs):
        assert argv[-1] == "scientisttwo.providers.worker"
        request = json.loads(kwargs["stdin"])
        prompt = json.loads(request["prompt"])
        role, inputs = prompt["role"], deepcopy(prompt["inputs"])
        inputs.pop("_scientisttwo_wire_contract")
        pending = inspect_api_ledger(self.store)["records"][-1]
        assert pending["receipt"] is None
        schema = pending["request"]["controller_request"]["schema"]
        candidates = [model for model in ROLE_SCHEMAS[role] if model.model_json_schema() == schema]
        assert len(candidates) == 1, role
        value = self.agents.respond(role, inputs)
        if role in {"initial_drafter", "draft_enhancer"}:
            # Exercise the API backend's declaration contract. The test/fixtures
            # remain explicitly synthetic; no live model claim is made here.
            value["synthetic_agents"] = inputs["synthetic_agents"]
        parsed = candidates[0].model_validate(value).model_dump(mode="json")
        _assert_supplied_values_preserved(value, parsed)
        codec = StructuredOutputCodec(schema)
        assert request["wire_schema"] == codec.wire_schema
        output = canonical(_wire_value(codec, schema, parsed)).decode()
        assert codec.decode_json(output) == parsed
        self.calls.append((role, candidates[0].__name__))
        receipt = {"version": "provider-http-v1", "status": "completed", "output_text": output,
            "usage": {"input_tokens": 10, "output_tokens": 2}, "usage_known": True,
            "request_sent": True, "http_status": 200, "reported_model": "synthetic-model@fixture",
            "response_id": "fixture-response-" + str(len(self.calls)),
            "response_sha256": digest(output.encode()), "error_code": None}
        return ProcessResult(0, canonical(receipt).decode(), "", .001)


def test_complete_api_research_cycle_refines_reviews_accounts_and_resumes(tmp_path):
    original, store, _, agents, experiment_transport = campaign(
        tmp_path, refine=True, rebuttal=True, editorial=True,
        baseline_mode="reconstruct_from_sources")
    values = original.config.model_dump(mode="json")
    values.update(model="synthetic-api-contract-model", search_rounds=2, success_target=4,
                  token_stop_threshold=1_000_000)
    values["workflow"]["seed_ideas"] = 3
    values["provider"] = ModelProvider(kind="openai_compatible", base_url="http://localhost:8000/v1",
                                       max_output_tokens=32000).model_dump(mode="json")
    config = LiveSpec.model_validate(values)
    legacy = agents.legacy_response

    def scripted_legacy(role, inputs):
        if role == "evolve":
            return {"idea": Hypothesis(title="synthetic-evolved-offset",
                hypothesis="Reduce the synthetic prediction offset using recorded attempts",
                novelty_score=0, novelty_scope="Synthetic control; no novelty claim",
                source_ids=["task"]).model_dump(),
                "change_plan": {"changed_files": ["custom.py"], "reason": "Synthetic method evolution"}}
        value = legacy(role, inputs)
        if role == "implement" and "parent" in inputs:
            before = value["edits"][0]["before"]
            offset = float(next(line.removeprefix(MARKER) for line in before.splitlines()
                                if line.startswith(MARKER)))
            value["edits"][0]["after"] = "\n".join(
                line for line in before.splitlines() if not line.startswith(MARKER)) + f"\n{MARKER}{offset / 2}\n"
        return value

    agents.legacy_response = scripted_legacy
    budget = Budget(store, config)
    budget.remaining()
    runner = DevelopmentRunner(store, config, original.adapter, budget, executor=experiment_transport)
    transport = ScriptedProviderTransport(store, agents)
    backend = APIBackend(store, config, allow_live=True, process_executor=transport)
    controller = ResearchController(store, config, original.adapter, backend, runner, budget=budget)
    try:
        state = controller.run()
        assert state["status"] == "complete", state
        assert state["outcome"] == "development_improvement"
        roles = {role for role, _ in transport.calls}
        assert {"limitations", "verify_limitations", "initial_idea", "additional_idea", "novelty_checker",
            "baseline_plan", "baseline_coder", "implement", "critic", "evolve", "full_set_coder",
            "selection_review", "ablation_planner", "ablation_coder", "control_reviewer",
            "ablation_critic", "method_disposition", "refine", "initial_drafter", "draft_enhancer",
            "writing_critic", "scientific_review", "rebuttal_planner", "rebuttal_coder", "meta"} <= roles
        assert {"Candidate", "ParentPatch", "MethodChange", "Hypothesis", "ManuscriptRecord",
                "ScientificReview", "WritingCriticReview"} <= {schema for _, schema in transport.calls}
        assert any(role == "draft_enhancer" and "manuscript_revision" in inputs for role, inputs in agents.seen)
        assert store.get("live_frozen") and store.get("live_final")["selected"]["status"] == "completed"
        assert "FINAL_STAGED_LABELS_MUST_NOT_ENTER_REFINEMENT" not in str(agents.seen)
        manuscript = ManuscriptRecord.model_validate(store.json(store.get("research_manuscript")))
        assert manuscript.concern_responses and manuscript.claims
        ledger = store.json(store.json(store.get("research_manuscript_bundle"))["ledger_ref"])
        verify_evidence_ledger(store, ledger)
        assert ledger["includes_final"] is False
        assert not store.verify()
        store.set("api_model_accounting_ref", store.put_json(backend.accounting(), "api-accounting"))
        accounting = verify_api_accounting(store, config)
        assert accounting["status"] == "pass" and accounting["calls"] == len(transport.calls)
        assert accounting["observed_tokens"] == accounting["controller_accounted_tokens"] == 12 * len(transport.calls)
        assert accounting["unknown_calls"] == 0
        before = capture_resume_snapshot(store)
        stages, calls = len(experiment_transport.stages), len(transport.calls)
        resumed = ResearchController(store, config, original.adapter, backend, runner, budget=budget)
        assert resumed.run() == state
        assert capture_resume_snapshot(store) == before
        assert len(experiment_transport.stages) == stages and len(transport.calls) == calls
        assert verify_api_accounting(store, config) == accounting
    finally:
        store.close()


@pytest.mark.parametrize("api_campaign", ["openai"], indirect=True)
@pytest.mark.parametrize("escaped", [False, True])
def test_controller_rejects_selected_credential_before_request_persistence(api_campaign, escaped):
    item = api_campaign
    credential = "synthetic-audit-credential"
    payload = json.dumps({"nested": credential})
    if escaped:
        payload = payload.replace("credential", "cred\\u0065ntial")
    before = {str(path.relative_to(item.store.root)): digest(path.read_bytes())
              for path in item.store.root.rglob("*") if path.is_file() and path.suffix not in {".sqlite", "-wal", "-shm"}}
    artifact_count = item.store.db.execute("SELECT count(*) FROM artifacts").fetchone()[0]
    with pytest.raises(BackendError, match="credential"):
        item.controller.ask("must-not-store-credential", "critic", {"nested_json": payload}, Answer)
    assert not item.sent
    assert item.store.db.execute("SELECT count(*) FROM live_operations").fetchone()[0] == 0
    assert item.store.db.execute("SELECT count(*) FROM artifacts").fetchone()[0] == artifact_count
    assert not any(credential.encode() in path.read_bytes() for path in item.store.root.rglob("*") if path.is_file())
    after = {str(path.relative_to(item.store.root)): digest(path.read_bytes())
             for path in item.store.root.rglob("*") if path.is_file() and path.suffix not in {".sqlite", "-wal", "-shm"}}
    assert after == before
