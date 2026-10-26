"""Bounded writer contracts use scripted responses and fake PDF transports only."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from scientisttwo.backends import BackendError
from scientisttwo.integrations.paperorchestra import controller_writer as bridge
from scientisttwo.integrations.paperorchestra.runtime import ModelRequest
from scientisttwo.live.output_contracts import AgentOutputError
from scientisttwo.models import AgentResult
from scientisttwo.research.policy import ResearchPolicy
from scientisttwo.store import IntegrityError, canonical, digest

import test_paperorchestra_controller_writer as writer_tests
from test_paperorchestra_pdf import FakeContainer, IMAGE

context = writer_tests.context


def request(controller, role="outline", kind="json"):
    return ModelRequest(role=role, prompt="Unchanged upstream prompt with its required schema and evidence IDs.",
                        model_name=controller.config.model, response_kind=kind)


def transport(context, *, key="bounded", cap=None):
    controller, ledger, _ = context
    controller.config.workflow.paperorchestra_max_response_chars = cap
    return bridge.ControllerTransport(controller, key, ledger, SimpleNamespace(synthetic=True))


def scripted_output(controller, text):
    calls = []

    def invoke(role, inputs, schema):
        calls.append({"role": role, "inputs": deepcopy(inputs), "schema": schema.model_json_schema()})
        return AgentResult(payload={"output": text}, backend=controller.backend.name,
                           model=controller.config.model, synthetic=True,
                           usage={"input_tokens": 0, "output_tokens": 0})

    controller.backend.invoke = invoke
    return calls


def test_none_preserves_prior_response_schema_and_prompt(context):
    controller, _, _ = context
    writer = transport(context)
    calls = scripted_output(controller, '{"complete":true}')
    result = writer(request(controller))
    assert result.raw_text == '{"complete":true}'
    assert writer.output_schema is bridge.PaperOrchestraResponse
    assert calls[0]["schema"]["properties"]["output"]["maxLength"] == 250_000
    assert "response_size_policy" not in calls[0]["inputs"]
    assert not controller.store.get("paperorchestra-response-policy/bounded/upstream/0001")
    assert ResearchPolicy().paperorchestra_max_response_chars is None


@pytest.mark.parametrize("cap", [0, -1, 250001, True, "10000"])
def test_response_character_budget_requires_explicit_valid_integer(cap):
    with pytest.raises((ValidationError, ValueError)):
        ResearchPolicy(paperorchestra_max_response_chars=cap)
    with pytest.raises(ValueError):
        bridge.response_size_policy(cap)


@pytest.mark.parametrize("role,kind", [("outline", "json"), ("section_writing", "latex"),
    ("content_refinement", "text"), ("peer_review", "json"), ("format_review", "json")])
def test_prompt_schema_and_artifact_share_one_frozen_budget(context, role, kind):
    controller, _, _ = context
    writer = transport(context, cap=10000)
    calls = scripted_output(controller, '{"complete":true,"evidence":"measurement:example"}')
    original = request(controller, role, kind)
    result = writer(original)
    assert result.raw_text == '{"complete":true,"evidence":"measurement:example"}'
    supplied = calls[0]
    policy = supplied["inputs"]["response_size_policy"]
    assert policy["max_output_characters"] == supplied["schema"]["properties"]["output"]["maxLength"] == 10000
    assert policy["field"] == "output"
    assert policy["truncate_response"] is False and policy["automatic_retry"] is False
    assert policy["provider_token_cap"] is False
    assert "10000" in supplied["inputs"]["instruction"]
    assert supplied["inputs"]["instruction"].startswith("Controller output-character budget")
    assert supplied["inputs"]["response_instruction"].startswith("Controller output-character budget")
    assert supplied["inputs"]["prompt"] == original.prompt
    assert "evidence and citation identifiers" in policy["preserve"]
    assert "numerical measurements" in policy["preserve"] and "required protocol facts" in policy["preserve"]
    proof = controller.store.json(controller.store.get("paperorchestra-response-policy/bounded/upstream/0001"))
    assert proof["policy"] == policy and proof["upstream_role"] == role
    assert proof["upstream_prompt_sha256"] == digest(original.prompt.encode())
    assert proof["output_schema_sha256"] == digest(canonical(supplied["schema"]))
    # The durable controller request carries the same schema, not an out-of-band hint.
    stored = controller.store.json(controller.store.get("request/bounded/upstream/0001"))
    assert stored["schema"] == supplied["schema"]


def test_fences_and_inner_json_escaping_count_toward_the_limit(context):
    controller, _, _ = context
    text = '```json\n' + json.dumps({"literal": '\\value "quoted"', "symbol": "🧪"}, ensure_ascii=False) + '\n```'
    writer = transport(context, cap=len(text))
    calls = scripted_output(controller, text)
    assert writer(request(controller)).raw_text == text
    assert len(calls) == 1
    assert writer.output_schema.model_validate({"output": text}).output == text
    with pytest.raises(ValidationError):
        writer.output_schema.model_validate({"output": text + " "})


def test_oversize_fails_closed_without_truncation_or_output_correction(context):
    controller, _, _ = context
    controller.config.agent_output_retries = 1  # Schema failures must not use this allowance.
    text = "x" * 81
    writer = transport(context, cap=80)
    calls = scripted_output(controller, text)
    with pytest.raises(BackendError, match="retry limit exhausted"):
        writer(request(controller))
    assert len(calls) == 1 and text == "x" * 81
    operations = controller.budget.summary()
    assert len(operations) == 1 and operations[0]["status"] == "failed"
    assert not controller.store.db.execute("SELECT 1 FROM calls WHERE key LIKE 'live-agent/%'").fetchone()
    assert writer.error is not None


def test_bounded_mode_rejects_automatic_transport_retry_configuration(context):
    controller, _, _ = context
    controller.config.model_retries = 1
    with pytest.raises(ValueError, match="model_retries=0"):
        transport(context, cap=10000)
    assert controller.budget.summary() == [] and controller.backend.roles == []
    # Opting out does not change the original configuration contract.
    assert transport(context, cap=None).output_schema is bridge.PaperOrchestraResponse


def test_partial_attempt_reuses_exact_bounded_response_without_second_call(context):
    controller, _, _ = context
    calls = scripted_output(controller, '{"complete":true}')
    original = request(controller)
    first = transport(context, cap=10000)(original)
    before = controller.budget.summary()
    replay = transport(context, cap=10000)(original)
    assert first == replay and len(calls) == 1 and controller.budget.summary() == before
    with pytest.raises(IntegrityError, match="response-size policy changed"):
        transport(context, cap=12000)(original)
    assert len(calls) == 1 and controller.budget.summary() == before


def test_complete_writer_records_adapter_deviation_and_cached_result(context):
    controller, ledger, inputs = context
    controller.config.workflow.paperorchestra_max_response_chars = 10000
    pdf = bridge.ContainerPdfTools(IMAGE, executor=FakeContainer())
    first = bridge.write_controller_manuscript(controller, "draft", inputs, ledger, pdf_tools=pdf)
    report = controller.store.json(first.evidence_ref)
    assert report["controller_identity"]["response_size_policy"]["max_output_characters"] == 10000
    assert report["controller_identity"]["response_size_policy"]["provider_token_cap"] is False
    before = controller.budget.summary(), list(controller.backend.roles), len(pdf.receipts)
    assert bridge.write_controller_manuscript(controller, "draft", inputs, ledger, pdf_tools=pdf) == first
    assert before == (controller.budget.summary(), controller.backend.roles, len(pdf.receipts))
    controller.config.workflow.paperorchestra_max_response_chars = 12000
    with pytest.raises(IntegrityError, match="inputs changed"):
        bridge.write_controller_manuscript(controller, "draft", inputs, ledger, pdf_tools=pdf)
    assert before == (controller.budget.summary(), controller.backend.roles, len(pdf.receipts))


@pytest.mark.parametrize("text", ['{"unfinished":', "Cannot complete the required artifact within this character budget."])
def test_malformed_inner_json_or_refusal_aborts_without_repair_call(context, text):
    controller, ledger, inputs = context
    controller.config.workflow.paperorchestra_max_response_chars = 10000
    calls = scripted_output(controller, text)
    with pytest.raises(AgentOutputError, match="PaperOrchestra writing failed"):
        bridge.write_controller_manuscript(controller, "malformed", inputs, ledger,
                                           pdf_tools=bridge.ContainerPdfTools(IMAGE, executor=FakeContainer()))
    assert len(calls) == 1
    assert sum(row["kind"] == "model" for row in controller.budget.summary()) == 1
    assert controller.store.get("paperorchestra-attempt/malformed/0001")
    assert not controller.store.db.execute("SELECT 1 FROM calls WHERE key='paperorchestra/controller/malformed'").fetchone()
