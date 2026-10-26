"""Offline provider-schema and lossless wire-map checks; CLI executables are fake."""
from copy import deepcopy
import hashlib
import json
import sys

from pydantic import Field, ValidationError
import pytest

from scientisttwo.backends import (
    BackendError, CodexBackend, StructuredOutputCodec, STRUCTURED_PROMPT_VERSION,
    strict_json_loads, structured_schema, STRUCTURED_WIRE_CONTRACT_KEY,
)
from scientisttwo.live.models import (
    Candidate, ControlPlans, Draft, Hypothesis, LimitationAssessment, LiveMetaReview,
    LiveReview, Revision, Seeds, SourceFile,
)
from scientisttwo.live.parent_edits import MethodChange, ParentPatch
from scientisttwo.live.supplementary import SupplementaryPlans
from scientisttwo.models import CriticDecision, Record, Refinement
from scientisttwo.research.contracts import (
    AblationDesigns, BaselinePlan, ControlReview, FullSetHandoff, MethodDisposition,
    SelectionReview, SupplementaryDesigns,
)
from scientisttwo.research.controller import PROMPTS
from scientisttwo.research.literature import NoveltyAssessment
from scientisttwo.research.manuscript import ManuscriptRecord
from scientisttwo.research.reviews import ScientificReview, WritingCriticReview
from scientisttwo.integrations.paperorchestra.controller_writer import PaperOrchestraResponse
from scientisttwo.integrations.paperorchestra.manuscript_binding import PaperOrchestraBindings
from scientisttwo.store import canonical


ROLE_SCHEMAS = {
    "limitations": [LimitationAssessment], "verify_limitations": [LimitationAssessment],
    "seeds": [Seeds], "implement": [Candidate, ParentPatch], "critic": [CriticDecision],
    "engineer": [Candidate, ParentPatch], "evolve": [MethodChange], "refine": [MethodChange],
    "ablations": [ControlPlans], "ablation_critic": [Refinement], "draft": [Draft],
    "review": [LiveReview], "rebuttal": [SupplementaryPlans], "meta": [LiveMetaReview],
    "initial_idea": [Hypothesis], "additional_idea": [Hypothesis],
    "novelty_checker": [NoveltyAssessment], "baseline_plan": [BaselinePlan],
    "baseline_coder": [Candidate], "baseline_engineer": [Candidate],
    "full_set_coder": [FullSetHandoff], "selection_review": [SelectionReview],
    "ablation_planner": [AblationDesigns], "ablation_coder": [ParentPatch],
    "control_reviewer": [ControlReview], "method_disposition": [MethodDisposition],
    "initial_drafter": [ManuscriptRecord], "writing_critic": [WritingCriticReview],
    "draft_enhancer": [ManuscriptRecord], "scientific_review": [ScientificReview],
    "rebuttal_coder": [ParentPatch], "rebuttal_planner": [SupplementaryDesigns],
    "paperorchestra_model": [PaperOrchestraResponse],
    "paperorchestra_bindings": [PaperOrchestraBindings],
}


def provider_contract(node):
    assert isinstance(node, dict)
    assert "default" not in node
    if node.get("type") == "object":
        assert isinstance(node["properties"], dict)
        assert node["additionalProperties"] is False
        assert set(node["required"]) == set(node["properties"])
        assert len(node["required"]) == len(node["properties"])
    for keyword in ("properties", "$defs", "definitions"):
        for child in node.get(keyword, {}).values():
            provider_contract(child)
    if "items" in node:
        provider_contract(node["items"])
    for child in node.get("anyOf", []):
        provider_contract(child)


def idea():
    return Hypothesis(title="Synthetic idea", hypothesis="Test source contracts only",
                      novelty_score=0, novelty_scope="No novelty claim", source_ids=["task"],
                      adapter_payload={"default": {"type": "json", "value": [1, True, None]},
                                       "unicode": "研究", "looks_like_json": '{"do_not_parse":true}',
                                       "nested": {"empty": {}, "precision": 0.125}})


def fake_cli(tmp_path, raw_output):
    path = tmp_path / "fake-codex"
    capture = tmp_path / "captured-wire-input.json"
    path.write_text(f'''#!{sys.executable}
import json,pathlib,sys
args=sys.argv
assert args[args.index('--model')+1]=='fake-model'
assert '--strict-config' in args and '--output-schema' in args
prompt=json.load(sys.stdin)
schema=json.loads(pathlib.Path(args[args.index('--output-schema')+1]).read_text())
assert prompt['prompt_version']=='scientisttwo-v1'
assert 'controller owns execution' in prompt['instructions']
pathlib.Path({str(capture)!r}).write_text(json.dumps({{'schema':schema,'prompt':prompt}}))
raw={raw_output!r}
pathlib.Path(args[args.index('--output-last-message')+1]).write_bytes(raw.encode())
print(json.dumps({{'type':'item.completed','item':{{'type':'agent_message','text':raw}}}}))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':7,'output_tokens':11}}}}))
''')
    path.chmod(0o700)
    return str(path), capture


def test_all_current_research_roles_have_an_explicit_schema_test():
    assert set(ROLE_SCHEMAS) == set(PROMPTS)


@pytest.mark.parametrize("role,model", [(role, model) for role, models in ROLE_SCHEMAS.items() for model in models])
def test_every_research_role_emits_closed_provider_objects_without_mutating_domain(role, model):
    original = model.model_json_schema()
    before = deepcopy(original)
    wire = structured_schema(original)
    provider_contract(wire)
    assert original == before
    assert wire["type"] == "object"
    assert role in PROMPTS


@pytest.mark.parametrize("model,role", [(Hypothesis, "initial_idea"), (Seeds, "seeds"), (Revision, "refine")])
def test_fake_cli_roundtrip_preserves_nonempty_maps_and_actual_wire_events(tmp_path, model, role):
    proposed = idea()
    if model is Hypothesis:
        domain = proposed.model_dump(mode="json")
        wire = deepcopy(domain)
        wire["adapter_payload"] = json.dumps(domain["adapter_payload"], ensure_ascii=False)
    elif model is Seeds:
        domain = Seeds(ideas=[proposed]).model_dump(mode="json")
        wire = deepcopy(domain)
        wire["ideas"][0]["adapter_payload"] = json.dumps(domain["ideas"][0]["adapter_payload"], ensure_ascii=False)
    else:
        domain = Revision(idea=proposed, candidate=Candidate(
            files=[SourceFile(path="train.py", content="# synthetic; never executed\n")],
            command=["python", "train.py"], description="Source fixture")).model_dump(mode="json")
        wire = deepcopy(domain)
        wire["idea"]["adapter_payload"] = json.dumps(domain["idea"]["adapter_payload"], ensure_ascii=False)
    raw = json.dumps(wire, ensure_ascii=False, indent=2).replace("\n", "\r\n")
    cli, capture = fake_cli(tmp_path, raw)
    backend = CodexBackend(tmp_path / "unused", "fake-model", executable=cli, allow_live=True)
    result = backend.invoke(role, {"purpose": "Explicit offline fake CLI test"}, model)
    assert result.payload == domain
    assert result.prompt_version == STRUCTURED_PROMPT_VERSION
    assert result.usage == {"input_tokens": 7, "output_tokens": 11}
    shown = json.loads(capture.read_text())
    assert shown["schema"] == StructuredOutputCodec(model.model_json_schema()).wire_schema
    provider_contract(shown["schema"])
    assert shown["prompt"]["role"] == role
    contract = shown["prompt"]["inputs"][STRUCTURED_WIRE_CONTRACT_KEY]
    assert contract["version"] == "json-map-wire-v1"
    assert contract["relay_envelope_version"] == shown["prompt"]["prompt_version"] == "scientisttwo-v1"
    assert "JSON strings containing one object" in contract["instruction"]
    assert backend.events[0]["item"]["text"] == raw
    receipt = backend.events[-1]
    assert receipt["origin"] == "local_backend"
    assert receipt["codec_version"] == "json-map-wire-v1"
    assert receipt["wire_output_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert receipt["decoded_output_sha256"] == hashlib.sha256(canonical(domain)).hexdigest()
    assert receipt["validated_payload_sha256"] == hashlib.sha256(canonical(domain)).hexdigest()


class TypedPayload(Record):
    mapping: dict[str, list[int]] | None
    rows: list[dict[str, int]]


def test_nested_array_ref_nullable_and_typed_mapping_constraints_are_preserved():
    class Envelope(Record):
        value: TypedPayload
    codec = StructuredOutputCodec(Envelope.model_json_schema())
    value = {"value": {"mapping": '{"a":[1,2]}', "rows": ['{"x":3}', '{}']}}
    decoded = codec.decode_json(json.dumps(value))
    assert Envelope.model_validate(decoded).model_dump() == {
        "value": {"mapping": {"a": [1, 2]}, "rows": [{"x": 3}, {}]}}
    value["value"]["mapping"] = None
    assert codec.decode_json(json.dumps(value))["value"]["mapping"] is None
    value["value"]["rows"] = ['{"x":"invalid integer"}']
    with pytest.raises(ValidationError):
        Envelope.model_validate(codec.decode_json(json.dumps(value)))


def test_schema_keyword_field_names_are_literal_properties_not_schema_nodes():
    schema = {"type": "object", "additionalProperties": False, "properties": {
        name: {"type": "string", "default": "fixture"}
        for name in ("default", "type", "properties", "$defs", "items", "anyOf")},
        "examples": [{"default": "literal example", "type": "not a schema"}]}
    original = deepcopy(schema)
    codec = StructuredOutputCodec(schema)
    assert codec.wire_schema["required"] == list(schema["properties"])
    assert set(codec.wire_schema["properties"]) == set(schema["properties"])
    assert codec.wire_schema["examples"] == schema["examples"]
    assert all("default" not in field for field in codec.wire_schema["properties"].values())
    assert codec.decode_json(json.dumps({name: "kept" for name in schema["properties"]})) == {
        name: "kept" for name in schema["properties"]}
    assert schema == original


def test_true_empty_closed_object_has_explicit_properties_and_accepts_only_empty_object():
    codec = StructuredOutputCodec({"type": "object", "additionalProperties": False})
    assert codec.wire_schema == {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    assert codec.decode_json('{}') == {}
    with pytest.raises(ValueError):
        codec.decode_json('{"not_empty":1}')


@pytest.mark.parametrize("bad", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}',
                                 '{"x":-Infinity}', '{"x":1e999}', '{"x":{"a":1,"a":2}}'])
def test_strict_json_rejects_duplicate_and_nonfinite_values(bad):
    with pytest.raises(ValueError):
        strict_json_loads(bad)
    codec = StructuredOutputCodec(Hypothesis.model_json_schema())
    wire = idea().model_dump(mode="json")
    wire["adapter_payload"] = bad
    with pytest.raises(ValueError):
        codec.decode_json(json.dumps(wire))


@pytest.mark.parametrize("bad", ['not JSON', '[]', 'null', '7', '"a JSON string"'])
def test_mapping_wire_string_must_decode_exactly_once_to_object(bad):
    codec = StructuredOutputCodec(Hypothesis.model_json_schema())
    wire = idea().model_dump(mode="json")
    wire["adapter_payload"] = bad
    with pytest.raises(ValueError):
        codec.decode_json(json.dumps(wire))


def test_recursive_model_refs_preserve_each_map_boundary():
    class Node(Record):
        label: str
        attributes: dict = Field(default_factory=dict)
        children: list["Node"] = Field(default_factory=list)
    Node.model_rebuild()
    codec = StructuredOutputCodec(Node.model_json_schema())
    provider_contract(codec.wire_schema)
    raw = {"label": "root", "attributes": '{"nested":{"retain":true}}',
           "children": [{"label": "leaf", "attributes": '{}', "children": []}]}
    decoded = codec.decode_json(json.dumps(raw))
    assert Node.model_validate(decoded).attributes == {"nested": {"retain": True}}
    assert decoded["children"][0]["attributes"] == {}


def test_direct_map_definition_reference_decodes_with_literal_json_pointer_escapes():
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"payload": {"$ref": "#/$defs/Map~1with~0name"}},
              "$defs": {"Map/with~name": {"type": "object", "additionalProperties": {"type": "integer"}}}}
    codec = StructuredOutputCodec(schema)
    assert codec.wire_schema["$defs"]["Map/with~name"]["type"] == "string"
    assert codec.decode_json('{"payload":"{\\"key\\":4}"}') == {"payload": {"key": 4}}


def test_ambiguous_mapping_string_union_fails_before_executor(tmp_path):
    class Ambiguous(Record):
        value: dict | str
    dispatched = []
    backend = CodexBackend(tmp_path, "fake-model", allow_live=True,
                           executor=lambda *args, **kwargs: dispatched.append(args))
    with pytest.raises(BackendError, match="no model request"):
        backend.invoke("initial_idea", {}, Ambiguous)
    assert dispatched == []


@pytest.mark.parametrize("bad_map", ['{"duplicate":1,"duplicate":2}', '{"value":NaN}',
                                     '{"value":1e999}', '[]'])
def test_fake_cli_invalid_mapping_response_is_rejected(tmp_path, bad_map):
    wire = idea().model_dump(mode="json")
    wire["adapter_payload"] = bad_map
    cli, _ = fake_cli(tmp_path, json.dumps(wire))
    backend = CodexBackend(tmp_path, "fake-model", executable=cli, allow_live=True)
    with pytest.raises(BackendError, match="invalid structured"):
        backend.invoke("initial_idea", {}, Hypothesis)
    assert all(event.get("type") != "scientisttwo.structured_output_codec" for event in backend.events)


def test_fake_cli_original_pydantic_constraints_still_reject_decoded_mapping(tmp_path):
    raw = json.dumps({"mapping": '{"a":["not an integer"]}', "rows": []})
    cli, _ = fake_cli(tmp_path, raw)
    backend = CodexBackend(tmp_path, "fake-model", executable=cli, allow_live=True)
    with pytest.raises(BackendError, match="invalid structured"):
        backend.invoke("synthetic", {}, TypedPayload)


def test_reserved_wire_metadata_collision_rejects_before_dispatch(tmp_path):
    inputs = {STRUCTURED_WIRE_CONTRACT_KEY: {"user_supplied": True}}
    previous = deepcopy(inputs)
    calls = []
    backend = CodexBackend(tmp_path, "fake-model", allow_live=True,
                           executor=lambda *args, **kwargs: calls.append(args))
    with pytest.raises(BackendError, match="Reserved structured-output metadata"):
        backend.invoke("critic", inputs, CriticDecision)
    assert inputs == previous and calls == []
