"""Writer metadata and source-context contracts; all experiment results are synthetic."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scientisttwo.integrations.paperorchestra import controller_writer as bridge
from scientisttwo.integrations.paperorchestra.writer_context import (
    build_writer_context, context_method_fact, verify_context_method_fact, _python_literals,
)
from scientisttwo.integrations.paperorchestra.manuscript_binding import (
    PaperOrchestraBindings, SourceSectionBinding, SourceClaimBinding, SourceNumericBinding,
    bind_paperorchestra_manuscript, binding_prompt_context,
)
from scientisttwo.integrations.paperorchestra.runtime import ModelRequest
from scientisttwo.models import Profile, TaskSpec, AgentResult
from scientisttwo.research.manuscript import LedgerClaim, build_evidence_ledger, verify_evidence_ledger
from scientisttwo.research.pilot_bridge import create_pilot_campaign
from scientisttwo.store import Store, IntegrityError, canonical, digest

from test_research_pilot_bridge import SyntheticContainer
import test_paperorchestra_controller_writer as writer_tests

context = writer_tests.context


@pytest.fixture
def pilot(tmp_path):
    store = Store(tmp_path / "pilot")
    task = TaskSpec(name="synthetic-moons-metadata", datasets=["moons"], subset_datasets=["moons"],
                    data_seed=20261025, split_seed=23, settings={"private_fixture": "not-for-writer",
                        "dataset_metadata": {"moons": {"generator": "sklearn.datasets.make_moons",
                            "n_samples": 600, "n_features": 2, "noise_standard_deviation": 0.2,
                            "private_note": "not-for-writer"}}})
    worker = SyntheticContainer(store)
    _, adapter, runner = create_pilot_campaign(store, task, Profile(), container_runner=worker)
    observation = runner.execute("baseline/full", adapter.baseline(), "full")
    ledger = build_evidence_ledger(store, [observation])
    yield store, observation, ledger
    store.close()


def metadata_ledger(store, observation, ledger):
    fact = context_method_fact(store, ledger)
    result = build_evidence_ledger(store, [observation], method_facts={"writer_context": fact})
    return result, fact


def test_context_exports_complete_source_and_counts_without_protected_payloads(pilot, monkeypatch):
    store, observation, ledger = pilot
    original = store.json(observation.legacy_result_ref)
    read = store.read

    def guarded(ref):
        assert ref not in {original["observations_ref"], original["logs_ref"]}
        return read(ref)

    monkeypatch.setattr(store, "read", guarded)
    value = build_writer_context(store, ledger)
    assert value["task"]["fields"]["data_seed"] == 20261025
    assert value["task"]["dataset_metadata"]["moons"]["noise_standard_deviation"] == 0.2
    assert value["partition_counts"]["datasets"]["moons"] == {
        "total_rows": 600, "train_rows": 360, "development_rows": 120, "test_rows": 120}
    sources = value["execution_and_source"]["sources"]
    assert sources and all(file["text_complete"] for item in sources for file in item["files"])
    assert any("LogisticRegression" in file["text"] for item in sources for file in item["files"])
    text = canonical(value).decode()
    assert "not-for-writer" not in text and str(store.root) not in text
    assert '"x_train"' not in text and '"y_train"' not in text and '"row_ids"' not in text
    assert value["protocols"][observation.id]["evaluation"]["split"] == "development"
    assert value["execution_and_source"]["observations"][0]["training_executions"][0]["measurement_origin"] == "synthetic_test"


def test_context_rejects_final_inventory_and_mismatched_frozen_task(pilot):
    store, _, ledger = pilot
    with pytest.raises(IntegrityError, match="non-final"):
        build_writer_context(store, {**ledger, "includes_final": True})
    task = store.json(store.get("task_ref"))
    task["split_seed"] += 1
    store.set("task_ref", store.put_json(task))
    with pytest.raises(IntegrityError, match="frozen task"):
        build_writer_context(store, ledger)


def test_method_metadata_is_rederived_and_cannot_include_outcome_scores(pilot):
    store, observation, ledger = pilot
    ledger, fact = metadata_ledger(store, observation, ledger)
    verify_evidence_ledger(store, ledger)
    assert fact["numeric_metadata"]
    assert all(field["context_path"][0] in {"task", "partition_counts", "protocols", "declared_source_parameters"}
               for field in fact["numeric_metadata"])
    forged = deepcopy(ledger)
    forged["entries"]["method:writer_context"]["numeric_metadata"][0]["value"] += 1
    with pytest.raises(IntegrityError, match="numeric metadata"):
        verify_evidence_ledger(store, forged)


@pytest.mark.parametrize("field", ["context_sha256", "source_ref", "source_sha256", "source_path", "context_path"])
def test_method_metadata_forged_paths_and_hashes_fail(pilot, field):
    store, observation, ledger = pilot
    ledger, fact = metadata_ledger(store, observation, ledger)
    forged = deepcopy(fact)
    if field == "context_sha256":
        forged[field] = "0" * 64
    else:
        forged["numeric_metadata"][0][field] = ["metrics", "accuracy"] if field.endswith("path") else "forged"
    with pytest.raises(IntegrityError):
        verify_context_method_fact(store, forged, ledger["observations"])


def bound_metadata(ledger, *, section="protocol", claim_kind="method", path=None,
                   quote="The screening fraction is 0.5 of the training partition.", number_text="0.5"):
    latex = "\\documentclass{article}\n\\title{Typed protocol}\n\\begin{document}\n" + quote + "\n\\end{document}"
    start = latex.index(quote)
    end = latex.index(r"\end{document}")
    body_start = latex.index(r"\begin{document}") + len(r"\begin{document}")
    number = latex.index(number_text)
    evidence = "method:writer_context"
    bindings = PaperOrchestraBindings(latex_sha256=digest(latex.encode()), title="Typed protocol",
        sections=[SourceSectionBinding(start=body_start, end=end, quote=latex[body_start:end], section=section)],
        claims=[SourceClaimBinding(start=start, end=start+len(quote), quote=quote,
            claim=LedgerClaim(id="protocol-fraction", kind=claim_kind, text=quote, evidence_ids=[evidence]))],
        numeric_values=[SourceNumericBinding(start=number, end=number+len(number_text), quote=number_text, evidence_id=evidence,
            location="method_metadata", metadata_path=path or ["task", "fields", "subset_fraction"])])
    return latex, bindings


def test_nonempirical_protocol_decimal_binds_without_reclassifying_it_as_a_score(pilot):
    store, observation, ledger = pilot
    ledger, _ = metadata_ledger(store, observation, ledger)
    latex, bindings = bound_metadata(ledger)
    manuscript = bind_paperorchestra_manuscript(bindings, latex, ledger, synthetic_agents=True)
    assert manuscript.claims[0].kind == "method" and manuscript.claims[0].value is None
    offered = binding_prompt_context(latex, ledger)
    options = next(iter(offered["numeric_candidate_groups"].values()))["matching_candidates"]
    assert any(item.get("metadata_path") == ["task", "fields", "subset_fraction"] for item in options)


def test_partition_count_in_protocol_table_has_explicit_nonempirical_binding(pilot):
    store, observation, ledger = pilot
    ledger, _ = metadata_ledger(store, observation, ledger)
    latex, bindings = bound_metadata(ledger, quote=r"\begin{tabular}{lr}Total rows & 600 \\ \end{tabular}",
        number_text="600", path=["partition_counts", "datasets", "moons", "total_rows"])
    value = bind_paperorchestra_manuscript(bindings, latex, ledger, synthetic_agents=True)
    assert value.claims[0].kind == "method"
    bindings.sections[0].section = "results"
    with pytest.raises(ValueError, match="empirical result"):
        bind_paperorchestra_manuscript(bindings, latex, ledger, synthetic_agents=True)


@pytest.mark.parametrize("section,kind", [("results", "method"), ("abstract", "method"),
                                         ("protocol", "measured"), ("method", "derived")])
def test_metadata_cannot_validate_result_claims_or_unclassified_result_sections(pilot, section, kind):
    store, observation, ledger = pilot
    ledger, _ = metadata_ledger(store, observation, ledger)
    latex, bindings = bound_metadata(ledger, section=section, claim_kind=kind)
    with pytest.raises(ValueError, match="empirical result"):
        bind_paperorchestra_manuscript(bindings, latex, ledger, synthetic_agents=True)


def test_metadata_path_cannot_reach_execution_metrics(pilot):
    store, observation, ledger = pilot
    ledger, _ = metadata_ledger(store, observation, ledger)
    latex, bindings = bound_metadata(ledger, path=["execution_and_source", "observations", 0, "metrics", "accuracy"])
    with pytest.raises(ValueError, match="metadata path"):
        bind_paperorchestra_manuscript(bindings, latex, ledger, synthetic_agents=True)


def test_concise_scope_is_frozen_in_controller_request_and_cached_proof(context):
    controller, ledger, _ = context
    controller.config.workflow.paperorchestra_writing_scope = "concise_note"
    seen = []
    def invoke(role, inputs, schema):
        seen.append(inputs)
        return AgentResult(payload={"output": '{"complete":true}'}, backend=controller.backend.name,
            model=controller.config.model, synthetic=True, usage={"input_tokens": 0, "output_tokens": 0})
    controller.backend.invoke = invoke
    request = ModelRequest("outline", "Frozen upstream prompt", controller.config.model, "json")
    first = bridge.ControllerTransport(controller, "scope", ledger, SimpleNamespace(synthetic=True))(request)
    replay = bridge.ControllerTransport(controller, "scope", ledger, SimpleNamespace(synthetic=True))(request)
    assert first == replay and len(seen) == 1
    assert seen[0]["writing_scope"] == "concise_note"
    assert "soft target" in seen[0]["instruction"] and "method_protocol_context" in seen[0]
    proof = controller.store.json(controller.store.get("paperorchestra-writer-context/scope/upstream/0001"))
    assert proof["writing_scope"] == "concise_note" and proof["includes_final"] is False
    controller.config.workflow.paperorchestra_writing_scope = "upstream"
    with pytest.raises(IntegrityError, match="writer context changed"):
        bridge.ControllerTransport(controller, "scope", ledger, SimpleNamespace(synthetic=True))(request)
    assert len(seen) == 1


def test_literal_constructor_parameters_are_not_executed_or_inferred():
    code = '''from sklearn.svm import SVC as Model
import sklearn.ensemble as ensemble
from sklearn import neighbors
unknown = 0.2
def build(seed):
    marker = "🧪"; model = Model(C=0.1, gamma=unknown, random_state=seed)
    forest = ensemble.RandomForestClassifier(n_estimators=80, max_depth=8)
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    return model
'''
    found = _python_literals(code)
    assert [(item["keyword"], item["value"]) for item in found] == [
        ("C", .1), ("n_estimators", 80), ("max_depth", 8), ("n_neighbors", 5)]
    assert found[0]["literal_text"] == "0.1"
    assert found[0]["call_target"] == "sklearn.svm.SVC"
    assert not _python_literals('score = 0.1\n# SVC(C=0.1)\nfoo(C=0.1)')


@pytest.mark.parametrize("shadow", ["SVC = other", "def SVC(**kwargs): return None",
    "from private_module import SVC", "def f(SVC): return SVC(C=.1)"])
def test_shadowed_constructor_names_are_unavailable(shadow):
    assert not _python_literals('from sklearn.svm import SVC\n' + shadow + '\nmodel = SVC(C=.1)')


def test_unrepresentable_numeric_source_literal_is_not_method_metadata():
    assert not _python_literals('from sklearn.svm import SVC\nmodel = SVC(C=' + '9' * 500 + ')')


def test_declared_source_parameter_metadata_binds_to_exact_literal(pilot):
    store, observation, ledger = pilot
    ledger, fact = metadata_ledger(store, observation, ledger)
    context = store.json(fact["evidence_ref"])
    assert context["declared_source_parameters"]
    parameter_index = next(index for index, item in enumerate(context["declared_source_parameters"])
                           if item["keyword"] == "max_iter")
    parameter = context["declared_source_parameters"][parameter_index]
    assert parameter["value"] == 2000 and parameter["literal_text"] == "2000"
    latex, bindings = bound_metadata(ledger, quote="The declared iteration limit is 2000 in the source.",
        number_text="2000", path=["declared_source_parameters", parameter_index, "value"])
    bind_paperorchestra_manuscript(bindings, latex, ledger, synthetic_agents=True)
    forged = deepcopy(ledger)
    field = next(item for item in forged["entries"]["method:writer_context"]["numeric_metadata"]
                 if item["context_path"] == ["declared_source_parameters", parameter_index, "value"])
    field["source_path"][-1] += 1
    with pytest.raises(IntegrityError):
        verify_evidence_ledger(store, forged)


def test_dataset_noise_is_declared_metadata_not_an_outcome(pilot):
    store, observation, ledger = pilot
    ledger, _ = metadata_ledger(store, observation, ledger)
    latex, bindings = bound_metadata(ledger, quote="Declared generator noise is 0.2 in the task specification.",
        number_text="0.2", path=["task", "dataset_metadata", "moons", "noise_standard_deviation"])
    bind_paperorchestra_manuscript(bindings, latex, ledger, synthetic_agents=True)
    bindings.claims[0].claim.kind = "measured"
    with pytest.raises(ValueError, match="empirical result"):
        bind_paperorchestra_manuscript(bindings, latex, ledger, synthetic_agents=True)
