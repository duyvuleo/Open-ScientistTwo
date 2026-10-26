"""Provider routing/configuration contracts; no provider or container is contacted."""
from contextlib import closing
import json
import sys
from types import ModuleType

import pytest
from typer.testing import CliRunner

from scientisttwo.models import AgentResult, Profile, TaskSpec
from scientisttwo.providers.config import ModelProvider
from scientisttwo.live import cli, suite
from scientisttwo.live.adapters import implementation_identity
from scientisttwo.live.models import LiveSpec
from scientisttwo.live.runtime import AmbiguousCall
from scientisttwo.process import ProcessResult
from scientisttwo.research import live_pilot, live_pilot_cli
from scientisttwo.research.controller import ResearchController
from scientisttwo.research.policy import ResearchPolicy
from scientisttwo.store import IntegrityError, Store

from test_native_live_execution import native_config
from test_study_media_relay import Answer


def forbidden(*args, **kwargs):
    raise AssertionError("No real model, Codex process or experiment is permitted")


def api_provider(kind="openai_compatible", **overrides):
    values = {"kind": kind, "max_output_tokens": 128}
    if kind == "openai_compatible":
        values["base_url"] = "http://127.0.0.1:8000/v1"
    return ModelProvider(**{**values, **overrides})


@pytest.fixture
def fake_api(monkeypatch):
    constructed = []
    module = ModuleType("scientisttwo.providers.backend")

    class APIBackend:
        synthetic = False

        def __init__(self, store, config, **kwargs):
            assert kwargs["allow_live"] is True
            self.store, self.config = store, config
            self.model, self.name = config.model, config.provider.kind
            self.calls, self.events = [], []
            constructed.append(self)

        def invoke(self, role, inputs, output_schema):
            self.calls.append(role)
            return AgentResult(payload={"answer": "synthetic API contract"}, backend=self.name,
                model=self.model, synthetic=False, usage={"input_tokens": 10, "output_tokens": 2})

        def accounting(self):
            return {"fixture": "synthetic API dispatch bookkeeping", "claimed_calls": len(self.calls),
                    "known_observed_tokens": len(self.calls) * 12, "usage_complete": True}

    module.APIBackend = APIBackend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(live_pilot, "CodexBackend", forbidden)
    monkeypatch.setattr(live_pilot, "DirectPilotExecutor", forbidden)
    return constructed


def one_role(self):
    self.store.set("live_spec_ref", self.store.put_json(self.config))
    answer = self.ask("api-contract", "critic", {"fixture": "synthetic routing"}, Answer)
    assert answer.answer == "synthetic API contract"
    state = {"status": "complete", "stage": "complete", "outcome": "no_improvement",
             "reason": "Synthetic provider-routing check; no experiments executed"}
    self.store.set("live_state", state)
    return state


@pytest.mark.parametrize("kind", ["openai", "anthropic", "openai_compatible"])
def test_native_api_execution_uses_provider_backend_and_completed_resume_is_noop(
    tmp_path, monkeypatch, fake_api, kind
):
    config = native_config(tmp_path, model="configured/custom-model", provider=api_provider(kind))
    monkeypatch.setattr(ResearchController, "run", one_role)
    output = tmp_path / "campaign"
    cli.execute(config, output, True)
    assert len(fake_api) == 1 and fake_api[0].name == kind
    with closing(Store(output)) as store:
        before = [dict(row) for row in store.db.execute("SELECT * FROM live_operations")]
        ref = store.get("api_model_accounting_ref")
        assert store.json(ref)["known_observed_tokens"] == 12
        assert store.get("live_pilot_model_accounting") is None
        deadline = store.get("live_deadline")
    monkeypatch.setattr(ResearchController, "run", forbidden)
    cli.execute(config, output, True)
    with closing(Store(output)) as store:
        assert [dict(row) for row in store.db.execute("SELECT * FROM live_operations")] == before
        assert store.get("api_model_accounting_ref") == ref
        assert store.get("live_deadline") == deadline
    assert len(fake_api) == 1


def test_api_cannot_use_legacy_codex_dispatch_or_codex_executable(tmp_path, monkeypatch, fake_api):
    config = native_config(tmp_path, provider=api_provider())
    with pytest.raises(ValueError, match="CLI executable"):
        cli.execute(config, tmp_path / "campaign", True, executable="another-cli")
    legacy = config.model_copy(update={"engine_version": "legacy-live-v1"})
    with pytest.raises(ValueError, match="legacy"):
        cli.execute(legacy, tmp_path / "legacy", True)
    assert not fake_api


@pytest.mark.parametrize("key", ["contract_repair_manifest_ref", "inherited_model_accounting_ref",
                                  "study_plan_ref", "study_compute_budget"])
def test_api_rejects_codex_specific_repair_and_study_stores(tmp_path, monkeypatch, fake_api, key):
    config = native_config(tmp_path, provider=api_provider())
    output = tmp_path / "campaign"
    with closing(Store(output)) as store:
        store.set(key, "synthetic-existing-codex-record")
    with pytest.raises(IntegrityError, match="Codex repair/study"):
        cli.execute(config, output, True)
    assert not fake_api


def test_paperorchestra_rejects_text_only_api_before_backend(tmp_path, fake_api):
    config = native_config(tmp_path, provider=api_provider())
    config.workflow = ResearchPolicy(writer_backend="paperorchestra", compiler_image="sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="vision"):
        cli.execute(config, tmp_path / "campaign", True)
    assert not fake_api


def test_native_check_is_api_specific_and_never_echoes_credentials(tmp_path, monkeypatch):
    provider = api_provider(api_key_env="SCIENTISTTWO_TEST_API_KEY")
    config = native_config(tmp_path, provider=provider)
    manifest = tmp_path / "live.json"
    manifest.write_text(config.model_dump_json())
    credential = "test-only" + "-credential-value"
    monkeypatch.setenv("SCIENTISTTWO_TEST_API_KEY", credential)
    monkeypatch.setattr(cli.shutil, "which", forbidden)
    monkeypatch.setattr(cli, "preflight", lambda native: {"ready": True, "errors": []})
    def process(argv, *args, **kwargs):
        assert argv[0] == "docker" and "image" in argv
        return ProcessResult(code=0, stdout="sha256:" + "a" * 64, stderr="", timed_out=False, seconds=0)
    monkeypatch.setattr(cli, "run_process", process)
    result = CliRunner().invoke(cli.app, ["check", str(manifest)])
    assert result.exit_code == 0, result.exception or result.output
    data = json.loads(result.stdout)
    assert data["provider_ready"] and data["provider"] == "openai_compatible"
    assert data["provider_configuration"]["credential_configured"]
    assert "codex_installed" not in data and credential not in result.output


def test_model_provider_is_frozen_in_live_pilot_manifest_without_backend(tmp_path, monkeypatch, fake_api):
    task = TaskSpec()
    options = live_pilot.LivePilotOptions(model="custom-org/long-custom-model",
        provider=api_provider(), max_calls=10, token_stop_threshold=10000,
        campaign_seconds=600, call_seconds=30, workflow=ResearchPolicy())
    task_path, options_path, output = (tmp_path / name for name in ("task.json", "options.json", "manifest.json"))
    task_path.write_text(task.model_dump_json())
    options_path.write_text(options.model_dump_json())
    result = CliRunner().invoke(live_pilot_cli.app, ["create", str(task_path), str(options_path), str(output),
        "--image", "sha256:" + "a" * 64])
    assert result.exit_code == 0, result.exception or result.output
    frozen = live_pilot_cli.LivePilotManifest.model_validate_json(output.read_text())
    assert frozen.options.provider == options.provider
    assert frozen.options.model == options.model and not fake_api
    assert not list(tmp_path.rglob("campaign.sqlite"))


def test_secret_like_invalid_fields_are_hidden_by_outer_manifest_validation(tmp_path):
    config = native_config(tmp_path)
    value = config.model_dump(mode="json")
    secret = "never-print" + "-fixture-credential"
    value["provider"] = {"kind": "openai", "max_output_tokens": 128, "api_key": secret}
    with pytest.raises(ValueError) as caught:
        LiveSpec.model_validate(value)
    assert secret not in str(caught.value)


def test_native_create_freezes_api_configuration_without_auth_or_calls(tmp_path, monkeypatch, fake_api):
    config = native_config(tmp_path)
    native = tmp_path / "native.json"
    native.write_text(config.native.model_dump_json())
    output = tmp_path / "api-manifest.json"
    monkeypatch.setattr(cli, "preflight", lambda spec: {"ready": True, "errors": []})
    monkeypatch.setattr(cli, "CodexBackend", forbidden)
    result = CliRunner().invoke(cli.app, ["create", str(native), str(output),
        "--model", "configured-model", "--max-calls", "5", "--seconds", "60",
        "--provider", "openai", "--max-output-tokens", "128", "--token-stop-threshold", "1000"])
    assert result.exit_code == 0, result.exception or result.output
    frozen = LiveSpec.model_validate_json(output.read_text())
    assert frozen.provider.kind == "openai" and frozen.provider.api_key_env == "OPENAI_API_KEY"
    assert frozen.model == "configured-model" and frozen.model_retries == 0
    assert frozen.provider.max_output_tokens == 128 and not fake_api


@pytest.mark.parametrize("bad_provider", [
    {"kind": "openai", "max_output_tokens": 128, "api_key": "fixture-value"},
    {"kind": "openai_compatible", "max_output_tokens": 128,
     "base_url": "https://user:fixture-value@example.org/v1"},
])
def test_live_pilot_create_rejects_credential_values_without_echo_or_output(tmp_path, bad_provider):
    task, options_path, output = (tmp_path / name for name in ("task.json", "options.json", "manifest.json"))
    task.write_text(TaskSpec().model_dump_json())
    options = live_pilot.LivePilotOptions(model="configured-model", max_calls=5,
        token_stop_threshold=1000, campaign_seconds=60, call_seconds=15, workflow=ResearchPolicy())
    value = options.model_dump(mode="json")
    value["provider"] = bad_provider
    options_path.write_text(json.dumps(value))
    result = CliRunner().invoke(live_pilot_cli.app, ["create", str(task), str(options_path),
        str(output), "--image", "sha256:" + "a" * 64])
    assert result.exit_code != 0 and not output.exists()
    assert "fixture-value" not in result.output + str(result.exception)


def test_live_pilot_check_needs_no_codex_for_api(tmp_path, monkeypatch):
    options = live_pilot.LivePilotOptions(model="local/model", provider=api_provider(),
        max_calls=5, token_stop_threshold=1000, campaign_seconds=60, call_seconds=15,
        workflow=ResearchPolicy())
    manifest = live_pilot_cli.LivePilotManifest(task=TaskSpec(), profile=Profile.named("offline"),
        options=options, runtime="docker", image="sha256:" + "a" * 64,
        implementation=implementation_identity())
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json())
    monkeypatch.setattr(cli.shutil, "which", forbidden)
    monkeypatch.setattr(live_pilot_cli, "run_process", lambda *a, **k: ProcessResult(
        code=0, stdout=manifest.image, stderr="", seconds=0, timed_out=False))
    result = CliRunner().invoke(live_pilot_cli.app, ["check", str(path)])
    assert result.exit_code == 0, result.exception or result.output
    value = json.loads(result.stdout)
    assert value["ready"] and value["provider"] == "openai_compatible"
    assert value["live_model_calls"] == 0


@pytest.mark.parametrize("updates,expected_ready", [
    ({}, True),
    ({"engine_version": "legacy-live-v1"}, False),
    ({"token_stop_threshold": None}, False),
    ({"model_retries": 1}, False),
])
def test_suite_api_preflight_checks_dispatch_requirements_without_calls(
    tmp_path, monkeypatch, updates, expected_ready
):
    from scientisttwo.live import workspace, workspace_task
    from scientisttwo import process

    config = native_config(tmp_path, provider=api_provider(), **updates)
    monkeypatch.setattr(suite.shutil, "which", forbidden)
    monkeypatch.setattr(workspace_task, "preflight", lambda native: {"ready": True, "errors": []})
    monkeypatch.setattr(workspace, "validate_baseline_contract", lambda adapter: None)
    monkeypatch.setattr(workspace, "live_readiness", lambda adapter: {"ready": True})

    def inspect_image(argv, *args, **kwargs):
        assert argv[0] == "docker" and "image" in argv
        return ProcessResult(code=0, stdout="sha256:" + "a" * 64,
            stderr="", timed_out=False, seconds=0)

    monkeypatch.setattr(process, "run_process", inspect_image)
    checked = suite.check_assignment(config)
    assert checked["ready"] is expected_ready
    assert checked["provider_ready"] is expected_ready
    if not expected_ready:
        assert any("token threshold and model_retries=0" in error for error in checked["errors"])


def test_suite_unknown_api_usage_blocks_other_assignments(tmp_path):
    config = native_config(tmp_path, provider=api_provider())
    suite_config = suite.SuiteSpec(name="synthetic", wall_seconds=600, max_calls=20,
        max_experiments=20, token_stop_threshold=1000,
        assignments=[suite.SuiteAssignment(id="first", repetition=0, campaign=config),
                     suite.SuiteAssignment(id="second", repetition=1, campaign=config)],
        implementation=suite.suite_identity())
    with closing(Store(tmp_path / "suite")) as store:
        budget = suite.SuiteBudget(store, suite_config)
        first = budget.for_assignment("first")
        first.reserve("model/unknown", "model", {"fixture": True})
        first.finish("model/unknown", "failed", {"usage_known": False, "error": "unknown_api_usage"}, 1)
        with pytest.raises(AmbiguousCall, match="unknown"):
            budget.for_assignment("second").reserve("model/next", "model", {"fixture": True})
        assert len(budget.summary()) == 1


def test_suite_unknown_api_usage_commits_blocked_summary_and_resume_adds_no_operations(tmp_path):
    config = native_config(tmp_path, provider=api_provider())
    suite_config = suite.SuiteSpec(name="synthetic", wall_seconds=600, max_calls=20,
        max_experiments=20, token_stop_threshold=1000,
        assignments=[suite.SuiteAssignment(id="first", repetition=0, campaign=config),
                     suite.SuiteAssignment(id="second", repetition=1, campaign=config)],
        implementation=suite.suite_identity())
    attempted = []

    def unknown_usage(config, output, budget):
        attempted.append(budget.assignment)
        budget.reserve("model/unknown", "model", {"fixture": True})
        budget.finish("model/unknown", "failed",
            {"usage_known": False, "error": "unknown_api_usage"}, 1)
        raise AmbiguousCall("Synthetic unknown usage")

    output = tmp_path / "suite"
    result = suite.run_suite(suite_config, output, allow_live=True, executor=unknown_usage)
    assert result["status"] == "blocked"
    assert result["assignments"][0]["outcome"] == "ambiguous_operation"
    assert result["assignments"][1]["status"] == "not_started"
    assert result["model_attempts"] == 1 and result["improvement_denominator"] == 2
    assert suite.inspect_suite(output)[1] == result
    with closing(Store(output)) as store:
        before = [dict(row) for row in store.db.execute("SELECT * FROM suite_operations")]
        assert store.json(store.get("suite_summary_ref")) == result
    resumed = suite.run_suite(suite_config, output, allow_live=True, executor=unknown_usage)
    with closing(Store(output)) as store:
        assert [dict(row) for row in store.db.execute("SELECT * FROM suite_operations")] == before
    assert resumed == result and attempted == ["first"]


def test_research_suite_api_assignment_uses_shared_accounting(tmp_path, monkeypatch, fake_api):
    config = native_config(tmp_path, model="custom/model", provider=api_provider())
    suite_config = suite.SuiteSpec(name="synthetic", wall_seconds=600, max_calls=20,
        max_experiments=20, token_stop_threshold=1000,
        assignments=[suite.SuiteAssignment(id="first", repetition=0, campaign=config)],
        implementation=suite.suite_identity())
    monkeypatch.setattr(ResearchController, "run", one_role)
    with closing(Store(tmp_path / "suite")) as store:
        budget = suite.SuiteBudget(store, suite_config)
        state = suite.execute_assignment(config, tmp_path / "child", budget.for_assignment("first"))
        assert state["status"] == "complete"
        operations = budget.summary()
        assert len(operations) == 1 and operations[0]["tokens"] == 12
        assert operations[0]["assignment"] == "first" and operations[0]["status"] == "success"
    assert len(fake_api) == 1 and fake_api[0].calls == ["critic"]


def test_live_pilot_policy_records_selected_provider(tmp_path):
    from scientisttwo.research.pilot_bridge import create_pilot_campaign
    options = live_pilot.LivePilotOptions(model="custom/model", provider=api_provider(),
        max_calls=10, token_stop_threshold=10000, campaign_seconds=600, call_seconds=30,
        workflow=ResearchPolicy())
    with closing(Store(tmp_path / "campaign")) as store:
        config, _, _ = create_pilot_campaign(store, TaskSpec(), Profile.named("offline"),
            image="sha256:" + "a" * 64, live_options=options)
        assert config.provider == options.provider
        assert store.get("live_pilot_execution_policy")["backend"] == "openai_compatible"


def test_provider_files_belong_to_source_identity():
    identity = implementation_identity()
    assert "providers/config.py" in identity
    assert all(not key.endswith(".pyc") for key in identity)
