"""External operator for a pinned study; execute this file in a fresh interpreter.

The scientific package stays byte-for-byte pinned. Only the explicitly hashed
continuation module is loaded from outside that package. Planning and auditing
never construct a backend; running requires an already admitted allowance.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def verify_package(package, manifest, expected_hash):
    package, manifest = Path(package).resolve(), Path(manifest)
    if manifest.is_symlink() or not manifest.is_file() or file_hash(manifest) != expected_hash:
        raise ValueError("Pinned source manifest differs from its declared hash")
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, dict) or not {"pyproject.toml", "uv.lock"} <= set(entries):
        raise ValueError("A complete pinned package manifest is required")
    for name, record in entries.items():
        rel = Path(name)
        path = package / rel
        if (rel.is_absolute() or ".." in rel.parts or path.is_symlink()
                or not path.is_file() or not path.resolve().is_relative_to(package)
                or path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]):
            raise ValueError("Pinned source differs: " + name)
    actual = {p.relative_to(package).as_posix() for p in (package / "src").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if actual != {name for name in entries if name.startswith("src/")}:
        raise ValueError("Pinned source closure contains missing or extra files")
    return package


def load_operator(module, expected_hash, package):
    module = Path(module)
    if (module.is_symlink() or not module.is_file() or file_hash(module) != expected_hash
            or module.resolve().is_relative_to(package)):
        raise ValueError("Continuation operator must be hash-bound and outside the pinned package")
    if "scientisttwo" in sys.modules:
        raise ValueError("Use a fresh interpreter; a scientific package is already loaded")
    sys.path.insert(0, str(package / "src"))
    import scientisttwo
    if not Path(scientisttwo.__file__).resolve().is_relative_to(package / "src"):
        raise ValueError("Scientific imports did not resolve to the pinned package")
    name = "scientisttwo.continuation"
    spec = importlib.util.spec_from_file_location(name, module)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


@contextmanager
def campaign_store(root):
    from scientisttwo.store import Store
    root = Path(root).resolve()
    if not (root / "campaign.sqlite").is_file():
        raise ValueError("An existing campaign is required; recovery cannot start a new study")
    with (root / ".study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Store(root)
        try:
            yield store
        finally:
            store.close()


def scientific_config(store):
    from scientisttwo.live.adapters import implementation_identity
    from scientisttwo.live.models import LiveSpec
    ref = store.get("live_spec_ref")
    if not ref:
        raise ValueError("Campaign has no frozen live specification")
    config = LiveSpec.model_validate(store.json(ref))
    if config.implementation != implementation_identity():
        raise ValueError("Use the original pinned scientific runtime and dependencies")
    return config


def construct_controller(store, config, queue, policy_path):
    if getattr(getattr(config, "provider", None), "kind", "codex") != "codex":
        raise ValueError("The remote study continuation transport supports only the Codex provider")
    from scientisttwo.backends import CodexBackend
    from scientisttwo.studies.budget import StudyBudget
    from scientisttwo.studies.cifar_adapter import CifarRecipeDevelopment
    from scientisttwo.studies.coordinator import StudyResearchController
    from scientisttwo.studies.media_relay import QueueExecutor, RelayPolicy, read_json
    from scientisttwo.studies.runner import StudyDevelopmentRunner
    compute = store.get("study_compute_budget")
    if not compute:
        raise ValueError("Missing original compute allowance")
    policy = RelayPolicy.from_mapping(read_json(Path(policy_path), 4096))
    if policy.model != config.model or policy.max_calls != config.max_calls:
        raise ValueError("Relay differs from the frozen model/call allowance")
    if policy.token_stop_threshold != config.token_stop_threshold:
        raise ValueError("Relay differs from the frozen observed-token allowance")
    adapter = CifarRecipeDevelopment(config.native, config.development)
    adapter.bind_source(store.get("authorized_source_snapshot"))
    if adapter.identity() != store.get("live_adapter_identity"):
        raise ValueError("Scientific adapter or prepared assets changed")
    budget = StudyBudget(store, config,
        max_experiment_seconds=compute["max_experiment_seconds"],
        max_final_seconds=compute["max_final_seconds_per_label"],
        max_final_wall_seconds=compute["max_final_wall_seconds"])
    runner = StudyDevelopmentRunner(store, config, adapter, budget)
    backend = CodexBackend(store.root, config.model, allow_live=True,
        timeout=config.call_seconds, executor=QueueExecutor(Path(queue), policy))
    return StudyResearchController(store, config, adapter, backend, runner, budget=budget)


def check_execution_lease(args, study, research_store):
    from scientisttwo import slurm_execution
    launch = study.get("paired_launch")
    if not launch or launch["execution_config_sha256"] != args.execution_config_sha256:
        raise ValueError("Continuation must use the unchanged approved execution lease")
    config = slurm_execution.load_config(args.execution_config, args.execution_config_sha256)
    if os.environ.get("SLURM_JOB_ID") != str(config["job_id"]):
        raise ValueError("Continue inside the original authorized Slurm allocation")
    if config["expires_at"] < launch["window_deadline"]:
        raise ValueError("Execution lease does not cover the original study window")
    closed = Path(args.package).resolve() / "evidence/worker-result.json"
    if (closed.is_symlink() or not closed.is_file()
            or file_hash(closed) != args.closed_worker_result_sha256):
        raise ValueError("Original outer worker must have a hash-bound closed cleanup receipt")
    receipt = json.loads(closed.read_text())
    if (receipt.get("cleanup", {}).get("status") != "pass"
            or not isinstance(receipt.get("finished_at"), (int, float))
            or receipt["finished_at"] > time.time()
            or receipt.get("lease", {}).get("driver_deadline") != launch["window_deadline"]):
        raise ValueError("Original worker cleanup is incomplete or belongs to a different window")
    helper = load_environment_helper()
    helper.restore_environment(research_store, args.execution_config, args.execution_config_sha256)


def load_environment_helper():
    path = Path(__file__).with_name("continuation_environment.py")
    name = "scientisttwo.continuation_environment"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finalize_study(args, continuation, research):
    """Use the original coordinator and a backend that cannot issue model calls."""
    from scientisttwo.live.models import LiveSpec
    from scientisttwo.studies.budget import StudyBudget
    from scientisttwo.studies.cifar_adapter import CifarRecipeDevelopment
    from scientisttwo.studies.coordinator import audit_study, release_finals
    from scientisttwo.studies.direct_control import DirectControl
    from scientisttwo.studies.runner import StudyDevelopmentRunner

    class NoModelCalls:
        name, synthetic = "codex", False

        def invoke(self, *args, **kwargs):
            raise ValueError("Model calls are forbidden during protected final evaluation")

    research.backend = NoModelCalls()
    with campaign_store(args.study) as study:
        plan = study.json(study.get("study_plan_ref"))
        if plan["arms"]["scientist"] != str(research.store.root):
            raise ValueError("Continuation does not belong to this paired study")
        before = continuation.audit_continuation(research.store, args.plan_ref, args.admission_ref)
        if before["integrity"] != "pass":
            raise ValueError("Continuation audit must pass before final access")
        prior = study.get("continuation/final-result/" + args.plan_ref)
        if prior:
            return study.json(prior)
        check_execution_lease(args, study, research.store)
        launch = study.get("paired_launch")
        if (not study.get("study_final_grant_ref") and time.time()
                + launch["max_final_wall_seconds"] + launch["audit_seconds"] + 120
                > launch["window_deadline"]):
            raise ValueError("Insufficient unchanged lease for final evaluation and audit")
        with campaign_store(Path(plan["arms"]["direct"])) as direct_store:
            config = LiveSpec.model_validate(plan["arm_configurations"]["direct"])
            adapter = CifarRecipeDevelopment(config.native, config.development)
            adapter.bind_source(direct_store.get("authorized_source_snapshot"))
            limits = direct_store.get("study_compute_budget")
            budget = StudyBudget(direct_store, config,
                max_experiment_seconds=limits["max_experiment_seconds"],
                max_final_seconds=limits["max_final_seconds_per_label"],
                max_final_wall_seconds=limits["max_final_wall_seconds"])
            runner = StudyDevelopmentRunner(direct_store, config, adapter, budget)
            direct = DirectControl(direct_store, config, adapter, NoModelCalls(), runner,
                                   budget=budget, allow_live=True)
            finals = release_finals(study, research, direct)
            audit_seconds = min(launch["audit_seconds"], launch["window_deadline"] - time.time() - 30)
            if audit_seconds <= 0:
                raise ValueError("No approved time remains for independent scoring audit")
            evidence_audit = audit_study(study, research, direct, reexecute_evaluators=True,
                max_score_replays=launch["max_score_replays"], max_audit_seconds=audit_seconds)
            chain = continuation.audit_continuation(research.store, args.plan_ref, args.admission_ref)
            result = {"status": "completed_with_unresolved_historical_usage"
                      if evidence_audit["status"] == "pass" and chain["integrity"] == "pass"
                      else "audit_failed", "finals": finals, "evidence_audit": evidence_audit,
                      "continuation_audit": chain, "live_model_calls": 0,
                      "original_total_token_ceiling_verified": False,
                      "comparison_limit": "Scientific protocols remain matched; exact total-token matching is unverified"}
            ref = study.put_json(result, "continued-study-final-result")
            study.set("continuation/final-result/" + args.plan_ref, ref)
            return result


def run(args):
    os.umask(0o077)
    sys.dont_write_bytecode = True
    package = verify_package(args.package, args.source_manifest, args.source_manifest_sha256)
    continuation = load_operator(args.operator_module, args.operator_sha256, package)
    from scientisttwo.backends import strict_json_loads
    with campaign_store(args.campaign) as store:
        config = scientific_config(store)
        if args.mode == "plan":
            proof = strict_json_loads(args.dispatch_proof.read_text()) if args.dispatch_proof else None
            ref = continuation.prepare_plan(store, args.failed_operation, args.continuation_id,
                                            failed_dispatch_proof=proof)
            binding = {"operator_sha256": file_hash(__file__),
                       "environment_adapter_sha256": file_hash(Path(__file__).with_name("continuation_environment.py")),
                       "continuation_module_sha256": args.operator_sha256,
                       "package": str(package), "source_manifest_sha256": args.source_manifest_sha256}
            key = "continuation-operator/" + ref
            if store.get(key) not in (None, binding):
                raise ValueError("Continuation operator binding changed")
            store.set(key, binding)
            return {"status": "planned_no_calls", "plan_ref": ref, "plan": store.json(ref),
                    "new_experiments": 0, "new_model_calls": 0}
        binding = {"operator_sha256": file_hash(__file__),
                   "environment_adapter_sha256": file_hash(Path(__file__).with_name("continuation_environment.py")),
                   "continuation_module_sha256": args.operator_sha256,
                   "package": str(package), "source_manifest_sha256": args.source_manifest_sha256}
        if store.get("continuation-operator/" + str(args.plan_ref)) != binding:
            raise ValueError("Pinned package or external continuation operator differs from the plan")
        if args.mode == "admit":
            if not args.allowance:
                raise ValueError("An explicit bounded allowance file is required")
            ref = continuation.admit_continuation(store, args.plan_ref,
                                                   strict_json_loads(args.allowance.read_text()))
            return {"status": "admitted_no_calls", "admission_ref": ref,
                    "new_experiments": 0, "new_model_calls": 0}
        if args.mode == "audit":
            return continuation.audit_continuation(store, args.plan_ref, args.admission_ref)
        if args.mode == "finalize" and not args.allow_final:
            raise ValueError("Protected final evaluation requires --allow-final")
        if args.mode == "run" and not args.allow_live:
            raise ValueError("No live dispatch: continuation run requires --allow-live")
        if not all((args.study, args.queue, args.policy, args.execution_config,
                    args.execution_config_sha256, args.admission_ref, args.closed_worker_result_sha256)):
            raise ValueError("Run requires study, new queue, pinned policy, execution lease and admission")
        from scientisttwo.integrations.paperorchestra.cli import readonly_campaign
        with readonly_campaign(args.study) as study:
            plan = study.json(study.get("study_plan_ref"))
            if str(store.root) != plan["arms"]["scientist"]:
                raise ValueError("Campaign does not match the original study's scientist arm")
            if args.mode == "run" and study.get("study_final_grant_ref"):
                raise ValueError("No recovery/model calls after protected-final release")
            # Sequential execution: the original driver must have closed the
            # direct arm before this worker can generate further experiments.
            with readonly_campaign(Path(plan["arms"]["direct"])) as direct:
                if (not direct.get("direct_result_ref")
                        or direct.db.execute("SELECT 1 FROM live_operations WHERE status='running'").fetchone()):
                    raise ValueError("Wait for the original direct-control arm to close")
            if args.mode == "run" or not study.get("continuation/final-result/" + args.plan_ref):
                check_execution_lease(args, study, store)
        controller = construct_controller(store, config, args.queue, args.policy)
        continuation.install_continuation(controller, args.plan_ref, args.admission_ref)
        if args.mode == "finalize":
            return finalize_study(args, continuation, controller)
        if store.get("live_state", {}).get("status") == "frozen":
            return {"status": "already_frozen", "research": store.get("live_state"),
                    "new_experiments": 0, "new_model_calls": 0,
                    "continuation_audit": continuation.audit_continuation(store, args.plan_ref, args.admission_ref)}
        load_environment_helper().install_cached_preflight(controller)
        result = controller.run()
        integrity = continuation.audit_continuation(store, args.plan_ref, args.admission_ref)
        return {"status": "continuation_returned", "research": result,
                "continuation_audit": integrity, "protected_final_access": False,
                "scope": "Research arm only; original both-arms final barrier remains in force"}


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("mode", choices=["plan", "admit", "run", "audit", "finalize"])
    for flag in ("package", "source-manifest", "operator-module", "campaign"):
        value.add_argument("--" + flag, type=Path, required=True)
    for flag in ("source-manifest-sha256", "operator-sha256"):
        value.add_argument("--" + flag, required=True)
    value.add_argument("--failed-operation")
    value.add_argument("--continuation-id")
    value.add_argument("--dispatch-proof", type=Path)
    value.add_argument("--plan-ref")
    value.add_argument("--admission-ref")
    value.add_argument("--allowance", type=Path)
    for flag in ("study", "queue", "policy", "execution-config"):
        value.add_argument("--" + flag, type=Path)
    value.add_argument("--execution-config-sha256")
    value.add_argument("--closed-worker-result-sha256")
    value.add_argument("--allow-live", action="store_true")
    value.add_argument("--allow-final", action="store_true")
    return value


if __name__ == "__main__":
    try:
        print(json.dumps(run(parser().parse_args()), indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "blocked", "error_type": type(error).__name__,
                          "reason": str(error), "recovery_completed": False}), file=sys.stderr)
        raise SystemExit(2)
