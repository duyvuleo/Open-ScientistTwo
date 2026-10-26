"""Synthetic failure records and fake stages; no models, containers or benchmarks."""
import importlib.util
from pathlib import Path

import pytest

from scientisttwo.artifacts import ArtifactSpec, freeze_bundle, load_bundle
from scientisttwo.live.models import Candidate, Observation, SourceFile
from scientisttwo.live.source_tree import candidate_snapshot, freeze_source_tree
from scientisttwo.research.execution_context import ConfigurationDomainError, build_execution_context
from scientisttwo.store import IntegrityError, Store, canonical, digest


BAD = ['{"optimizer":"rmsprop"}', '{"epochs":"11"}', '{not json}']
NONSTRING_OPTIMIZERS = ['{"optimizer":[]}', '{"optimizer":{}}']
BASE = '{"epochs":5,"max_epochs":10,"max_steps":1000,"batch_size":128,"seed":0}'


def fixture(tmp_path, text, *, status='execution_failed', base_text=BASE, accepted_outputs=False,
            accepted_reference=None):
    store = Store(tmp_path / 'campaign')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'method.py').write_text('VALUE = 1\n')
    (source / 'training.json').write_text(base_text)
    (source / 'training_config.json').write_text('{}')
    base = freeze_source_tree(store, source, ['training_config.json'])
    candidate = Candidate(base_snapshot_ref=base,
        files=[SourceFile(path='training_config.json', content=text)],
        command=['python', 'method.py'], description='Synthetic candidate configuration')
    tree = candidate_snapshot(store, candidate)
    tree_ref, code_ref = store.put_json(tree), store.put_json(candidate)
    protocol = {'phase': 'subset', 'seeds': [0]}
    log = store.put(b'Synthetic original training-loader failure', 'log')
    execution = store.put_json({'stage': 'dev-' + digest(b'attempt0')[:24],
        'job_ref': store.put_json({'seed': 0}), 'returncode': 0 if status == 'success' else 1,
        'timed_out': False, 'seconds': .01, 'log_ref': log})
    bundles = []
    if accepted_outputs:
        output = tmp_path / 'outputs'
        (output / 'checkpoint').mkdir(parents=True)
        (output / 'checkpoint/config.json').write_text(BASE)
        bundles = [freeze_bundle(store, output, [ArtifactSpec(path='checkpoint', kind='directory', role='checkpoint')])]
    accepted = {}
    if accepted_reference:
        ref = store.put_json({'synthetic': True, 'accepted_output': accepted_reference})
        accepted[accepted_reference] = [{'checkpoint/output': ref}] if accepted_reference == 'artifact_refs' else [ref]
    result = Observation(id='attempt', phase='subset', status=status,
        metric='quality', value=.5 if status == 'success' else None,
        values=[.5] if status == 'success' else [],
        prediction_refs=([store.put_json({'synthetic': True})] if status == 'success'
                         else accepted.pop('prediction_refs', [])),
        reason='Synthetic original loader failure' if status != 'success' else 'Synthetic success',
        protocol_hash=digest(canonical(protocol)), protocol_ref=store.put_json(protocol),
        code_ref=code_ref, source_tree_ref=tree_ref, log_refs=[execution], bundle_refs=bundles,
        **accepted)
    store.cache('development/attempt', {}, result)
    store.cache('materialization/development/attempt', {}, {'source_tree_ref': tree_ref, 'code_ref': code_ref})
    return store, result, tree


def test_import_uses_isolated_package():
    import scientisttwo.research.execution_context as module
    assert Path(module.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / 'src')


@pytest.mark.parametrize('text', BAD + NONSTRING_OPTIMIZERS + ['[]', '{"learning_rate":NaN}'])
def test_failed_candidate_domain_error_is_explicit_unknown_not_default_fallback(tmp_path, text):
    store, result, _ = fixture(tmp_path, text)
    try:
        before = list(store.db.iterdump())
        context = build_execution_context(store, [result])
        item = context['observations'][0]
        assert context['schema_version'] == 2
        assert item['status'] == 'execution_failed'
        assert item['candidate_overrides'] is None
        assert item['configuration_status'] == 'invalid_candidate_overrides'
        assert item['configuration_errors'][0]['status'] == 'invalid'
        assert item['configuration_errors'][0]['artifact_ref'] == item['configuration_provenance']['candidate_overrides']['ref']
        assert item['base_configuration']['epochs'] == 5
        assert item['executions'][0]['configured'] == {'seed': 0}
        assert item['executions'][0]['optimizer_parameters'] == {}
        assert item['executions'][0]['returncode'] == 1
        assert 'epochs' in item['unknown_settings']
        assert store.json(store.db.execute("SELECT output_ref FROM calls WHERE key='development/attempt'").fetchone()[0])['reason'] == result.reason
        assert list(store.db.iterdump()) == before
    finally:
        store.close()


@pytest.mark.parametrize('text', BAD + NONSTRING_OPTIMIZERS + ['[]', '{"learning_rate":NaN}'])
def test_successful_observation_with_invalid_candidate_configuration_remains_fatal(tmp_path, text):
    store, result, _ = fixture(tmp_path, text, status='success')
    try:
        with pytest.raises(ConfigurationDomainError):
            build_execution_context(store, [result])
    finally:
        store.close()


@pytest.mark.parametrize('base_text', ['{broken', '{"optimizer":"rmsprop"}'])
def test_trusted_base_configuration_errors_are_not_suppressed(tmp_path, base_text):
    store, result, _ = fixture(tmp_path, BAD[0], base_text=base_text)
    try:
        with pytest.raises(ConfigurationDomainError):
            build_execution_context(store, [result])
    finally:
        store.close()


def test_verified_file_integrity_failure_is_not_a_configuration_domain_error(tmp_path):
    store, result, tree = fixture(tmp_path, BAD[0])
    try:
        ref = load_bundle(store, tree.bundle_ref, verify=False).files['training_config.json'].ref
        (store.root / ref).write_text('{}')
        with pytest.raises(IntegrityError) as caught:
            build_execution_context(store, [result])
        assert not isinstance(caught.value, ConfigurationDomainError)
    finally:
        store.close()


def test_partial_accepted_outputs_with_unparseable_source_still_fail(tmp_path):
    store, result, _ = fixture(tmp_path, BAD[2], accepted_outputs=True)
    try:
        with pytest.raises(ConfigurationDomainError):
            build_execution_context(store, [result])
    finally:
        store.close()


@pytest.mark.parametrize('reference', ['prediction_refs', 'evaluation_refs', 'artifact_refs', 'timing_refs'])
def test_failed_observation_with_accepted_output_references_cannot_be_downgraded(tmp_path, reference):
    store, result, _ = fixture(tmp_path, NONSTRING_OPTIMIZERS[0], accepted_reference=reference)
    try:
        with pytest.raises(ConfigurationDomainError):
            build_execution_context(store, [result])
    finally:
        store.close()


def test_numeric_limit_violation_stays_visible_as_a_declared_setting(tmp_path):
    store, result, _ = fixture(tmp_path, '{"epochs":11}')
    try:
        item = build_execution_context(store, [result])['observations'][0]
        assert item['candidate_overrides'] == {'epochs': 11}
        assert item['status'] == 'execution_failed'
        assert 'configuration_errors' not in item
    finally:
        store.close()


@pytest.mark.parametrize('text', BAD)
def test_controller_can_present_failure_engineer_and_resume_without_reexecution(tmp_path, text):
    from scientisttwo.live.models import Hypothesis
    from scientisttwo.process import ProcessResult
    from test_research_workflow import campaign, with_offset
    import scientisttwo.research.execution_context as execution_context

    controller, store, runner, agents, transport = campaign(tmp_path)
    original_response, original_executor = agents.respond, runner.raw_executor
    contract_path = Path(execution_context.__file__).parents[1] / 'task_families/templates/training_contract.py'
    spec = importlib.util.spec_from_file_location('trusted_test_loader', contract_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    base_path = tmp_path / 'trusted-training.json'
    base_path.write_text(BASE)
    bad = with_offset(controller.adapter.baseline(), .5)
    bad.files.append(SourceFile(path='training_config.json', content=text))
    failures, engineers = [], []

    def respond(role, inputs):
        if role == 'implement':
            agents.seen.append((role, inputs))
            return bad.model_dump()
        if role == 'critic' and inputs['evidence']['observation']['status'] != 'success':
            agents.seen.append((role, inputs))
            failures.append(inputs)
            item = next(o for o in inputs['evidence_package']['effective_execution']['observations']
                        if o['id'] == inputs['evidence']['observation']['id'])
            assert item['configuration_status'] == 'invalid_candidate_overrides'
            assert item['configuration_errors'] and item['candidate_overrides'] is None
            return {'decision': 'Engineer', 'reason': 'Repair the explicit failed configuration'}
        if role == 'engineer':
            agents.seen.append((role, inputs))
            engineers.append(inputs)
            repaired = Candidate.model_validate(inputs['candidate'])
            for file in repaired.files:
                if file.path == 'training_config.json':
                    file.content = '{}\n'
            return repaired.model_dump()
        return original_response(role, inputs)

    def execute(argv, timeout, **kwargs):
        if 'run' in argv:
            mounts = {value.rsplit(':', 2)[1]: Path(value.rsplit(':', 2)[0])
                      for index, value in enumerate(argv) if index and argv[index - 1] == '-v'}
            work = mounts.get('/workspace')
            if work and (work / 'training_config.json').is_file():
                try:
                    loader.load_configuration(base_path, work, 'cifar10')
                except (ValueError, TypeError) as error:
                    transport.stages.append(('synthetic-loader-rejection', mounts))
                    return ProcessResult(1, 'Synthetic loader failure', str(error), .01)
        return original_executor(argv, timeout, **kwargs)

    agents.responder, runner.raw_executor = respond, execute
    idea = Hypothesis(title='Synthetic repair', hypothesis='Repair configuration', novelty_score=0,
                      novelty_scope='No novelty claim', source_ids=['task'])

    def run():
        controller.trials = []
        runner.preflight()
        baseline = controller.adapter.baseline()
        controller.baseline_candidate = baseline
        controller.baselines = {phase: runner.execute('baseline/' + phase, baseline, phase)
                                for phase in ('subset', 'full')}
        return controller.evaluate('candidate', idea)

    try:
        store.set('live_state', {'stage': 'search', 'status': 'running'})
        result = run()
        assert result is not None and result['result']['status'] == 'success'
        assert len(failures) == len(engineers) == 1
        assert engineers[0]['feedback']['observation']['status'] == 'execution_failed'
        failed = store.json(store.db.execute("SELECT output_ref FROM calls WHERE key='development/candidate/subset/0'").fetchone()[0])
        assert failed['status'] == 'execution_failed' and failed['value'] is None
        before = (controller.budget.summary(), len(agents.seen), len(transport.stages))
        assert run() == result
        assert (controller.budget.summary(), len(agents.seen), len(transport.stages)) == before
    finally:
        store.close()
