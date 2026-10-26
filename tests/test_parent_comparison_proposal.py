"""Explicitly synthetic records; no research models, containers or data runs."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scientisttwo.live.evaluation import compare_observations
from scientisttwo.live.models import Candidate, DevelopmentSpec, Observation, SourceFile
from scientisttwo.live.runtime import Budget
from scientisttwo.live.source_tree import candidate_snapshot
from scientisttwo.research.controller import ResearchController
from scientisttwo.research.parent_comparison import CONTEXT_VERSION, resolve_parent_comparison
from scientisttwo.store import IntegrityError, Store, canonical, digest


def test_imports_are_from_isolated_proposal():
    import scientisttwo.research.controller as controller
    import scientisttwo.research.parent_comparison as parent
    expected = Path(__file__).resolve().parents[1] / 'src'
    assert Path(controller.__file__).resolve().is_relative_to(expected)
    assert Path(parent.__file__).resolve().is_relative_to(expected)


def source(value, *, command=None, description='Synthetic candidate'):
    return Candidate(files=[SourceFile(path='method.py', content=f'VALUE = {value}\n')],
                     command=command or ['python', 'method.py'], description=description)


def observation(store, key, candidate, value, *, phase='subset', protocol_tag='same',
                receipts=True, source_binding=True, status='success', seeds=None):
    seeds = [0] if seeds is None else seeds
    tree_ref = store.put_json(candidate_snapshot(store, candidate))
    protocol = {'phase': phase, 'seeds': seeds, 'tag': protocol_tag}
    log = store.put(b'Synthetic execution receipt; no code executed', 'log')
    execution = store.put_json({'stage': 'dev-' + digest((key + '0').encode())[:24],
        'job_ref': store.put_json({'seed': 0}), 'returncode': 0 if status == 'success' else 1, 'timed_out': False,
        'seconds': 0.0, 'log_ref': log})
    result = Observation(id=key, phase=phase, status=status, metric='quality',
        value=value if status == 'success' else None,
        values=[value] * len(seeds) if status == 'success' else [],
        code_ref=store.put_json(candidate), source_tree_ref=tree_ref if source_binding else None,
        protocol_ref=store.put_json(protocol), protocol_hash=digest(canonical(protocol)),
        log_refs=[execution] if receipts else [],
        prediction_refs=[store.put_json({'synthetic': True, 'quality': value, 'seed': seed})
                         for seed in seeds] if status == 'success' else [])
    store.cache('development/' + key, {}, result)
    store.cache('materialization/development/' + key, {},
                {'source_tree_ref': tree_ref, 'code_ref': result.code_ref})
    return result


@pytest.fixture
def case(tmp_path):
    store = Store(tmp_path / 'synthetic')
    config = DevelopmentSpec(metric='quality', direction='maximize', seeds=[0])
    parent = source(2)
    full = observation(store, 'parent-full', parent, .6, phase='full')
    baseline = observation(store, 'fixed-reference', source(1), .4687)
    trial = {'id': full.id, 'candidate': parent.model_dump(), 'result': full.model_dump()}
    value = SimpleNamespace(store=store, config=config, parent=parent, full=full,
        baseline=baseline, child=None, trial=trial,
        compare=lambda left, right: compare_observations(config, left, right))
    yield value
    store.close()


def resolve(case):
    ensure_child(case)
    return resolve_parent_comparison(case.store, case.trial, case.full.source_tree_ref,
                                     case.child, case.compare)


def ensure_child(case):
    if case.child is None:
        case.child = observation(case.store, 'revised', source(3), .6207)


def test_actual_regression_values_use_bound_parent_not_baseline_or_prose(case):
    matched = observation(case.store, 'parent-screening', case.parent, .56)
    context, actual = resolve(case)
    assert context['schema_version'] == CONTEXT_VERSION
    assert context['status'] == 'available' and actual == matched
    assert context['comparison']['delta'] == pytest.approx(.0607)
    assert case.compare(case.child, case.baseline)['delta'] == pytest.approx(.152)
    assert context['reference']['id'] == matched.id
    assert context['reference']['execution_refs'] == matched.log_refs
    assert context['reference']['prediction_refs'] == matched.prediction_refs
    assert context['reference']['protocol_hash'] == case.child.protocol_hash


def test_missing_parent_phase_does_not_fall_back_to_global_baseline(case):
    context, actual = resolve(case)
    assert actual is None and context['status'] == 'unavailable'
    assert context['reference'] is context['comparison'] is None
    assert context['new_experiments'] == 0
    assert 'separate explicit admission' in context['reason']


@pytest.mark.parametrize('kind', ['protocol', 'source', 'command', 'missing_receipts'])
def test_incompatible_parent_reference_is_not_used(case, kind):
    candidate = source(999) if kind == 'source' else case.parent
    if kind == 'command':
        candidate = source(2, command=['python', 'other.py'])
    observation(case.store, 'incompatible', candidate, .9,
                protocol_tag='different' if kind == 'protocol' else 'same',
                receipts=kind != 'missing_receipts')
    context, actual = resolve(case)
    assert actual is None and context['status'] == 'unavailable'
    assert context['comparison'] is None
    reason = {'protocol': 'protocol_mismatch', 'source': 'source_or_command_mismatch',
              'command': 'source_or_command_mismatch', 'missing_receipts': 'execution_evidence_incomplete'}[kind]
    assert context['rejected_reference_counts'][reason] >= 1


def test_earliest_compatible_parent_is_reused_without_score_cherry_picking(case):
    earliest = observation(case.store, 'parent-first', case.parent, .56)
    observation(case.store, 'parent-later', case.parent, .9)
    context, actual = resolve(case)
    assert actual.id == earliest.id
    assert context['comparison']['delta'] == pytest.approx(.0607)


def test_descriptive_provenance_does_not_change_source_equivalence(case):
    observation(case.store, 'parent-alias', source(2, description='Different prose only'), .56)
    assert resolve(case)[0]['status'] == 'available'


def test_corrupt_parent_receipt_remains_fatal(case):
    matched = observation(case.store, 'parent-screening', case.parent, .56)
    (case.store.root / matched.log_refs[0]).write_text('{}')
    with pytest.raises(IntegrityError):
        resolve(case)


def test_parent_execution_cannot_be_rebound_to_other_source(case):
    ensure_child(case)
    other = case.store.put_json(candidate_snapshot(case.store, source(999)))
    with pytest.raises(IntegrityError, match='parent source differs'):
        resolve_parent_comparison(case.store, case.trial, other, case.child, case.compare)


@pytest.mark.parametrize('missing', ['source', 'failed_execution'])
def test_original_parent_requires_successful_executed_source_binding(case, missing):
    original = observation(case.store, 'unusable-original-parent', case.parent, .6, phase='full',
                           source_binding=missing != 'source',
                           status='execution_failed' if missing == 'failed_execution' else 'success')
    # An otherwise compatible earlier observation cannot manufacture the missing
    # execution/source provenance of the actually declared parent.
    observation(case.store, 'compatible-screening', case.parent, .56)
    case.trial = {**case.trial, 'id': original.id, 'result': original.model_dump()}
    context, actual = resolve(case)
    assert actual is None and context['status'] == 'unavailable'
    assert context['reference'] is context['comparison'] is None
    assert context['parent_original_execution_status'] == original.status
    assert context['parent_original_source_tree_ref'] == original.source_tree_ref
    assert 'candidate source alone is insufficient' in context['reason']


def test_a_batch_receipt_can_cover_multiple_matched_seeds(case):
    parent = observation(case.store, 'matched-grid', case.parent, .56, seeds=[0, 1, 2])
    case.child = observation(case.store, 'revised-grid', source(3), .6207, seeds=[0, 1, 2])
    context, actual = resolve(case)
    assert actual == parent and context['status'] == 'available'
    assert len(parent.log_refs) == 1 and len(parent.values) == len(parent.prediction_refs) == 3
    assert context['comparison']['delta'] == pytest.approx(.0607)


def test_final_evidence_is_never_a_parent_control(case):
    observation(case.store, 'protected-final', case.parent, .99, phase='final')
    context, actual = resolve(case)
    assert actual is None and 'protected-final' not in str(context)
    final = observation(case.store, 'bad-child-final', source(3), .99, phase='final')
    with pytest.raises(IntegrityError, match='protected final'):
        resolve_parent_comparison(case.store, case.trial, case.full.source_tree_ref, final, case.compare)


def test_context_replay_adds_no_model_or_experiment_operations(case):
    observation(case.store, 'parent-screening', case.parent, .56)
    ensure_child(case)
    limits = SimpleNamespace(campaign_seconds=600, max_calls=10, max_experiments=10, token_stop_threshold=None)
    budget = Budget(case.store, limits)
    budget.reserve('synthetic-existing-model', 'model', {})
    budget.finish('synthetic-existing-model', 'success', {'synthetic': True}, 0, 0)
    budget.reserve('synthetic-existing-experiment', 'experiment', {})
    budget.finish('synthetic-existing-experiment', 'success', {'synthetic': True}, 0)
    before = budget.summary()
    first = resolve(case)
    second = resolve(case)
    assert first == second and budget.summary() == before


def controller_shell(case):
    ensure_child(case)
    controller = ResearchController.__new__(ResearchController)
    controller.store = case.store
    controller.research_enabled = True
    controller.trials = []
    controller.comparison = case.compare
    controller.parent_input = lambda trial: (None, case.full.source_tree_ref, None, None)
    return controller


def test_initial_seed_critic_context_is_unchanged(case):
    controller = controller_shell(case)
    original = controller.evidence_package([case.baseline, case.child], trials=[])
    assert controller.critic_evidence(case.child, case.baseline) == {'evidence_package': original}


def test_parent_reference_is_structured_in_actual_critic_context(case):
    parent = observation(case.store, 'parent-screening', case.parent, .56)
    controller = controller_shell(case)
    context = controller.critic_evidence(case.child, case.baseline, case.trial)
    observations = context['evidence_package']['observations']
    assert [row['id'] for row in observations] == [case.baseline.id, case.child.id, parent.id]
    supplied = observations[-1]
    assert supplied['metrics'] == {'quality': .56}
    assert supplied['protocol']['hash'] == case.child.protocol_hash
    assert supplied['training_executions'][0]['receipt_ref'] == parent.log_refs[0]
    assert supplied['source']['source_tree_ref'] == parent.source_tree_ref
    assert context['parent_comparison']['comparison']['delta'] == pytest.approx(.0607)


def test_later_parent_execution_cannot_rewrite_an_earlier_critic_context(case):
    initial = resolve(case)
    assert initial[0]['status'] == 'unavailable'
    observation(case.store, 'later-matched-parent', case.parent, .56)
    assert resolve(case) == initial


@pytest.mark.parametrize('veto', [False, True])
def test_controller_preserves_fixed_baseline_veto_and_completed_replay(tmp_path, veto):
    from scientisttwo.live.models import Hypothesis
    from scientisttwo.live.parent_edits import ParentChangePlan
    from test_research_workflow import campaign

    controller, store, runner, agents, transport = campaign(tmp_path)
    original_response = agents.respond

    def respond(role, inputs):
        if role == 'critic' and 'parent_comparison' in inputs and veto:
            agents.seen.append((role, inputs))
            return {'decision': 'Bad', 'reason': 'Synthetic methodological veto despite numerical gains'}
        return original_response(role, inputs)

    agents.responder = respond
    idea = Hypothesis(title='Synthetic initial', hypothesis='Fixture offset', novelty_score=0,
                      novelty_scope='No novelty claim', source_ids=['task'])

    def run():
        controller.trials = []
        runner.preflight()
        baseline = controller.adapter.baseline()
        controller.baseline_candidate = baseline
        controller.baselines = {phase: runner.execute('baseline/' + phase, baseline, phase)
                                for phase in ('subset', 'full')}
        first = controller.evaluate('first', idea)
        assert first is not None
        return controller.evaluate('derived', idea, implementation_parent=first,
            change_plan=ParentChangePlan(changed_files=['custom.py'], reason='Synthetic offset change'))

    try:
        store.set('live_state', {'stage': 'search', 'status': 'running'})
        result = run()
        assert (result is None) is veto
        critics = [inputs for role, inputs in agents.seen if role == 'critic']
        initial = [inputs for inputs in critics if 'parent_comparison' not in inputs]
        derived = [inputs for inputs in critics if 'parent_comparison' in inputs]
        assert len(initial) == 2 and len(derived) == (1 if veto else 2)
        for inputs in derived:
            context = inputs['parent_comparison']
            assert context['status'] == 'available'
            assert context['reference']['id'].startswith('first/')
            assert inputs['comparison']['delta'] != context['comparison']['delta']
            assert inputs['comparison']['improved'] and context['comparison']['improved']
            assert inputs['evidence_package']['observations'][-1]['id'] == context['reference']['id']
        before = (controller.budget.summary(), len(agents.seen), len(transport.stages))
        again = run()
        assert (again is None) is veto
        assert (controller.budget.summary(), len(agents.seen), len(transport.stages)) == before
    finally:
        store.close()
