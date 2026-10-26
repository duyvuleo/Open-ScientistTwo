"""Purely synthetic provenance contracts; no historical research records.

No model, experiment, container or remote connection is invoked. A new explicitly
rebound claim is checked against synthetic Store commitments created in the test.
"""
from copy import deepcopy

import pytest

from scientisttwo.live.models import Candidate, Hypothesis, Observation, SourceFile
from scientisttwo.models import CriticDecision
from scientisttwo.research.manuscript import (
    LedgerClaim, ManuscriptRecord, build_evidence_ledger, fixture_manuscript,
    freeze_research_history, render_manuscript, validate_manuscript,
    verify_evidence_ledger, writer_history_contract,
)
from scientisttwo.store import IntegrityError, Store, canonical, digest


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path/'synthetic-campaign')
    yield value
    value.close()


def commit(store, name='search/0/subset/0', *, phase='subset', value=.5,
           status='success', decision='Bad', reason='Synthetic scientific rejection'):
    candidate = Candidate(files=[SourceFile(path='train.py', content="print('synthetic; never executed')\n")],
                          command=['python', 'train.py'], description='Synthetic provenance-only candidate')
    protocol = {'phase': phase, 'seeds': [0], 'adapter': 'synthetic-provenance-test'}
    protocol_ref = store.put_json(protocol, 'synthetic-protocol')
    observation = Observation(id=name, phase=phase, status=status, metric='accuracy',
        value=value if status=='success' else None, values=[value] if status=='success' else [],
        code_ref=store.put_json(candidate), protocol_ref=protocol_ref, protocol_hash=digest(canonical(protocol)),
        prediction_refs=[store.put_json({'synthetic_prediction': True})] if status=='success' else [],
        log_refs=[store.put_json({'synthetic_execution_receipt': True})], reason='Synthetic test observation')
    store.cache('development/'+name, {'synthetic': True}, observation)
    trial = {'id':name,
             'idea':Hypothesis(title='Synthetic control', hypothesis='Synthetic hypothesis, no research claim',
                               novelty_score=0, novelty_scope='Synthetic unit test', source_ids=['task']).model_dump(),
             'candidate':candidate.model_dump(), 'result':observation.model_dump(),
             'decision':CriticDecision(decision=decision, reason=reason).model_dump(), 'parents':[]}
    store.cache('live-trial/'+name, {'key':name}, trial)
    return trial


def history_entry(ledger, trial_id):
    return next((key,value) for key,value in ledger['entries'].items()
                if value.get('kind')=='research_history' and value['trial_id']==trial_id)


def test_history_separates_rejected_science_from_execution_and_exposes_citable_ids(store):
    trial=commit(store)
    refs=freeze_research_history(store,[trial])
    ledger=build_evidence_ledger(store,[],history_refs=refs)
    key,entry=history_entry(ledger,trial['id'])
    assert ledger['schema_version']=='evidence-ledger-v2'
    assert ledger['research_history']['schema_version']=='research-history-context-v1'
    assert entry['execution_status']=='success' and entry['decision']['decision']=='Bad'
    assert entry['trial_ref']==refs[0] and entry['observation_ref']
    assert len(entry['measurement_ids'])==1
    measured=ledger['entries'][entry['measurement_ids'][0]]
    assert measured['kind']=='measured' and measured['value']==.5
    manuscript=fixture_manuscript(ledger)
    manuscript.claims.append(LedgerClaim(id='history',kind='interpretation',text='The successful control was scientifically rejected.',evidence_ids=[key]))
    manuscript.discussion.claim_ids.append('history')
    assert validate_manuscript(manuscript,ledger).status=='pass'
    contract=writer_history_contract(ledger)
    assert contract['schema_version']=='research-writer-context-v2'
    assert key in contract['history_ids']
    verify_evidence_ledger(store,ledger)


def test_failed_execution_is_citable_history_but_never_a_numeric_measurement(store):
    trial=commit(store,status='invalid',decision='Bad')
    ledger=build_evidence_ledger(store,[],history_refs=freeze_research_history(store,[trial]))
    key,entry=history_entry(ledger,trial['id'])
    assert entry['execution_status']=='invalid' and entry['measurement_ids']==[]
    assert not any(value['kind']=='measured' for value in ledger['entries'].values())
    manuscript=fixture_manuscript(ledger)
    manuscript.claims.append(LedgerClaim(id='invented_score',kind='measured',text='Recorded score',evidence_ids=[key],value=.9))
    manuscript.results.claim_ids.append('invented_score')
    assert validate_manuscript(manuscript,ledger).status=='fail'


@pytest.mark.parametrize('mutation',['offered_decision','offered_value','missing_trial','modified_trial','modified_observation','foreign','final_phase','final_id'])
def test_fabricated_missing_modified_foreign_and_final_history_rejected(store,tmp_path,mutation):
    if mutation=='final_phase':
        trial=commit(store,phase='final')
    elif mutation=='final_id':
        trial=commit(store,name='final/disguised',phase='full')
    else:
        trial=commit(store)
    ref=store.db.execute('SELECT output_ref FROM calls WHERE key=?',('live-trial/'+trial['id'],)).fetchone()[0]
    if mutation=='offered_decision':
        trial['decision']['decision']='Good'
    elif mutation=='offered_value':
        trial['result']['value']=.99
    elif mutation=='missing_trial':
        store.db.execute('DELETE FROM calls WHERE key=?',('live-trial/'+trial['id'],))
        store.db.commit()
    elif mutation=='modified_trial':
        (store.root/ref).write_text('{}')
    elif mutation=='modified_observation':
        observed=store.db.execute('SELECT output_ref FROM calls WHERE key=?',('development/'+trial['id'],)).fetchone()[0]
        (store.root/observed).write_text('{}')
    elif mutation=='foreign':
        other=Store(tmp_path/'foreign')
        try:
            foreign=commit(other,'foreign/subset/0')
            foreign_ref=other.db.execute('SELECT output_ref FROM calls WHERE key=?',('live-trial/'+foreign['id'],)).fetchone()[0]
            imported=store.put(other.read(foreign_ref))
            with pytest.raises(IntegrityError,match='not committed'):
                build_evidence_ledger(store,[],history_refs=[imported])
            return
        finally:
            other.close()
    with pytest.raises((IntegrityError,ValueError)):
        freeze_research_history(store,[trial])


def test_missing_history_artifact_and_duplicate_refs_rejected(store):
    trial=commit(store)
    refs=freeze_research_history(store,[trial])
    with pytest.raises(IntegrityError,match='duplicated'):
        build_evidence_ledger(store,[],history_refs=refs+refs)
    (store.root/refs[0]).unlink()
    with pytest.raises(IntegrityError):
        build_evidence_ledger(store,[],history_refs=refs)


def test_later_trials_cannot_expand_frozen_history_or_change_earlier_verification(store):
    early=commit(store)
    ledger=build_evidence_ledger(store,[],history_refs=freeze_research_history(store,[early]))
    before=canonical(ledger)
    later=commit(store,'search/later/full/0',phase='full',value=.8,decision='Good')
    commit(store,'final/later',phase='final',value=.9,decision='Good')
    verify_evidence_ledger(store,ledger)
    assert canonical(ledger)==before
    assert later['id'] not in ledger['observations']
    assert len(ledger['research_history']['trial_refs'])==1
    rendered=render_manuscript(store,fixture_manuscript(ledger),ledger)
    assert rendered.compilation.status=='unavailable'
    assert not store.verify()


def test_rebound_commitment_is_not_silently_substituted_into_old_history(store):
    trial=commit(store)
    ledger=build_evidence_ledger(store,[],history_refs=freeze_research_history(store,[trial]))
    changed=deepcopy(trial)
    changed['decision']['reason']='Different decision after the fact'
    ref=store.put_json(changed)
    store.db.execute('UPDATE calls SET output_ref=? WHERE key=?',(ref,'live-trial/'+trial['id']))
    store.db.commit()
    with pytest.raises(IntegrityError,match='not committed'):
        verify_evidence_ledger(store,ledger)


def test_legacy_v1_ledger_remains_unchanged_and_does_not_adopt_history(store):
    trial=commit(store)
    ledger=build_evidence_ledger(store,[trial['result']])
    before=canonical(ledger)
    assert ledger['schema_version']=='evidence-ledger-v1' and 'research_history' not in ledger
    commit(store,'later/full',phase='full')
    verify_evidence_ledger(store,ledger)
    assert canonical(ledger)==before
    changed=deepcopy(ledger)
    changed['research_history']={'schema_version':'research-history-context-v1','trial_refs':[]}
    with pytest.raises(IntegrityError,match='Legacy'):
        verify_evidence_ledger(store,changed)


def test_unbound_synthetic_history_requires_a_new_explicit_grounding(store):
    trials = [
        commit(store, 'synthetic/0/subset/0', value=.25),
        commit(store, 'synthetic/1/subset/0', value=.75),
    ]
    legacy = build_evidence_ledger(store, [trial['result'] for trial in trials])
    draft = fixture_manuscript(legacy)
    draft.claims.append(LedgerClaim(id='unbound-history', kind='interpretation',
        text='Both synthetic trials executed successfully but were scientifically rejected.'))
    draft.discussion.claim_ids.append('unbound-history')
    original_bytes = canonical(draft)
    original_ledger = canonical(legacy)
    old = ManuscriptRecord.model_validate_json(original_bytes)
    failure = validate_manuscript(old, legacy)
    assert failure.errors == ['Interpretation unbound-history requires evidence or a source and remains unverified']

    history = build_evidence_ledger(store, [], history_refs=freeze_research_history(store, trials))
    verify_evidence_ledger(store, history)
    context = deepcopy(legacy)
    context['entries'].update(history['entries'])
    # Available history never silently repairs an already serialized claim.
    assert validate_manuscript(old, context).status == 'fail'
    rebound = old.model_copy(deep=True)
    claim = next(claim for claim in rebound.claims if claim.id == 'unbound-history')
    claim.evidence_ids = [history_entry(history, trial['id'])[0] for trial in trials]
    assert validate_manuscript(rebound, context).status == 'pass'
    assert canonical(old) == original_bytes and canonical(legacy) == original_ledger
    assert claim.text == next(claim.text for claim in old.claims if claim.id == 'unbound-history')


def test_writer_receives_versioned_citable_negative_history_from_controller(tmp_path):
    from test_research_workflow import campaign
    controller, store, _, agents, _ = campaign(tmp_path, reject=True)
    try:
        outcome=controller.run()
        assert outcome['status']=='complete' and outcome['outcome']=='no_improvement'
        writers=[inputs for role,inputs in agents.seen if role=='initial_drafter']
        assert writers
        for inputs in writers:
            assert inputs['writer_evidence_contract']['schema_version']=='research-writer-context-v2'
            assert inputs['ledger']['schema_version']=='evidence-ledger-v2'
            assert inputs['scientific_history']
            for item in inputs['scientific_history']:
                assert item['evidence_id'] in inputs['ledger']['entries']
                assert item['execution_status']=='success' and item['decision']['decision']=='Bad'
                assert store.json(item['trial_ref'])['decision']==item['decision']
                assert all(identifier in inputs['ledger']['entries'] for identifier in item['measurement_ids'])
            verify_evidence_ledger(store,inputs['ledger'])
        snapshots=[store.json(row[0]) for row in store.db.execute(
            "SELECT output_ref FROM calls WHERE key LIKE 'writer-history/%'")]
        assert snapshots and all(item['schema_version']=='research-history-context-v1' for item in snapshots)
    finally:
        store.close()
