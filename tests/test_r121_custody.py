import copy
import pytest
from experiments.source_ceiling_r121 import adjudication as j,core as c,run as r


@pytest.fixture(scope='module')
def evidence():
    cases,rows=r.raw()
    return cases,rows,r.labels(cases,rows)


@pytest.mark.parametrize('damage',['missing_row','duplicate_row','different_arm','new_label','removed_label','bad_label'])
def test_cannot_change_population_or_blind_assignments(evidence,damage):
    cases,raw,gold=evidence
    rows=list(raw)
    judgments=dict(gold)
    if damage=='missing_row':
        rows.pop()
    elif damage=='duplicate_row':
        rows[5]=rows[4]
    elif damage=='different_arm':
        rows[0]=dict(rows[0],arm='unknown')
    elif damage=='new_label':
        judgments['new']='V'
    elif damage=='removed_label':
        judgments.pop(next(iter(judgments)))
    elif damage=='bad_label':
        judgments[next(iter(judgments))]='probably'
    with pytest.raises(ValueError):
        c.classify(cases,rows,judgments)


@pytest.mark.parametrize('damage',['unreviewed','duplicate','out_of_range','bool_index','unknown_label'])
def test_vote_table_fail_closed(evidence,damage):
    task=c.read(j.BLIND)
    votes=copy.deepcopy(c.read(j.VOTES))
    k=next(k for k in votes if votes[k])
    if damage=='unreviewed':
        votes[k].pop()
    elif damage=='duplicate':
        votes[k].append(votes[k][0])
    elif damage=='out_of_range':
        votes[k][0]=len(task)
    elif damage=='bool_index':
        votes[k][0]=False
    else:
        votes['maybe']=[]
    with pytest.raises(ValueError):
        j.judgments(task,votes)


def test_public_seal_and_both_fresh_execution_receipts(evidence):
    from scripts.review_r121 import inspect
    d=inspect()
    assert (d['verdict']=='SOURCE_CEILING_FEASIBLE_ONLY')==any(z['ceiling_pass'] for z in d['arms'].values())
    assert d['wp2']=='CLOSED'


@pytest.mark.parametrize('damage', ['failed_exit','stale_packet','stale_weights','shared_process','wrong_token_cost'])
def test_replay_receipt_mutations_fail_without_touching_evidence(monkeypatch,damage):
    original=c.read
    initial_pid=original(r.OUT/'initial.json')['pid']
    def shadow(path):
        obj=original(path)
        if path==r.OUT/'replay.json':
            if damage=='failed_exit':
                obj['exit']=1
            if damage=='shared_process':
                obj['pid']=initial_pid
        if path==r.OUT/'replay'/'receipt.json':
            if damage=='shared_process':
                obj['pid']=initial_pid
            elif damage=='stale_packet':
                obj['packet']='other'
            elif damage=='stale_weights':
                obj['source']='other'
            elif damage=='wrong_token_cost':
                obj['costs']['fields']['output_tokens']+=1
        return obj
    monkeypatch.setattr(c,'read',shadow)
    with pytest.raises(ValueError):
        r.raw()
