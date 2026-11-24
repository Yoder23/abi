import copy
import pytest
from experiments.project_binding_r120 import run as r
from experiments.project_binding_r120 import analysis


@pytest.fixture(scope='module')
def evidence():
    p=r.read(r.OUT/'packet.json')
    live=r.receipts('teacher')
    return p, live, r.check_gold(live,p), r.receipts('nli')


@pytest.mark.parametrize('damage', ['missing','duplicate','payload','binding','stale_ids','missing_logits','infinite','verdict','live_swap','missing_label','swapped_slot','relabel','alias_order','swap_gap','fake_parse','alter_body','fake_cache','changed_body_logits'])
def test_mutations_rejected(evidence,damage):
    p,live,gold,pred=evidence
    pred=pred.copy()
    pred[0]=copy.deepcopy(pred[0])
    live=live.copy()
    if damage=='missing':
        pred.pop()
    elif damage=='duplicate':
        pred[1]=pred[0]
    elif damage=='payload':
        pred[0]['candidate_sha256']='0'*64
    elif damage=='binding':
        pred[0]['input_sha256']='0'*64
    elif damage=='stale_ids':
        pred[0]['ids'][1]=[]
    elif damage=='missing_logits':
        pred[0]['logits'].pop()
    elif damage=='infinite':
        pred[0]['logits'][1][0]=float('inf')
    elif damage=='verdict':
        pred[0]['decisions']['arms']['abi']['reason']='invented'
    elif damage=='swapped_slot':
        pred[0]['decisions']['slots'][0]['roles']=['nobody','anybody']
    elif damage=='relabel':
        gold=dict(gold,rows=[dict(gold['rows'][0],text_sha256='stale'),*gold['rows'][1:]])
    elif damage=='alias_order':
        pred[0]['logits'][3][1] += 10
    elif damage=='swap_gap':
        pred[0]['decisions']['gaps'][0] += 1
    elif damage=='fake_parse':
        pred[0]['decisions']['bindings'][1]['reason']='invented'
    elif damage=='alter_body':
        pred[0]['decisions']['bindings'][1]['body_sha256']='0'*64
    elif damage=='fake_cache':
        pred[0]['new_slots']=[999]
    elif damage=='changed_body_logits':
        pred[0]['logits'][11][1]+=10
    elif damage=='live_swap':
        live[0]=live[1]
    else:
        gold=dict(gold,rows=gold['rows'][:-1])
    with pytest.raises(ValueError):
        analysis.analyze(p,live,gold,pred)


@pytest.mark.parametrize('field,value', [('review_group',-1),('group_sha256','stale'),('label','?')])
def test_blind_judgment_edits_rejected(evidence,field,value):
    from scripts.review_r120 import verify_review_groups
    packet,live,gold,_=evidence
    rows=[dict(gold['rows'][0],**{field:value}),*gold['rows'][1:]]
    with pytest.raises(ValueError,match='prediction-blind judgment'):
        verify_review_groups(packet,live,dict(gold,rows=rows))
