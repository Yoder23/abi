import copy
import pytest
from experiments.voice_converse_r119 import run as r
from experiments.voice_converse_r119 import analysis


@pytest.fixture(scope='module')
def evidence():
    p=r.read(r.OUT/'packet.json')
    live=r.receipts('teacher')
    return p, live, r.check_gold(live,p), r.receipts('nli')


@pytest.mark.parametrize('damage', ['missing','duplicate','payload','binding','stale_ids','missing_logits','infinite','verdict','live_swap','missing_label','swapped_slot','relabel','alias_order','swap_gap'])
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
    elif damage=='live_swap':
        live[0]=live[1]
    else:
        gold=dict(gold,rows=gold['rows'][:-1])
    with pytest.raises(ValueError):
        analysis.analyze(p,live,gold,pred)
