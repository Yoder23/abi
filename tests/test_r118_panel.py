from collections import Counter
import re
from experiments.modality_roles_r118 import panel


def test_counterfactual_design_without_filter_predictions():
    p=panel.build()
    assert len(p['contexts'])==288 and len(p['empty'])==96 and p==panel.build()
    grid=Counter((x['family'],x['state'],x['count']) for x in p['contexts'])
    assert len(grid)==48 and set(grid.values())=={6}
    for x in p['contexts']:
        ch=x['challenges']
        assert len(ch)==8 and len(set(ch.values()))==8
        assert set(x['roster'])==set(re.findall(r'\b(?:'+'|'.join(x['roster'])+r')\b',ch['valid']))
        assert Counter(ch['actor_swap'].split())!=Counter() and sorted(re.findall(r'\w+',ch['valid']))==sorted(re.findall(r'\w+',ch['actor_swap']))
        assert x['target'] in dict(x['notes']) and len(x['notes'])==x['count']
        assert dict(x['notes'])[x['target']]!=ch['valid'] and x['target'] not in ch['off_target']
