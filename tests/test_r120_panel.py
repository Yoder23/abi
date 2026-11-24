from collections import Counter
import re
from experiments.project_binding_r120 import panel


def test_prospective_panel_and_swaps():
    p=panel.build()
    assert len(p['contexts'])==288 and len(p['empty'])==96 and p==panel.build()
    cells=Counter((x['family'],x['state'],x['count']) for x in p['contexts'])
    assert len(cells)==48 and set(cells.values())=={6}
    for c in p['contexts']:
        ch=c['challenges']
        assert len(set(ch.values()))==len(ch)==19
        assert len(c['notes'])==c['count'] and c['target'] not in ch['off_target']
        a,b=c['roster']
        pat=r'\b(?:'+a+'|'+b+r')\b'
        for v in panel.VOICES:
            pos,neg=ch['valid_'+v],ch['swap_'+v]
            assert sorted(re.findall(pat,pos))==[a,b] and c['target'] in pos
            assert re.sub(pat,lambda m:b if m[0]==a else a,pos)==neg and pos!=neg
            assert pos!=dict(c['notes'])[c['target']]
        exact=dict(c['notes'])[c['target']]
        body=exact.split(': ',1)[1]
        assert ch['valid_exact']==exact and ch['valid_prefix']=='Project '+exact
        assert ch['valid_label']=='Project ID: '+c['target']+'. '+body and ch['missing']==body
        for good,bad in [('valid_prefix','wrong_prefix'),('valid_label','wrong_label')]:
            wrong=re.search(r'P[0-9A-F]{8}',ch[bad])[0]
            assert wrong!=c['target'] and ch[bad]==ch[good].replace(c['target'],wrong)
            if bad=='wrong_label':
                assert sum(a!=b for a,b in zip(wrong,c['target']))==1
        assert ch['conflicting'].startswith('Project '+c['target']+': Project ')
    assert sum(c['family'] in panel.RENTALS for c in p['contexts'])==144
