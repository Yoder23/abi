from collections import Counter
import re
from experiments.voice_converse_r119 import panel


def test_prospective_panel_and_swaps():
    p=panel.build()
    assert len(p['contexts'])==288 and len(p['empty'])==96 and p==panel.build()
    cells=Counter((x['family'],x['state'],x['count']) for x in p['contexts'])
    assert len(cells)==48 and set(cells.values())=={6}
    for c in p['contexts']:
        ch=c['challenges']
        assert len(set(ch.values()))==len(ch)==12
        assert len(c['notes'])==c['count'] and c['target'] not in ch['off_target']
        a,b=c['roster']
        pat=r'\b(?:'+a+'|'+b+r')\b'
        for v in panel.VOICES:
            pos,neg=ch['valid_'+v],ch['swap_'+v]
            assert sorted(re.findall(pat,pos))==[a,b] and c['target'] in pos
            assert re.sub(pat,lambda m:b if m[0]==a else a,pos)==neg and pos!=neg
            assert pos!=dict(c['notes'])[c['target']]
