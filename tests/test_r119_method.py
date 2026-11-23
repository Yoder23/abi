import pytest
from experiments.voice_converse_r119 import method as m

T='P1234CAFE'
S=T+': Kim called Ray.'
N=[(T,S)]
R=['Kim','Ray']
E=[-1,2,0]
NO=[2,0,-1]
L=[E]*3+[E,E,NO,NO]*2


def test_aligned_and_counterfactual_are_distinct_and_simultaneous():
    p=m.pairs(T,N,R,S)
    assert len(p)==11 and p[3]==(T+': Alice called Bob.',)*2
    assert p[5]==(p[3][0],T+': Bob called Alice.') and p[7]==(p[5][1],)*2
    assert p[6]==p[5][::-1]
    assert m.decide(T,N,R,S,L)['arms']['abi']['reason']=='accept'


def test_tie_blocks_only_contrast():
    d=m.decide(T,N,R,S,[E]*11)['arms']
    assert d['status_only']['reason']==d['alias_only']['reason']=='accept'
    assert d['abi']['reason']=='swap_not_separated'


@pytest.mark.parametrize('text',[T+': Kim called.',T+': Kim called Ray and Bob.',T+': Kim plans to call Ray.',''])
def test_fail_closed(text):
    assert m.decide(T,N,R,text,L)['arms']['abi']['retained_sha256'] is None


def test_uncertain_single_assignment_is_enough_to_withhold():
    bad=L.copy()
    bad[8]=NO
    assert m.decide(T,N,R,S,bad)['arms']['alias_only']['reason']=='aligned_not_entailed'
    with pytest.raises(ValueError):
        m.decide(T,N,R,S,L[:-1])
    with pytest.raises(ValueError):
        m.decide(T,N,R,S,L[:-1]+[[0,1,float('nan')]])
