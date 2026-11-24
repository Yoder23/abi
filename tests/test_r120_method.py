import pytest
from experiments.project_binding_r120 import method as m

T='P1234ABCD'
B='Nia called Tess.'
S=T+': '+B
N=[(T,S)]
R=['Nia','Tess']
E,C=[0,3,1],[3,0,1]
L=[E]*3+[E,E,C,C]*4


@pytest.mark.parametrize('head',[T+': ','Project '+T+': ','Project ID: '+T+'. ','For project '+T+', '])
def test_only_delimited_prefix_is_removed(head):
    text=head+B
    assert m.parse(T,text)==(B,'bound')
    p=m.pairs(T,N,R,text)
    assert len(p)==19 and p[11]==('Alice called Bob.','Alice called Bob.')
    assert m.decide(T,N,R,text,L)['arms']['abi']['retained_sha256']==m.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize('text',[B,'P1234ABCE: '+B,T+'9: '+B,'project '+T.lower()+': '+B,
                                S+' P1234ABCE.',S+' '+T,T+':  ','Status: '+T+'. '+B,
                                'P1234ABCE: '+S,T+' is planned: '+B])
def test_bad_bindings_cannot_be_overridden(text):
    assert m.parse(T,text)[0] is None
    assert m.decide(T,N,R,text,L)['arms']['abi']['reason']!='accept'


def test_guard_and_semantic_interventions_separate():
    l=L.copy()
    l[3]=C
    out=m.decide(T,N,R,S,l)['arms']
    assert out['r119']==out['strict_raw'] and out['r119']['reason']!='accept' and out['abi']['reason']=='accept'
    with pytest.raises(ValueError):
        m.decide(T,N,R,S,L[:-1])
