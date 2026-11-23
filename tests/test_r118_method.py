import pytest
from experiments.modality_roles_r118 import method as m

T='P1234ABCD'
N=['Ari','Zed']


@pytest.mark.parametrize('s,k', [('Ari may sing to Zed.','authorized'),('Ari plans to sing to Zed.','planned'),
                                ('Ari chanted to Zed.','occurred'),('Ari never chanted to Zed.',None),
                                ('Ari plans to get permission.',None),('Ari and Zed.',None)])
def test_states(s,k):
    assert m.state(s)==k


@pytest.mark.parametrize('s,k', [('To Zed, Ari chanted.',N),('Zed was serenaded by Ari.',N),
                                ('Zed chanted to Ari.',N[::-1]),('Zed and Ari.',None),('Ari chanted.',None)])
def test_roles(s,k):
    assert m.roles(s,N)==k


def test_masks_do_not_change_baseline():
    f=T+': Ari plans to sing to Zed.'
    for text,guard in [(T+': Ari may sing to Zed.','state'),(T+': Zed plans to sing to Ari.','roles')]:
        out=m.decide(T,[(T,f)],N,text,[[0,3,0]]*3)['arms']
        assert out['r117']['reason']=='accept' and out['abi']['reason']==guard+'_mismatch'
