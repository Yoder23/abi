import pytest
from experiments.grounded_normalization_r117 import method as m

T = 'P0000000A'
F = 'P0000000B'
N = [(T, T+' has a lamp.'), (F, F+' has a window.')]
E, C, TIE = [0, 3, 0], [2, 0, 1], [1, 1, 0]


def test_method_controls():
    assert m.pairs(T, N, 'x') == [(N[0][1]+'\n'+N[1][1], 'x'), (N[0][1], 'x'), ('x', N[0][1])]
    for a, b, reason in [(E, E, 'accept'), (TIE, E, 'fact_not_entailing'), (E, C, 'incomplete')]:
        v = m.decide(T, N, N[0][1], [E, a, b])
        assert v['abi']['reason'] == reason and v['one_way']['reason'] == 'accept'
    assert m.decide(T, [], T, [E]*3)['abi']['reason'] == 'no_bound_fact'
    assert m.decide(T, N, T+' '+F, [E]*3)['lexical']['reason'] == 'lexical'
    with pytest.raises(ValueError):
        m.decide(T, N, T, [[2, float('nan'), 1]]*3)
    with pytest.raises(ValueError):
        m.pairs(T, N+N, T)
