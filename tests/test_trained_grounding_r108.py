"""No model calls: prospective populations, conservative gates, hostile raw edits."""
import copy

import pytest

from experiments.trained_grounding_r108 import core as c
from tests.test_evaluator_control_r104 import Tokenizer


def fixture():
    ref = {'id': 'fixture-a', 'pair': 'fixture', 'panel': 'a', 'task': 'rewriting',
           'prompt': 'NOTEQA is coordinated by PersonQA for Monday. Approval is pending.',
           'required': ['NOTEQA', 'PersonQA', 'Monday']}
    tok = Tokenizer()
    prompt = c.render(ref, 'parent', tok)
    tokens = tok.encode(ref['prompt']) + [0]
    route = c.ROUTES.index(ref['task'])
    raw = {**{k: ref[k] for k in ('id', 'pair', 'panel', 'task')}, 'arm': 'parent', 'prompt': prompt,
           'prompt_ids': tok.encode(prompt), 'token_ids': tokens, 'output': ref['prompt'],
           'screen': c.screen(ref, ref['prompt']), 'output_utf8_bytes': len(ref['prompt'].encode()),
           'output_characters': len(ref['prompt']), 'wall_seconds': .01, 'stop_reason': 'eos',
           'execution_routes': [{'cakes': [route], 'adapters': [[route]]*6}]*len(tokens)}
    return ref, raw, tok


def synthetic_panel():
    return [{**ref, 'screen': {'mechanical_pass': True, 'collapse': False}, 'stop_reason': 'eos',
             'prompt_ids': [1], 'token_ids': [2], 'output_utf8_bytes': 1, 'output': 'x'} for ref in c.catalog()]


def test_prospective_pairs_cover_same_trained_tasks():
    rows = c.catalog()
    assert rows == c.catalog() and len(rows) == 96
    assert len({r['prompt'] for r in rows}) == 96
    assert not {r['code'] for r in rows} & {r['code'] for r in c.prior.catalog('qualification')}
    for a, b in zip(rows[::2], rows[1::2]):
        assert a['pair'] == b['pair'] and a['task'] == b['task'] and a['code'] != b['code']
        assert all(v in r['prompt'] for r in (a, b) for v in r['required'])
    for panel in ('a', 'b'):
        assert all(sum(r['panel'] == panel and r['task'] == t for r in rows) == 6 for t in c.TASKS)


def test_no_gate_lowering_and_extended_audit():
    value = c.audit()
    assert len(value['cases']) == 164 and not value['mismatches']
    for case in value['cases']:
        if case['screen']['mechanical_pass']:
            assert c.prior.screen(case['row'], case['answer'])['mechanical_pass']


def test_both_members_and_every_task_must_pass():
    rows = synthetic_panel()
    assert c.summarize(rows)['qualified_mechanical_only']
    task = rows[0]['task']
    for row in [r for r in rows if r['task'] == task and r['panel'] == 'a'][:2]:
        row['screen']['mechanical_pass'] = False
    assert not c.summarize(rows)['qualified_mechanical_only']
    assert c.summarize(rows)['both_pass']['passing'] == 46


def test_pair_gate_stricter_than_separate_panel_gate():
    rows = synthetic_panel()
    for panel, task in zip(('a', 'b'), (c.TASKS[0], c.TASKS[0])):
        selected = [r for r in rows if r['panel'] == panel and r['task'] == task]
        selected[0 if panel == 'a' else 1]['screen']['mechanical_pass'] = False
    assert all(c.prior.qualified([r for r in rows if r['panel'] == p]) for p in ('a', 'b'))
    assert not c.summarize(rows)['qualified_mechanical_only']


@pytest.mark.parametrize('kind', ['collapse', 'budget', 'missing', 'duplicate', 'reordered'])
def test_incomplete_or_collapsed_panels_never_qualify(kind):
    rows = synthetic_panel()
    if kind == 'collapse':
        rows[0]['screen']['collapse'] = True
    elif kind == 'budget':
        rows[0]['stop_reason'] = 'budget'
    elif kind == 'missing':
        rows.pop()
    elif kind == 'duplicate':
        rows[-1] = rows[0]
    else:
        rows.reverse()
    if kind in ('missing', 'duplicate', 'reordered'):
        with pytest.raises(RuntimeError):
            c.summarize(rows)
    else:
        assert not c.summarize(rows)['qualified_mechanical_only']


def test_fixed_interface_and_matched_contrast():
    ref, raw, tok = fixture()
    assert c.render(ref, 'parent', tok) == c.prior.base.model_prompt(ref, 'distillation')
    assert c.render(ref, 'control', tok).endswith('assistant:')
    c.validate([raw], [ref], 'parent', tok)
    rows = synthetic_panel()
    assert c.contrast(rows, rows)['ci95'] == [0, 0]
    worse = copy.deepcopy(rows)
    for row in worse:
        row['screen']['mechanical_pass'] = False
    assert c.contrast(rows, worse)['ci95'] == [1, 1]


@pytest.mark.parametrize('kind', ['rows', 'id', 'pair', 'panel', 'arm', 'prompt', 'tokens', 'text', 'screen',
                                  'stop', 'early_eos', 'route', 'time_nan', 'time_zero', 'bytes'])
def test_raw_edits_fail_closed(kind):
    ref, raw, tok = fixture()
    rows = [raw]
    if kind == 'rows':
        rows = []
    elif kind in ('id', 'pair', 'panel', 'arm', 'prompt'):
        raw[kind] = 'wrong'
    elif kind == 'tokens':
        raw['token_ids'][0] = -1
    elif kind == 'text':
        raw['output'] += 'changed'
    elif kind == 'screen':
        raw['screen']['mechanical_pass'] = False
    elif kind == 'stop':
        raw['stop_reason'] = 'budget'
    elif kind == 'early_eos':
        raw['token_ids'][0] = 0
    elif kind == 'route':
        raw['execution_routes'] = None
    elif kind.startswith('time_'):
        raw['wall_seconds'] = float('nan') if kind == 'time_nan' else 0
    else:
        raw['output_utf8_bytes'] += 1
    with pytest.raises(RuntimeError):
        c.validate(rows, [ref], 'parent', tok)


def test_missing_freeze_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        c.binding()
    c.write(tmp_path/'freeze.json', {'files': {}})
    with pytest.raises(RuntimeError):
        c.binding()
