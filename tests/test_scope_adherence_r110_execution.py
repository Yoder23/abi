"""Development/hostile software checks, never the prospective challenge score."""
import ast
import copy
import inspect
from types import SimpleNamespace

import pytest

from experiments.scope_adherence_r110 import challenge, design as d, evaluator as e, execution as x, run


def test_all_original_development_fixtures_and_known_regressions():
    observations = x.development_rows()
    assert len(observations) == 171
    for row in observations:
        actual = row['observed']['mechanical_pass'] if row['kind'] == 'historical_fixture' else row['observed']['classification']
        assert actual == row['expected'], row['id']


@pytest.mark.parametrize('text,label', [
    ('Nothing is confirmed yet.', e.SAFE),
    ('We will proceed once formal approval is granted.', e.SAFE),
    ('The arrangements are not confirmed.', e.SAFE),
    ('Approval is still pending, not confirmed.', e.SAFE),
    ('If the details are confirmed, the plans are approved.', e.UNSAFE),
    ('The plans are approved, if the details are confirmed.', e.UNSAFE),
    ('Nothing is confirmed yet; the plans are approved.', e.UNSAFE),
    ('Nothing is confirmed yet and it will be approved.', e.UNSAFE),
    ('Approval is no longer pending.', e.UNSAFE),
    ('Approval is not still pending.', e.UNSAFE),
    ('Perhaps the plans are approved.', e.UNKNOWN),
    ('It is false that the plans are approved.', e.UNKNOWN),
    ('"The plans are approved" is a quotation.', e.UNKNOWN),
    ('I do not think the plans are approved.', e.UNKNOWN),
    ('Approval materialized mysteriously.', e.UNKNOWN),
    ('The plans might be confirmed.', e.UNKNOWN),
    ('The plans are not only confirmed.', e.UNKNOWN),
    ('The plans are not necessarily pending.', e.UNKNOWN),
    ('Nothing is not confirmed.', e.UNKNOWN),
    ('Nothing is pending.', e.UNSAFE),
    ('Status: provisional', e.SAFE),
])
def test_development_scope_boundaries_and_abstentions(text, label):
    result = e.classify(text)
    assert result['classification'] == label
    for span in result['evidence']:
        assert text[span['start']:span['end']] == span['text']
        assert span['clause_start'] <= span['start'] < span['end'] <= span['clause_end']
        assert span['reason']


def test_unresolved_never_passes_and_other_observations_are_unchanged():
    fixtures = x.h.audit()['cases']
    fixed = ('required_literal_coverage', 'exact_required_literals', 'surface_form', 'nonempty_bounded',
             'requested_sentence_count', 'unknown_identifiers', 'extra_numbers', 'unsupported_days',
             'collapse', 'negated_pending_status')
    for fixture in fixtures:
        before = d.reference.screen(fixture['row'], fixture['answer'])
        after = e.screen(fixture['row'], fixture['answer'])
        assert all(before[k] == after[k] for k in fixed)
    row = fixtures[0]['row']
    assert not e.screen(row, 'Perhaps the plans are approved.')['mechanical_pass']


def test_classifier_has_no_catalog_labels_paths_or_ids():
    assert list(inspect.signature(e.classify).parameters) == ['answer']
    source = inspect.getsource(e.classify)
    for forbidden in ('expected', 'read(', 'open(', 'sha(', "['id']", "['class']"):
        assert forbidden not in source
    assert 'evaluator' not in inspect.getsource(challenge)


def test_challenge_structure_only_no_candidate_scoring():
    # These are structural generator tests, not evaluations of any generated case.
    packet = challenge.generate('a'*40, 'b'*64)
    assert packet == challenge.generate('a'*40, 'b'*64)
    assert len(packet['cases']) == len({v['text'] for v in packet['cases']}) == 600
    pools = challenge.pools()
    for group in challenge.LABELS:
        selected = [v for v in packet['cases'] if v['class'] == group]
        assert len(selected) == 100
        assert {v['family'] for v in selected} == set(pools[group])
        assert all(v['expected'] == challenge.LABELS[group] for v in selected)
    assert challenge.generate('c'*40, 'b'*64) != packet


@pytest.mark.parametrize('free,used,allowed', [(10*1024**3, 1, True), (10*1024**3-1, 1, False),
                                             (11*1024**3, 1024**3+1, False)])
def test_storage_guard(monkeypatch, free, used, allowed):
    monkeypatch.setattr(x, 'storage_observation', lambda: {'free_bytes': free, 'evidence_bytes': used,
        'minimum_free_bytes': 10*1024**3, 'maximum_evidence_bytes': 1024**3})
    if allowed:
        assert x.guard_storage()['free_bytes'] == free
    else:
        with pytest.raises(RuntimeError, match='Storage guard'):
            x.guard_storage()


@pytest.mark.parametrize('entry', [run.materialize, run.scope, run.live, lambda: run.execute('plain', 'initial')])
def test_storage_stop_precedes_source_access(monkeypatch, entry):
    def blocked():
        raise RuntimeError('Storage guard')
    monkeypatch.setattr(x, 'guard_storage', blocked)
    monkeypatch.setattr(x, 'freeze_binding', lambda *a, **k: pytest.fail('source accessed after storage stop'))
    with pytest.raises(RuntimeError, match='Storage guard'):
        entry()


def fake_scope_packet():
    # Synthetic verifier fixtures, deliberately not the challenge generator.
    items = [{'id': f'fixture-{i}', 'kind': 'historical_fixture', 'expected': True,
              'output': '', 'observed': {'mechanical_pass': True}} for i in range(164)]
    items += [{'id': f'known-{i}', 'kind': 'known_regression', 'expected': e.SAFE,
               'output': '', 'observed': {'classification': e.SAFE}} for i in range(7)]
    items += [{'id': f'{group}-{i}', 'kind': 'challenge', 'class': group, 'expected': label,
               'output': '', 'observed': {'classification': label}}
              for group, label in challenge.LABELS.items() for i in range(100)]
    storage = [{'free_bytes': 11*1024**3, 'evidence_bytes': 1000}]*771
    receipts = [{'repeat': repeat, 'pid': i+100, 'rows': 771, 'model_calls': 0,
                 'elapsed_seconds': 1., 'cpu_seconds': .5, 'storage_observations': storage}
                for i, repeat in enumerate(('initial', 'replay'))]
    return items, receipts, [{'returncode': 0}]*2, {'elapsed_seconds': 3.}, {'budget_seconds': 1800}


@pytest.mark.parametrize('mutation', ['missing_row', 'changed_label', 'forged_pass', 'missing_replay',
                                    'missing_cost', 'reused_pid', 'timeout', 'bad_operation', 'missing_storage'])
def test_scope_verifier_hostile_mutations(monkeypatch, mutation):
    expected, receipts, operations, complete, launch = fake_scope_packet()
    first, replay = copy.deepcopy(expected), copy.deepcopy(expected)
    monkeypatch.setattr(x, 'scope_rows', lambda packet: expected)
    assert x.validate_scope_packet({}, first, replay, receipts, operations, complete, launch)['verdict'] == 'FINITE_SCOPE_PREREQUISITE_PASSED'
    if mutation == 'missing_row': first.pop()
    if mutation == 'changed_label': first[-1]['expected'] = e.SAFE
    if mutation == 'forged_pass': first[-1]['observed']['classification'] = e.SAFE
    if mutation == 'missing_replay': replay.clear()
    if mutation == 'missing_cost': receipts[0]['cpu_seconds'] = 0
    if mutation == 'reused_pid': receipts[0]['pid'] = receipts[1]['pid']
    if mutation == 'timeout': complete['elapsed_seconds'] = 1801
    if mutation == 'bad_operation': operations[0] = {'returncode': 1}
    if mutation == 'missing_storage': receipts[0]['storage_observations'] = []
    with pytest.raises(RuntimeError):
        x.validate_scope_packet({}, first, replay, receipts, operations, complete, launch)


def test_negative_scope_is_failure_not_missing_evidence():
    observed, *_ = fake_scope_packet()
    observed[171]['observed']['classification'] = e.SAFE
    result = x.scope_summary(observed)
    assert result['verdict'] == 'SCOPE_PREREQUISITE_FAILED'
    assert result['false_safe_assertions'] == [observed[171]['id']]


def test_failed_scope_blocks_gpu_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(x, 'OUT', tmp_path)
    monkeypatch.setattr(x, 'guard_storage', lambda: {})
    monkeypatch.setattr(x, 'freeze_binding', lambda: {})
    def failed():
        raise RuntimeError('Scope prerequisite did not pass')
    monkeypatch.setattr(x, 'require_scope_pass', failed)
    monkeypatch.setattr(run, 'child', lambda *a, **k: pytest.fail('child executed'))
    with pytest.raises(RuntimeError, match='Scope prerequisite'):
        run.live()
    assert not (tmp_path/'live/launch.json').exists()


def fake_live():
    result = []
    for row in d.population():
        screen = {'mechanical_pass': True, 'collapse': False, 'requested_sentence_count': True}
        result.append({**{k: row[k] for k in ('id', 'pair', 'panel', 'task')}, 'screen': screen,
            'legacy_screen': dict(screen), 'stop_reason': 'eos', 'prompt_ids': [1], 'token_ids': [2],
            'output_utf8_bytes': 1, 'output': 'x', 'prompt': row['prompt']})
    return result


def test_bootstrap_unit_is_pair_not_replay():
    plain, contract = fake_live(), fake_live()
    for row in plain:
        if row['task'] in d.INTERVENTION_TASKS:
            row['screen']['requested_sentence_count'] = False
    result = x.bootstrap_delta(contract, plain, sentence=True)
    assert result['paired_clusters'] == 12 and result['mean_difference'] == 1 and result['ci95'] == [1., 1.]
    assert x.bootstrap_delta(contract, contract, True)['ci95'] == [0., 0.]
    assert x.bootstrap_delta(contract, plain)['paired_clusters'] == 48


def test_absolute_gates_and_negative_controls_not_replaced_by_effect():
    plain, contract = fake_live(), fake_live()
    for row in plain:
        if row['task'] in d.INTERVENTION_TASKS:
            row['screen']['requested_sentence_count'] = False
    assert x.live_decision(plain, contract)['verdict'] == 'BOUNDED_INTERFACE_IMPROVEMENT'
    target = next(r for r in contract if r['task'] not in d.INTERVENTION_TASKS)
    target['token_ids'] = [3]
    assert x.live_decision(plain, contract)['verdict'] == 'UNCHANGED_INPUT_EXECUTION_DISCREPANCY'
    target['token_ids'] = [2]
    for row in contract:
        if row['task'] == 'tone_control': row['screen']['mechanical_pass'] = False
    assert x.live_decision(plain, contract)['verdict'] != 'BOUNDED_INTERFACE_IMPROVEMENT'


@pytest.mark.parametrize('method', [x.freeze_binding, x.verify_scope, x.require_scope_pass, run.derive_live, run.verify])
def test_missing_packages_rows_hashes_fail_closed(tmp_path, monkeypatch, method):
    monkeypatch.setattr(x, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        method()


def test_source_driver_no_training_recipients_or_hardware_setters():
    source = inspect.getsource(run)
    tree = ast.parse(source)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('backward', 'step', 'train', 'load_layercake_core') for n in ast.walk(tree))
    for forbidden in ('--power-limit', '--lock-gpu-clocks', 'prepare_source', 'snapshot_download'):
        assert forbidden not in source
    assert "do_sample=False" in source and 'max_new_tokens=640' in source
    assert 'dtype=torch.bfloat16' in source
    assert 'stop_sampler.set()' in source


class RecordedTokenizer:
    eos_token_id = 2

    def __len__(self):
        return 200000

    def encode(self, prompt, add_special_tokens=False):
        return next(r['prompt_ids'] for r in x.inputs('plain') if r['prompt'] == prompt)

    def decode(self, tokens, skip_special_tokens=True):
        return 'Approval is pending.'


def valid_negative_live_rows(monkeypatch):
    # A fake tokenizer plus explicit negative model responses tests recomputation,
    # not quality. No source model is loaded and no actual generation is claimed.
    tokenizer = RecordedTokenizer()
    result = []
    rendered = x.inputs('plain')
    monkeypatch.setattr(d, 'render', lambda row, interface, tok: next(v['prompt'] for v in rendered if v['id'] == row['id']))
    for ref, inp in zip(d.population(), rendered):
        answer = tokenizer.decode([3, 2])
        result.append({**{k: ref[k] for k in ('id', 'pair', 'panel', 'task')}, 'interface': 'plain',
            'prompt': inp['prompt'], 'prompt_ids': inp['prompt_ids'], 'token_ids': [3, 2],
            'decoded_output': answer, 'output': answer, 'screen': e.screen(ref, answer),
            'legacy_screen': x.h.screen(ref, answer), 'stop_reason': 'eos',
            'output_utf8_bytes': len(answer.encode()), 'output_characters': len(answer), 'wall_seconds': .1,
            'storage_before': {'free_bytes': 11*1024**3, 'evidence_bytes': 1000}})
    return result, tokenizer


@pytest.mark.parametrize('mutation', ['row', 'identity', 'token', 'prompt', 'decoded', 'screen', 'legacy',
                                    'stop', 'time', 'storage', 'partial_budget', 'early_eos'])
def test_raw_live_rows_hostile_mutations(monkeypatch, mutation):
    observed, tokenizer = valid_negative_live_rows(monkeypatch)
    x.validate_live(observed, 'plain', tokenizer)
    if mutation == 'row': observed.pop()
    if mutation == 'identity': observed[0]['id'] = 'wrong'
    if mutation == 'token': observed[0]['token_ids'] = [-1, 2]
    if mutation == 'prompt': observed[0]['prompt'] = 'different'
    if mutation == 'decoded': observed[0]['decoded_output'] = 'rewritten'
    if mutation == 'screen': observed[0]['screen']['mechanical_pass'] = True
    if mutation == 'legacy': observed[0]['legacy_screen']['mechanical_pass'] = True
    if mutation == 'stop': observed[0]['stop_reason'] = 'budget'
    if mutation == 'time': observed[0]['wall_seconds'] = float('nan')
    if mutation == 'storage': observed[0]['storage_before']['free_bytes'] = 0
    if mutation == 'partial_budget': observed[0].update(token_ids=[3], stop_reason='budget')
    if mutation == 'early_eos': observed[0]['token_ids'] = [2, 3, 2]
    with pytest.raises(RuntimeError):
        x.validate_live(observed, 'plain', tokenizer)
