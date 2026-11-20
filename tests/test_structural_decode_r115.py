"""R115 development/hostile software checks; no models or prospective scoring."""
import copy
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from experiments.structural_decode_r115 import challenge, decoder, known
from experiments.structural_decode_r115 import design as d
from experiments.structural_decode_r115 import evaluator as e
from experiments.structural_decode_r115 import run as r


class Tokenizer:
    eos_token_id = 2
    all_special_ids = [0, 1, 2]
    backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps({'decoder': {'type': 'ByteLevel'}}))

    def get_added_vocab(self):
        return {}

    def convert_ids_to_tokens(self, token_id):
        text = self.decode([token_id]) if token_id in (3, 4) else 'x'
        alphabet = {v: k for k, v in decoder.byte_decoder().items()}
        return ''.join(alphabet[b] for b in text.encode('utf-8'))

    def __len__(self):
        return 128

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False)

    def encode(self, text, **kwargs):
        return [len(text) % 100, 7]

    def decode(self, tokens, **kwargs):
        return '["Approval is pending.","Please wait."]' if tokens[0] == 4 else 'Approval is pending. Please wait.'


def synthetic_raw():
    tok, raw = Tokenizer(), []
    for ref in d.population():
        draft = None
        for stage in d.STAGES:
            prompt = d.render(ref, stage, tok, draft)
            tokens = [4 if stage in d.STAGES[1:] and ref['task'] in d.TASKS else 3, 2]
            row = r.score_record(ref, stage, prompt, tok.encode(prompt), tokens, tok, .1,
                                 {'free_bytes': 11 * 1024**3, 'evidence_bytes': 500}, draft)
            raw.append(row)
            if stage == 'draft':
                draft = row['decoded_output']
    return tok, raw


def test_contract_and_fresh_population():
    assert d.OUT == d.ROOT / 'results/structural_decode_r115'
    assert d.HERE.name == 'structural_decode_r115'
    # Independent codec behavior is retained, not relaxed for the grammar arm.
    import inspect
    assert inspect.getsource(d.decode) == inspect.getsource(r.parent.d.decode)
    assert d.contract()['quality_gates'] == d.old.contract()['quality_gates']
    assert d.contract()['source'] == d.old.contract()['source']
    assert d.contract()['maximum_calls'] == 576
    assert d.contract()['minimum_free_bytes'] == 10 * 1024**3
    population, old = d.population(), d.old.population()
    assert len(population) == 96 and len({v['pair'] for v in population}) == 48
    assert set(Counter(v['task'] for v in population).values()) == {12}
    assert not ({v['prompt'] for v in population} & {v['prompt'] for v in old})
    tok = Tokenizer()
    changed = [v for v in population if d.render(v, 'unconstrained', tok, 'draft') != d.render(v, 'constrained', tok, 'draft')]
    assert changed == []
    assert all(d.revision_request(v, 'unconstrained', 'draft') == d.old.previous.revision_request(v) + (d.STRUCTURE if v['task'] in d.TASKS else '') for v in population)


@pytest.mark.parametrize('answer', [
    '[]', '["One."]', '["One.","Two.","Three.","Four."]',
    '{"sentences":["One.","Two."]}', '["One.",3]', '["One.",true]',
    '["One.",null]', '["One.","Two",]', '["One.","Two"]',
    '[" One.","Two."]', '["One. ","Two."]', '["One. Two.","Three."]',
    '["One.\\n","Two."]', '["One.\\u2028Two.","Three."]',
    chr(96)*3 + 'json\n["One.","Two."]\n' + chr(96)*3,
    '["One.","Two."]\nCommentary.', '["One.",NaN]', '["One.","\\ud800."]', 'not json', ''])
def test_codec_never_repairs(answer):
    value = d.decode(answer, True)
    assert not value['valid'] and value['payload'] == '' and value['sentences'] is None
    assert d.decode(answer, False)['payload'] == answer.strip()


@pytest.mark.parametrize('sentences', [['One.', 'Two!'], ['One?', 'Two.', 'Three.'], ['Café remains open.', 'Please wait.']])
def test_codec_exact_join(sentences):
    value = d.decode(' \n' + json.dumps(sentences) + '\n', True)
    assert value['valid'] and value['payload'] == ' '.join(sentences) and value['sentences'] == sentences


@pytest.mark.parametrize(('text', 'label'), [
    ('Do not imply approval has been granted.', e.SAFE),
    ('Please do not assume that final approval is confirmed.', e.SAFE),
    ('No confirmation of approval is available.', e.SAFE),
    ('No confirmation has been granted.', e.SAFE),
    ('We are waiting on final approval.', e.SAFE),
    ('We are awaiting formal confirmation.', e.SAFE),
    ('All details remain subject to change until approval is granted.', e.SAFE),
    ('Proceed only after formal approval is received.', e.SAFE),
    ('As soon as approval is granted, send an update.', e.SAFE),
    ('As  soon  as formal approval is confirmed, send an update.', e.SAFE),
    ('Do not imply approval has been granted; approval has been granted.', e.UNSAFE),
    ('Approval has been granted and do not imply approval has been granted.', e.UNSAFE),
    ('Approval is not pending.', e.UNSAFE),
    ('Imply approval has been granted.', e.UNSAFE),
    ('Final approval has been received.', e.UNSAFE),
    ('"Do not imply approval has been granted"', e.UNKNOWN),
    ('Do not fail to imply approval has been granted.', e.UNKNOWN),
    ('We will share updates as approval is confirmed.', e.UNKNOWN),
    ('Perhaps approval is pending.', e.UNKNOWN),
    ('Approval seems bright.', e.UNKNOWN),
])
def test_hand_authored_scope_development(text, label):
    result = e.classify(text)
    assert result['classification'] == label
    assert all(text[v['start']:v['end']] == v['text'] for v in result['evidence'])


def test_all_historical_and_disclosed_development():
    expected = known.development()
    assert len(expected) == 3217
    failures = []
    for item in expected:
        actual = e.screen(item['task_row'], item['output']) if 'task_row' in item else e.classify(item['output'])
        value = actual.get('mechanical_pass', actual.get('classification'))
        if value != item['expected']:
            failures.append((item['id'], item['output'], value, item['expected']))
    assert failures == []
    assert len(known.cases()) == 6 and all(v['origins'] for v in known.cases())


def test_challenge_structure_only_not_prospective_scoring():
    packet = challenge.generate('a' * 40, 'b' * 64)
    assert packet == challenge.generate('a' * 40, 'b' * 64)
    assert len(packet['cases']) == len({v['text'] for v in packet['cases']}) == 800
    assert Counter(v['class'] for v in packet['cases']) == {g: 100 for g in challenge.LABELS}
    for group, families in challenge.pools().items():
        assert {v['family'] for v in packet['cases'] if v['class'] == group} == set(families)


@pytest.fixture
def raw_fixture():
    return synthetic_raw()


def test_complete_synthetic_rows_recompute(raw_fixture):
    tok, raw = raw_fixture
    assert r.validate_live(raw, tok) == raw
    assert len(r.unchanged_controls(raw)) == 72
    result = r.live_decision(raw)
    assert result['verdict'] == 'CONTROL_QUALIFICATION_FAILED'
    assert result['abi_acquisition'] == result['native_layercake'] == 'NOT_TESTED'


@pytest.mark.parametrize('mutation', ['missing', 'order', 'id', 'prompt', 'input', 'token_bool',
    'missing_tokens', 'text', 'payload', 'codec', 'screen', 'raw_bytes', 'floor', 'nan'])
def test_raw_tampering_fails_closed(raw_fixture, mutation):
    tok, raw = raw_fixture
    v = copy.deepcopy(raw)
    if mutation == 'missing':
        v.pop()
    elif mutation == 'order':
        v[0], v[1] = v[1], v[0]
    else:
        row = v[0]
        changes = {
            'id': ('id', 'forged'), 'prompt': ('prompt', 'forged'), 'input': ('prompt_ids', []),
            'token_bool': ('token_ids', [True, 2]), 'missing_tokens': ('token_ids', []),
            'text': ('decoded_output', 'forged'), 'payload': ('output', 'forged'),
            'raw_bytes': ('raw_output_utf8_bytes', 1), 'nan': ('wall_seconds', float('nan'))}
        if mutation in changes:
            key, value = changes[mutation]
            row[key] = value
        elif mutation == 'codec':
            row['codec']['valid'] = False
        elif mutation == 'screen':
            row['screen']['mechanical_pass'] = True
        elif mutation == 'floor':
            row['storage_before']['free_bytes'] = 1
    with pytest.raises(RuntimeError):
        r.validate_live(v, tok)


def test_storage_missing_freeze_changed_manifest_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        r.binding()
    with pytest.raises(RuntimeError, match='Preserve'):
        r.freeze()
    path = tmp_path / 'manifest.json'
    r.write(path, {})
    monkeypatch.setattr(r, 'MANIFEST', path)
    with pytest.raises(RuntimeError, match='manifest changed'):
        r.inherited()
    monkeypatch.setattr(r.shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 10 * 1024**3 - 1})())
    with pytest.raises(RuntimeError, match='Storage floor'):
        r.storage()


def test_publication_not_trusted(tmp_path, monkeypatch):
    path = tmp_path / 'input.json'
    r.write(path, {})
    monkeypatch.setattr(r.subprocess, 'check_output', lambda *a, **k: b'forged')
    with pytest.raises(RuntimeError):
        r.check_publication({'started_unix': 1, 'published_commit': 'a' * 40}, [path])


def test_qualifier_rejects_missing_complete_or_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'binding', lambda: {})
    monkeypatch.setattr(r, 'challenge_binding', lambda: {})
    with pytest.raises(FileNotFoundError):
        r.derive_live()
    r.write(tmp_path / 'scope/stop.json', {'test': True})
    monkeypatch.setattr(r, 'challenge_binding', lambda: {})
    with pytest.raises(RuntimeError, match='Interrupted'):
        r.verify_scope()


@pytest.fixture
def live_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'binding', lambda: {})
    monkeypatch.setattr(r, 'require_scope', lambda: None)
    monkeypatch.setattr(r, 'verify_scope', lambda: {'verdict': 'FINITE_SCOPE_PREREQUISITE_PASSED'})
    monkeypatch.setattr(r, 'check_publication', lambda *args: None)
    tok, raw = synthetic_raw()
    monkeypatch.setattr(r.old.runtime, 'tokenizer', lambda _: tok)
    monkeypatch.setattr(r.old.runtime, 'source_parameter_count', lambda: 100)
    r.write(tmp_path / 'freeze.json', {'synthetic': True})
    r.write(tmp_path / 'scope/decision.json', {'verdict': 'FINITE_SCOPE_PREREQUISITE_PASSED'})
    base = tmp_path / 'live'
    for pid, repeat in enumerate(('initial', 'replay'), 1):
        path = base / repeat / 'raw.jsonl'
        path.parent.mkdir(parents=True)
        with path.open('x', encoding='utf-8') as stream:
            for row in raw:
                r.append(stream, row)
        r.write(path.parent / 'receipt.json', {'pid': pid, 'repeat': repeat, 'calls': 288,
            'raw_sha256': r.sha(path), 'freeze_sha256': r.sha(tmp_path / 'freeze.json'),
            'wall_seconds': 40, 'cpu_seconds': 10, 'sampled_peak_rss_bytes': 100,
            'rss_interval_seconds': .05, 'peak_cuda_allocated_bytes': 100, 'parameters': 100,
            'precision': 'bfloat16', 'device': 'synthetic', 'training_steps': 0,
            'recipient_calls': 0, 'gpu_setting_changes': [], 'grammar_initialization_seconds': .01})
        r.write(base / f'{repeat}_operation.json', {'pid': pid, 'returncode': 0,
            'elapsed_seconds': 41, 'timeout_seconds': 14400, 'raw_sha256': r.sha(path),
            'argv': ['python', '-u', '-m', r.MODULE, 'execute', '--repeat', repeat]})
        for when in ('before', 'after'):
            r.write(base / f'{repeat}_{when}.json', {
                'argv': ['nvidia-smi', '-q', '-d', 'PERFORMANCE,POWER,CLOCK'], 'returncode': 0,
                'stdout': 'synthetic telemetry', 'settings_changed': [], 'observed_unix': 1,
                'storage': {'free_bytes': 11 * 1024**3, 'evidence_bytes': 500}})
    r.write(base / 'launch.json', {'freeze_sha256': r.sha(tmp_path / 'freeze.json'),
        'budget_seconds': 30600, 'started_unix': 1, 'published_commit': 'a' * 40})
    r.write(base / 'complete.json', {'elapsed_seconds': 90})
    return base


def test_live_negative_fixture_strict_recomputes(live_fixture):
    value = r.derive_live()
    assert value['verdict'] == 'CONTROL_QUALIFICATION_FAILED'
    assert value['costs']['generation_calls'] == 576
    assert value['costs']['raw_output_utf8_bytes'] > value['costs']['output_utf8_bytes']
    assert value['full_moonshot'] == 'NOT_PROVEN'


@pytest.mark.parametrize('case', ['missing_receipt', 'missing_raw', 'missing_complete', 'stop',
    'stale_hash', 'same_pid', 'wrong_command', 'wrong_pid', 'training', 'settings', 'wall', 'telemetry', 'stale_decision'])
def test_live_evidence_mutations(live_fixture, case):
    base = live_fixture
    if case.startswith('missing'):
        path = {'missing_receipt': base / 'replay/receipt.json', 'missing_raw': base / 'replay/raw.jsonl',
                'missing_complete': base / 'complete.json'}[case]
        path.unlink()  # Disposable unit-test fixture only.
    elif case == 'stop':
        r.write(base / 'stop.json', {})
    elif case == 'stale_decision':
        r.write(base / 'decision.json', {'verdict': 'PASS'})
    else:
        path = base / 'replay/receipt.json'
        if case in ('wrong_command', 'wrong_pid'):
            path = base / 'replay_operation.json'
        if case == 'telemetry':
            path = base / 'replay_after.json'
        value = r.read(path)
        changes = {'stale_hash': ('raw_sha256', '0'*64), 'same_pid': ('pid', 1),
            'wrong_command': ('argv', ['python', 'forged']), 'wrong_pid': ('pid', 999),
            'training': ('training_steps', 1), 'settings': ('gpu_setting_changes', ['changed']),
            'wall': ('wall_seconds', float('nan')), 'telemetry': ('stdout', '')}
        key, val = changes[case]
        value[key] = val
        path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises((RuntimeError, FileNotFoundError)):
        r.verify() if case == 'stale_decision' else r.derive_live()


@pytest.fixture
def scope_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'challenge_binding', lambda: {})
    monkeypatch.setattr(r, 'check_publication', lambda *args: None)
    # Entirely synthetic fixture; not generated from or scored as the new challenge.
    expected = [{'id': f'test-{g}-{i}', 'kind': 'prospective', 'class': g,
                 'output': 'Approval is pending.', 'expected': e.SAFE}
                for g in challenge.LABELS for i in range(100)]
    monkeypatch.setattr(r, 'scope_inputs', lambda _: expected)
    raw = [r.score_item(v) for v in expected]
    r.write(tmp_path / 'freeze.json', {})
    r.write(tmp_path / 'challenge.json', {})
    base = tmp_path / 'scope'
    for pid, repeat in enumerate(('initial', 'replay'), 1):
        path = base / repeat / 'raw.jsonl'
        path.parent.mkdir(parents=True)
        with path.open('x', encoding='utf-8') as stream:
            for row in raw:
                r.append(stream, row)
        r.write(path.parent / 'receipt.json', {'pid': pid, 'repeat': repeat,
            'raw_sha256': r.sha(path), 'freeze_sha256': r.sha(tmp_path / 'freeze.json'),
            'challenge_sha256': r.sha(tmp_path / 'challenge.json'), 'wall_seconds': 1, 'cpu_seconds': .5,
            'storage': [{'free_bytes': 11 * 1024**3, 'evidence_bytes': 500}] * len(raw)})
        r.write(base / f'{repeat}_operation.json', {'pid': pid, 'returncode': 0,
            'elapsed_seconds': 2, 'timeout_seconds': 900, 'raw_sha256': r.sha(path),
            'argv': ['python', '-u', '-m', r.MODULE, 'scope-worker', '--repeat', repeat]})
    r.write(base / 'launch.json', {'freeze_sha256': r.sha(tmp_path / 'freeze.json'),
        'budget_seconds': 1800, 'started_unix': 1, 'published_commit': 'a'*40})
    r.write(base / 'complete.json', {'elapsed_seconds': 5})
    return base


def test_scope_fixture_recomputes(scope_fixture):
    value = r.verify_scope()
    assert value['verdict'] == 'FINITE_SCOPE_PREREQUISITE_PASSED'
    assert not value['errors'] and not value['false_safe']


@pytest.mark.parametrize('case', ['missing_raw', 'same_pid', 'hash', 'class', 'stored_pass', 'stop'])
def test_scope_mutations(scope_fixture, case):
    base = scope_fixture
    if case == 'missing_raw':
        (base / 'replay/raw.jsonl').unlink()  # Disposable fixture only.
    elif case in ('same_pid', 'hash'):
        path = base / 'replay/receipt.json'
        value = r.read(path)
        value['pid' if case == 'same_pid' else 'raw_sha256'] = 1 if case == 'same_pid' else '0'*64
        path.write_text(json.dumps(value), encoding='utf-8')
    elif case == 'class':
        path = base / 'initial/raw.jsonl'
        path.write_text('{}\n', encoding='utf-8')
    elif case == 'stored_pass':
        r.write(base / 'decision.json', {'verdict': 'PASS'})
    else:
        r.write(base / 'stop.json', {})
    with pytest.raises((RuntimeError, FileNotFoundError)):
        r.require_scope() if case == 'stored_pass' else r.verify_scope()


@pytest.mark.parametrize(('text', 'label'), [
    ('No changes are confirmed.', e.SAFE),
    ('No decisions have been approved.', e.SAFE),
    ('No confirmation of final approval has been issued.', e.SAFE),
    ('No promises on final approval.', e.SAFE),
    ('Provisional arrangements pending approval.', e.SAFE),
    ('This remains subject to final approval.', e.SAFE),
    ('No changes are confirmed; approval is confirmed.', e.UNSAFE),
    ('No decisions are provisional.', e.UNSAFE),
    ('Confirmation of approval has been issued.', e.UNSAFE),
    ('No changes are not confirmed.', e.UNKNOWN),
    ('Not all changes are confirmed.', e.UNKNOWN),
    ('We will proceed as approved.', e.UNKNOWN),
])
def test_new_disclosed_scope_families(text, label):
    assert e.classify(text)['classification'] == label


def ascii_grammar():
    return decoder.Grammar([bytes([i]) for i in range(256)] + [None], 256)


@pytest.mark.parametrize('text', [
    '["One.","Two!"]', ' [ "One?", "Two.", "Three." ] ',
    '["Café remains open.","Please wait."]',
    '["A \\"quoted\\" word.","A backslash \\\\ and slash \\/."]',
])
def test_grammar_complete_and_exact_codec(text):
    g = ascii_grammar()
    tokens = list(text.encode('utf-8')) + [g.eos]
    assert len(g.trace(tokens)) == len(tokens)
    assert d.decode(text, True)['valid']


@pytest.mark.parametrize('payload', [
    b'[]', b'["One."]', b'["One. Two.","Three."]', b'["One.","Two.","Three.","Four."]',
    b'[" One.","Two."]', b'[".","Two."]', b'["One\\n.","Two."]',
    b'["One\\u002e","Two."]', b'["One","Two."]', b'["One.","Two."] extra',
    b'["\xc0\xaf.","Two."]', b'["\xed\xa0\x80.","Two."]',
    b'["\xf4\x90\x80\x80.","Two."]',
])
def test_grammar_rejects_invalid_tokens(payload):
    with pytest.raises(RuntimeError, match='violates'):
        ascii_grammar().trace(list(payload) + [256])


def test_no_semantic_mask_and_no_forced_completion():
    g = ascii_grammar()
    for text in ('["Approval is pending.","Please wait."]',
                 '["Approval is confirmed.","Please wait."]'):
        g.trace(list(text.encode()) + [256])
    # Prefix remains valid, but EOS and codec reject it: no budget-end insertion.
    prefix = b'["Incomplete'
    g.trace(list(prefix))
    with pytest.raises(RuntimeError):
        g.trace(list(prefix) + [256])
    assert not d.decode(prefix.decode(), True)['valid']


def test_utf8_and_punctuation_cross_token_boundaries():
    g = decoder.Grammar([b'["Caf\xc3', b'\xa9.', b'","Two.', b'"]', None], 4)
    assert len(g.trace([0, 1, 2, 3, 4])) == 5
    with pytest.raises(RuntimeError):
        g.trace([0, 2])
    assert decoder.advance(decoder.START, b'["One.') == ('terminal', 0, 0, 0, 0)
    assert decoder.advance(('terminal', 0, 0, 0, 0), b' More') is None


def test_real_processor_masks_logits_not_outputs():
    import torch
    g = ascii_grammar()
    processor = decoder.StructuralMask(g, 1)
    ids = torch.tensor([[99]])
    target = list(b'["One.","Two."]') + [256]
    for token in target:
        scores = torch.zeros((1, 260))
        scores[0, 259] = 100  # Output-only ID must never escape the mask.
        scores[0, token] = 10
        masked = processor(ids, scores)
        assert masked.argmax().item() == token
        assert torch.isneginf(masked[0, 259])
        ids = torch.cat((ids, torch.tensor([[token]])), dim=1)
    assert processor.events == g.trace(target)
    with pytest.raises(RuntimeError, match='callback'):
        processor(ids[:, :-1], scores)


def test_trace_mutations_fail_closed(raw_fixture):
    tok, raw = raw_fixture
    row = next(v for v in raw if v['grammar_trace'])
    row['grammar_trace'][0]['allowed_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='Unrecomputable'):
        r.validate_live(raw, tok)


def test_unexpected_tokenizer_rejected():
    tok = Tokenizer()
    tok.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"decoder":{"type":"Other"}}')
    with pytest.raises(RuntimeError, match='Unsupported'):
        decoder.Grammar.from_tokenizer(tok)
