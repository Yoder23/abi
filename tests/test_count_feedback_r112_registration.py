"""R112 design/custody checks only; no prospective scope scoring or model calls."""
import json
from collections import Counter

import pytest

from experiments.count_feedback_r112 import design as d
from experiments.count_feedback_r112 import register as r


def test_unchanged_gates_source_and_bounded_budget():
    c = d.contract()
    assert c['quality_gates'] == d.previous.contract()['quality_gates']
    assert c['source'] == d.previous.contract()['source']
    assert c['maximum_calls'] == 96 * 3 * 2
    assert c['maximum_output_tokens'] == 576 * 640
    assert c['minimum_free_bytes'] == 10 * 1024**3
    assert c['gpu_setting_changes'] == []
    assert c['training_steps'] == c['recipient_calls'] == c['automatic_retries'] == 0


def test_population_is_recomputable_paired_and_fresh():
    rows = d.population()
    assert rows == d.population() and len(rows) == 96
    assert len({r['id'] for r in rows}) == 96
    assert set(Counter(r['pair'] for r in rows).values()) == {2}
    assert set(Counter((r['task'], r['panel']) for r in rows).values()) == {6}
    assert len({r['task'] for r in rows}) == 8
    assert sum(r['task'] in d.TASKS for r in rows) == 24
    old_prompts = {r['prompt'] for r in d.previous.population()}
    assert not old_prompts & {r['prompt'] for r in rows}
    assert all('N112' in r['code'] and 'N111' not in r['prompt'] for r in rows)


@pytest.mark.parametrize('text,count', [('', 0), ('A.', 1), ('A. B? C!', 3),
                                      ('A.B', 1), ('A; B. C. D. E.', 4),
                                      (' A.\nB?\n', 2)])
def test_counter_is_the_frozen_mechanical_rule(text, count):
    assert d.sentence_count(text) == count


def test_only_count_feedback_differs_on_sentence_tasks():
    for row in d.population():
        plain = d.revision_request(row, 'revision', 'One. Two. Three. Four.')
        assert plain == d.previous.revision_request(row)
        feedback = d.revision_request(row, 'count_feedback', 'One. Two. Three. Four.')
        expected = plain + (d.FEEDBACK.format(count=4) if row['task'] in d.TASKS else '')
        assert feedback == expected


class Messages:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
        return messages


def test_both_revisions_use_the_same_complete_unmodified_draft():
    row = next(r for r in d.population() if r['task'] in d.TASKS)
    draft = '  One. Two. Three. Four.\n'
    for stage in ('revision', 'count_feedback'):
        messages = d.render(row, stage, Messages(), draft)
        assert messages[1]['content'] == row['prompt']
        assert messages[2] == {'role': 'assistant', 'content': draft}
        assert len(messages) == 4
    assert len(d.render(row, 'draft', Messages())) == 2
    with pytest.raises(ValueError):
        d.render(row, 'draft', Messages(), draft)
    with pytest.raises(ValueError):
        d.render(row, 'count_feedback', Messages())


@pytest.mark.parametrize('value', [-1, True, 1.5, None])
def test_invalid_storage_observation_rejected(value):
    with pytest.raises(ValueError):
        r.resource_observation(value)


@pytest.mark.parametrize('free,blocked', [(0, True), (10 * 1024**3 - 1, True),
                                       (10 * 1024**3, False)])
def test_resource_floor_does_not_authorize_execution(free, blocked):
    receipt = r.resource_observation(free)
    assert receipt['verdict'].startswith('RESOURCE_BLOCKED') is blocked
    assert receipt['model_calls'] == 0
    assert receipt['implementation_frozen'] is False
    assert receipt['scope_challenge_executed'] is False


def test_write_once_receipt(tmp_path):
    path = tmp_path / 'receipt.json'
    r.write_new(path, {'preserved': 1})
    with pytest.raises(FileExistsError):
        r.write_new(path, {'replaced': 1})
    assert json.loads(path.read_text()) == {'preserved': 1}


def test_registration_cannot_overwrite_existing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'OUT', tmp_path)
    with pytest.raises(RuntimeError, match='Preserve existing'):
        r.register()


def test_challenge_seed_binds_code_and_protocol():
    first = d.challenge_seed('a' * 40, 'b' * 64)
    assert len(first) == 64
    assert first != d.challenge_seed('c' * 40, 'b' * 64)
    assert first != d.challenge_seed('a' * 40, 'd' * 64)
    with pytest.raises(ValueError):
        d.challenge_seed('bad', 'b' * 64)


@pytest.mark.parametrize('failure', ['missing', 'size', 'digest', 'escape'])
def test_parent_dependency_audit_fails_closed(tmp_path, monkeypatch, failure):
    root = tmp_path / 'abi_release'
    root.mkdir()
    payload = root / 'payload'
    payload.write_bytes(b'parent')
    decision = root / 'decision.json'
    decision.write_bytes(b'{}')
    name = '../outside' if failure == 'escape' else 'payload'
    item = {'bytes': 0 if failure == 'size' else 6,
            'sha256': '0' * 64 if failure == 'digest' else r.sha(payload)}
    manifest = root / 'manifest.json'
    r.write_new(manifest, {'files': {name: item}, 'host_inputs': {}})
    monkeypatch.setattr(d, 'ROOT', root)
    monkeypatch.setattr(r, 'MANIFEST', 'manifest.json')
    monkeypatch.setattr(r, 'DECISION', 'decision.json')
    monkeypatch.setattr(d, 'R111_MANIFEST', r.sha(manifest))
    monkeypatch.setattr(d, 'R111_DECISION', r.sha(decision))
    if failure == 'missing':
        payload.unlink()
    with pytest.raises(RuntimeError):
        r.audit_parent()
