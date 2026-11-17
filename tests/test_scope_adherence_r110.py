"""Preregistration/input-design checks only, not evaluator or model qualification."""
import ast
import copy
import inspect
from types import SimpleNamespace

import pytest

from experiments.scope_adherence_r110 import design as d
from experiments.scope_adherence_r110 import register as r


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return '\n'.join(m['role']+': '+m['content'] for m in messages)+'\nassistant:'

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(range(len(text.split())))


def test_fixed_contract_and_no_gpu_execution_command():
    value = d.contract()
    r.validate_contract(value)
    assert value['resources']['maximum_generation_calls'] == 4*96
    assert value['population']['rows'] == 96 and value['population']['counterfactual_pairs'] == 48
    assert value['evaluator_gate']['historical_fixtures'] == 164
    assert value['evaluator_gate']['known_r109_scope_regressions'] == 7
    assert value['evaluator_gate']['labels_are_independent_human_ratings'] is False
    assert value['source_process_order'] == [
        ['plain', 'initial'], ['response_contract', 'initial'], ['response_contract', 'replay'], ['plain', 'replay']]
    assert value['analysis']['bootstrap_draws'] == 10000
    assert value['analysis']['support_requires_lower_bound_strictly_above'] == 0
    assert {'output_repair', 'forced_eos', 'training', 'recipient_execution', 'native_host_changes'} <= set(value['prohibited'])
    for module in (d, r):
        tree = ast.parse(inspect.getsource(module))
        forbidden = {'generate', 'execute', 'pilot', 'train', 'prepare_source'}
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr in forbidden for n in ast.walk(tree))


def test_deterministic_known_template_instantiations():
    first = d.population()
    assert first == d.population() and len(first) == 96
    assert len({v['id'] for v in first}) == len({v['prompt'] for v in first}) == 96
    assert not {v['prompt'] for v in first} & {v['prompt'] for v in d.reference.catalog()}
    assert all('N108' not in v['prompt'] and 'N110' in v['prompt'] for v in first)
    assert all(v['split'] == 'registered-instantiations-of-known-templates' for v in first)
    for panel in ('a', 'b'):
        assert all(sum(v['panel'] == panel and v['task'] == task for v in first) == 6 for task in d.reference.TASKS)
    for a, b in zip(first[::2], first[1::2]):
        assert a['pair'] == b['pair'] and a['task'] == b['task'] and a['panel'] == 'a' and b['panel'] == 'b'
        assert a['required'] != b['required']
    for row in first:
        assert all(v in row['prompt'] for v in row['required'])
        assert all(fact in row['prompt'] for fact in row['plan']['facts'])
        assert row['plan']['request'] in row['prompt']


def test_only_twenty_four_inputs_change_without_removing_any_notes():
    rows = d.population()
    before = copy.deepcopy(rows)
    changed = []
    for row in rows:
        plain = d.user_message(row, 'plain')
        structured = d.user_message(row, 'response_contract')
        assert plain == row['prompt'] and structured.startswith(plain)
        if structured != plain:
            changed.append(row)
            assert row['task'] in d.INTERVENTION_TASKS
            payload = d.response_contract(row)
            assert payload['required_literals'] == row['required']
            assert payload['sentence_count'] == 'two or three' and payload['approval_status'] == 'pending'
            payload['required_literals'].clear()
            assert row['required']
        else:
            assert d.response_contract(row) is None
    assert len(changed) == 24 and len({v['pair'] for v in changed}) == 12
    assert rows == before


def test_rendered_matrix_and_negative_controls():
    matrix = r.rendered_inputs(Tokenizer())
    assert len(matrix) == 192
    unchanged = 0
    for a, b in zip(matrix[::2], matrix[1::2]):
        assert a['id'] == b['id'] and a['interface'] == 'plain' and b['interface'] == 'response_contract'
        if a['task'] not in d.INTERVENTION_TASKS:
            assert a['prompt'] == b['prompt'] and a['prompt_ids'] == b['prompt_ids']
            unchanged += 1
    assert unchanged == 72


def test_context_cannot_be_enlarged_to_fit_contract():
    tokenizer = Tokenizer()
    tokenizer.encode = lambda *a, **kw: list(range(385))
    with pytest.raises(RuntimeError, match='context budget'):
        r.rendered_inputs(tokenizer)


def test_unregistered_interface_rejected():
    with pytest.raises(ValueError):
        d.user_message(d.population()[0], 'new_prompt')


@pytest.mark.parametrize('field', ['panel_minimum_passes', 'task_minimum_passes', 'pair_minimum_both_passes',
                                  'task_pair_minimum_both_passes', 'maximum_collapses', 'maximum_truncations',
                                  'maximum_words', 'minimum_words', 'sentence_range', 'exact_literal_case_required'])
def test_quality_gate_mutations_rejected(field):
    value = d.contract()
    value['quality_gates'][field] = None
    with pytest.raises(RuntimeError):
        r.validate_contract(value)


@pytest.mark.parametrize('field', ['maximum_generation_calls', 'maximum_generated_tokens', 'model_process_seconds',
                                  'controller_seconds', 'minimum_free_bytes', 'maximum_evidence_bytes',
                                  'maximum_new_download_bytes', 'gpu_setting_changes', 'automatic_retries'])
def test_resource_mutations_rejected(field):
    value = d.contract()
    value['resources'][field] = ['changed'] if field == 'gpu_setting_changes' else -1
    with pytest.raises(RuntimeError):
        r.validate_contract(value)


@pytest.mark.parametrize('field', ['maximum_errors_in_each_class', 'maximum_false_safe_assertions',
                                  'unresolved_is_a_pass', 'first_challenge_failure_stops_study', 'challenge_rows'])
def test_evaluator_gate_cannot_be_weakened(field):
    value = d.contract()
    old = value['evaluator_gate'][field]
    value['evaluator_gate'][field] = not old if type(old) is bool else old+1
    with pytest.raises(RuntimeError):
        r.validate_contract(value)


def test_challenge_seed_is_content_derived_not_a_selectable_integer():
    seed = d.challenge_seed('a'*40, 'b'*64)
    assert len(seed) == 64 and seed == d.challenge_seed('a'*40, 'b'*64)
    assert seed != d.challenge_seed('c'*40, 'b'*64)
    with pytest.raises(ValueError):
        d.challenge_seed('master', 'b'*64)


@pytest.mark.parametrize('section,field,value', [
    ('source', 'cpu_threads', 8), ('source', 'greedy', False), ('source', 'tf32', True),
    ('population', 'rows', 48), ('population', 'seed', 7),
    ('analysis', 'bootstrap_draws', 100), ('analysis', 'confidence_level', .8),
    ('analysis', 'support_requires_lower_bound_strictly_above', -.1),
    ('analysis', 'all_task_quality_noninferiority_lower_bound', -.2),
    ('analysis', 'unchanged_interface_rows_must_match_tokens', 0),
])
def test_source_population_analysis_changes_rejected(section, field, value):
    changed = d.contract()
    changed[section][field] = value
    with pytest.raises(RuntimeError):
        r.validate_contract(changed)


def test_registration_missing_or_overwrite_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        r.verify()
    r.historical.write(tmp_path/'population.json', [])
    with pytest.raises(RuntimeError, match='Preserve'):
        r.seal()


def test_storage_stop_precedes_source_access(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'OUT', tmp_path)
    monkeypatch.setattr(r.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    monkeypatch.setattr(r, 'historical_bindings', lambda: pytest.fail('source accessed after disk stop'))
    with pytest.raises(RuntimeError, match='free disk'):
        r.seal()
