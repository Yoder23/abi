"""Seal/recompute a preregistration only. This module cannot launch model runs."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

from . import design as d
from experiments.trained_grounding_r109 import core as historical
from experiments.trained_grounding_r109 import run as historical_run

IDENTITIES = {
    'results/trained_grounding_r109/freeze.json': '88149d5be624ba13072659b2d2b70855b2d14125211fe5700b32bda905b1bc1e',
    'results/trained_grounding_r109/decision.json': '57126b03256ca20623e6a527d9b2f38110f755ff003f0ccf170b3042e1762725',
    'results/review_2026_11_17/r109_manifest.json': '109bba3f6d8962437f0a8e889a1bb191fc794f57ee447872cfb11c445b9e345d',
}
DESIGN_FILES = ('design.py', 'register.py', 'PROTOCOL.md', 'contract.json')
PACKET_FILES = ('population.json', 'rendered_inputs.json')


def validate_contract(value):
    expected_gates = {
        'panel_rows': 48, 'panel_minimum_passes': 44, 'task_rows_per_panel': 6,
        'task_minimum_passes': 5, 'pair_rows': 48, 'pair_minimum_both_passes': 44,
        'task_pair_minimum_both_passes': 5, 'maximum_collapses': 0,
        'maximum_truncations': 0, 'exact_scientific_replay_required': True,
        'maximum_words': 120, 'minimum_words': 3, 'sentence_range': [2, 3], 'exact_literal_case_required': True,
    }
    if (value['quality_gates'] != expected_gates or value['source']['model'] != historical.SOURCE
            or value['source']['revision'] != historical.REVISION
            or value['source']['maximum_new_tokens'] != 640 or value['source']['context_tokens'] != 1024
            or value['source']['generation_seed'] != historical.SEED
            or value['source']['precision'] != 'bfloat16'
            or value['source']['cpu_threads'] != 4 or value['source']['greedy'] is not True
            or value['source']['tf32'] is not False
            or value['interfaces'] != list(d.INTERFACES)
            or value['intervention_tasks'] != list(d.INTERVENTION_TASKS)):
        raise RuntimeError('Changed quality, source or interface contract')
    if (value['population']['seed'] != 110001 or value['population']['rows'] != 96
            or value['population']['counterfactual_pairs'] != 48 or value['population']['tasks'] != 8
            or value['population']['pairs_per_task'] != 6
            or value['source_process_order'] != [['plain', 'initial'], ['response_contract', 'initial'],
                                                 ['response_contract', 'replay'], ['plain', 'replay']]
            or value['analysis']['bootstrap_draws'] != 10000 or value['analysis']['bootstrap_seed'] != 108001
            or value['analysis']['confidence_level'] != .95
            or value['analysis']['support_requires_lower_bound_strictly_above'] != 0
            or value['analysis']['all_task_quality_noninferiority_lower_bound'] != -.10
            or value['analysis']['unchanged_interface_rows_must_match_tokens'] != 72
            or value['analysis']['score_both_frozen_legacy_and_new_evaluators'] is not True):
        raise RuntimeError('Changed population, ordering or primary analysis')
    resource = value['resources']
    evaluator = value['evaluator_gate']
    if (resource['maximum_generation_calls'] != 384 or resource['maximum_generated_tokens'] != 384*640
            or resource['source_process_seconds']*resource['source_process_count'] != resource['model_process_seconds']
            or resource['model_process_seconds'] != 57600 or resource['controller_seconds'] != 59400
            or resource['minimum_free_bytes'] != 10*1024**3 or resource['maximum_evidence_bytes'] != 1024**3
            or resource['maximum_new_download_bytes'] != 0 or resource['gpu_setting_changes'] != []
            or resource['automatic_retries'] != 0 or resource['offline_evaluator_seconds'] != 1800
            or evaluator['challenge_rows'] != 600 or evaluator['challenge_rows_per_class'] != 100
            or len(set(evaluator['challenge_classes'])) != 6
            or evaluator['maximum_errors_in_each_class'] != 0 or evaluator['maximum_false_safe_assertions'] != 0
            or evaluator['unresolved_is_a_pass'] is not False
            or evaluator['first_challenge_failure_stops_study'] is not True):
        raise RuntimeError('Changed bounded resource or evaluator gate')
    prohibited = {'historical_rescoring', 'output_repair', 'sentence_truncation', 'forced_eos',
                  'best_of_sampling', 'model_sweep', 'prompt_sweep', 'training', 'recipient_execution',
                  'native_host_changes', 'automatic_follow_on'}
    if set(value['prohibited']) != prohibited:
        raise RuntimeError('Changed prohibited operations')


def historical_bindings():
    for relative, expected in IDENTITIES.items():
        if historical.sha(d.ROOT/relative) != expected:
            raise RuntimeError(f'R109 identity changed: {relative}')
    frozen = historical.binding()
    files = dict(frozen['files'])
    for path in historical.OUT.rglob('*'):
        if path.is_file():
            files[path.relative_to(d.ROOT).as_posix()] = historical.sha(path)
    files.update(IDENTITIES)
    return files, frozen


def rendered_inputs(tokenizer):
    result = []
    for row in d.population():
        for interface in d.INTERFACES:
            prompt = d.render(row, interface, tokenizer)
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens)+640 > 1024:
                raise RuntimeError('Input contract exceeds unchanged context budget')
            result.append({'id': row['id'], 'pair': row['pair'], 'task': row['task'],
                           'panel': row['panel'], 'interface': interface,
                           'user_message': d.user_message(row, interface),
                           'prompt': prompt, 'prompt_ids': tokens,
                           'input_tokens': len(tokens)})
    return result


def required_files():
    paths = [*(d.HERE/n for n in DESIGN_FILES), *(d.OUT/n for n in PACKET_FILES),
             d.ROOT/'tests/test_scope_adherence_r110.py']
    return {os.path.relpath(p, d.ROOT).replace(chr(92), '/') for p in paths}


def seal():
    target = d.OUT/'registration.json'
    if target.exists() or any((d.OUT/n).exists() for n in PACKET_FILES):
        raise RuntimeError('Preserve prior registration inputs')
    value = d.contract()
    validate_contract(value)
    free = shutil.disk_usage(d.ROOT).free
    if free < value['resources']['minimum_free_bytes']:
        raise RuntimeError('Insufficient free disk; do not delete evidence')
    files, frozen = historical_bindings()
    # Recompute the historical negative result; this performs zero generation.
    old_decision = historical_run.verify()
    if old_decision['verdict'] != 'CONTROL_QUALIFICATION_FAILED':
        raise RuntimeError('Unexpected historical scientific result')
    source = historical_run.inventory(historical_run.snapshot())
    if source != frozen['source_files']:
        raise RuntimeError('Cached source differs from preserved R109')
    tokenizer = historical_run.tokenizer('control')
    inputs = rendered_inputs(tokenizer)
    historical.write(d.OUT/'population.json', d.population())
    historical.write(d.OUT/'rendered_inputs.json', inputs)
    files.update({name: historical.sha(d.ROOT/name) for name in required_files()})
    historical.write(target, {
        'format': 'r110-registration/1', 'registered_unix': time.time(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=d.ROOT, text=True).strip(),
        'files': files, 'contract': value, 'source_files': source,
        'source_model': historical.SOURCE, 'source_revision': historical.REVISION,
        'population_rows': 96, 'input_renderings': 192,
        'maximum_input_tokens': max(r['input_tokens'] for r in inputs),
        'free_disk_bytes_at_registration': free,
        'status_at_registration': 'PREREGISTERED_NOT_EXECUTED',
        'implementation_at_registration': 'SCORER_CHALLENGE_AND_LIVE_DRIVER_NOT_IMPLEMENTED',
        'new_model_calls': 0, 'new_training_steps': 0, 'new_recipient_calls': 0,
        'gpu_setting_changes': [],
        'next_gate': 'Separate execution authorization and published implementation freeze before the first challenge',
    })
    verify()


def verify():
    registered = historical.read(d.OUT/'registration.json')
    validate_contract(d.contract())
    if (registered['format'] != 'r110-registration/1' or registered['contract'] != d.contract()
            or registered['status_at_registration'] != 'PREREGISTERED_NOT_EXECUTED'
            or registered['implementation_at_registration'] != 'SCORER_CHALLENGE_AND_LIVE_DRIVER_NOT_IMPLEMENTED'
            or any(registered[k] != 0 for k in ('new_model_calls', 'new_training_steps', 'new_recipient_calls'))
            or registered['gpu_setting_changes'] != []):
        raise RuntimeError('Invalid registration identity or scope')
    inherited, frozen = historical_bindings()
    expected_names = set(inherited) | required_files()
    if set(registered['files']) != expected_names:
        raise RuntimeError('Missing or unexpected registration binding')
    for name, digest in registered['files'].items():
        if historical.sha(d.ROOT/name) != digest or (name in inherited and digest != inherited[name]):
            raise RuntimeError(f'Missing or changed registration input: {name}')
    source = historical_run.inventory(historical_run.snapshot())
    if (registered['source_model'] != historical.SOURCE or registered['source_revision'] != historical.REVISION
            or registered['source_files'] != source or source != frozen['source_files']):
        raise RuntimeError('Missing or changed source package')
    inputs = rendered_inputs(historical_run.tokenizer('control'))
    if (historical.read(d.OUT/'population.json') != d.population()
            or historical.read(d.OUT/'rendered_inputs.json') != inputs
            or registered['population_rows'] != 96 or registered['input_renderings'] != len(inputs)
            or registered['maximum_input_tokens'] != max(r['input_tokens'] for r in inputs)):
        raise RuntimeError('Unrecomputable input population or context accounting')
    print(json.dumps({'status': 'REGISTRATION_INTEGRITY_ONLY', 'scientific_result': 'NOT_EXECUTED',
           'registration_sha256': historical.sha(d.OUT/'registration.json'),
           'files': len(expected_names), 'input_renderings': len(inputs),
           'maximum_input_tokens': registered['maximum_input_tokens']}), flush=True)
    return registered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('seal', 'verify'))
    command = parser.parse_args().command
    {'seal': seal, 'verify': verify}[command]()
