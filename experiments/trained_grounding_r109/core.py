"""R109 reuses R108 scientific functions verbatim; resources and namespace differ."""
from pathlib import Path
import os

from experiments.trained_grounding_r108 import core as previous

ROOT, HOST = previous.ROOT, previous.HOST
HERE = Path(__file__).resolve().parent
OUT = ROOT/'results/trained_grounding_r109'
HOST_REVISION = '763e6da3cca1aa7564e6bce4b64dbee4effa552d'
SOURCE, REVISION, SOURCE_FILES = previous.SOURCE, previous.REVISION, previous.SOURCE_FILES
ARMS, TASKS, ROUTES = previous.ARMS, previous.TASKS, previous.ROUTES
MAX_NEW, CONTEXT, SEED = previous.MAX_NEW, previous.CONTEXT, previous.SEED
FILES = previous.FILES
sha, read, rows, write, append = previous.sha, previous.read, previous.rows, previous.write, previous.append
checkpoint, training_provenance = previous.checkpoint, previous.training_provenance
catalog, screen, audit, render = previous.catalog, previous.screen, previous.audit, previous.render
validate, pair_rows, summarize, contrast = previous.validate, previous.pair_rows, previous.summarize, previous.contrast

CONTROL_SECONDS, RECIPIENT_SECONDS = 14400, 3600
TOTAL_SECONDS, MODEL_PROCESS_SECONDS = 59400, 57600
MIN_FREE_BYTES, MAX_EVIDENCE_BYTES = 10*1024**3, 1024**3
R108_FREEZE_SHA256 = 'b8c58cb2d2809df0f92ab4ba321b5b8edaf2821017abdfa43baf531c29f3509d'


def process_budget(arm):
    if arm not in ('control', *ARMS):
        raise ValueError('Unregistered arm')
    return CONTROL_SECONDS if arm == 'control' else RECIPIENT_SECONDS


def binding():
    frozen = read(OUT/'freeze.json')
    if (not frozen.get('files') or frozen['source_model'] != SOURCE or frozen['source_revision'] != REVISION
            or frozen['resource_policy'] != resource_policy()):
        raise RuntimeError('Missing or wrong R109 freeze')
    if sha(previous.OUT/'freeze.json') != R108_FREEZE_SHA256:
        raise RuntimeError('Historical R108 freeze changed')
    ancestor = read(previous.OUT/'freeze.json')
    if (any(frozen['files'].get(p) != digest for p, digest in ancestor['files'].items())
            or frozen['source_files'] != ancestor['source_files']):
        raise RuntimeError('Missing or changed inherited bindings')
    required = [HERE/'core.py', HERE/'run.py', HERE/'PROTOCOL.md',
                ROOT/'tests/test_trained_grounding_r109.py', OUT/'source.json',
                OUT/'population.json', OUT/'evaluator_audit.json', OUT/'matched_training_provenance.json',
                OUT/'resources/preflight.json', previous.OUT/'freeze.json',
                previous.OUT/'control/initial/raw.jsonl', previous.OUT/'stop.json',
                previous.OUT/'interruption.json']
    if any(os.path.relpath(p, ROOT).replace(chr(92), '/') not in frozen['files'] for p in required):
        raise RuntimeError('Missing mandatory evidence binding')
    for name, digest in frozen['files'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError(f'Changed frozen input: {name}')
    for name, expected in (('population.json', catalog()), ('evaluator_audit.json', audit()),
                           ('matched_training_provenance.json', training_provenance())):
        if read(OUT/name) != expected or sha(OUT/name) != sha(previous.OUT/name):
            raise RuntimeError(f'Changed scientific input: {name}')
    if audit()['mismatches']:
        raise RuntimeError('Evaluator prerequisite failed')
    return frozen


def resource_policy():
    return {'control_process_seconds': CONTROL_SECONDS, 'recipient_process_seconds': RECIPIENT_SECONDS,
            'controller_seconds': TOTAL_SECONDS, 'model_process_seconds': MODEL_PROCESS_SECONDS,
            'minimum_free_bytes': MIN_FREE_BYTES, 'maximum_evidence_bytes': MAX_EVIDENCE_BYTES,
            'maximum_generation_calls': 960, 'new_download_bytes': 0, 'gpu_setting_changes': []}
