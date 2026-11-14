"""Additive read-only hardware observation; no power/thermal setting changes."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.trained_grounding_r108 import core as c


def inspect(name):
    import psutil
    commands = []
    for argv in (['nvidia-smi', '-q', '-d', 'PERFORMANCE,POWER,CLOCK'],
                 ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,clocks.sm,power.draw,temperature.gpu', '--format=csv,noheader']):
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        commands.append({'argv': argv, 'returncode': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr})
    battery = psutil.sensors_battery()
    path = c.OUT/'control/initial/raw.jsonl'
    # Only complete flushed JSON lines; preserve an in-progress last line separately.
    raw, tail = [], None
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                tail = line
    elapsed = time.time()-c.read(c.OUT/'launch.json')['start_unix']
    value = {'format': 'r108-live-resource-observation/1', 'observed_unix': time.time(),
        'freeze_sha256': c.sha(c.OUT/'freeze.json'), 'commands': commands,
        'battery': battery._asdict() if battery else None, 'free_disk_bytes': shutil.disk_usage(ROOT).free,
        'virtual_memory': psutil.virtual_memory()._asdict(), 'launch_elapsed_seconds': elapsed,
        'completed_rows': len(raw), 'incomplete_last_line': tail,
        'observed_generation_seconds': sum(r['wall_seconds'] for r in raw),
        'observed_output_tokens': sum(len(r['token_ids']) for r in raw),
        'linear_96_prompt_generation_seconds_projection': sum(r['wall_seconds'] for r in raw)*96/len(raw) if raw else None,
        'projection_scope': 'Extrapolation from completed rows only, not a measured complete-run cost',
        'settings_changed': [], 'scope': 'Resource diagnosis, no scientific qualification or performance superiority claim'}
    c.write(c.OUT/f'resource_{name}.json', value)
    print(json.dumps({k: value[k] for k in ('completed_rows', 'launch_elapsed_seconds', 'observed_generation_seconds',
                                          'observed_output_tokens', 'linear_96_prompt_generation_seconds_projection')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=('initial', 'final'))
    inspect(parser.parse_args().name)
