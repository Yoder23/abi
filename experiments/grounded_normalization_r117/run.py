"""Write-once live workers; public raw evidence and conditional scoring."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.evaluator_scope_r105 import core as nli
from experiments.trained_grounding_r108 import run as teacher
from . import method, panel

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'results/grounded_normalization_r117'
R116 = ROOT/'results/progress_grounding_r116'
PREREG = 'dd7884dd46af51bb0dff0cab57c1bf8aeb306329'
INPUT = 'abi-inputs-r117-2026-11-23'
GOLD = 'abi-gold-r117-2026-11-23'
MODULE = 'experiments.grounded_normalization_r117.run'


def need(cond, why):
    if not cond:
        raise ValueError(why)


def dump(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(dump(x))


def read(p):
    return json.loads(p.read_bytes())


def lines(p):
    return [json.loads(s) for s in p.read_bytes().splitlines()]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def filehash(p):
    return sha(p.read_bytes())


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def binding(tag, paths):
    for p in paths:
        need(git('show', tag+':'+p.relative_to(ROOT).as_posix()) == p.read_bytes(), 'changed '+str(p))


def guard():
    need(shutil.disk_usage(ROOT).free >= 10*2**30, 'disk floor')
    need(sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) <= 2**30, 'evidence cap')


def packet():
    binding(PREREG, [HERE/'PROTOCOL.md', HERE/'WP2_DESIGN.md'])
    binding(panel.METHOD, [HERE/'method.py', ROOT/'experiments/grounded_english_r99/core.py'])
    need(filehash(R116/'decision.json') == 'd42f8c8b71bf8fde4891641ea2374a30289073e046ff15f6eb0a4e2fffdee279' and
         filehash(R116/'packet.json') == '532829b576c99da7c2cf31e579f8a0bf91e9d168ab47d7bee59b5c663ec58aa1', 'R116')
    nf = nli.OUT/'freeze.json'
    need(filehash(nf) == '893718abe540aca2335000636ac75f0f0b4376b86f6eef1bf7b97443a775e741', 'R105')
    return dict(panel.build(), method=panel.METHOD, prereg=PREREG, r116=filehash(R116/'decision.json'),
                source=read(R116/'packet.json')['identities']['source'], classifier=read(nf)['model_files'],
                versions={p:importlib.metadata.version(p) for p in ('torch','transformers','tokenizers','numpy','psutil')})


def frozen():
    binding(INPUT, [HERE/p for p in ('panel.py','run.py','analysis.py')]+[OUT/'packet.json'])
    p = read(OUT/'packet.json')
    need(dump(p) == dump(packet()), 'packet reconstruct')
    return p


def records(p, live):
    need(len(live) == len(p['contexts']) == 192 and len(p['empty']) == 96, 'population')
    for ctx, r in zip(p['contexts'], live):
        need(r['id'] == ctx['id'] and sha(r['text'].encode()) == r['text_sha256'], 'live binding')
        for k, txt in list(sorted(ctx['challenges'].items()))+[('live',r['text'])]:
            yield dict(id=ctx['id']+'/'+k, kind=k, text=txt, **{k:ctx[k] for k in ('family','cluster','count','actual','target','notes')})
    for ctx in p['empty']:
        yield dict(ctx, kind='empty')


def check_gold(live, p):
    gold = read(OUT/'gold.json')
    need(gold['teacher_raw'] == filehash(OUT/'teacher-initial/raw.jsonl') and len(gold['rows']) == 192, 'gold length or binding')
    for g, r, c in zip(gold['rows'], live, p['contexts']):
        need(g['id'] == r['id'] == c['id'] and g['text_sha256'] == r['text_sha256'] and g['label'] in ('V','U','O','Q')
             and g['rationale'].strip() and (r['stop'] == 'eos' or g['label'] == 'Q'), 'gold rubric')
    return gold


def receipts(stage):
    paths = []
    for rep in ('initial', 'replay'):
        path = OUT/(stage+'-'+rep)
        rec, op = read(path/'receipt.json'), read(OUT/(path.name+'.json'))
        need(rec['pid'] == op['pid'] and rec['stage'] == stage and rec['rep'] == rep and op['exit'] == 0 and
             0 < rec['seconds'] <= op['seconds'] <= (1800 if stage == 'teacher' else 900) and
             rec['calls'] == (192 if stage == 'teacher' else 3744) and rec['raw_sha256'] == filehash(path/'raw.jsonl') and
             rec['packet'] == filehash(OUT/'packet.json') and rec['peak_cuda'] > 0 and rec['peak_rss'] > 0, 'receipt '+path.name)
        paths.append((rec['raw_sha256'], rec['pid']))
    need(paths[0][0] == paths[1][0] and paths[0][1] != paths[1][1], stage+' replay')
    return lines(OUT/(stage+'-initial/raw.jsonl'))


def teacher_rows(p, tok, model, torch):
    tok.padding_side = 'left'
    eos = model.generation_config.eos_token_id
    ends = eos if isinstance(eos, list) else [eos]
    for i in range(0,192,8):
        guard()
        contexts = p['contexts'][i:i+8]
        prompts = [tok.apply_chat_template([{'role':'user','content':c['prompt']}], tokenize=False,
                                           add_generation_prompt=True) for c in contexts]
        inputs = tok(prompts, add_special_tokens=False, return_tensors='pt', padding=True, truncation=False)
        width = inputs.input_ids.shape[1]
        need(width+80 <= 1024, 'context overflow')
        generated = model.generate(**inputs.to('cuda'), do_sample=False, max_new_tokens=80,
                                   num_beams=1, pad_token_id=tok.pad_token_id, use_cache=True)
        for c, prompt, inp, mask, seq in zip(contexts,prompts,inputs.input_ids.tolist(),inputs.attention_mask.tolist(),generated.tolist()):
            out = seq[width:]
            end = next((j for j,v in enumerate(out) if v in ends), None)
            if end is not None:
                out = out[:end+1]
            text = tok.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            yield dict(id=c['id'], prompt=prompt, input_ids=[t for t,m in zip(inp,mask) if m],
                       output_ids=out, eos_ids=ends, text=text, text_sha256=sha(text.encode()), stop='eos' if end is not None else 'length')


def nli_rows(p, live, tok, model, torch):
    rows = list(records(p, live))
    need(len(rows) == 1248, 'NLI population')
    for i in range(0,len(rows),5):
        guard()
        batch = rows[i:i+5]
        pairs = [pair for r in batch for pair in method.pairs(r['target'], r['notes'], r['text'])]
        a,b = zip(*pairs)
        inputs = tok(list(a),list(b), return_tensors='pt',padding=True,truncation=False)
        need(inputs.input_ids.shape[1] <= 512 and len(pairs) <= 16, 'NLI bounds')
        logits = model(**inputs.to('cuda')).logits.tolist()
        masks = inputs.attention_mask.tolist()
        ids = [[t for t,m in zip(s,mask) if m] for s,mask in zip(inputs.input_ids.tolist(),masks)]
        for j, row in enumerate(batch):
            sl = slice(3*j,3*j+3)
            yield dict(id=row['id'], target=row['target'], input_sha256=sha(dump(pairs[sl]).encode()),
                       candidate_sha256=sha(row['text'].encode()), ids=ids[sl], logits=logits[sl],
                       decisions=method.decide(row['target'],row['notes'],row['text'],logits[sl]))


def worker(stage, rep):
    import psutil
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    started = time.monotonic()
    proc = psutil.Process()
    p = frozen()
    guard()
    live = None
    if stage == 'nli':
        binding(GOLD, [OUT/'gold.json'])
        live = receipts('teacher')
        check_gold(live,p)
    torch.manual_seed(117001)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    src = stage == 'teacher'
    path = teacher.snapshot() if src else nli.MODEL
    weights = teacher.inventory(path) if src else nli.inventory(path,nli.MODEL_FILES)
    need(weights == p['source' if src else 'classifier'], 'model identity')
    options = dict(local_files_only=True,trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(path, **options)
    dtype = torch.bfloat16 if src else torch.float32
    cls = AutoModelForCausalLM if src else AutoModelForSequenceClassification
    model = cls.from_pretrained(path,dtype=dtype,**options).to('cuda').eval()
    need(all(v.dtype == dtype for v in model.parameters()), 'precision')
    if not src:
        need(model.config.id2label == dict(enumerate(nli.LABELS)), 'label order')
    cell = OUT/(stage+'-'+rep)
    counter = inputs = outputs = text_bytes = 0
    with torch.inference_mode(), (cell/'raw.jsonl').open('x',encoding='utf-8',newline='\n') as f:
        stream = teacher_rows(p,tok,model,torch) if src else nli_rows(p,live,tok,model,torch)
        for r in stream:
            counter += 1 if src else 3
            inputs += len(r['input_ids']) if src else sum(map(len,r['ids']))
            if src:
                outputs += len(r['output_ids'])
                text_bytes += len(r['text'].encode())
            f.write(dump(r))
            f.flush()
    need(counter == (192 if src else 3744), 'incomplete worker')
    write(cell/'receipt.json',dict(stage=stage,rep=rep,pid=os.getpid(),raw_sha256=filehash(cell/'raw.jsonl'),
        packet=filehash(OUT/'packet.json'),seconds=time.monotonic()-started,cpu_seconds=sum(proc.cpu_times()[:2]),
        model_manifest_sha256=sha(dump(weights).encode()), device=torch.cuda.get_device_name(),calls=counter,
        peak_cuda=torch.cuda.max_memory_allocated(),peak_rss=proc.memory_info().peak_wset,input_tokens=inputs,
        generated_tokens=outputs,generated_utf8_bytes=text_bytes,trainable_updates=0))


def launch(stage):
    need(stage in ('teacher','nli'), 'stage')
    guard()
    frozen()
    if stage == 'nli':
        binding(GOLD, [OUT/'gold.json'])
        check_gold(receipts('teacher'), read(OUT/'packet.json'))
    for rep in ('initial','replay'):
        cell = OUT/(stage+'-'+rep)
        cell.mkdir(exist_ok=False)
        start = time.monotonic()
        with (cell/'console.txt').open('xb') as f:
            proc = subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker',stage,rep],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                                    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1'))
            try:
                proc.wait(timeout=1800 if stage == 'teacher' else 900)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        write(OUT/(cell.name+'.json'),dict(pid=proc.pid,exit=proc.returncode,seconds=time.monotonic()-start))
        need(proc.returncode == 0, 'worker stopped; preserve '+str(cell))
    receipts(stage)


if __name__ == '__main__':
    action,*args = sys.argv[1:]
    if action == 'prepare':
        guard()
        write(OUT/'packet.json',packet())
    elif action == 'launch':
        launch(*args)
    elif action == 'worker':
        worker(*args)
    elif action == 'score':
        from .analysis import derive
        write(OUT/'decision.json', derive())
    elif action == 'verify':
        from .analysis import derive
        need(read(OUT/'decision.json') == derive(), 'reported result does not recompute')
        print('R117 raw reconstruction and exact fresh-process replays verified.')
    else:
        raise SystemExit('unknown operation')
