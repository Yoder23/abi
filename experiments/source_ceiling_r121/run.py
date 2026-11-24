"""Write-once source workers and standalone evidence review. No host or filter."""
import os
import shutil
import subprocess
import sys
import time
from . import core as c

ROOT,HERE,OUT=c.ROOT,c.HERE,c.OUT
INPUT='abi-inputs-r121-2026-11-24'
GOLD='abi-labels-r121-2026-11-24'
FILES=('core.py','run.py','PROTOCOL.md')
P=OUT/'packet.json'


def filehash(p):
    return c.sha(p.read_bytes())


def bind(ref,files):
    for p in files:
        blob=subprocess.check_output(['git','show',ref+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
        c.need(blob==p.read_bytes(),'freeze mismatch '+str(p))


def guard():
    c.need(shutil.disk_usage(ROOT).free>=10*2**30 and sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=256*2**20,'storage limits')


def prepare():
    old=ROOT/'results/project_binding_r120'
    c.need(filehash(old/'decision.json')==c.OLD and filehash(ROOT/'experiments/grounded_normalization_r117/WP2_DESIGN.md')==c.WP2,'historical bindings')
    bind(c.REG,[HERE/'PROTOCOL.md'])
    prev=c.read(old/'packet.json')
    cases=c.panel()
    c.need(not {x[k] for x in cases for k in ('target','foil')}.intersection(k for x in prev['contexts'] for k,_ in x['notes']),'old IDs')
    c.need(not {x['prompts']['direct'] for x in cases}.intersection(x['prompt'] for x in prev['contexts']),'old prompts')
    return dict(cases=cases,source=prev['source'],versions=prev['versions'],old=c.OLD,wp2_design=c.WP2,registration=c.REG,
                code={name:filehash(HERE/name) for name in FILES})


def frozen():
    bind(INPUT,[P]+[HERE/n for n in FILES])
    p=c.read(P)
    c.need(p==prepare(),'changed packet')
    return p


def worker(rep):
    import importlib.metadata
    import psutil
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from experiments.trained_grounding_r108 import run as old
    start=time.monotonic()
    proc=psutil.Process()
    p=frozen()
    c.need({k:importlib.metadata.version(k) for k in p['versions']}==p['versions'],'environment')
    loc=old.snapshot()
    c.need(old.inventory(loc)==p['source'],'source weights')
    torch.manual_seed(121001)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    tok=AutoTokenizer.from_pretrained(loc,local_files_only=True,trust_remote_code=False)
    tok.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(loc,dtype=torch.bfloat16,local_files_only=True,trust_remote_code=False).to('cuda').eval()
    c.need(all(w.dtype==torch.bfloat16 for w in model.parameters()),'precision')
    ends=model.generation_config.eos_token_id
    ends=ends if isinstance(ends,list) else [ends]
    path=OUT/rep/'raw.jsonl'
    costs={a:dict(calls=0,input_tokens=0,output_tokens=0,output_bytes=0,batches=0) for a in c.ARMS}
    with torch.inference_mode(),path.open('x',encoding='utf-8',newline='\n') as f:
        for group in c.schedule(p['cases']):
            guard()
            prompts=[tok.apply_chat_template([dict(role='user',content=x['prompts'][a])],tokenize=False,add_generation_prompt=True) for a,x in group]
            inps=tok(prompts,add_special_tokens=False,return_tensors='pt',padding=True,truncation=False)
            width=inps.input_ids.shape[1]
            c.need(width+80<=1024 and len(group)<=8,'model bounds')
            out=model.generate(**inps.to('cuda'),do_sample=False,max_new_tokens=80,num_beams=1,pad_token_id=tok.pad_token_id,use_cache=True)
            costs[group[0][0]]['batches']+=1
            for (a,x),s,inp,mask,seq in zip(group,prompts,inps.input_ids.tolist(),inps.attention_mask.tolist(),out.tolist()):
                gen=seq[width:]
                stop=next((i for i,t in enumerate(gen) if t in ends),None)
                gen=gen if stop is None else gen[:stop+1]
                txt=tok.decode(gen,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                ids=[t for t,m in zip(inp,mask) if m]
                v=dict(arm=a,id=x['id'],prompt=s,input_ids=ids,output_ids=gen,ends=ends,
                       stop='length' if stop is None else 'eos',text=txt,text_sha256=c.sha(txt.encode()))
                f.write(c.dump(v))
                z=costs[a]
                z['calls']+=1
                z['input_tokens']+=len(ids)
                z['output_tokens']+=len(gen)
                z['output_bytes']+=len(txt.encode())
            f.flush()
    c.put(OUT/rep/'receipt.json',dict(pid=os.getpid(),rep=rep,raw=filehash(path),packet=filehash(P),
        source=c.sha(c.dump(p['source']).encode()),device=torch.cuda.get_device_name(),costs=costs,
        seconds=time.monotonic()-start,cpu_seconds=sum(proc.cpu_times()[:2]),peak_cuda=torch.cuda.max_memory_allocated(),
        peak_rss=proc.memory_info().peak_wset,trainable_updates=0))


def launch():
    frozen()
    guard()
    for rep in ('initial','replay'):
        d=OUT/rep
        d.mkdir(exist_ok=False)
        start=time.monotonic()
        with (d/'console.txt').open('xb') as f:
            proc=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'worker',rep],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                                env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1'))
            try:
                proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        c.put(OUT/(rep+'.json'),dict(pid=proc.pid,exit=proc.returncode,seconds=time.monotonic()-start))
        c.need(proc.returncode==0,'stopped; preserve '+rep)
    raw()


def raw(tokenizers=False):
    import json
    p=c.read(P)
    c.need(p['cases']==c.panel() and p['old']==c.OLD and p['wp2_design']==c.WP2 and
           p['registration']==c.REG and p['code']=={n:filehash(HERE/n) for n in FILES},'panel or code')
    hashes=[]
    pids=[]
    for rep in ('initial','replay'):
        d=OUT/rep
        rec,op=c.read(d/'receipt.json'),c.read(OUT/(rep+'.json'))
        c.need(rec['rep']==rep and rec['pid']==op['pid'] and op['exit']==0 and
               0<rec['seconds']<=op['seconds']<=1800 and rec['packet']==filehash(P) and
               rec['source']==c.sha(c.dump(p['source']).encode()) and rec['raw']==filehash(d/'raw.jsonl') and
               rec['trainable_updates']==0 and min(rec['peak_cuda'],rec['peak_rss'])>0,'worker receipt')
        hashes.append(rec['raw'])
        pids.append(rec['pid'])
    c.need(hashes[0]==hashes[1] and pids[0]!=pids[1],'fresh replay')
    rows=[json.loads(l) for l in (OUT/'initial/raw.jsonl').read_bytes().splitlines()]
    ordered=[pair for g in c.schedule(p['cases']) for pair in g]
    c.need(len(rows)==len(ordered)==576,'population')
    if tokenizers:
        from transformers import AutoTokenizer
        from experiments.trained_grounding_r108 import run as old
        tok=AutoTokenizer.from_pretrained(old.snapshot(),local_files_only=True,trust_remote_code=False)
    for x,(a,case) in zip(rows,ordered):
        ids,gen=x['input_ids'],x['output_ids']
        c.need((x['arm'],x['id'])==(a,case['id']) and x['text_sha256']==c.sha(x['text'].encode()) and
               case['prompts'][a] in x['prompt'] and 0<len(ids)<=944 and 0<len(gen)<=80 and
               all(type(n) is int and n>=0 for n in ids+gen) and not set(gen[:-1]).intersection(x['ends']) and
               ((x['stop']=='length' and len(gen)==80 and gen[-1] not in x['ends']) or
                (x['stop']=='eos' and gen[-1] in x['ends'])),'source row')
        if tokenizers:
            s=tok.apply_chat_template([dict(role='user',content=case['prompts'][a])],tokenize=False,add_generation_prompt=True)
            c.need(s==x['prompt'] and tok.encode(s,add_special_tokens=False)==ids and
                   tok.decode(gen,skip_special_tokens=True,clean_up_tokenization_spaces=False)==x['text'],'token reconstruction')
    for a in c.ARMS:
        data=[x for x in rows if x['arm']==a]
        costs=dict(calls=len(data),input_tokens=sum(len(x['input_ids']) for x in data),output_tokens=sum(len(x['output_ids']) for x in data),
                   output_bytes=sum(len(x['text'].encode()) for x in data),batches=sum(g[0][0]==a for g in c.schedule(p['cases'])))
        for rep in ('initial','replay'):
            c.need(costs==c.read(OUT/rep/'receipt.json')['costs'][a],'wrong costs')
    return p['cases'],rows


def groups(cases,rows):
    bag={}
    for group in c.schedule(cases):
        for a,x in group:
            v=next(rows)
            c.need((a,x['id'])==(v['arm'],v['id']),'order')
            g,_,_=c.bucket(x,a,v)
            if g is not None:
                bag[c.sha(c.dump(g).encode())]=g
    # Display near-identical sentences together; never show arm, record, IDs of cases or arm totals.
    return [dict(key=k,**bag[k]) for k in sorted(bag,key=lambda k:(bag[k]['notes'][0],bag[k]['final'],bag[k]['notes'][1:]))]


def labels(cases,rows):
    task=groups(cases,iter(rows))
    c.need(c.read(OUT/'blind.json')==task,'blind packet')
    gold=c.read(OUT/'labels.json')
    c.need(gold['blind']==filehash(OUT/'blind.json') and gold['raw']==filehash(OUT/'initial/raw.jsonl') and
           [x['key'] for x in gold['rows']]==[x['key'] for x in task],'labels detached')
    return {x['key']:x['label'] for x in gold['rows']}


def derive(tokenizers=False):
    cases,rows=raw(tokenizers)
    summary=c.metrics(c.classify(cases,rows,labels(cases,rows)))
    paths=['packet.json','blind.json','labels.json']+[v for rep in ('initial','replay') for v in (rep+'.json',rep+'/raw.jsonl',rep+'/receipt.json')]
    return dict(summary,evidence={s:filehash(OUT/s) for s in paths})


if __name__=='__main__':
    action,*args=sys.argv[1:]
    if action=='prepare':
        c.put(P,prepare())
    elif action=='launch':
        launch()
    elif action=='worker':
        worker(*args)
    elif action=='blind':
        cases,rows=raw()
        c.put(OUT/'blind.json',groups(cases,iter(rows)))
    elif action=='score':
        frozen()
        bind(GOLD,[OUT/'blind.json',OUT/'labels.json'])
        c.put(OUT/'decision.json',derive())
    elif action=='verify':
        d=derive('--tokenizers' in args)
        c.need(d==c.read(OUT/'decision.json'),'decision does not recompute')
        print(d['verdict'],d['paired'],'WP2',d['wp2'])
    else:
        raise SystemExit('unknown stage')
