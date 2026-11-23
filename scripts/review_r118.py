"""CPU evidence audit. No GPU, weights, HF cache, sibling host or Git required."""
import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.modality_roles_r118 import analysis as a  # noqa: E402
from experiments.modality_roles_r118 import run as r  # noqa: E402

DECISION = '4a4e3294e2a245dddc1e8b62136967b99b72499fb5d9c05e46ada04c836b9633'
REG = '378caf60a4ae3bf27424155e92740195720070e7aa7ca19924bfa0f9314faf4e'
WP2 = '5be167cf386d0cb2e14fa2722870eeb0e3e8d325039bb62d0c514fd1a84ee9f3'


def inspect(tokenizers=False):
    decision = r.OUT/'decision.json'
    r.need(r.filehash(decision)==DECISION and r.filehash(r.HERE/'PROTOCOL.md')==REG
           and r.filehash(ROOT/'experiments/grounded_normalization_r117/WP2_DESIGN.md')==WP2, 'changed seal/design')
    d=r.read(decision)
    for key,h in d['evidence'].items():
        r.need(r.filehash(r.OUT/key)==h, 'changed or missing evidence: '+key)
    p=r.read(r.OUT/'packet.json')
    from experiments.modality_roles_r118 import panel
    r.need(r.dump({k:p[k] for k in ('contexts','empty')})==r.dump(panel.build()), 'prospective generation')
    source, predictions=r.receipts('teacher'),r.receipts('nli')
    gold=r.check_gold(source,p)
    providers={stage:r.sha(r.dump(p[key]).encode()) for stage,key in [('teacher','source'),('nli','classifier')]}
    for stage in ('teacher','nli'):
        for rep in ('initial','replay'):
            rec=r.read(r.OUT/(stage+'-'+rep)/'receipt.json')
            r.need(rec['model_manifest_sha256']==providers[stage] and rec['trainable_updates']==0,'worker identity')
            src=stage=='teacher'
            r.need(rec['input_tokens']==(sum(len(x['input_ids']) for x in source) if src else
                                         sum(sum(map(len,x['ids'])) for x in predictions)) and
                   rec['generated_tokens']==(sum(len(x['output_ids']) for x in source) if src else 0) and
                   rec['generated_utf8_bytes']==(sum(len(x['text'].encode()) for x in source) if src else 0),'accounting')
    obs=list(r.records(p,source))
    if tokenizers:
        from transformers import AutoTokenizer
        options=dict(local_files_only=True,trust_remote_code=False)
        tt=AutoTokenizer.from_pretrained(r.teacher.snapshot(),**options)
        nt=AutoTokenizer.from_pretrained(r.nli.MODEL,**options)
    for c,x in zip(p['contexts'],source):
        ids=x['output_ids']
        r.need(c['prompt'] in x['prompt'] and 0<len(ids)<=80 and 0<len(x['input_ids'])<=944
               and all(isinstance(t,int) and t>=0 for t in x['input_ids']+ids)
               and not any(t in x['eos_ids'] for t in ids[:-1]) and
               (ids[-1] in x['eos_ids'] if x['stop']=='eos' else x['stop']=='length' and len(ids)==80),'teacher stop')
        if tokenizers:
            prompt=tt.apply_chat_template([{'role':'user','content':c['prompt']}],tokenize=False,add_generation_prompt=True)
            r.need(prompt==x['prompt'] and x['input_ids']==tt.encode(prompt,add_special_tokens=False) and
                   tt.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==x['text'],'teacher tokens')
    if tokenizers:
        for row,z in zip(obs,predictions):
            left,right=zip(*r.method.prior.pairs(row['target'],row['notes'],row['text']))
            r.need(nt(list(left),list(right),truncation=False)['input_ids']==z['ids'],'NLI tokens')
    summary=a.analyze(p,source,gold,predictions)
    r.need(summary=={k:v for k,v in d.items() if k!='evidence'},'metrics or decision do not recompute')
    return summary


def bundle(dest):
    # Include the exact imported historical Python helpers, not their datasets or weights.
    files={Path(m.__file__).resolve() for name,m in sys.modules.items() if name.startswith('experiments.')
           and getattr(m,'__file__',None)}
    files.update(p for p in r.OUT.rglob('*') if p.is_file())
    files.update((r.HERE/'blind.py',r.HERE/'annotation.py',r.HERE/'PROTOCOL.md',ROOT/'experiments/grounded_normalization_r117/WP2_DESIGN.md',Path(__file__).resolve(),
                  ROOT/'docs/releases/R118_REVIEW.md',ROOT/'LICENSE'))
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix())
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizers',action='store_true',help='also verify every token against the pinned local tokenizer caches')
    parser.add_argument('--bundle',type=Path,help='create a new small standalone evidence-review ZIP')
    options=parser.parse_args()
    d=inspect(options.tokenizers)
    if options.bundle:
        bundle(options.bundle)
    print(f"{d['verdict']}; WP2 {d['wp2']}; {d['arms']['abi']['complete_clusters']}/48 complete clusters; decision {DECISION}")
