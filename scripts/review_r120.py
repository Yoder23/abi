"""CPU evidence audit. No GPU, weights, HF cache, sibling host or Git required."""
import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.project_binding_r120 import analysis as a  # noqa: E402
from experiments.project_binding_r120 import run as r  # noqa: E402

DECISION = 'd81387e9a726e4ee93b7192cf89dd1b4d562aa1895554875c81cf4868136d427'
REG = '5e7908dd93d080251f46fd8789a7216b7ac34f49a28fcaec8008805c54d7ecd4'
WP2 = '5be167cf386d0cb2e14fa2722870eeb0e3e8d325039bb62d0c514fd1a84ee9f3'


def verify_review_groups(packet, source, gold):
    """Bind every investigator judgment to its exact prediction-blind group."""
    from experiments.project_binding_r120 import annotation as ann
    groups = ann.groups(packet, source)
    assignments = [(int(i), code) for code, ids in ann.INDEX.items() for i in ids.split()]
    r.need(sorted(i for i, _ in assignments) == list(range(len(groups))), 'incomplete group review')
    by_id = {g['id']: g for g in gold['rows']}
    seen = set()
    for i, code in assignments:
        key, ids = groups[i]
        label, reason = ann.KEY[code]
        digest = r.sha(r.dump(key).encode())
        for rid in ids:
            g = by_id[rid]
            r.need(rid not in seen and (g['label'], g['rationale'], g['review_group'], g['group_sha256'])
                   == (label, reason, i, digest), 'changed prediction-blind judgment')
            seen.add(rid)
    r.need(len(seen) == len(source) == len(gold['rows']), 'incomplete label coverage')


def inspect(tokenizers=False):
    decision = r.OUT/'decision.json'
    r.need(r.filehash(decision)==DECISION and r.filehash(r.HERE/'PROTOCOL.md')==REG
           and r.filehash(ROOT/'experiments/grounded_normalization_r117/WP2_DESIGN.md')==WP2, 'changed seal/design')
    d=r.read(decision)
    for key,h in d['evidence'].items():
        r.need(r.filehash(r.OUT/key)==h, 'changed or missing evidence: '+key)
    p=r.read(r.OUT/'packet.json')
    from experiments.project_binding_r120 import panel
    r.need(r.dump({k:p[k] for k in ('contexts','empty')})==r.dump(panel.build()), 'prospective generation')
    source, predictions=r.receipts('teacher'),r.receipts('nli')
    gold=r.check_gold(source,p)
    verify_review_groups(p,source,gold)
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
            left,right=zip(*r.method.pairs(row['target'],row['notes'],row['roster'],row['text']))
            r.need(nt(list(left),list(right),truncation=False)['input_ids']==z['ids'],'NLI tokens')
    summary=a.analyze(p,source,gold,predictions)
    r.need(summary=={k:v for k,v in d.items() if k!='evidence'},'metrics or decision do not recompute')
    # Independent paired-unit tally from the public inputs and recomputed predictions.
    for arm in r.method.ARMS:
        lookup={z['id']:z['decisions']['arms'][arm]['retained_sha256'] is not None for z in predictions}
        all_correct=0
        pairs={v:0 for v in panel.VOICES}
        for c in p['contexts']:
            answers=[lookup[c['id']+'/valid_'+v] and not lookup[c['id']+'/swap_'+v] for v in panel.VOICES]
            all_correct+=all(answers)
            for v,correct in zip(panel.VOICES,answers):
                pairs[v]+=correct
        check=summary['arms'][arm]
        r.need(all_correct==check['all_six'] and all(k==check['voices'][v]['paired'] for v,k in pairs.items()),'independent pair counts')
        for pos,neg in [('valid_prefix','wrong_prefix'),('valid_label','wrong_label')]:
            paired=sum(lookup[c['id']+'/'+pos] and not lookup[c['id']+'/'+neg] for c in p['contexts'])
            r.need(paired==check['prefix'][pos]['paired'],'independent prefix pairing')
        rentals=[c for c in p['contexts'] if c['family'] in panel.RENTALS]
        r.need(sum(lookup[c['id']+'/valid_converse'] and not lookup[c['id']+'/swap_converse'] for c in rentals)
               ==check['rental']['paired'],'independent rental pairing')
    for rep in ('initial','replay'):
        rec=r.read(r.OUT/('nli-'+rep)/'receipt.json')
        r.need(all(rec[k]==v for k,v in summary['cost'].items()),'physical model accounting')
    return summary


def bundle(dest):
    # Include the exact imported historical Python helpers, not their datasets or weights.
    files={Path(m.__file__).resolve() for name,m in sys.modules.items() if name.startswith('experiments.')
           and getattr(m,'__file__',None)}
    # Only publish enumerated scientific evidence; do not sweep local scratch files.
    decision=r.OUT/'decision.json'
    files.add(decision)
    files.update(r.OUT/name for name in r.read(decision)['evidence'])
    files.update((r.HERE/'blind.py',r.HERE/'annotation.py',r.HERE/'PROTOCOL.md',ROOT/'experiments/grounded_normalization_r117/WP2_DESIGN.md',Path(__file__).resolve(),
                  ROOT/'docs/releases/R120_REVIEW.md',ROOT/'LICENSE'))
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix())
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizers',action='store_true',help='also verify every token against the pinned local tokenizer caches')
    parser.add_argument('--bundle',type=Path,help='create a new small standalone evidence-review ZIP')
    parser.add_argument('--paired',action='store_true',help='show registered prefix and rental paired contrasts')
    options=parser.parse_args()
    d=inspect(options.tokenizers)
    if options.bundle:
        bundle(options.bundle)
    if options.paired:
        print(r.dump(d['paired_contrasts']).strip())
    print(f"{d['verdict']}; WP2 {d['wp2']}; {d['arms']['abi']['complete_clusters']}/48 complete clusters; decision {DECISION}")
