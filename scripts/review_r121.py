"""Review frozen source-only R121 from public evidence using Python and NumPy."""
import argparse
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.source_ceiling_r121 import adjudication as j, run as r, core as c

SEAL='f5e3c4155d1ed1c71a969fa3099cf05a96f7be6b6ef64acda8c59653489bf69e'
PACKET='dfa3a7c1a51376e301a3b3f7633a45157b2d7059192f8228c805297e297cad14'


def inspect(tokenizers=False):
    path=c.OUT/'decision.json'
    c.need(r.filehash(path)==SEAL and r.filehash(r.P)==PACKET,'changed result or input freeze')
    gold=c.read(c.OUT/'labels.json')
    c.need(gold['votes']==r.filehash(j.VOTES) and
           j.judgments(c.read(j.BLIND),c.read(j.VOTES))==gold['rows'],'unbound blind judgments')
    d=r.derive(tokenizers)
    c.need(d==c.read(path),'decision or raw evidence changed')
    # Compute the ceilings a second way, from the paired per-cell class counts.
    for a,z in d['arms'].items():
        v={k:n['V'] for k,n in z['cells'].items()}
        c.need(len(v)==48 and all(sum(n.values())==6 for n in z['cells'].values()),'cell sizes')
        windows={s:sum(n for k,n in v.items() if k.split('/')[-1] in (s,'occurred')) for s in c.STATES[:2]}
        c.need(all(n==z['windows'][s]['valid'] for s,n in windows.items()) and
               sum(v.values())==z['totals']['V'] and
               z['ceiling_pass']==(sum(v.values())>=216 and min(v.values())>=3 and min(windows.values())>=144),
               'independent oracle ceiling')
    for rep in ('initial','replay'):
        receipt=c.read(c.OUT/rep/'receipt.json')
        c.need(receipt['costs']['direct']['calls']==receipt['costs']['fields']['calls']==288,'equal sequences')
    return d


def bundle(path,d):
    files={r.HERE/n for n in (*r.FILES,'adjudication.py')}
    files.update(c.OUT/n for n in d['evidence'])
    files.update([j.VOTES,c.OUT/'decision.json',Path(__file__).resolve(),ROOT/'docs/releases/R121_REVIEW.md',ROOT/'LICENSE'])
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            i=zipfile.ZipInfo(p.relative_to(ROOT).as_posix())
            i.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(i,p.read_bytes())


def output_diagnostics():
    """Descriptive only: do not restore missing IDs, repair or re-score any row."""
    from collections import Counter
    cases,rows=r.raw()
    scores=c.classify(cases,rows,r.labels(cases,rows))
    counts={a:Counter() for a in c.ARMS}
    for (case,a,label,_),x in zip(scores,rows):
        final,why=c.parse(a,x['text'],x['stop'])
        if final is None:
            counts[a]['unscorable_'+why]+=1
        elif case['target'] not in final:
            counts[a]['target_absent_from_final']+=1
            counts[a]['target_absent_class_'+label]+=1
            if a=='fields' and case['target'] in x['text'][:len(x['text'])-len(final)]:
                counts[a]['target_in_workspace_but_not_final']+=1
    return {a:dict(n) for a,n in counts.items()}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokenizers',action='store_true')
    ap.add_argument('--bundle',type=Path)
    ap.add_argument('--diagnostics',action='store_true')
    args=ap.parse_args()
    d=inspect(args.tokenizers)
    for a,z in d['arms'].items():
        print(a,z['totals'],'windows', {k:t['valid'] for k,t in z['windows'].items()},
              'cell misses',sum(n['V']<3 for n in z['cells'].values()),'ceiling',z['ceiling_pass'])
    print(d['verdict'],'WP2',d['wp2'],'paired',d['paired'],'SHA256',SEAL)
    if args.diagnostics:
        print('Descriptive ID counts:',output_diagnostics())
    if args.bundle:
        bundle(args.bundle,d)
