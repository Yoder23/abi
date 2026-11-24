"""Display blinded groups; attach only explicit judgments to immutable group hashes."""
import argparse
from . import run as r
from . import core as c

BLIND=c.OUT/'blind.json'
VOTES=c.OUT/'manual_votes.json'


def judgments(task,votes):
    ids={}
    c.need(set(votes)==set('VUOQ'),'unknown or missing rating class')
    for label,numbers in votes.items():
        for idx in numbers:
            c.need(type(idx) is int and 0<=idx<len(task) and idx not in ids,'invalid/duplicate group')
            ids[idx]=label
    c.need(set(ids)==set(range(len(task))),'unreviewed group')
    return [dict(index=i,key=g['key'],label=ids[i]) for i,g in enumerate(task)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['show','seal'])
    ap.add_argument('start',nargs='?',type=int,default=0)
    ap.add_argument('end',nargs='?',type=int,default=10000)
    args=ap.parse_args()
    task=c.read(BLIND)
    if args.action=='show':
        print(len(task),'groups')
        for i,g in enumerate(task):
            if args.start<=i<args.end:
                print(f"{i:03} {' / '.join(g['notes'])} || {g['final']!a}")
        return
    c.need(not (c.OUT/'decision.json').exists(),'score already exists')
    cases,rows=r.raw()
    c.need(r.groups(cases,iter(rows))==task,'stale blind review')
    c.put(c.OUT/'labels.json',dict(blind=r.filehash(BLIND),raw=r.filehash(c.OUT/'initial/raw.jsonl'),
                provenance='Prediction- and arm-masked investigator judgments, not independent or human.',
                votes=r.filehash(VOTES),rows=judgments(task,c.read(VOTES))))


if __name__=='__main__':
    main()
