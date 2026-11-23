"""Frozen analysis: no selective acceptance or deletion of unparsed clauses."""
from collections import Counter
from experiments.grounded_normalization_r117.analysis import tally, bootstrap
from . import run as r
from . import method, panel


def subset(rows, **keys):
    return [x for x in rows if all(x[k]==v for k,v in keys.items())]


def inv(t):
    return sum(t['accepted'][k] for k in 'UOQ')


def calc(table):
    authored=[x for x in table if x['kind']!='live']
    live=subset(table,kind='live')
    kinds={k:tally(subset(table,kind=k)) for k in sorted({x['kind'] for x in table})}
    families,cells,transitions,windows={},{},{},{}
    for f in sorted(panel.BANK):
        rows=subset(authored,family=f)
        complete=0
        for i in range(6):
            cluster=[x for x in rows if x['cluster']==f+'-'+str(i) and x['kind']!='empty']
            r.need(len(cluster)==48,'cluster shape')
            complete+=all(x['ok']==(x['label']=='V') for x in cluster)
        families[f]=dict(complete=complete,authored=tally(rows),live=tally(subset(live,family=f)),
                         valid=tally(subset(rows,kind='valid')))
        for count in (1,2):
            for state in panel.STATES:
                cells[f'{f}/{count}/{state}']=tally(subset(live,family=f,count=count,state=state))
    for s in panel.STATES:
        transitions[s]={k:tally(subset(authored,state=s,kind=k)) for k in
                        ['actor_swap','joint']+['to_'+t for t in panel.STATES if t!=s]}
    for other in ('authorized','planned'):
        window=[x for x in live if x['state'] in (other,'occurred')]
        t=tally(window)
        windows[other]=dict(live=t,passed=(t['n']==192 and t['accepted']['V']>=144 and
            (t['valid_retention'] or 0)>=.9 and all((tally(subset(window,family=f))['valid_retention'] or 0)>=.8 for f in families)))
    a,s=tally(authored),tally(live)
    n=sum(v['complete'] for v in families.values())
    gates=dict(clusters=n>=44,every_family=all(v['complete']>=5 for v in families.values()),
               zero_authored_invalid=inv(a)==0,no_notes=kinds['empty']['withheld']==96,
               live_usable=s['accepted']['V']>=216,live_retention=(s['valid_retention'] or 0)>=.9,
               family_retention=all((v['live']['valid_retention'] or 0)>=.8 for v in families.values()),
               live_cell_supply=all(x['n']==6 and x['gold']['V']>=3 for x in cells.values()),
               zero_live_invalid=inv(s)==0,original_windows=all(x['passed'] for x in windows.values()))
    return dict(authored=a,live=s,by_kind=kinds,by_family=families,cells=cells,directed=transitions,
                windows=windows,complete_clusters=n,gates=gates,absolute_pass=all(gates.values()),
                nonverbatim=tally([x for x in live if dict(x['notes'])[x['target']]!=x['text']]),
                reasons=dict(Counter(x['reason'] for x in table)))


def analyze(p,live,gold,predictions):
    expected=list(r.records(p,live))
    r.need(len(expected)==len(predictions)==2688 and len({x['id'] for x in expected})==2688,'population')
    r.need(len(gold['rows'])==288,'label population')
    for c,x,g in zip(p['contexts'],live,gold['rows']):
        ids=x['output_ids']
        r.need(x['id']==g['id']==c['id'] and x['text_sha256']==g['text_sha256'] and g['label'] in ('V','U','O','Q')
               and g['rationale'] and c['prompt'] in x['prompt'] and 0<len(x['input_ids'])<=944 and 0<len(ids)<=80
               and not any(t in x['eos_ids'] for t in ids[:-1]) and (ids[-1] in x['eos_ids'] if x['stop']=='eos'
               else x['stop']=='length' and len(ids)==80 and g['label']=='Q'),'teacher-label binding')
    labels={g['id']:g['label'] for g in gold['rows']}
    tables={arm:[] for arm in method.ARMS}
    for x,z in zip(expected,predictions):
        pairs=method.prior.pairs(x['target'],x['notes'],x['text'])
        r.need(z['id']==x['id'] and z['target']==x['target'] and z['candidate_sha256']==r.sha(x['text'].encode())
               and z['input_sha256']==r.sha(r.dump(pairs).encode()) and len(z['ids'])==3 and
               all(0<len(seq)<=512 and all(type(t) is int and t>=0 for t in seq) for seq in z['ids']),'record custody')
        d=method.decide(x['target'],x['notes'],x['roster'],x['text'],z['logits'])
        r.need(d==z['decisions'],'invented decision or slots')
        kind=x['kind']
        label=(labels[x['id'].rsplit('/',1)[0]] if kind=='live' else 'V' if kind=='valid' else
               'O' if kind in ('omission','off_target') else 'U')
        for arm in tables:
            why=d['arms'][arm]['reason']
            tables[arm].append(dict(x,label=label,ok=why=='accept',reason=why))
    arms={k:calc(v) for k,v in tables.items()}
    ci=bootstrap(tables,sorted(panel.BANK))
    passed=arms['abi']['absolute_pass']
    superiority={}
    for ctl in ('one_way','r117'):
        delta=ci['authored']['abi_minus_control'][ctl]
        superiority[ctl]=passed and delta['valid_retention'][0]>=-.05 and delta['invalid_acceptance'][1]<0
    return dict(arms=arms,ci95=ci,superiority=superiority,wp2='CLOSED',
                verdict='R118_BOUNDED_PASS' if passed else 'R118_NOT_QUALIFIED')


def derive():
    p=r.frozen()
    r.binding(r.GOLD,[r.OUT/'gold.json'])
    live,preds=r.receipts('teacher'),r.receipts('nli')
    d=analyze(p,live,r.check_gold(live,p),preds)
    d['evidence']={name:r.filehash(r.OUT/name) for name in ['packet.json','gold.json']+
                   [s+'-'+rep+'/'+f for s in ('teacher','nli') for rep in ('initial','replay')
                    for f in ('raw.jsonl','receipt.json')]+[s+'-'+rep+'.json' for s in ('teacher','nli') for rep in ('initial','replay')]}
    return d
