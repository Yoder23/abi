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
            r.need(len(cluster)==114,'cluster shape')
            complete+=all(x['ok']==(x['label']=='V') for x in cluster)
        families[f]=dict(complete=complete,authored=tally(rows),live=tally(subset(live,family=f)),
                         valid=tally([x for x in rows if x['kind'].startswith('valid_')]))
        for count in (1,2):
            for state in panel.STATES:
                cells[f'{f}/{count}/{state}']=tally(subset(live,family=f,count=count,state=state))
    for s in panel.STATES:
        transitions[s]={k:tally(subset(authored,state=s,kind=k)) for k in
                        ['swap_'+v for v in panel.VOICES]+['joint']+['to_'+t for t in panel.STATES if t!=s]}
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
    by_id={x['id']:x for x in authored}
    contexts=[x['id'].rsplit('/',1)[0] for x in authored if x['kind']=='valid_active']
    r.need(len(contexts)==288,'matched population')
    def matched(c,v):
        return by_id[c+'/valid_'+v]['ok'] and not by_id[c+'/swap_'+v]['ok']
    voices={v:dict(n=len(contexts),valid=kinds['valid_'+v]['accepted']['V'],swaps=kinds['swap_'+v]['accepted']['U'],
                   paired=sum(matched(c,v) for c in contexts)) for v in panel.VOICES}
    def isok(c,k):
        return by_id[c+'/'+k]['ok']
    rent=[c for c in contexts if by_id[c+'/valid_converse']['family'] in panel.RENTALS]
    prefix={k:dict(n=len(contexts),valid=sum(isok(c,k) for c in contexts),
                   wrong=sum(isok(c,neg) for c in contexts),paired=sum(isok(c,k) and not isok(c,neg) for c in contexts))
            for k,neg in [('valid_prefix','wrong_prefix'),('valid_label','wrong_label')]}
    rental=dict(n=len(rent),valid=sum(isok(c,'valid_converse') for c in rent),
               swaps=sum(isok(c,'swap_converse') for c in rent),paired=sum(matched(c,'converse') for c in rent))
    return dict(prefix=prefix,rental=rental,voices=voices,all_six=sum(all(matched(c,v) for v in panel.VOICES) for c in contexts),
                authored=a,live=s,by_kind=kinds,by_family=families,cells=cells,directed=transitions,
                windows=windows,complete_clusters=n,gates=gates,absolute_pass=all(gates.values()),
                nonverbatim=tally([x for x in live if dict(x['notes'])[x['target']]!=x['text']]),
                reasons=dict(Counter(x['reason'] for x in table)))


def analyze(p,live,gold,predictions):
    expected=list(r.records(p,live))
    r.need(len(expected)==len(predictions)==5856 and len({x['id'] for x in expected})==5856,'population')
    r.need(len(gold['rows'])==288,'label population')
    for c,x,g in zip(p['contexts'],live,gold['rows']):
        ids=x['output_ids']
        r.need(x['id']==g['id']==c['id'] and x['text_sha256']==g['text_sha256'] and g['label'] in ('V','U','O','Q')
               and g['rationale'] and c['prompt'] in x['prompt'] and 0<len(x['input_ids'])<=944 and 0<len(ids)<=80
               and not any(t in x['eos_ids'] for t in ids[:-1]) and (ids[-1] in x['eos_ids'] if x['stop']=='eos'
               else x['stop']=='length' and len(ids)==80 and g['label']=='Q'),'teacher-label binding')
    labels={g['id']:g['label'] for g in gold['rows']}
    tables={arm:[] for arm in method.ARMS}
    cache={}
    physical_tokens=0
    for idx,(x,z) in enumerate(zip(expected,predictions)):
        pairs=method.pairs(x['target'],x['notes'],x['roster'],x['text'])
        r.need(z['id']==x['id'] and z['target']==x['target'] and z['candidate_sha256']==r.sha(x['text'].encode())
               and z['input_sha256']==r.sha(r.dump(pairs).encode()) and len(z['ids'])==19 and
               all(0<len(seq)<=512 and all(type(t) is int and t>=0 for t in seq) for seq in z['ids']),'record custody')
        fresh=[]
        for j,(pair,ids,vec) in enumerate(zip(pairs,z['ids'],z['logits'])):
            if pair not in cache:
                cache[pair]=(ids,vec)
                fresh.append(j)
                physical_tokens+=len(ids)
            else:
                r.need(cache[pair]==(ids,vec),'nonidentical cached observation')
        r.need(z['new_slots']==fresh,'invalid cache accounting')
        d=method.decide(x['target'],x['notes'],x['roster'],x['text'],z['logits'])
        r.need(d==z['decisions'],'invented decision or slots')
        kind=x['kind']
        label=(labels[x['id'].rsplit('/',1)[0]] if kind=='live' else 'V' if kind.startswith('valid_') else
               'O' if kind in ('omission','off_target') else 'U')
        for arm in tables:
            why=d['arms'][arm]['reason']
            tables[arm].append(dict(x,label=label,ok=why=='accept',reason=why))
    arms={k:calc(v) for k,v in tables.items()}
    ci=bootstrap(tables,sorted(panel.BANK))
    passed=arms['abi']['absolute_pass']
    superiority={}
    for ctl in ('r119','strict_raw','status_only'):
        delta=ci['authored']['abi_minus_control'][ctl]
        superiority[ctl]=passed and delta['valid_retention'][0]>=-.05 and delta['invalid_acceptance'][1]<0
    batches=0
    for i in range(0,len(predictions),16):
        group=predictions[i:i+16]
        n=(sum(len(z['new_slots']) for z in group)+15)//16
        r.need([z['new_batches'] for z in group]==[n]+[0]*(len(group)-1),'batch accounting')
        batches+=n
    focus=paired_contrasts(p,predictions)
    gain=focus['strict_raw']['prefix']['ci95'][0]>0 and all(v['wrong']==0 for v in arms['abi']['prefix'].values())
    return dict(arms=arms,ci95=ci,superiority=superiority,paired_contrasts=focus,bounded_prefix_gain=gain,
                cost=dict(evaluated_nli_pairs=len(cache),evaluated_nli_tokens=physical_tokens,nli_forward_batches=batches),wp2='CLOSED',
                verdict='R120_BOUNDED_PASS' if passed else 'R120_NOT_QUALIFIED')


def paired_contrasts(p,predictions):
    import numpy as np
    scores={z['id']:z['decisions']['arms'] for z in predictions}
    def accepted(ctx,kind,arm):
        return scores[ctx['id']+'/'+kind][arm]['reason']=='accept'
    out={}
    for ctl in ('r119','strict_raw'):
        out[ctl]={}
        for label,families in [('prefix',sorted(panel.BANK)),('rental',sorted(panel.RENTALS))]:
            grid=[]
            for family in families:
                for i in range(6):
                    cells=[c for c in p['contexts'] if c['cluster']==family+'-'+str(i)]
                    def count(arm):
                        if label=='prefix':
                            return sum(accepted(c,k,arm) for c in cells for k in ('valid_prefix','valid_label'))/12
                        return sum(accepted(c,'valid_converse',arm) and not accepted(c,'swap_converse',arm) for c in cells)/6
                    grid.append(count('abi')-count(ctl))
            nf=len(families)
            draw=np.random.default_rng(117001).integers(6,size=(10000,nf,6))+np.arange(nf)[None,:,None]*6
            out[ctl][label]=dict(delta=float(np.mean(grid)),ci95=np.quantile(np.array(grid)[draw].mean(axis=(1,2)),[.025,.975]).tolist())
    return out


def derive():
    p=r.frozen()
    r.binding(r.GOLD,[r.OUT/'gold.json'])
    live,preds=r.receipts('teacher'),r.receipts('nli')
    d=analyze(p,live,r.check_gold(live,p),preds)
    for rep in ('initial','replay'):
        receipt=r.read(r.OUT/('nli-'+rep)/'receipt.json')
        r.need(all(receipt[k]==v for k,v in d['cost'].items()),'physical NLI costs')
    d['evidence']={name:r.filehash(r.OUT/name) for name in ['packet.json','gold.json']+
                   [s+'-'+rep+'/'+f for s in ('teacher','nli') for rep in ('initial','replay')
                    for f in ('raw.jsonl','receipt.json')]+[s+'-'+rep+'.json' for s in ('teacher','nli') for rep in ('initial','replay')]}
    return d
