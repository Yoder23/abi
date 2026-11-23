"""Blind-label join only after inference. Stratified paired cluster inference."""
from collections import Counter
import numpy as np
from . import run as r
from . import method


def rate(a, b):
    return a/b if b else None


def tally(rows):
    counts = Counter(i['label'] for i in rows)
    accepted = Counter(i['label'] for i in rows if i['ok'])
    n, v, k = len(rows), counts['V'], sum(accepted.values())
    bad = k-accepted['V']
    return dict(n=n, gold={s:counts[s] for s in 'VUOQ'}, accepted={s:accepted[s] for s in 'VUOQ'},
                valid_rejected=v-accepted['V'],withheld=n-k,valid_retention=rate(accepted['V'],v),
                invalid_acceptance=rate(bad,n-v),contamination=rate(bad,k),coverage=rate(k,n),
                source_omission_rate=rate(counts['O'],n))


def metric_rows(table, subset):
    return [x for x in table if (x['kind']=='live' if subset=='live' else x['kind']!='live')]


def bootstrap(tables, families):
    # One resampled set of cluster IDs shared by every arm, stratum and metric.
    rng = np.random.default_rng(117001)
    draw = rng.integers(6,size=(10000,8,6))+np.arange(8)[None,:,None]*6
    result = {}
    for subset in ('authored','live'):
        vals = {}
        for arm, table in tables.items():
            rows = metric_rows(table,subset)
            grid = []
            for f in families:
                for i in range(6):
                    xs=[x for x in rows if x['cluster']==f+'-'+str(i)]
                    good=[x for x in xs if x['label']=='V']
                    bad=[x for x in xs if x['label']!='V']
                    grid.append((sum(x['ok'] for x in good),len(good),sum(x['ok'] for x in bad),len(bad)))
            t = np.array(grid)[draw].sum(axis=(1,2)).astype(float)
            with np.errstate(invalid='ignore',divide='ignore'):
                vals[arm]=t[:,[0,2]]/t[:,[1,3]]

        def intervals(array):
            ret = {}
            for i,name in enumerate(('valid_retention','invalid_acceptance')):
                a = array[:,i]
                # Do not infer a CI for a rate with undefined resampled denominators.
                ret[name]=np.quantile(a,[.025,.975]).tolist() if np.isfinite(a).all() else None
            return ret
        result[subset] = {a:intervals(v) for a,v in vals.items()}
        result[subset]['abi_minus_control']={a:intervals(vals['abi']-v) for a,v in vals.items() if a!='abi'}
    return result


def analyze(p, live, gold, predictions):
    expected=list(r.records(p,live))
    r.need(len(expected)==len(predictions)==1248 and len({x['id'] for x in expected})==1248, 'complete unique population')
    labels={g['id']:g['label'] for g in gold['rows']}
    r.need(len(labels)==192,'live labels')
    tables={arm:[] for arm in method.ARMS}
    for x,z in zip(expected,predictions):
        pairs=method.pairs(x['target'],x['notes'],x['text'])
        r.need(z['id']==x['id'] and z['target']==x['target'] and z['candidate_sha256']==r.sha(x['text'].encode())
               and z['input_sha256']==r.sha(r.dump(pairs).encode()) and len(z['ids'])==3
               and all(0<len(s)<=512 and all(type(t) is int and t>=0 for t in s) for s in z['ids']), 'row binding')
        d=method.decide(x['target'],x['notes'],x['text'],z['logits'])
        r.need(d==z['decisions'], 'unrecomputable predictions')
        label=(labels[x['id'].rsplit('/',1)[0]] if x['kind']=='live' else
               {'valid':'V','opposite':'U','mixed':'U','omission':'O','off_target':'O','empty':'U'}[x['kind']])
        for arm in tables:
            tables[arm].append(dict(x,label=label,ok=d[arm]['reason']=='accept'))
    families=sorted({x['family'] for x in expected})
    r.need(len(families)==8,'family population')
    result={}
    for arm,tab in tables.items():
        auth=metric_rows(tab,'authored')
        source=metric_rows(tab,'live')
        by_kind={k:tally([x for x in tab if x['kind']==k]) for k in sorted({x['kind'] for x in tab})}
        by_family, cells={},{}
        for f in families:
            a=[x for x in auth if x['family']==f]
            s=[x for x in source if x['family']==f]
            complete=sum(all(x['ok']==(x['label']=='V') for x in a if x['cluster']==f+'-'+str(i) and x['kind']!='empty') for i in range(6))
            by_family[f]=dict(authored=tally(a),live=tally(s),complete=complete)
            for count in (1,2):
                for actual in (0,1):
                    def select(seq):
                        return [x for x in seq if (x['count'],x['actual'])==(count,actual)]
                    cells[f'{f}/{count}/{actual}']=dict(authored=tally(select(a)),live=tally(select(s)))
        a,s=tally(auth),tally(source)
        complete=sum(v['complete'] for v in by_family.values())
        tests=dict(clusters=complete>=44,every_family=all(v['complete']>=5 for v in by_family.values()),
            zero_authored_invalid=sum(a['accepted'][k] for k in 'UOQ')==0,no_notes=by_kind['empty']['withheld']==96,
            live_usable=s['accepted']['V']>=144,live_retention=(s['valid_retention'] or 0)>=.90,
            family_retention=all((v['live']['valid_retention'] or 0)>=.8 for v in by_family.values()),
            live_cell_supply=all(v['live']['gold']['V']>=3 for v in cells.values()),
            zero_live_invalid=sum(s['accepted'][k] for k in 'UOQ')==0)
        result[arm]=dict(authored=a,live=s,complete_clusters=complete,by_kind=by_kind,by_family=by_family,
                         cells=cells,gates=tests,absolute_pass=all(tests.values()))
    ci=bootstrap(tables,families)
    delta=ci['authored']['abi_minus_control']['one_way']
    qualified=result['abi']['absolute_pass']
    return dict(arms=result,ci95=ci,verdict='WP1_PASS' if qualified else 'WP1_FAIL',wp2='ELIGIBLE_FOR_AUTHORIZATION' if qualified else 'CLOSED',
                superiority_to_one_way=bool(qualified and delta['valid_retention'][0]>=-.05 and delta['invalid_acceptance'][1]<0))


def derive():
    p=r.frozen()
    r.binding(r.GOLD,[r.OUT/'gold.json'])
    live,preds=r.receipts('teacher'),r.receipts('nli')
    gold=r.check_gold(live,p)
    for ctx,x in zip(p['contexts'],live):
        ids=x['output_ids']
        r.need(ctx['prompt'] in x['prompt'] and 0<len(ids)<=80 and 0<len(x['input_ids'])<=944 and
               not any(t in x['eos_ids'] for t in ids[:-1]) and
               (ids[-1] in x['eos_ids'] if x['stop']=='eos' else x['stop']=='length' and len(ids)==80), 'teacher record')
    ans=analyze(p,live,gold,preds)
    ans['evidence']={name:r.filehash(r.OUT/name) for name in ['packet.json','gold.json']+
                     [stage+'-'+rep+'/'+file for stage in ('teacher','nli') for rep in ('initial','replay')
                      for file in ('raw.jsonl','receipt.json')]+[stage+'-'+rep+'.json' for stage in ('teacher','nli') for rep in ('initial','replay')]}
    return ans
