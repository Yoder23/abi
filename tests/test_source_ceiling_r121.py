import pytest
from experiments.source_ceiling_r121 import core as c


def test_balanced_panel():
    cases=c.panel()
    assert len(cases)==len({x['id'] for x in cases})==288 and cases==c.panel()
    assert len({x[k] for x in cases for k in ('target','foil')})==96
    sched=list(c.schedule(cases))
    assert len(sched)==72
    for i in range(0,72,2):
        a,b=sched[i:i+2]
        assert [x for _,x in a]==[x for _,x in b]
        assert [a[0][0],b[0][0]]==list(c.ARMS[::1 if (i//2)%2==0 else -1])
    for x in cases:
        d,f=(x['prompts'][a] for a in c.ARMS)
        assert f.replace(c.EXTRA,'')==d and x['state'] not in c.EXTRA
        assert len(x['notes'])==x['count'] and len(x['roles'])==2


def test_parser_is_not_an_answer_repair():
    assert c.parse('fields','FIELDS: inaccurate scratch\nFINAL: Fine.','eos')==('Fine.','ok')
    assert c.parse('fields','note\nFINAL: Fine.\nExtra.','eos')[0]=='Fine.\nExtra.'
    assert c.parse('direct','wrong\nFINAL: Fine.','eos')[0]=='wrong\nFINAL: Fine.'


@pytest.mark.parametrize('text,stop,reason',[('fine','eos','marker_count'),('FINAL: fine','eos','no_workspace'),
    ('fields\nFINAL:','eos','empty'),('fields\nFINAL: x\nFINAL: y','eos','marker_count'),('records\nFINAL: fine','length','length')])
def test_invalid_envelope(text,stop,reason):
    assert c.parse('fields',text,stop)==(None,reason)


def test_copy_does_not_erase_an_added_fact():
    x=c.panel()[0]
    note=next(n for n in x['notes'] if n.startswith(x['target']))
    for prefix in ('','Project ','Project ID '):
        assert c.bucket(x,'direct',dict(text=prefix+note,stop='eos'))[1:] == ('V','source_copy')
    group,label,_=c.bucket(x,'direct',dict(text=note+' It was authorized.',stop='eos'))
    assert label is None and 'authorized.' in group['final'] and x['roles'][0] not in group['final']


def test_supply_ceiling_rejects_hidden_cell_shortfall():
    cases=c.panel()
    rows=[(x,a,'U' if x['family']=='sale' and x['count']==2 and x['state']=='occurred' and x['cluster']<4 else 'V','manual')
          for x in cases for a in c.ARMS]
    d=c.metrics(rows)
    assert d['verdict']=='STOP_SOURCE_RECIPE' and d['arms']['direct']['totals']['V']==284
    assert not d['arms']['direct']['gates']['each_cell']
    assert d['paired']['all']['ci95']==[0,0] and d['wp2']=='CLOSED'


def test_window_cannot_be_rescued_by_global_yield():
    cases=c.panel()
    def good(x):
        return not (x['state'] in ('authorized','occurred') and x['cluster']<2)
    rows=[(x,a,'V' if a=='fields' or good(x) else 'U','manual') for x in cases for a in c.ARMS]
    d=c.metrics(rows)
    assert d['arms']['direct']['gates']==dict(usable=True,each_cell=True,windows=False)
    assert d['supported_fields_gain'] and d['wp2']=='CLOSED'
