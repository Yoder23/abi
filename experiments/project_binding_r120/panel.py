"""Prospective stimuli, authored after the method freeze. No imports by method."""
import random

METHOD='3a3e46df0cc6feb723c498ef4c882378a1a3a3e7'
STATES=('authorized','planned','occurred')
VOICES=('active','passive','converse')
# Each entry has four (infinitive-complement, past-tense clause) pairs:
# note, valid active, valid passive, valid converse. Both names appear once.
RENTALS=('keyboard','marina','sublease','tiller')
BANK={
'keyboard':(
 ('{a} to rent the keyboard out to {b}','{a} rented the keyboard out to {b}'),
 ('{a} to rent out the keyboard to {b}','{a} rented out the keyboard to {b}'),
 ('the keyboard to be rented out by {a} to {b}','the keyboard was rented out by {a} to {b}'),
 ('{b} to rent the keyboard from {a}','{b} rented the keyboard from {a}')),
'marina':(
 ('{a} to lease the marina slip to {b}','{a} leased the marina slip to {b}'),
 ('{a} to lease out the marina slip to {b}','{a} leased out the marina slip to {b}'),
 ('the marina slip to be leased to {b} by {a}','the marina slip was leased to {b} by {a}'),
 ('{b} to lease the marina slip from {a}','{b} leased the marina slip from {a}')),
'sublease':(
 ('{a} to sublet the studio to {b}','{a} sublet the studio to {b}'),
 ('{a} to grant {b} a sublease of the studio','{a} granted {b} a sublease of the studio'),
 ('the studio to be sublet to {b} by {a}','the studio was sublet to {b} by {a}'),
 ('{b} to take the studio on sublease from {a}','{b} took the studio on sublease from {a}')),
'tiller':(
 ('{a} to hire out the tiller to {b}','{a} hired out the tiller to {b}'),
 ('{a} to rent the tiller out to {b}','{a} rented the tiller out to {b}'),
 ('the tiller to be hired out by {a} to {b}','the tiller was hired out by {a} to {b}'),
 ('{b} to hire the tiller from {a}','{b} hired the tiller from {a}')),
'sale':(
 ('{a} to sell the compass to {b}','{a} sold the compass to {b}'),
 ('{a} to sell {b} the compass','{a} sold {b} the compass'),
 ('the compass to be sold by {a} to {b}','the compass was sold by {a} to {b}'),
 ('{b} to buy the compass from {a}','{b} bought the compass from {a}')),
'loan':(
 ('{a} to lend the lantern to {b}','{a} lent the lantern to {b}'),
 ('{a} to lend {b} the lantern','{a} lent {b} the lantern'),
 ('the lantern to be lent to {b} by {a}','the lantern was lent to {b} by {a}'),
 ('{b} to borrow the lantern from {a}','{b} borrowed the lantern from {a}')),
'gift':(
 ('{a} to give {b} the medal as a gift','{a} gave {b} the medal as a gift'),
 ('{a} to gift the medal to {b}','{a} gifted the medal to {b}'),
 ('the medal to be given as a gift to {b} by {a}','the medal was given as a gift to {b} by {a}'),
 ('{b} to receive the medal as a gift from {a}','{b} received the medal as a gift from {a}')),
'match':(
 ('{a} to defeat {b} in the exhibition match','{a} defeated {b} in the exhibition match'),
 ('{a} to beat {b} in the exhibition match','{a} beat {b} in the exhibition match'),
 ('{b} to be beaten by {a} in the exhibition match','{b} was beaten by {a} in the exhibition match'),
 ('{b} to lose the exhibition match to {a}','{b} lost the exhibition match to {a}')),
}
NAMES='Ada Amos Ana Basil Cleo Dana Edgar Elsie Farah Felix Gwen Hugo Iris Ivan Jade Jules Kira Leon Lila Luca Mara Milo Nadia Neil Nora Omar Orla Owen Pia Remy Rosa Theo Vera Wade Yara Zara'.split()


def build():
    rng=random.Random(int(METHOD,16))
    contexts,empty,seen=[],[],set()
    for family,bank in BANK.items():
        names=rng.sample(NAMES,12)
        for i in range(6):
            cluster=f'{family}-{i}'
            pair=[f'P{rng.getrandbits(32):08X}' for _ in range(2)]
            assert len(set(pair)-seen)==2
            seen.update(pair)
            target,foil=pair
            near=target[:-1]+'0123456789ABCDEF'[(int(target[-1],16)+1)%16]
            assert near not in seen
            seen.add(near)
            roster=sorted(names[i*2:i*2+2])
            a,b=roster[::1 if i%2==0 else -1]
            def render(state,form,swap=False,project=target):
                infinitive,past=bank[form]
                names={'a':b if swap else a,'b':a if swap else b}
                if state=='occurred':
                    return (project+': ' if form==0 else 'For '+project+', ')+past.format(**names)+'.'
                if form==0:
                    return project+': The '+state+' arrangement is for '+infinitive.format(**names)+'.'
                return 'For '+project+', the '+state+' arrangement is for '+infinitive.format(**names)+'.'
            for count in (1,2):
                for j,state in enumerate(STATES):
                    alt=STATES[(j+1+i%2)%3]
                    notes=[[target,render(state,0)]]
                    wrong=render(alt,0,True,foil)
                    if count==2:
                        notes.append([foil,wrong])
                        rng.shuffle(notes)
                    ch={k+v:render(state,n,swap) for n,v in enumerate(VOICES,1)
                        for k,swap in [('valid_',False),('swap_',True)]}
                    # Balance the voice of the other controls within each family.
                    form=1+(i+j+count)%3
                    ch.update({'to_'+s:render(s,form) for s in STATES if s!=state})
                    ch.update(joint=render(alt,form,True),mixed=render(state,form)[:-1]+', and both people signed a contract.',
                              omission='This update concerns '+target+'.',off_target=wrong)
                    exact=render(state,0)
                    body=exact.split(': ',1)[1]
                    ch.update(valid_exact=exact,valid_prefix='Project '+exact,valid_label='Project ID: '+target+'. '+body,
                              wrong_prefix='Project '+foil+': '+body,wrong_label='Project ID: '+near+'. '+body,
                              conflicting='Project '+target+': Project '+foil+': '+body,missing=body)
                    prompt=('Give one natural English restatement of the supplied note for '+target+
                            '. Include the project ID and both names. Preserve the direction and whether the '
                            'arrangement is authorized, planned or reported to have happened. Do not add other facts.\nNotes:\n'
                            +'\n'.join(s for _,s in notes))
                    contexts.append(dict(id=f'{cluster}-{count}-{state}',cluster=cluster,family=family,count=count,
                        state=state,target=target,roster=roster,notes=notes,prompt=prompt,challenges=ch))
            for k in range(2):
                empty.append(dict(id=cluster+'-empty-'+str(k),cluster=cluster,family=family,count=0,state=STATES[k],
                                  target=target,roster=roster,notes=[],text=render(STATES[k],1+i%3)))
    rng.shuffle(contexts)
    rng.shuffle(empty)
    assert len({c['prompt'] for c in contexts})==288
    return dict(contexts=contexts,empty=empty)
