"""Prospective stimuli, authored after the method freeze. No imports by method."""
import random

METHOD='7f5a6f81e8856615a52adcfb2d0edd926a103053'
STATES=('authorized','planned','occurred')
VOICES=('active','passive','converse')
# Each entry has four (infinitive-complement, past-tense clause) pairs:
# note, valid active, valid passive, valid converse. Both names appear once.
BANK={
'sale':(
 ('{a} to sell the easel to {b}','{a} sold the easel to {b}'),
 ('{a} to sell {b} the easel','{a} sold {b} the easel'),
 ('the easel to be sold to {b} by {a}','the easel was sold to {b} by {a}'),
 ('{b} to buy the easel from {a}','{b} bought the easel from {a}')),
'loan':(
 ('{a} to lend the atlas to {b}','{a} lent the atlas to {b}'),
 ('{a} to lend {b} the atlas','{a} lent {b} the atlas'),
 ('the atlas to be lent by {a} to {b}','the atlas was lent by {a} to {b}'),
 ('{b} to borrow the atlas from {a}','{b} borrowed the atlas from {a}')),
'rental':(
 ('{a} to rent the booth out to {b}','{a} rented the booth out to {b}'),
 ('{a} to rent out the booth to {b}','{a} rented out the booth to {b}'),
 ('the booth to be rented out to {b} by {a}','the booth was rented out to {b} by {a}'),
 ('{b} to rent the booth from {a}','{b} rented the booth from {a}')),
'order':(
 ('{a} to immediately precede {b} in the parade','{a} immediately preceded {b} in the parade'),
 ('{a} to appear immediately before {b} in the parade','{a} appeared immediately before {b} in the parade'),
 ('{b} to be immediately preceded by {a} in the parade','{b} was immediately preceded by {a} in the parade'),
 ('{b} to follow immediately after {a} in the parade','{b} followed immediately after {a} in the parade')),
'duel':(
 ('{a} to defeat {b} in the stage duel','{a} defeated {b} in the stage duel'),
 ('{a} to beat {b} in the stage duel','{a} beat {b} in the stage duel'),
 ('{b} to be defeated by {a} in the stage duel','{b} was defeated by {a} in the stage duel'),
 ('{b} to lose the stage duel to {a}','{b} lost the stage duel to {a}')),
'succession':(
 ('{a} to directly succeed {b} as chair','{a} directly succeeded {b} as chair'),
 ('{a} to take over as chair directly from {b}','{a} took over as chair directly from {b}'),
 ('{b} to be directly succeeded as chair by {a}','{b} was directly succeeded as chair by {a}'),
 ('{b} to be the immediate predecessor of {a} as chair','{b} was the immediate predecessor of {a} as chair')),
'repayment':(
 ('{a} to repay {b} the entire debt','{a} repaid {b} the entire debt'),
 ('{a} to repay the whole debt to {b}','{a} repaid the whole debt to {b}'),
 ('the entire debt to be repaid to {b} by {a}','the entire debt was repaid to {b} by {a}'),
 ('{b} to receive full repayment of the debt from {a}','{b} received full repayment of the debt from {a}')),
'tutoring':(
 ('{a} to tutor {b} during the lesson','{a} tutored {b} during the lesson'),
 ('{a} to give {b} tutoring during the lesson','{a} gave {b} tutoring during the lesson'),
 ('{b} to be tutored by {a} during the lesson','{b} was tutored by {a} during the lesson'),
 ('{b} to receive tutoring from {a} during the lesson','{b} received tutoring from {a} during the lesson')),
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
            roster=sorted(names[i*2:i*2+2])
            a,b=roster[::1 if i%2==0 else -1]
            def render(state,form,swap=False,project=target):
                infinitive,past=bank[form]
                names={'a':b if swap else a,'b':a if swap else b}
                if state=='occurred':
                    return (project+': ' if form==0 else 'For '+project+', ')+past.format(**names)+'.'
                if form==0:
                    return project+': The '+state+' arrangement is for '+infinitive.format(**names)+'.'
                return 'The arrangement '+state+' for '+project+' is for '+infinitive.format(**names)+'.'
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
