"""Evaluation-only prose bank, authored after ca6022e; never imported by method."""
from hashlib import sha256
import random

METHOD='ca6022edfb01e5df09ecec799c49c03d553957aa'
STATES=('authorized','planned','occurred')
# Ordered source and paraphrase per state. a/b are the action initiator and
# other participant, not the authority granting permission.
BANK={
'loan':(
 ('{a} is permitted to lend the projector to {b}.','Permission to lend the projector to {b} has been granted to {a}.'),
 ('{a} intends to lend the projector to {b}.','{b} is the intended borrower under {a}’s plan to lend the projector.'),
 ('{a} lent the projector to {b}.','{b} borrowed the projector from {a}.')),
'invitation':(
 ('{a} has clearance to invite {b} to the reception.','An invitation to the reception may be extended to {b} by {a}.'),
 ('{a} plans to invite {b} to the reception.','{a} intends to extend a reception invitation to {b}.'),
 ('{a} invited {b} to the reception.','{b} received a reception invitation from {a}.')),
'brief':(
 ('{a} has permission to explain the diagram to {b}.','{a} is allowed to walk {b} through the diagram.'),
 ('{a} is scheduled to explain the diagram to {b}.','The schedule calls for {a} to walk {b} through the diagram.'),
 ('{a} explained the diagram to {b}.','{b} was given an explanation of the diagram by {a}.')),
'voucher':(
 ('{a} is authorized to issue a voucher to {b}.','For {b}, a voucher may be issued by {a}.'),
 ('{a} intends to issue a voucher to {b}.','It is {a}’s intention to provide {b} with a voucher.'),
 ('{a} issued a voucher to {b}.','A voucher was issued to {b} by {a}.')),
'introduction':(
 ('{a} is allowed to introduce {b} at the assembly.','{a} has the go-ahead to present {b} at the assembly.'),
 ('{a} plans to introduce {b} at the assembly.','An introduction of {b} by {a} is planned for the assembly.'),
 ('{a} introduced {b} at the assembly.','At the assembly, {b} was introduced by {a}.')),
'escort':(
 ('{a} has permission to escort {b} to the studio.','{a} has been cleared to accompany {b} to the studio as escort.'),
 ('{a} intends to escort {b} to the studio.','{a} plans to act as escort for {b} on the way to the studio.'),
 ('{a} escorted {b} to the studio.','{b} was accompanied to the studio by {a} as escort.')),
'invoice':(
 ('{a} is permitted to bill {b} for the repairs.','{a} has authorization to invoice {b} for the repairs.'),
 ('{a} plans to bill {b} for the repairs.','{b} is to be invoiced for the repairs under {a}’s plan.'),
 ('{a} billed {b} for the repairs.','The repair bill went from {a} to {b}.')),
'nomination':(
 ('{a} is authorized to nominate {b} for the chair.','Permission has been given for {a} to put {b} forward for the chair.'),
 ('{a} intends to nominate {b} for the chair.','{a} plans to put {b} forward for the chair.'),
 ('{a} nominated {b} for the chair.','{b} was put forward for the chair by {a}.')),
}
NAMES='Ada Amos Ana Basil Cleo Dana Edgar Elsie Farah Felix Gwen Hugo Iris Ivan Jade Jules Kira Leon Lila Luca Mara Milo Nadia Neil Nora Omar Orla Owen Pia Remy Rosa Theo Vera Wade Yara Zara'.split()


def build():
    rng=random.Random(int(METHOD,16))
    contexts,empty,seen=[],[],set()
    for family,forms in BANK.items():
        names=rng.sample(NAMES,12)
        for i in range(6):
            cluster=f'{family}-{i}'
            pair=[f'P{rng.getrandbits(32):08X}' for _ in range(2)]
            assert len(set(pair)-seen)==2
            seen.update(pair)
            t,foil=pair
            roster=sorted(names[i*2:i*2+2])
            a,b=roster[::1 if i%2==0 else -1]
            def render(state,which,rev=False,project=t):
                return project+': '+forms[state][which].format(a=b if rev else a,b=a if rev else b)
            for count in (1,2):
                for j,state in enumerate(STATES):
                    alt=(j+1+i%2)%3
                    notes=[[t,render(j,0)]]
                    wrong_project=render(alt,0,True,foil)
                    if count==2:
                        notes.append([foil,wrong_project])
                        rng.shuffle(notes)
                    good=render(j,1)
                    ch={'valid':good,'actor_swap':render(j,1,True),'joint':render(alt,1,True),
                        'mixed':good[:-1]+', and both people signed a written agreement.',
                        'omission':t+' is the subject of this update.','off_target':wrong_project}
                    ch.update({'to_'+s:render(k,1) for k,s in enumerate(STATES) if k!=j})
                    # Prompt supplies neither a multiple-choice status list nor role labels.
                    prompt=('Restate only the milestone for '+t+' in one natural English sentence. '
                            'Keep the project ID, both names, who does what, and the exact status. '
                            'Use your own wording without adding facts or commentary.\nNotes:\n'+'\n'.join(s for _,s in notes))
                    contexts.append(dict(id=f'{cluster}-{count}-{state}',cluster=cluster,family=family,count=count,
                        state=state,target=t,roster=roster,notes=notes,prompt=prompt,challenges=ch))
            for k in range(2):
                empty.append(dict(id=cluster+'-empty-'+str(k),cluster=cluster,family=family,count=0,state=STATES[k],
                                  target=t,roster=roster,notes=[],text=render(k,1)))
    rng.shuffle(contexts)
    rng.shuffle(empty)
    assert len({sha256(c['prompt'].encode()).digest() for c in contexts})==288
    return dict(contexts=contexts,empty=empty)
