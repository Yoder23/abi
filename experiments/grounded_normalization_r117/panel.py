"""Prospective benchmark. No imports from the filter, no model calls."""
import random

METHOD = 'dd56e033f95407c828588381b571ad918a884e29'
# source nonactual, paraphrase, source actual, paraphrase, unsupported addition
BANK = (
 ('archive', 'The records team intends to archive the folder for {p}.', 'Archiving the {p} folder is the records team’s intention.', 'The records team archived the folder for {p}.', 'The folder for {p} was archived by the records team.', 'the originals were destroyed'),
 ('handoff', 'The coordinator has permission to hand over the keys for {p}.', 'The coordinator is allowed to transfer the keys for {p}.', 'The coordinator handed over the keys for {p}.', 'For {p}, the coordinator transferred the keys.', 'the recipient signed a receipt'),
 ('briefing', 'A staff briefing about {p} is planned.', 'There are plans for a briefing of staff on {p}.', 'Staff attended a briefing about {p}.', 'A briefing on {p} was attended by staff.', 'the director chaired the session'),
 ('posting', 'The clerk is authorized to post the notice for {p}.', 'For {p}, the clerk has permission to put up the notice.', 'The clerk posted the notice for {p}.', 'The notice for {p} was put up by the clerk.', 'the notice drew three objections'),
 ('inspection', 'An inspection of the workspace for {p} is scheduled.', 'There is a scheduled workspace inspection for {p}.', 'The workspace for {p} underwent an inspection.', 'An inspection of the {p} workspace took place.', 'the workspace passed every check'),
 ('filing', 'The office plans to file the return for {p}.', 'Filing the {p} return is in the office’s plans.', 'The office filed the return for {p}.', 'The return for {p} was filed by the office.', 'a late fee was charged'),
 ('dispatch', 'The team has clearance to send the binder for {p}.', 'The team is permitted to dispatch the binder for {p}.', 'The team sent the binder for {p}.', 'The binder for {p} was dispatched by the team.', 'the binder arrived yesterday'),
 ('interview', 'The panel intends to interview the applicant for {p}.', 'The panel plans to meet the applicant for a {p} interview.', 'The panel interviewed the applicant for {p}.', 'The applicant for {p} was interviewed by the panel.', 'the applicant received an offer'),
)


def build():
    rng = random.Random(int(METHOD, 16))
    used, contexts, empty = set(), [], []
    for family, nf, nv, af, av, addition in BANK:
        for rep in range(6):
            ids = ['P%08X' % rng.getrandbits(32) for _ in range(2)]
            assert len(set(ids)-used) == 2
            used.update(ids)
            target, foil = ids
            cluster = f'{family}-{rep}'
            for count in (1, 2):
                for actual in (0, 1):
                    fact, valid, opposite = (af, av, nv) if actual else (nf, nv, av)
                    fmt = lambda s: s.format(p=target)
                    other = (nf if actual else af).format(p=foil)
                    notes = [(target, fmt(fact))]
                    if count == 2:
                        notes.append((foil, other))
                        rng.shuffle(notes)
                    prompt = ('Write one English sentence paraphrasing only the supplied milestone for '+target+
                              '. Preserve the project identifier and what is known about whether it happened. '
                              'Do not add information or commentary.\nNotes:\n'+'\n'.join(v for _, v in notes))
                    contexts.append(dict(id=f'{cluster}-{count}-{actual}', cluster=cluster, family=family,
                        count=count, actual=actual, target=target, notes=notes, prompt=prompt, challenges={
                        'valid':fmt(valid), 'opposite':fmt(opposite), 'mixed':fmt(valid)[:-1]+', and '+addition+'.',
                        'omission':'This update concerns '+target+'.', 'off_target':other}))
            for k, phrase in enumerate((nv, av)):
                empty.append(dict(id=f'{cluster}-empty-{k}', cluster=cluster, family=family,
                                  count=0, actual=k, target=target, notes=[], text=phrase.format(p=target)))
    rng.shuffle(contexts)
    rng.shuffle(empty)
    return {'contexts':contexts, 'empty':empty}
