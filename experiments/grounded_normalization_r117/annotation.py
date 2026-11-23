"""Investigator's row-by-row transcription of teacher output, BEFORE NLI.

Each block of four judgments is one project pair. Within the block the order
is one project/prospective, one/actual, two/prospective, two/actual. No default
V label; one explicit judgment for each entire response. Not human rating.
"""
from . import run as r

REVIEW = {
    'archive':'VVVV VVVV VVVV VVVV VVVD VVVV',
    'briefing':'VVVV VJVV VJVV VVVV VVVV VVTV',
    'handoff':'HVNV MVVV HVVV MDVV MVVV MVVV',
    'posting':'MRMR VVVV MVVV MIVV MVVI MIMI',
    'filing':'VVVV VAVV VVVV AVVD VVVV VVVV',
    'inspection':'VVVD VVVV VDVD VVVV VVVD VDVD',
    'dispatch':'VIVV VIVV VSVV AIVV AIVD VIVV',
    'interview':'VVVV VVVM VVVV VVVV VVVV VVVV',
}
KEY = {
    'V':('V','Target, milestone, participants and prospective/actual status preserved; no extra assertion.'),
    'D':('U','The project itself has become the affected object instead of its keys, binder, return, workspace or folder.'),
    'A':('U','The project is assigned an action or permission belonging to the named office or team.'),
    'I':('O','Reports the event but leaves out the supplied actor (team or clerk).'),
    'M':('U','Asserts an occurrence from permission alone, or asserts a plan in place of the reported occurrence.'),
    'N':('U','Permission does not establish that the handover has not yet occurred.'),
    'R':('U','Being notified or noticed is not the asserted posting of a notice by the clerk.'),
    'T':('U','It is the briefing that is planned; the notes do not say the project itself is planned.'),
    'J':('Q','Briefing regarding this milestone ambiguously changes what the briefing was about.'),
    'H':('Q','The coordinator may be the authorizing party or the person handing over the keys; permission holder unclear.'),
    'S':('Q','Submitted can imply a formal submission, which is not specified by sent.'),
}


def main():
    p=r.frozen()
    live=r.receipts('teacher')
    r.need(not any(r.OUT.glob('nli-*')), 'annotation must precede predictions')
    ids={}
    for family, blocks in REVIEW.items():
        need_blocks=blocks.split()
        r.need(len(need_blocks)==6 and all(len(b)==4 for b in need_blocks),'one label per cell')
        for i,b in enumerate(need_blocks):
            for (count,actual),code in zip(((1,0),(1,1),(2,0),(2,1)),b):
                ids[f'{family}-{i}-{count}-{actual}']=KEY[code]
    r.need(set(ids)=={x['id'] for x in live} and len(ids)==192,'review population')
    rows=[dict(id=x['id'],text_sha256=x['text_sha256'],label=ids[x['id']][0],rationale=ids[x['id']][1]) for x in live]
    r.write(r.OUT/'gold.json',dict(teacher_raw=r.filehash(r.OUT/'teacher-initial/raw.jsonl'),rows=rows,
            provenance='Codex investigator read all 192 texts and target facts before filter predictions; not independent or human.'))
    r.check_gold(live,p)


if __name__ == '__main__':
    main()
