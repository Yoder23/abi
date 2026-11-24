"""Investigator judgments recorded before R120 scoring. No default V label."""
from . import run as r
from .blind import groups

KEY={
 'V':('V','The project, event, status and direction are preserved with no new assertion.'),
 'F':('O','Adds a true statement about the other project although the request is for this project alone.'),
 'S':('U','Adds, negates or substitutes a status not established for the target; a true extra foil does not cure it.'),
 'D':('U','Changes participants or attributes a reversed event to the requested project without support.'),
 'I':('U','Attributes the event to a different/malformed project identifier.'),
 'Q':('Q','The generated answer reached the token limit mid-sentence; cannot qualify.'),
}
# Numeric identifiers refer to the frozen blind grouping procedure. Exhaustive
# review including incorrect, off-target and truncated responses. Explanatory
# metadata repeating project/name fields is not itself a change of meaning.
INDEX={
 'V':'''0 1 2 7 8 9 10 11 14 15 16 17 18 19 24 25 26 27 28 29 30 31 32 34 36 38 40 41 42 43 44 45 48 49 52 53 57 58 59
61 64 67 68 69 73 76 77 78 80 87 88 89 90 91 92 95 96 100 101 102 103 104 105 109 111 115 116 117 118 119''',
 'F':'''3 12 13 33 46 54 55 56 62 63 72 75 84 85 86 99 106 114''',
 'S':'''4 5 6 20 22 23 35 37 39 47 50 51 60 65 66 70 71 79 81 83 93 94 97 98 108 112''',
 'D':'''21 74 82 107 110 113''',
}


def main():
    p=r.frozen()
    live=r.receipts('teacher')
    r.need(not any(r.OUT.glob('nli-*')),'predictions must be absent')
    groupset=groups(p,live)
    labels={}
    for code,numbers in INDEX.items():
        for i in map(int,numbers.split()):
            r.need(i not in labels,'duplicate judgment')
            labels[i]=code
    r.need(len(groupset)==120 and set(labels)==set(range(len(groupset))),'one judgment per reviewed group')
    tagged={}
    for i,(key,ids) in enumerate(groupset):
        label,why=KEY[labels[i]]
        for rid in ids:
            tagged[rid]=dict(label=label,rationale=why,review_group=i,group_sha256=r.sha(r.dump(key).encode()))
    rows=[dict(id=x['id'],text_sha256=x['text_sha256'],**tagged[x['id']]) for x in live]
    r.write(r.OUT/'gold.json',dict(teacher_raw=r.filehash(r.OUT/'teacher-initial/raw.jsonl'),rows=rows,
            provenance='Prediction-blind Codex review of 120 groups covering 288 answers, not human or independent gold.'))
    r.check_gold(live,p)


if __name__=='__main__':
    main()
