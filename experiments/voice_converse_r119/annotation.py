"""Investigator judgments recorded before R119 scoring. No default V label."""
from . import run as r
from .blind import groups

KEY={
 'V':('V','The project, event, status and direction are preserved with no new assertion.'),
 'F':('O','Adds a true statement about the other project although the request is for this project alone.'),
 'S':('U','Adds or substitutes authorization or a plan not supported for this target event.'),
 'D':('U','Changes participants or attributes a reversed event to the requested project without support.'),
 'I':('U','Attributes the event to a different/malformed project identifier.'),
 'Q':('Q','The generated answer reached the token limit mid-sentence; cannot qualify.'),
}
# Numeric identifiers refer to the frozen blind grouping procedure. Exhaustive
# review including incorrect, off-target and truncated responses. Explanatory
# metadata repeating project/name fields is not itself a change of meaning.
INDEX={
 'V':'''0 1 7 11 13 14 15 16 17 20 21 27 28 29 30 31 34 36 37 42 43 44
 46 47 48 49 50 51 54 59 63 64 65 66 67 70 71 72 73 82 86 88 89 90
 91 92 93 94 95 96 97 101 102 103 104 105 109 113 114 115 117 118 121 125 129 130''',
 'F':'''8 9 26 39 40 41 45 52 61 62 68 69 74 75 87 106 116 119 120 126 127 128 131 132''',
 'S':'''3 4 5 6 10 19 23 24 25 32 33 35 38 53 56 57 58 60 76 77 78 79 80 81
 83 84 85 99 100 107 108 110 111 112 122 123 124''',
 'D':'12 18 22 98',
 'I':'2',
 'Q':'55',
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
    r.need(len(groupset)==133 and set(labels)==set(range(len(groupset))),'one judgment per reviewed group')
    tagged={}
    for i,(key,ids) in enumerate(groupset):
        label,why=KEY[labels[i]]
        for rid in ids:
            tagged[rid]=dict(label=label,rationale=why,review_group=i,group_sha256=r.sha(r.dump(key).encode()))
    rows=[dict(id=x['id'],text_sha256=x['text_sha256'],**tagged[x['id']]) for x in live]
    r.write(r.OUT/'gold.json',dict(teacher_raw=r.filehash(r.OUT/'teacher-initial/raw.jsonl'),rows=rows,
            provenance='Prediction-blind Codex review of 133 groups covering 288 answers, not human or independent gold.'))
    r.check_gold(live,p)


if __name__=='__main__':
    main()
