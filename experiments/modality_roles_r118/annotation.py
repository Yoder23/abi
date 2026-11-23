"""Investigator transcription; one explicit judgment per reviewed alpha-equivalent group.

No teacher or NLI self-rating. No implicit V/default label. Unsupported extra
claims take precedence over a missing identifier; ambiguous scope cannot pass.
V is semantic completeness, not independent stylistic approval.
"""
from . import run as r
from .blind import groups

KEY={
'V':('V','Project, action, direction, participants and modality preserved.'),
'I':('O','The required target project ID is absent.'),
'H':('U','Unchanged implies historical information not supplied in the note.'),
'P':('U','Pending or not started is not established by the milestone.'),
'A':('U','Approval or permission does not follow from occurrence or intention.'),
'X':('U','Adds a project-level, operational, ongoing or temporal status absent from the note.'),
'Q':('Q','Unqualified status field could describe this milestone or a broader status; not safely usable.'),
'M':('Q','The project identifier is presented as the status value, making the binding unclear.'),
}
INDEX={
'V': '''3 4 5 6 7 10 11 18 19 23 24 25 28 29 32 38 41 42 44 46 53 56
 58 61 63 64 67 68 73 75 76 77 80 83 84 85 87 88 89 93 95 97 98 99
 101 103 104 105 110 116 118 121 122 127 128 133 135 137 138 144 145 146
 147 149 150 151 153 156 157 159 161 162 163''',
'I': '1 8 16 21 27 36 43 48 57 66 70 86 90 96 100 123 125 148',
'H': '0 9 13 17 45 72 79 82 92 102 107 109 114 115 124 139 140 143',
'P': '2 12 14 15 35 37 39 51 59 60 62 65 78 81 113 117 119 129 131 132 136 141 158 160',
'A': '47 120 130 134 142 154',
'X': '26 31 34 40 55 69 71 106 111 155',
'Q': '20 22 30 33 49 52 54 74 91 94 108 112 126 152',
'M': '50',
}


def main():
    p=r.frozen()
    live=r.receipts('teacher')
    r.need(not any(r.OUT.glob('nli-*')), 'prediction-blind order')
    grouped=groups(p,live)
    labels={}
    for code,numbers in INDEX.items():
        for i in map(int,numbers.split()):
            r.need(i not in labels,'duplicate group')
            labels[i]=code
    r.need(set(labels)==set(range(len(grouped))) and len(grouped)==164,'explicit complete review')
    tagged={}
    for i,(key,ids) in enumerate(grouped):
        label,reason=KEY[labels[i]]
        for row in ids:
            tagged[row]=dict(label=label,rationale=reason,review_group=i,group_sha256=r.sha(r.dump(key).encode()))
    rows=[dict(id=x['id'],text_sha256=x['text_sha256'],**tagged[x['id']]) for x in live]
    r.write(r.OUT/'gold.json',dict(teacher_raw=r.filehash(r.OUT/'teacher-initial/raw.jsonl'),rows=rows,
            provenance='164 text groups covering all 288 full answers, reviewed by Codex before filter scoring. No human or independent gold.'))
    r.check_gold(live,p)


if __name__=='__main__':
    main()
