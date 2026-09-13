# R70 preregistration: explicit two-step choice reasoning acquisition

R69 reduced source verbosity but Phi still confused the starting and terminal
classes. R70 makes the operation unambiguous without revealing the answer: each
prompt supplies two invented class rules, identifies a starting class, directs
the source to apply rule 1 and rule 2 once each, and asks it to select one of
the three supplied class codes. The correct terminal code remains derived from
the rules.

The catalog has 2,000 unique search-only prompts, eight premise phrasings, four
instruction phrasings, compact unique nonce codes, a 16-token ceiling,
contains-conclusion evaluators, empty domain labels/claims, and no real facts.
Code and catalog are committed before source access. Pinned Phi, CUDA capture,
full accounting, segregation, and candidate inaccessibility remain unchanged.

At least 1,800/2,000 rows and every premise/instruction family must reach 85%
source pass rate. Post-hoc authoritative tokenization must find zero collapse.
Only then may passing search rows be composed for one LayerCake successor. R70
is acquisition evidence only, not a host or moonshot certificate.

