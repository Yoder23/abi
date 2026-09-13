# R59 v1a rejected

R59 v1a produced 1,278/1,400 functional rows but failed with 20 collapses. It
rejected 21 lexical boundary delimiters after the preceding word's BPE pieces
had already been accepted, leaving the returned prefix itself collapsed.

The raw audit also revealed that v1a omitted R55's inherited threshold-1
post-generation lexical truncation. The v1a output therefore was not a valid
R55-plus-one-invariant comparison. No final-test was run.

Raw evaluation SHA-256:
`3355f972d22010550db759d8ac63d983ea56640f0f2d2a4e5a975af432243bb2`.
