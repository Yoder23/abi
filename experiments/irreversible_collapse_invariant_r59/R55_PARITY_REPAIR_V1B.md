# R59 v1b R55-parity repair

R59 v1a still failed zero-collapse because it rejected the delimiter that made
a prior multi-token word's repetition visible, but returned that word. The
subsequent raw-row comparison exposed a more fundamental baseline mismatch:
R55's inherited R49 generation function always applies
`_truncate_novel_lexical_repetition(..., threshold=1)` after generation. R55's
candidate metadata says the lexical threshold is zero, so this runtime behavior
was not represented in the candidate manifest even though it is present in all
R55 screen evidence.

R59 v1b does not silently rewrite that history. It explicitly binds and counts
the inherited R55 threshold-1 lexical postprocessor, restoring exact screen
parity. The sole new R59 intervention is now the measured condition absent from
that postprocessor: terminate before accepting a sixth identical token in a
row. The lexical transition experiment from v1/v1a is removed from runtime.

All v1 and v1a rows remain negative evidence. The repaired screen runs in a new
directory. Validation must reproduce all 1,400 historical R55 outputs exactly,
with zero new run-boundary rejections. Final-test may differ only on historical
R55 rows whose maximum identical-token run already violated the locked gate.
No threshold was selected or changed after observing an answer.
