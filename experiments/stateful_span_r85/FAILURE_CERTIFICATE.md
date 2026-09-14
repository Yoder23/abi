# R85 failure certificate

R85 is closed and **not promoted**.

- Candidate checkpoint: `65187f3372dcffb8afffd1d1b2c9e7ade14ee1d93f60157452b0c223e6effe31`
- Candidate metadata: `7076960f2300d6891fd1f10bda8475e7640b17ff2e5d5291028f0aed20601ac9`
- Candidate binding claim: `e7c8693d939644427204950e85cdd49acdb2895828194674bec49af04ef4b3e0`
- Candidate binding file: `e3a56dcc7c2681c8a883445890e6ae1197aa1979e3928f8f3c2c31b9576e68a1`
- Screen result file: `b1a781f16f3f99f15b684167ca33e066990e6776a164d552109ae528f3472e83`
- Raw 1,400-row evidence: `7910e3177cd32445f25f71b6cccb23cb845ba0679e128d02f2f18e58e533030c`
- Recomputable evidence digest: `b4aec79712c51783ac8f0a7959f85843b2e9c98c0a21b4bd7d222aed5a74cf15`

The fresh pinned source passed 1,371/1,400. R85 passed 954/1,400 versus
10/1,400 for the unchanged R78 parent and 0/1,400 for a deterministic random
bridge. The paired candidate-minus-parent gain was 0.67429 with 95% bootstrap
interval `[0.64929, 0.69929]`. This is a real causal transfer effect, but it is
below the preregistered 1,330 quality gate. Source retention was 946/1,371
(0.69001), the lowest family passed 82/200, and 41 rows collapsed to an empty
span. Physical sparse execution, artifact immutability, source quality,
teacher absence, parent gain, and randomized-bridge controls passed.

Failure analysis found balanced performance across destination display
positions. Most semantic errors selected the subject code rather than a fixed
candidate position. All 41 collapse rows selected a whitespace-only tail, and
some semantically correct selections included adjacent punctuation because the
global length head did not localize the end token. The evidence-supported
successor is a fresh, prospectively screened bridge with (1) teacher-record-
preserving lexical renaming augmentation to prevent surface-code shortcuts and
(2) a joint start/end span objective. This is a changed measured mechanism,
not a nearby weight or size sweep.

R85 proves a bounded causal teacher-to-artifact-to-neural-package effect. It
does not prove unrestricted English, domains, minimality, LoRA/distillation
superiority, or the full ABI moonshot. The full ABI moonshot remains `OPEN`.
