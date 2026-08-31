# R11 native neural-ISA teacher-to-student protocol

R11 is an additive construction test of the actual ABI moonshot mechanism. It
does not use the R10 symbolic canonical interpreter to own answers.

## Hypothesis

A teacher learning event changes only a compact model-independent transition
state. A frozen zero-parameter neural ISA executes that state. ABI extracts the
teacher state into one immutable package. The identical package bytes are then
supplied to capability-blind host codecs that were frozen before the held-out
capability existed. Frozen source and recipient native output heads must produce
the same canonical UTF-8 result with no post-reveal recipient optimization.

## Claim ceiling

A pass establishes only Level-1 output equivalence for a synthetic capability
learned by an ABI-native teacher substrate. It is the construction proof needed
before arbitrary open-weight teacher extraction. It does not establish English,
arbitrary domains, distribution equivalence, information minimality, or the
full moonshot.

## Native boundary

- The package contains only a normalized `3 x 8 x 8` learned neural transition
  state and immutable provenance.
- The package contains no prompt, answer, row ID, rule seed, offsets, solver,
  model/tokenizer identity, hidden width, or host matrix.
- The input adapter may parse the registered grammar into start/operator IDs;
  it cannot compute a result.
- A frozen recurrent neural module applies the learned transition state.
- A pre-capability frozen host codec injects that neural state into the frozen
  recipient's final residual stream.
- The recipient's native output embedding/head produces full-vocabulary logits.
- Removing the package, backend, or codec must remove the behavior. Restoring
  the same bytes must restore it without optimization.

## Gates

- Four held-out capabilities generated only after code and host codecs freeze.
- Teacher BEFORE at most 0.25; teacher AFTER exactly 1.0 on 512 new depth-4-to-7
  prompts per capability and improves by at least 0.70.
- Every recipient AFTER and RESTORED output is byte-identical to teacher AFTER
  on every row.
- BEFORE, WRONG, ZERO, RANDOM, and SHUFFLED are at most 0.30.
- REMOVED, BACKEND_REMOVED, and CODEC_REMOVED exactly equal BASE row by row.
- Same package SHA-256 across source and all recipients.
- Source and recipient base weights and frozen codecs are byte-identical before
  and after. Recipient optimizer steps are zero.
- Teacher process is absent during recipient execution.
- Raw evidence, packages, hashes, replay, and hostile controls fail closed.
