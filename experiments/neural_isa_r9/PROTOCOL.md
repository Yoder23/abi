# R9 neural-ISA recipient-realization protocol

R9 is additive. It does not modify R7 or reinterpret the sealed R8 Level 0
negative. R8 established exact canonical extraction for its synthetic family
and failed to make a frozen Pythia recipient use that representation.

## Definitions

The canonical observable is the UTF-8 encoding of the deterministic output.
For the registered one-step synthetic task, output equivalence and decision
equivalence are both exact equality of the emitted canonical digit. We report
distribution equivalence separately as total-variation distance between the
teacher and recipient distributions over the eight canonical outputs.

"Lossless" is reserved for 100% exact output/decision equivalence on every
registered row. Passing a diagnostic threshold below 100% is not lossless.

## Gate A: capability-specific expressivity diagnostic

Gate A intentionally permits a capability-specific Pythia backend. The frozen
Pythia parameters do not change. A neural adapter consumes Pythia hidden states
and the exact R8 canonical package, then adds a residual to Pythia's own
full-vocabulary logits. It receives no parsed program, row ID, or answer at
inference. Development training can use labels and may encode the capability in
backend weights. Consequently, Gate A can test recipient-path expressivity and
package-conditioned causality, but cannot prove a universal ABI decoder or that
the package alone contains all realized behavior.

The backend trains on 288 public rows at depths 1-3 and is evaluated on 1,024
distinct public rows at depths 4-7. Pythia and the R8 package are hash-bound.

Gate A passes only if:

- AFTER exact decision accuracy is at least 0.95;
- paired AFTER-minus-BASE gain is at least 0.70 and its 95% bootstrap lower
  bound is above zero;
- BEFORE, WRONG, ZERO, RANDOM, SHUFFLED, and REMOVED are each no more than
  BASE + 0.05;
- the frozen Pythia state is byte-identical before and after;
- the saved backend is byte-identical across evaluation; and
- all raw rows, hashes, and registered inputs are independently recomputable.

Exact 1.0 accuracy is reported as lossless output/decision equivalence; it is
not required to call the expressivity diagnostic positive.

### Preserved v1 result and v2 repair

V1 used only Pythia's final hidden state and a 274,184-parameter GRU backend.
It completed 5,000 steps but failed both fit and generalization: a post-run live
training-row check was 12.5%, while independently verified unseen-depth AFTER
accuracy was 10.449% versus 7.324% BASE. This is a backend
optimization/representation failure, not proof that the canonical package is
insufficient.

V2 is one bounded, measured repair. It exposes the pinned recipient embedding
state and final state, uses a width-256 backend, records raw training-fit rows,
and requires at least 98% training accuracy before unseen-depth performance can
open Gate B. It does not relax any v1 quality or causality threshold.

## Gate B: capability-blind backend

Gate B is forbidden unless Gate A passes. It must use many pre-freeze public
capabilities to teach one general backend how to execute the canonical IR. The
backend must then freeze before a newly generated capability and package exist.
No capability-specific recipient optimization is allowed after freeze. The same
package bytes must work across independently pinned heterogeneous recipients.

A bounded public pilot may use the existing 24 R8 meta capabilities and four
development capabilities. A moonshot claim requires a separately registered,
larger capability universe, post-freeze hidden capabilities, three recipient
families, exact byte-output testing, distributional testing, removal/restore
causality, composition, information accounting, and matched LoRA/distillation
baselines.

## Anti-runtime boundary

The ABI package may not contain an executable solver, prompt-to-answer table,
recipient weights, tokenizer identity, or host-specific matrices. The scorer
may interpret the package only to reconstruct teacher reference distributions.
The deployed path must be package -> frozen neural backend -> recipient neural
state/logits. A non-neural runtime that computes the answer and asks the host to
print it is a different system and fails this protocol.
