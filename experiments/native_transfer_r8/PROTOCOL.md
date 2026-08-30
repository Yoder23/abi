# R8 preregistered protocol

## Hypothesis

Training one neural source on a previously absent opaque capability produces a
change that a frozen extractor can serialize into one byte-identical canonical
artifact. Generic bridges frozen before capability reveal then cause previously
absent behavior in frozen heterogeneous recipients through recipient logits.

## Primary family

Each capability assigns one hidden modular offset in `[0, 7]` to each of three
opaque operators. Programs apply one or more operators to a starting state in
`Z/8Z`. Operation names contain no semantic clues. Source training uses depths
1-3. Evaluation uses unseen examples at depths 4-7, counterfactual programs,
and adversarially similar programs.

The package contains a fixed `3 x 8 x 8` float32 canonical transition tensor
extracted from source logits on frozen atomic probes. It contains no executable
solver, test row, answer, seed, model identifier, tokenizer identifier, hidden
width, layer count, or host-specific matrix.

## Frozen order

1. Commit this protocol and the held-out secret commitment.
2. Generate only meta-train and development capabilities.
3. Train generic bridges; freeze and hash all models, extractor code, and
   bridges.
4. Verify the held-out reveal file is absent from the freeze environment.
5. Reveal the committed secret and derive held-out capabilities.
6. Train source `T_before -> T_after`, extract `P_before` and `P_after`, seal
   packages, then generate private evaluation labels.
7. Evaluate recipients in a label-free worker and score in a separate process.

## Pre-reveal amendment

The preserved v1 draft requested 768 unique source rows at depths 1--3, but the
finite universe contains only `8 * (3 + 9 + 27) = 312` start/program pairs.
Before any held-out reveal or freeze, v2 corrected the request to 288 unique
rows and made the generator sample the finite universe without replacement.
No threshold, evaluation depth, capability count, seed commitment, or model was
changed.

The v1/v2 seed commitment was later disqualified before any model training
because its preimage was found in a unit test. V2 also contained a duplicate
JSON key, and the registered 48-token limit truncated some T5 prompts. V3 is
the sole active preregistration: it uses a fresh unrevealed commitment, unique
JSON keys, and a 128-token runtime limit. The disqualified drafts remain as
negative protocol history.

Public revision 002 then falsified the source soft-prefix acquisition method:
most public capabilities remained near eight-way chance at unseen composition
depths. That result is preserved. V4 changes only source capability acquisition
to rank-8 LoRA on all 24 frozen-base GPT-2 Conv1D modules. Each capability gets
an isolated adapter state; the source base remains byte-identical. The frozen
extractor, canonical package, recipient path, held-out commitment, and every
scientific threshold remain unchanged.

A public rank-8 LoRA pilot then fit shallow examples while degrading unseen
depth-4--7 accuracy from 2.7% to 4.3%, so the full LoRA sweep was not launched.
V5 uses a disclosed differentiable stochastic-transition adapter as the source
model's family-specific neural inductive bias. It is trained only from examples,
receives no hidden rule table, composes learned stochastic matrices, and adds
its state through the source model's canonical-token logits. The public pilot
reached 100% unseen-depth accuracy. This makes the claim narrower—not broader:
R8 still has to show that unchanged extracted state creates native behavior in
three frozen, independently trained general-purpose recipients.

## Primary gates

- Eight held-out capability seeds and at least 512 evaluation rows each.
- `T_before <= 0.25`, `T_after >= 0.85`, and source gain at least 0.50 for
  every held-out capability.
- At least three pinned recipient families.
- `P_after - BASE >= 0.30` for every recipient, with paired bootstrap 95%
  confidence intervals excluding zero and no negative recipient effect.
- Median transfer efficiency relative to target-specific LoRA at least 0.50.
- BEFORE, WRONG, RANDOM, ZERO, SHUFFLED, and REMOVED each within +0.05 of BASE.
- Median causal fraction at least 0.75.
- One exact package SHA per capability across every recipient.
- Zero held-out recipient/bridge optimizer steps and no package mutation.
- No material preregistered unrelated-task degradation.

Composition behavior is meta-trained only on public capability pairs before
freeze. Held-out composition concatenates two independently extracted package
prefixes; it never creates or fits a joint package. Nested information budgets
retain a preregistered prefix of canonical transition rows and replace omitted
rows with the corresponding `P_before` rows.

Physical isolation is a separate hard gate. In-process socket blocking and a
staged subprocess are useful controls but do not satisfy it. A host without a
working container/sandbox runtime is capped below Level 2.

The stricter +0.40 gain and 0.70 transfer-efficiency thresholds are always
reported but do not replace the primary thresholds.

## Verdict ceiling

The pilot cannot exceed the evidence it actually produces. Missing open-weight
host diversity, physical isolation, a required baseline, intervention evidence,
or statistical depth forces `NOT YET ESTABLISHED`. Stored pass strings are not
scientific inputs.
