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
