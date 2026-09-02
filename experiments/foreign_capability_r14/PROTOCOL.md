# R14 non-exhaustive foreign-capability extraction protocol

## Purpose

R14 tests whether the foreign frontend can recover a compositional capability
without enumerating its behavior. It changes no R11 package, executor, codec,
or recipient mechanism. R13-B remains the finite-table control.

The registered claim is:

> A conventional open-weight Qwen source, trained through ordinary LoRA, can
> carry statistically verified compositional signal. A fixed zero-parameter
> ABI frontend can select the correct canonical program from at most 256 mixed,
> non-atomic source observations and reproduce that capability exactly on
> 10,000 prompt-disjoint, unseen-depth cases and 1,000 order counterfactuals,
> without the source or recipient training.

This is capability recovery, not lossless cloning of every Qwen output.

## Public development boundary

Exactly two public formulations were run before preregistration:

- V1 used full-vocabulary next-token loss. Qwen reached 0.1364 on 10,000
  unseen-depth cases and 0.105 on 1,000 order counterfactuals. The frontend did
  recover the exact capability, but V1 was rejected as insufficient source
  generalization.
- V2 materially changed the source objective to eight-way capability loss plus
  a 0.1-weight native-token constraint and extended the training curriculum.
  Qwen reached 0.2302 and 0.255 respectively, while the same frontend recovered
  10,000/10,000 and 1,000/1,000. V2 fixes the held-out formulation.

The two immutable public receipts and their file hashes are bound in the
configuration. No third public formulation is authorized in this lineage.

## Capability and information boundary

- A capability contains three distinct, noncommuting affine permutations of
  eight states. Operator words are opaque.
- The registered evaluation depths 13 through 18 contain 4,642,668,576
  possible `(start, program)` cases per capability.
- The frontend receives exactly 256 preregistered mixed programs per
  capability: no depth-one/atomic prompts, answers, source-training rows,
  held-out rows, capability operations, oracle, or adaptive queries.
- The frontend's fixed candidate family has 27,552 latent programs. Selection
  maximizes summed log source probability. It has zero learned parameters and
  takes zero optimization steps.
- Training, query, evaluation, and counterfactual program identities are all
  disjoint. Evaluation depths are absent from source training and extraction.
- Order-counterfactual pairs preserve start state and operation multiset while
  changing order and oracle answer.

The source is synthetically taught so ground truth can be measured. R14 does
not test extraction of knowledge that existed in the pretrained base model.

## Frozen execution

- Source: `Qwen/Qwen2.5-0.5B`, revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- Source adapter: rank-16 LoRA; 4,000 fixed AdamW steps; learning rate 0.0002;
  depth-balanced batch 24; no held-out checkpoint selection.
- R11: sealed tag `abi-r11-bounded-neural-construction-2026-08-31`, commit
  `d89d8b607a705c47f7e2046b7dd4dbd40c396a5f`.
- Recipients: frozen Pythia, Qwen2, and T5 model revisions and frozen R11
  codecs; 512 unseen-depth cases per capability and all registered causality
  conditions; zero recipient optimization.
- Three independently derived capabilities are required to pass.

## Gates

For every capability:

- the 95% Wilson lower bound of Qwen's canonical exact accuracy must exceed
  chance (0.125) on the query, 10,000-case unseen-depth evaluation, and
  1,000-case order-counterfactual suite;
- the selected latent program must equal the post-reveal secret program;
- extractor log-likelihood margin must be positive;
- the package must score exactly 1.0 on unseen-depth and order-counterfactual
  oracle evaluation;
- every frozen recipient must score exactly 1.0 in AFTER and RESTORED;
- registered negative controls must be at most 0.30;
- removal conditions must equal BASE exactly; and
- source base weights, recipient states, and codecs must remain unchanged.

Stored verdict/status booleans are untrusted. Strict verification recomputes
commitments, splits, probabilities, source confidence bounds, latent selection,
package behavior, artifact hashes, recipient rows, and causality gates.

## Claim ceiling

A pass is `NON_EXHAUSTIVE_FOREIGN_CAPABILITY_RECOVERY`. It is not lossless
teacher-function transplantation, behavioral equivalence to Qwen, extraction
of pre-existing factual/domain/English knowledge, a LayerCake product result,
information minimality, or superiority to LoRA, distillation, or fine-tuning.
