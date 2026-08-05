# ABI capability-compiler Phase 3 sequence-successor report

Status date: 2026-08-05

## Decision

The preregistered V6 B0-B4 initial-seed branch is **COMPLETE_FAILED**. The
continuous prompt-conditioned sequence bridge materially improves on the
failed output-only architecture, but the registered labeled candidate does not
pass the locked absolute, causal-control, teacher-relative, or repetition
gates. Phase 3 remains uncertified, Phase 4 remains locked, and final material
was not accessed.

The controlling decision is
`results/abi_capability_compiler_phase3_sequence/conditional_decision_v1.json`
(file SHA-256
`8b91a36b5f8707c7480b5180dca964c84ac9888e24aff6c9d84c4cc7501ae76d`,
internal evidence SHA-256
`c1464ea8927894012020791738fd40a01fa235d7d76ae9ca93b7a5d34e16263a`).
A second deterministic analysis is byte-identical.

## Bound experiment

- Frozen three-block LayerCake host; frozen-state identity is exact before and
  after all five training runs.
- 1,556,998 trainable bridge parameters out of 62,613,008 total (2.4867%).
- Continuous prompt summary plus rank-128 nonlinear residual before each of
  the three frozen blocks.
- No source parameters copied, no source block retained, and no teacher at
  training or inference; only the certified cached Phase 1 IR was used.
- B0-B4 consumed the same 28,000-record successful sequence SHA-256
  `b4ac23e86611c88515db8c17e72e78290ec41329059cb0546731c703ffeff28e`.
- One locked seed, 1,400 distinct development prompts, 100 per capability,
  Wilson intervals, and 10,000 prompt-paired stratified bootstrap replicates.

## Autonomous development results

| System | Definition | Passes | Rate | Collapses |
| --- | --- | ---: | ---: | ---: |
| B0 | labeled semantic routes | 1,148/1,400 | 82.00% | 43 |
| B1 | label-free prompt-hash routes | 1,224/1,400 | 87.43% | 60 |
| B2 | within-capability deranged targets | 277/1,400 | 19.79% | 100 |
| B3 | no teacher-response loss | 4/1,400 | 0.29% | 634 |
| B4 | monolithic output route | 1,165/1,400 | 83.21% | 57 |
| T0 | frozen teacher reference | 1,237/1,400 | 88.36% | 64 |

B0's lowest capability scores are abstention 46/100, tone control 69/100,
conversation 72/100, fluent realization 75/100, and prompt grounding 77/100.
It therefore fails the per-capability floors and the stricter abstention,
prompt-grounding, and instruction-following family. Its 43 collapses violate
the zero-collapse gate.

## Paired causal comparisons

| Comparison | Difference | Stratified-bootstrap 95% CI | Gate |
| --- | ---: | ---: | --- |
| B0 - B1 | -5.43 points | -7.50 to -3.43 | FAIL |
| B0 - B2 | +62.21 points | +60.14 to +64.29 | PASS |
| B0 - B3 | +81.71 points | +79.86 to +83.57 | PASS |
| B0 - B4 | -1.21 points | -3.14 to +0.71 | FAIL |
| B0 - T0 | -6.36 points | -8.64 to -4.07 | FAIL |

Actual response supervision and correctly paired targets are necessary in this
architecture: derangement and removal of response loss fail catastrophically.
However, semantic segmentation is not shown to cause the sequence-bridge gain.
The label-free control significantly beats B0, and B0 does not beat the
monolithic control. A claim that ABI labels or specialized output routes caused
the improved quality is therefore prohibited.

## Resource accounting

| System | Response tokens seen | Train seconds | Generation seconds |
| --- | ---: | ---: | ---: |
| B0 | 723,739 | 246.37 | 159.43 |
| B1 | 723,739 | 252.78 | 159.85 |
| B2 | 726,062 | 247.28 | 177.42 |
| B3 | 0 | 198.98 | 1,125.30 |
| B4 | 723,739 | 236.79 | 154.96 |

B2's token total differs only because its response targets are deliberately
deranged. B3 generated 268,800 evaluation tokens because the unsupervised
generator frequently reached the locked cap; this is measured negative
evidence, not a timeout or omitted run.

## Measured bottleneck and stop rule

Sequence-level prompt conditioning is a large improvement over A0's 27.07%
score, so the prior sequence-realization diagnosis was correct. The remaining
failure is not evidence for more data, steps, rank, or nearby route sweeps.
Within V6, semantic route specialization hurts aggregate development quality,
does not reduce collapse, and is unnecessary for the observed gain.

The exact V6 branch is closed. Seeds 130363 and 155921 are not authorized. A
future successor must be separately preregistered and test a shared/direct
sequence transformation without post-hoc selection, while retaining target
derangement, no-response, frozen-parent, and teacher comparisons. It must keep
the same final-data firewall and cannot inherit quality from B1 or T0.

## Verification

- Seven focused sequence-bridge/analysis/verifier tests pass.
- All five training manifests bind the same successful sample sequence.
- All five receipts bind 1,400 distinct prompts and their output-file hashes.
- The analysis verifies manifests, receipts, output hashes, aggregate counts,
  frozen-state identity, source absence, and registered-tensor confinement.
- Independent recomputation is byte-identical.
- The independent verifier rehashed all five persisted checkpoints and rejects
  six rehashed mutations: teacher presence, copied source parameters, frozen
  state drift, an unregistered tensor, false promotion, and unauthorized seeds.
- Full in-scope ABI suite: 491 passed, with the known separate-repository
  identity check explicitly deselected rather than weakened.
- Final-test access: `false`.
