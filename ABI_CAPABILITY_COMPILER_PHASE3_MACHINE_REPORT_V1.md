# ABI capability-compiler conditional Phase 3 machine report

Status date: 2026-08-04

## Decision

The initial conditional Phase 3 architecture branch is **COMPLETE_FAILED**.
It establishes a bounded causal labeled teacher-payload signal, but it does
not produce a fluent candidate and does not certify Phase 3.

- Phase 2 human ratings: `DEFERRED_NOT_PASSED`
- Phase 3 certificate: `NOT_ISSUED`
- Phase 4: `LOCKED`
- Final-test access: `false`
- Candidate promotion: `false`
- Remaining two seeds: not authorized by the preregistered stop rule

The controlling machine decision is
`results/abi_capability_compiler_phase3/conditional_decision_v3.json` (file
SHA-256 `6f11d41d092f45907efc7f9d909018311cbb3bcfa86227f628588f945ae08d1b`,
internal evidence SHA-256
`103b255a73bc5736ae0ff9aadfc8668d1474231dea484ee0d2192c7a8ce1f21c`).
An independent deterministic recomputation is byte-identical.

## Experimental identity

- Frozen teacher: `microsoft/Phi-3-mini-4k-instruct` revision
  `f39ac1d28e925b323eae81227eaba4464caced4e`
- Frozen LayerCake parent checkpoint SHA-256:
  `9e0e6b9add32b4c460f7b570a32584f380e59bf6d631e313ff813069d24e09e1`
- Certified input IR: 7,000 records, 500 for each of 14 English capabilities
- Selected IR accounting: 215,647 authoritative teacher output tokens;
  576,925 teacher input tokens; 1,381,925 raw prompt bytes; 671,297 raw output
  bytes; 463,312 unique normalized output UTF-8 bytes
- Candidate bridge: 606,730 trainable parameters, 0.9841% of the host
- Training: 7,000 successful batch-4 steps, maximum 512 tokens, seed 104729
- Hardware: NVIDIA GeForce RTX 3080 Laptop GPU
- Teacher at training or inference: absent; cached normalized records only
- Source parameters copied: zero

## Repair and evidence lineage

1. The initial 256-token preflight excluded too many registered records. The
   preregistered repair increased only the context limit to 512 and retained
   7,000/7,000 records.
2. A3 initially violated frozen-cake identity because AdamW weight decay can
   alter zero-gradient parameters. The V3 emitter amendment corrected the
   no-payload control without changing candidate capacity or thresholds.
3. V3 results were invalidated after authoritative token accounting proved
   that mixed-precision skipped steps advanced the sampler differently across
   systems. Those outputs remain historical, non-causal evidence.
4. V4 retries the identical batch after every skipped optimizer step and
   advances only after success. A0-A4 all bind the same 28,000-record sequence
   SHA-256 `b4ac23e86611c88515db8c17e72e78290ec41329059cb0546731c703ffeff28e`.
5. The first V4 analysis file contained correct numeric results but stale V3
   counts in one narrative field. It remains preserved as historical numeric
   evidence. V5 derives that prose from current evidence and emits the new
   immutable V3 decision; no experiment or statistic changed.

## Autonomous development results

Each system was evaluated on the same 1,400 distinct development prompts, 100
per capability.

| System | Definition | Passes | Pass rate | Wilson 95% CI | Collapses |
| --- | --- | ---: | ---: | ---: | ---: |
| A0 | labeled teacher payload, six routes | 379/1,400 | 27.07% | 24.81%-29.46% | 150 |
| A1 | label-free prompt-hash routing | 113/1,400 | 8.07% | 6.76%-9.62% | 210 |
| A2 | within-capability deranged targets | 271/1,400 | 19.36% | 17.37%-21.51% | 216 |
| A3 | no teacher response loss | 4/1,400 | 0.29% | 0.11%-0.73% | 634 |
| A4 | monolithic route 0 | 252/1,400 | 18.00% | 16.08%-20.10% | 141 |
| T0 | frozen teacher reference | 1,237/1,400 | 88.36% | Phase 2 evidence | 64 |

A0 scored 0/100 on coherence and fluent realization. It failed the locked
per-capability floor, the stricter prompt-grounding/instruction/abstention
floor, teacher-relative noninferiority, and zero-collapse requirements.

## Paired causal comparisons

The following are prompt-paired pass-rate differences from 10,000 stratified
bootstrap replicates, seed 1729.

| Comparison | Difference | 95% CI | Gate |
| --- | ---: | ---: | --- |
| A0 - A1 | +19.00 points | +16.79 to +21.21 | PASS |
| A0 - A2 | +7.71 points | +5.43 to +10.00 | PASS |
| A0 - A3 | +26.79 points | +24.71 to +28.86 | PASS |
| A0 - A4 | +9.07 points | +7.00 to +11.14 | PASS |
| A0 - T0 | -61.29 points | -63.64 to -58.86 | FAIL |

The positive control comparisons support three bounded causal conclusions:
the actual teacher responses matter, the declared destination labels/routing
matter, and segmentation helps relative to one monolithic route. They do not
show that the resulting model is useful or that ABI beats LoRA or distillation.

## Resource accounting

| System | Teacher response tokens seen | Train seconds | Generation seconds | Peak RSS | Peak CUDA |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 723,739 | 181.79 | 192.50 | 1.50 GiB | 1.74 GiB |
| A1 | 723,739 | 185.49 | 268.17 | 1.49 GiB | 1.74 GiB |
| A2 | 726,062 | 182.14 | 217.27 | 1.49 GiB | 1.65 GiB |
| A3 | 0 | 132.41 | 1,003.27 | 1.49 GiB | 0.78 GiB |
| A4 | 723,739 | 172.82 | 209.51 | 1.49 GiB | 1.73 GiB |

A2 uses the identical record sequence but has a different authoritative
response-token total because its targets are deliberately deranged within each
capability. A3 has zero teacher response tokens by construction.

## Measured bottleneck and stop decision

The 606,730-parameter output-side bridge can memorize training responses and
route held-out prompts, but lacks prompt-conditioned sequence-transformation
capacity. More data, more steps, additional seeds, or nearby cake sweeps are
not justified by this result and are prohibited for this branch.

A future Phase 3 proposal must be separately preregistered and add materially
different prompt-conditioned sequence realization while preserving the frozen
host, source absence, exact imported-information accounting, identical sample
exposure, matched controls, and absolute autonomous-quality gates. Final data
must remain unopened until a development candidate passes every gate and prior
phase governance permits promotion.

## Verification

- Focused Phase 0/Phase 3 tests: 18 passed.
- Full in-scope ABI suite: 484 passed, 1 external-control test deselected.
- The last unmodified full-suite audit before the added mutation guard reported
  483 passed and 1 fail-closed external LayerCake
  identity mismatch. The contract-bound LayerCake commit remains an ancestor
  of the separate repository's current commit; the verifier was not weakened.
- JSON parsing and `git diff --check`: pass.
