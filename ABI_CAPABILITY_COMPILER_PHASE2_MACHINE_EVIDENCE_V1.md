# ABI capability-compiler Phase 2 machine evidence

Status date: 2026-08-04

## Verdict

The preregistered Phase 2 machine baseline campaign is complete and
content-verified. Phase 2 itself is **not complete**. It is
**BLOCKED_EXTERNAL_HUMAN_RATINGS**, and Phase 3 remains **LOCKED**.

The fail-closed report is
`results/abi_capability_compiler_phase2/machine_evidence_v1.json` (SHA-256
`589d882668e5ccd5c620ec8ad7b0528fff991e818a5252d928fb4f415becd9ec`).
No final prompts were accessed. No ABI candidate or LayerCake candidate was
trained.

## Headline autonomous quality

Each trainable system was retrained and evaluated on the same 1,400
development prompts for seeds 104729, 130363, and 155921.

| System | Functional passes by seed | Mean rate | Repetition collapses by seed | Source at inference |
| --- | --- | ---: | --- | --- |
| T0 | 1,237/1,400 | 88.36% | 64 | yes |
| L0 | 1,342; 1,302; 1,329 | 94.60% | 19; 46; 41 | yes; oracle capability route |
| L1 | 1,355; 1,243; 1,344 | 93.86% | 16; 37; 46 | yes; prompt router |
| D0 | 83; 111; 87 | 6.69% | 105; 242; 73 | no |
| D1 | 21; 0; 0 | 0.50% | 230; 267; 188 | no |
| D2 | 38; 25; 25 | 2.10% | 191; 320; 172 | no |

L1 routed 4,200/4,200 headline prompts correctly. The capability-stratified
paired bootstrap used 10,000 resamples per seed. Every D0/D1/D2 seed is 80–88
percentage points below T0, with its 95% interval wholly below zero. L0 is
significantly above T0 on all three seeds. L1 is above T0 on two seeds; seed
130363 is indistinguishable from T0 (`[-0.0129, 0.0214]`).

The students fail the frozen 90% functional point estimate, 85% Wilson lower
bound, and zero-collapse requirements. Their speed cannot be promoted as a
matched-quality result.

## Runtime from frozen seed-104729 checkpoints

Bytes/second and characters/second are the primary cross-model metrics.
Token throughput is diagnostic only and uses actual runtime token IDs.

| System | Warm median B/s (n=20) | Ratio vs T0 | Warm TTFT | Peak RSS | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0 | 88.26 | 1.00× | 40.2 ms | 1.02 GB | 7.79 GB |
| L0 | 74.03 | 0.84× | 75.5 ms | 1.75 GB | 7.84 GB |
| L1 | 73.03 | 0.83× | 93.7 ms | 2.45 GB | 7.89 GB |
| D0 | 867.22 | 9.83× | 2.62 ms | 0.91 GB | 0.124 GB |
| D1 | 524.01 | 5.94× | 2.65 ms | 0.91 GB | 0.124 GB |
| D2 | 603.94 | 6.84× | 2.59 ms | 0.91 GB | 0.124 GB |

Each cold result came from a fresh process and one request. There was no
separate load probe.

| System | Model load | First output from process start | Total from process start |
| --- | ---: | ---: | ---: |
| T0 | 12.761 s | 13.056 s | 13.480 s |
| L0 | 14.720 s | 14.973 s | 15.456 s |
| L1 | 14.703 s | 14.974 s | 15.472 s |
| D0 | 9.323 s | 9.474 s | 9.909 s |
| D1 | 9.349 s | 9.500 s | 9.973 s |
| D2 | 9.247 s | 9.400 s | 9.430 s |

The student load path includes the preregistered local source-snapshot rehash
and tokenizer verification. No p95 or p99 is claimed: 20 observations support
neither threshold under the frozen contract.

## Imported-information and artifact accounting

- Packed examples: 1,181.
- Packed input tokens: 799,572.
- Response/EOS label positions: 222,647.
- Clean uninterrupted top-64 cache: 14,249,408 stored logit values.
- One-time top-k source inference: 122.117 seconds; wall: 418.761 seconds.
- Source parameters: 3,821,079,552; source weight bytes: 7,642,181,880.
- Same-size student: 11,060,800 parameters; 22,121,600 deployed bf16 bytes;
  1.0184× the sealed LayerCake active-byte reference.
- Headline checkpoint files: L0 705,106,816 bytes each; L1 1,409,752,528
  bytes each; students 44,246,000 bytes each in float32 safetensors.

Large checkpoints and top-k shards remain local content-addressed research
artifacts. Ordinary Git contains their hashes, receipts, analysis, runtime
evidence, and blinded packet—not multi-gigabyte weight payloads.

## Human gate

`results/abi_capability_compiler_phase2/human_rating_packet_v1/manifest.json`
(SHA-256
`06c1e9e0f8f4f938cd020686fff7ac8470e8cff4a3d3fe294a0b0c003e2c69aa`)
binds three exactly counterbalanced 7,000-pair forms and a separate restricted
answer key. It requires 21,000 judgments total.

Assign exactly one form to each of three independent people. Raters must not
see the answer key or one another's forms until all three completed forms are
immutable. This work cannot be performed by the research agent without
fabricating independence.

## Adversarial verification

- Focused Phase 2 hostile suite: 17 passed.
- Full ABI suite: 471 passed, 1 failed.
- The sole failure is the intentionally exact external LayerCake HEAD control:
  the contract binds `04cf2927a16fba686cd640e18a78708e5658bbda`, while the
  separate clean LayerCake repository is at
  `c7fc6db78229a022e82dcd481122bc118fa629ea`. The bound commit exists and is an
  ancestor. The exact-HEAD verifier was not weakened.
- The evidence verifier rehashes all 1,181 top-k shards, all 15 headline
  checkpoints, every response file, every runtime/candidate binding, paired
  analysis inputs, and all blinded packet files.

## Preserved operational and negative evidence

- The original vocabulary-conformance preflight failure remains hash-bound.
- The resumed top-k cache remains preserved; a clean uninterrupted replay
  produced identical decoded arrays and authoritative timing.
- The four-exposure L0 and low-rank/four-exposure L1 branches lost to their
  one-exposure finalists and remain preserved.
- All failed student grid, full-depth, and headline seeds remain preserved.
- A D0 invocation that incorrectly supplied the top-k channel was rejected
  before training by the import firewall.
- A PowerShell-scoping mistake produced a T0-only paired-analysis v1; it remains
  preserved and was superseded by complete v2 rather than overwritten.

## Claim boundary

Phase 2 establishes credible matched baseline machine behavior and a measured
quality-speed Pareto split. It does not establish ABI transfer, a fluent
teacher-free core, ABI superiority, or LayerCake integration. Phase 3 cannot
open until the external human gate and final Phase 2 certificate pass.
