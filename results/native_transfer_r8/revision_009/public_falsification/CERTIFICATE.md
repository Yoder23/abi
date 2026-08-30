# ABI R8 native neural transfer: public falsification certificate

Exact answer: **NO**

Verdict: **LEVEL 0 — FAIL**

R8 v10 registered architecture; public prerequisite only; held-out reveal was never opened

R7 is unchanged. This certificate neither relabels nor broadens R7.

## Recomputed recipient evidence

| Public capability | BASE | BEFORE | AFTER | ZERO | RANDOM | WRONG | AFTER−BASE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r8-development-000-1f39465a340dcd5a7274` | 0.1172 | 0.1133 | 0.1172 | 0.1172 | 0.1094 | 0.1523 | +0.0000 |
| `r8-development-001-c3d8a4a59c4eb8b0d35c` | 0.1016 | 0.1133 | 0.1211 | 0.1016 | 0.1406 | 0.1016 | +0.0195 |
| `r8-development-002-e4271d3719f2ca945c70` | 0.1172 | 0.1250 | 0.1172 | 0.1172 | 0.1211 | 0.1289 | +0.0000 |
| `r8-development-003-f6034c87b38ab269d561` | 0.1406 | 0.1367 | 0.1289 | 0.1406 | 0.1523 | 0.1406 | -0.0117 |

Paired AFTER−BASE across 1,024 raw rows: +0.0020; 95% bootstrap CI [-0.0195, +0.0244].

Canonical atomic extraction accuracy was 100.0% on meta-train and 100.0% on development.

## Gate ledger

| Gate | Result |
| --- | ---: |
| `canonical_extraction_exact` | PASS |
| `heldout_remained_unrevealed` | PASS |
| `matched_baselines` | FAIL / NOT RUN |
| `neural_causality` | FAIL / NOT RUN |
| `physical_isolation` | FAIL / NOT RUN |
| `pythia_public_native_transfer` | FAIL / NOT RUN |
| `source_raw_rows_recomputable` | FAIL / NOT RUN |
| `source_summary_hash_bound` | PASS |
| `three_recipient_families` | FAIL / NOT RUN |

## Decision

Stop before held-out reveal and do not spend Qwen/T5 compute: the smallest recipient failed every public development capability.

The held-out commitment was never revealed. Qwen, T5, causal, baseline, composition, and external gates were not run because the smallest recipient failed the public prerequisite. The source summary is hash-bound but lacks raw public source rows, so the verifier correctly leaves that raw-recomputation gate failed.

## Exact required answer

> Did information acquired by training one neural model become an immutable capability object that caused previously absent, generalizing behavior to emerge in several independently trained neural models, without capability-specific training of those recipient models, and with the recipient neural computation causally necessary for the behavior?

**NO**

R7 is unchanged. R8 does not establish native neural capability transfer, recipient independence, teacher extraction, or superiority to LoRA/distillation.
