# R14 non-exhaustive foreign-capability result

Updated: 2026-09-02

## Verdict

`R14_NOT_CERTIFIED`

The preregistered held-out result is a bounded partial success: ABI recovered
one of three fresh capabilities exactly. The registered all-capability claim
failed. R7 remains the controlling public release, R13-B remains the bounded
enumerable Track B pass, and R14 does not justify a new tag or release.

## What changed from R13

R13's 24 extractor probes enumerated the entire `3 x 8` transition table. R14
forbade atomic probes. Each hidden capability instead contained three distinct,
noncommuting affine permutations over eight states. The fixed frontend saw 256
mixed programs and Qwen's eight-way probabilities, but no answers, training
rows, held-out rows, hidden operations, oracle, or adaptive queries.

Evaluation used 10,000 prompt-disjoint programs at depths 13–18 plus 1,000
order-counterfactual rows per capability. The corresponding evaluation space
contains 4,642,668,576 possible cases per capability, so the extractor observed
only `5.514e-8` of that space. The R11 package format and executor were unchanged.

## Public development

Two public formulations were preserved before the held-out commitment was
used:

| Formulation | Qwen unseen | Qwen order CF | Package unseen | Package order CF | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| V1 full-vocabulary loss | 0.1364 | 0.1050 | 1.0000 | 1.0000 | Rejected: weak source |
| V2 canonical + native loss | 0.2302 | 0.2550 | 1.0000 | 1.0000 | Selected and frozen |

No third public formulation or secret-specific schedule adjustment was run.
Commit `b10963a` froze the implementation, protocol, gates, two public receipts,
and secret commitment before the private reveal was used.

## Held-out results

| Capability | Qwen query | Qwen unseen | Qwen order CF | Program recovered | Package unseen | Package order CF |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `r14-heldout-000-2bbf25e357e2f3c21997` | 0.2461 | 0.1180 | 0.1550 | No | 0.2449 | 0.2520 |
| `r14-heldout-001-ecb733710b44154969d6` | 0.3477 | 0.1589 | 0.1560 | Yes | 1.0000 | 1.0000 |
| `r14-heldout-002-84f5c2d2da8de40c09c3` | 0.2422 | 0.1703 | 0.1840 | No | 0.2657 | 0.2340 |

The 95% Wilson lower bounds exceeded 0.125 chance for capability 2 on all
source suites and for capability 3 on all source suites. Capability 1's unseen
lower bound was 0.1118 and failed. More importantly, the latent selector chose
the wrong program for capabilities 1 and 3. Recipient execution correctly did
not run after the aggregate source/extraction gate failed.

This result proves that the non-exhaustive mechanism can work on a fresh
capability: capability 2 was recovered from 256 answer-free mixed observations
and generalized exactly to 11,000 unseen/counterfactual cases. It does not prove
reliable capability-family extraction because only one of three passed.

## Evidence integrity

- Raw source observations: 36,840.
- Frozen source adapters: three, 35,229,008 bytes each.
- Fresh replay: 3/3 observation files byte-exact in three distinct processes,
  with zero optimizer steps.
- Hostile audit: 7/7 expected outcomes, including rejection of a
  hash-consistent probability edit and immunity to a forged stored `PASS`.
- Outcome diagnosis evidence:
  `40bd07f35718f08cc9998d8c5195751428505c12703b376f19a53d0824cbb5b0`.
- Live verification evidence:
  `c047aaa46bb2bd4f3bde57705c971d14e66d9ca17329e2ab54e87718100abbb6`.
- Hostile-audit evidence:
  `3bbdb3126a1f466f5c4b2784c0a9fcc44a8e8cb6d94f26505ebadb277603ca57`.
- Final negative certificate evidence:
  `c7cd7b69f164989e1fa9b0e412728c725c687cc487159be7cbbfec72445fd7ae`.

The 141,145,748-byte local evidence set is content-addressed by
`results/foreign_capability_r14/ARTIFACT_MANIFEST_V1.json`. Its bulk adapters,
raw rows, and replay rows are not published, so public reproducibility is not
claimed.

## Scientific conclusion

R14 rules out the claim that the current output-probability frontend reliably
recovers this non-exhaustive capability family. The observed bottleneck is the
foreign frontend/source representation, not the frozen R11 runtime: whenever
the correct latent program was selected, the package was exact. The next
materially different question is whether controlled access to foreign neural
state—weights, activations, residual representations, or training deltas—can
recover information that sparse output behavior does not identify reliably.

## Claim ceiling

R14 is not lossless Qwen cloning, extraction of pre-existing teacher knowledge,
English or domain extraction, LayerCake acceptance, information minimality, or
superiority to LoRA/distillation. It does not complete the ABI moonshot.
