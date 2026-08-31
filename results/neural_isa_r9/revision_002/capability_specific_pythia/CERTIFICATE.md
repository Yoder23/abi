# R9 Gate A capability-specific recipient diagnostic

Date: 2026-08-30

Verdict: **FAIL — UNIVERSAL GATE REMAINS CLOSED**

This certificate is additive. It does not modify the R7 release or the sealed
R8 Level 0 result.

## Registered question

Can a capability-specific neural backend make frozen Pythia-160M realize the
exact R8 canonical package strongly enough to justify training a general,
capability-blind neural-ISA backend?

## Immutable order

- V1 preregistration commit: `449aa5e`
- V2 repair preregistration commit: `f39d165`
- Pinned host: `EleutherAI/pythia-160m` at
  `50f5173d932e8e61f858120bcb800b97af589f46`
- Canonical latent SHA-256:
  `028f4f41ee6a1d4c642ff1f7be7cee4575d4ddb9db83421c8ed2d90b1732b3fc`
- V2 configuration SHA-256:
  `728db3286218fc1ffb3183f30bec78d11887fc418ea368d94eadfbfe50453e06`

V1's 274,184-parameter final-state backend completed 5,000 steps but failed to
fit. Its independently verified unseen-depth AFTER accuracy was 0.104492 versus
0.073242 BASE. The result is preserved under revision 001.

V2 was the sole bounded repair. It used embedding plus final recipient states,
1,039,880 backend parameters, 10,000 optimizer steps, and raw training-fit
observations. Pythia received zero optimizer steps and remained byte-identical.

## Strictly recomputed v2 results

| Measurement | Result | Gate |
| --- | ---: | ---: |
| Training fit | 0.128472 | >= 0.98 |
| Unseen-depth BASE | 0.073242 | reference |
| Unseen-depth AFTER | 0.125000 | >= 0.95 |
| AFTER - BASE | +0.051758 | >= +0.70 |
| Paired bootstrap 95% CI | [0.036133, 0.068359] | lower > 0 only |
| ZERO | 0.125000 | <= BASE + 0.05 |
| Mean AFTER teacher-recipient TV | 0.873909 | descriptive |
| Exact output/decision equivalence | false | 1.0 required |
| Distribution equivalence | false | max TV <= 0.001 |

The small paired gain excludes zero but is scientifically insufficient and is
not package-causal: ZERO matches AFTER. Gate A therefore fails.

## Evidence integrity

- Backend SHA-256:
  `078b8e3dad253246800513e727b003a911b09c91cb3280867fe247c7cca62080`
- Evaluation rows: 8,192; SHA-256:
  `a070b0af0d1880feefcd640ac1a6191a0acc042408cbfb3326bedf222d77ca5c`
- Training rows: 288; SHA-256:
  `6a993caa450bfb264a465b0b779b90c4cd3703229370c9a9b4beefebf9c95bf8`
- Verification evidence SHA-256:
  `0c59a6b42be664f41ad8f2465484820a3dc901832b8aacee8b5c774fbb874a92`
- Live replay: 8,480/8,480 rows reproduced exactly.
- Hostile verifier: 5/5 expected rejections, zero unexpected acceptances.
- Trusted scientific booleans consumed: 0.

## Decision

Close the registered recipient-state GRU branch. Do not spend compute on the
universal backend. A successor must use a materially different canonical IR or
recipient injection architecture and must first pass a newly preregistered
capability-specific fit, realization, and causality control.

This result does not prove that every neural ISA is impossible. It does prove
that R9 v2 is not lossless neural function transplantation and cannot support
teacher-to-recipient, heterogeneous-host, LoRA/distillation-superiority, or
ABI-moonshot claims.

