# ABI Capability Compiler Phase 3 Fit Diagnostic V26

Status date: 2026-08-06

Status: **COMPLETE — V23 AND V24 REMAIN FAILED; PHASE 3 UNCERTIFIED**

## Scientific question

V26 replayed the immutable V23 and V24 checkpoints without training. It asked
whether their failures primarily reflected inability to fit the acquisition
targets, autonomous state drift after otherwise strong fit, or held-out action
generalization. It also counted teacher targets that each sealed action surface
could not express.

Attempt 1 wrote no result because an empty held-out capability stratum caused a
division by zero in reporting. That failure is preserved. V27 authorized one
unchanged retry that reports empty metrics as `null`.

## Result

| Metric | V23 fixed actions | V24 pointer-supervised |
|---|---:|---:|
| Training targets representable | 7,000 / 7,000 | 7,000 / 7,000 |
| Training action accuracy | 95.30% | 94.93% |
| Training exact-sequence rate | 49.43% | 61.43% |
| Training mean action NLL | 0.2594 | 0.2359 |
| Autonomous training sample exact | 136 / 280 | 167 / 280 |
| Autonomous mean correct-prefix fraction | 61.32% | 68.67% |
| Autonomous repetition collapses | 13 | 11 |
| Autonomous generation errors | 1 | 0 |
| Development targets representable | 657 / 1,400 | 941 / 1,400 |
| Development representable rate | 46.93% | 67.21% |
| Development action accuracy on representable targets | 85.12% | 77.98% |
| Development exact rate on representable targets | 26.64% | 21.57% |

Both designs miss the preregistered 99% training-action and 90%
training-exact thresholds. Because training fit is insufficient, V26 does not
label autonomous failure as an isolated state-drift mechanism. Both designs
also show a preregistered held-out generalization drop.

V23 rejects 743 held-out teacher targets as unrepresentable; V24 rejects 459.
Both reject all 100 coherence targets and all 100 fact-free-reasoning targets.
Pointer supervision improves coverage but does not create an open-vocabulary
target surface and does not repair fit.

## Ownership

Primary owner: **ABI model fit, capacity, optimization, or target
representation**.

Secondary owner: **ABI held-out representability and generalization**.

No LayerCake regression was identified. V26 replays the historical v1 private
checkpoint graph solely to diagnose the failed ABI artifacts. The separately
certified LayerCake v2 Unicode host remains unchanged and contributes no
inherited quality or performance claim.

## Decision

V23 and V24 remain closed. No nearby pointer, fixed-vocabulary, loss-weight,
hidden-size, or data-expansion run is authorized.

The next bounded work is a no-training, hash-bound representation bake-off.
Every candidate representation must losslessly express all 7,000 acquisition
targets and all 1,400 development teacher targets through Unicode-atomic
actions before any new fit run can be proposed. The final split remains sealed.

Phase 3 is not certified. ABI has not been shown stronger than LoRA or
distillation.

## Evidence

- Raw diagnostic: `results/abi_capability_compiler_phase3_fit_diagnostic/fit_generalization_v26.json`
- Raw file SHA-256: `10103f6eceeb1fed2f627ba92ec37bbc25255d4aabf394736458d58384457871`
- Embedded evidence SHA-256: `fd8508074c3539c50869624ecf2a3a4a444b354ddf37f0929c2064050d7df15b`
- Decision: `results/abi_capability_compiler_phase3_fit_diagnostic/fit_decision_v26.json`
- Decision file SHA-256: `e77b4a5471a0768111ce099cb3e2b356a8478856864f64ad52f4a34e6303845a`
- Decision evidence SHA-256: `07cc833d724eeda5614f359374073ddf7dc78f0bd13d9e9d76bb045a128b4f06`

The independent verifier recomputes the complete GPU diagnostic. Pure
mutation tests reject altered certification state, action accuracy, checkpoint
identity, or LayerCake ownership.

The full ABI suite reports 535 passed and one expected fail-closed historical
external LayerCake commit mismatch. That historical contract remains bound to
its original sealed LayerCake commit; the current external repository is the
later v2 host lineage and is intentionally not made equivalent by weakening
the verifier.
