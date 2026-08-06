# ABI Phase 3 shared-output successor report

Status date: 2026-08-05

## Decision

The preregistered V11 C0-C4 initial matrix is **COMPLETE_FAILED**. C0 passes
all paired causal comparisons and the teacher-relative aggregate
noninferiority gate, but fails per-capability quality, critical-capability, and
zero-collapse gates. Remaining seeds are prohibited; Phase 3 is uncertified,
Phase 4 is locked, and final material was not accessed.

The controlling decision is
`results/abi_capability_compiler_phase3_shared_output/conditional_decision_v1.json`
(file SHA-256
`f6fbf0fd4cc7c3e998c3df1fe67abea8679c82eb65ba46df24f248656d069a4d`,
internal evidence SHA-256
`5b83d75470021d5599ceeeedabf599680971613615639f83cc34d5ce74d362f5`).
Independent recomputation is byte-identical.

## Bound experiment

- Frozen three-block LayerCake host.
- 1,057,798 trainable parameters, 1.6867% of 62,712,848 total.
- Three shared rank-128 prompt-conditioned pre-block residuals.
- One rank-64 output residual shared by every English capability.
- Semantic labels condition only the persistent shared sequence trajectory.
- Training-only margin applied only when a wrong greedy prediction repeated a
  teacher token from the preceding eight response positions.
- No source parameters or blocks copied; teacher absent during training and
  inference; certified cached Phase 1 IR only.
- C0-C4 bind the same 28,000-record successful sequence SHA-256
  `b4ac23e86611c88515db8c17e72e78290ec41329059cb0546731c703ffeff28e`.

## Autonomous development results

| System | Definition | Passes | Rate | Collapses |
| --- | --- | ---: | ---: | ---: |
| C0 | semantic sequence conditioning | 1,207/1,400 | 86.21% | 51 |
| C1 | prompt-hash conditioning | 1,090/1,400 | 77.86% | 75 |
| C2 | deranged targets | 242/1,400 | 17.29% | 49 |
| C3 | no teacher-response loss | 10/1,400 | 0.71% | 776 |
| C4 | monolithic condition | 1,107/1,400 | 79.07% | 58 |
| T0 | frozen teacher reference | 1,237/1,400 | 88.36% | 64 |

## Paired causal and teacher comparisons

| Comparison | Difference | 95% stratified-bootstrap CI | Gate |
| --- | ---: | ---: | --- |
| C0 - C1 | +8.36 points | +6.50 to +10.14 | PASS |
| C0 - C2 | +68.93 points | +67.07 to +70.86 | PASS |
| C0 - C3 | +85.50 points | +83.79 to +87.21 | PASS |
| C0 - C4 | +7.14 points | +5.36 to +8.86 | PASS |
| C0 - T0 | -2.14 points | -4.21 to -0.07 | PASS noninferiority margin |

This establishes a bounded development causal result: semantic labels,
correctly paired cached teacher responses, and response supervision each
matter in the shared-output architecture. C0 is aggregate-noninferior to T0
under the locked five-point margin. It does not establish broad fluency or ABI
superiority because the absolute gates still fail.

## Tail failures and collapse

| Capability | Passes | Collapses |
| --- | ---: | ---: |
| abstention | 55/100 | 0 |
| tone control | 66/100 | 3 |
| fluent realization | 75/100 | 5 |
| conversation | 77/100 | 1 |
| email drafting from notes | 79/100 | 41 |
| supplied-text summarization | 81/100 | 0 |
| coherence | 83/100 | 0 |

The remaining seven capabilities range from 97 to 100 passes. Forty-one of
C0's 51 collapses occur in email drafting. The teacher-forced wrong-repeat
penalty did not solve autonomous long-form exposure bias and cannot be tuned
inside V11.

## Stop decision and next measured hypothesis

V11 is closed. Seeds 130363 and 155921 are not authorized. Do not sweep the
repeat-loss weight/window/margin, rank, steps, data, decoding, or output cake.

The measured next hypothesis is bounded self-prefix recovery: keep the exact
shared-output architecture, but train recovery from a small preregistered set
of the model's own incorrect prefix tokens before predicting the cached
teacher continuation. This directly tests autonomous exposure bias in email,
conversation, realization, tone, summarization, and coherence. Abstention must
receive a separate matched uncertainty/abstention control rather than being
hidden by aggregate quality. Any such branch requires a new protocol and full
C0-C4 controls sealed before execution.

## Claim boundary

This is the strongest causal ABI development result so far, including locked
aggregate teacher noninferiority. It is not a Phase 3 certificate, a fluent
English core, evidence of fewer required tokens, superiority over LoRA or
distillation, permission to access final data, or permission to open Phase 4.

Verification reports 497 in-scope ABI tests passed with the known
separate-repository identity check explicitly deselected rather than weakened.
