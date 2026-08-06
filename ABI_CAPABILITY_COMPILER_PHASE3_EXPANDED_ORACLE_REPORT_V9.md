# ABI capability compiler Phase 3 expanded-bridge oracle report V9

Status date: 2026-08-06

## Decision

V22 is complete and failed its preregistered capacity gates. The expanded ABI
integration bridge scored 1,248/1,400 (89.14%) on the same contaminated
development pairs used for training and produced 75 repetition collapses. The
required gates were at least 99% aggregate, at least 95% for every capability,
and zero collapse.

This is not an acquisition result and cannot promote a model. It is a deliberately
contaminated upper-bound diagnostic. It establishes that doubling the sequence
rank from 128 to 256 and the shared-output rank from 64 to 128 is insufficient
to make the current ABI-to-LayerCake integration reliable, even when it can fit
the exact evaluation pairs directly.

## Isolation and accounting

- The exact V11 C0 subspace was transplanted into the expanded bridge.
- The sample-logit transplant difference was `1.52587890625e-5`.
- Only the registered bridge changed during training.
- The frozen LayerCake state hash was identical before and after:
  `709fd6875689886c3f5e5c62e6339872c46f927ec390c66f7ac29dfb8352c235`.
- The bridge contains 2,238,982 trainable parameters, 1,181,184 more than V11.
- Training took 46.55 seconds on the declared RTX 3080 Laptop GPU.
- The 255,586,760-byte checkpoint is bound by SHA-256
  `ab5813934095822bf1c53803256fa01eb6fad3fcae7ff146f5f0723a4c98e10e`.
- No CPU-speed, memory, TTFT, sparsity, or quality claim is inherited from the
  sealed LayerCake host.

## Autonomous result

| Capability | Passes / 100 | Collapses |
| --- | ---: | ---: |
| abstention | 75 | 0 |
| clarification | 100 | 1 |
| coherence | 100 | 0 |
| conversation | 88 | 0 |
| email drafting from notes | 61 | 37 |
| fact-free reasoning | 100 | 25 |
| fluent realization | 72 | 1 |
| format control | 100 | 0 |
| grammar | 100 | 0 |
| instruction following | 91 | 9 |
| prompt grounding | 100 | 0 |
| rewriting | 90 | 0 |
| supplied-text summarization | 82 | 2 |
| tone control | 89 | 0 |

Relative to the smaller V20 oracle, V22 gains 19 passes and reduces collapse
by 14, but remains far outside every discriminating gate. Capacity expansion
therefore does not repair autonomous integration.

## Failure ownership

The evidence does **not** show a sealed LayerCake regression: the exact host
identity and frozen tensor hash pass. It also does not test and therefore does
not blame ABI extraction or labeling, because the oracle sees the exact
development teacher pairs.

The current owner is the ABI-to-LayerCake integration design. This could be a
bridge optimization/architecture limitation or a missing host integration
surface. The experiment does not prove a fundamental LayerCake representational
ceiling. Resolving that distinction now requires a separately governed
LayerCake investigation rather than more ABI data, labeling, recovery-loss, or
nearby rank experiments.

## Governance consequence

ABI acquisition experimentation stops here. Phase 3 remains uncertified,
Phase 4 remains locked, final data remain unopened, and ABI cannot claim
superiority over LoRA or distillation. A future ABI retry requires a qualified
LayerCake integration interface, a new preregistration, fresh acquisition
evidence, and complete same-candidate speed and memory certification.

The raw decision recomputes from the bound metadata, receipt, checkpoint, and
1,400 output rows. Six adversarial mutations—fabricated aggregate success,
fabricated zero collapse, capacity promotion, Phase 3 certification,
superiority, and false LayerCake regression—are rejected.
