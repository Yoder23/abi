# R12-A conventional foreign-teacher frontend: public result

## Verdict

`BOUNDED_PUBLIC_FRONTEND_CONSTRUCTION_PASSED_STRICT_TEACHER_GATE_FAILED`

R12-A is an additive public experiment, not a new release. R7 remains the
controlling public release and R11 remains frozen at tag
`abi-r11-bounded-neural-construction-2026-08-31`.

R12-A established a narrow but real new result: after conventional Qwen was
trained on a fresh synthetic capability, a fixed zero-parameter extractor
read 24 atomic native-logit probes and emitted an unchanged 2,053-byte R11 ABI
package. The package executor produced the exact answer for all 512 unseen,
prompt-disjoint compositions. The source teacher was absent from package
execution.

The strict promotion gate nevertheless failed. The best observed native Qwen
checkpoint answered 510/512 public compositions exactly; the registered gate
was 512/512. No held-out secret was created or revealed.

## Public progression

| Revision | Native Qwen final | Native Qwen best | Atomic probes | ABI package executor |
|---|---:|---:|---:|---:|
| v1 | 68/512 (0.1328125) | 68/512 | 16/24 | 100/512 (0.1953125) |
| v2 | 282/512 (0.55078125) | 362/512 (0.70703125) | 24/24 | 512/512 (1.0) |
| v3 | 505/512 (0.986328125) | 505/512 | 24/24 | 512/512 (1.0) |
| v4 | 509/512 (0.994140625) | 510/512 (0.99609375) | 24/24 | 512/512 (1.0) |

V3 and V4 used 15,192 unique depth-1 through depth-7 training prompts and 512
unique depth-6/7 evaluation prompts, with zero prompt overlap. V4 was the one
preregistered low-learning-rate continuation from the hash-bound V3 adapter.

## What this proves

- A conventionally trained open-weight transformer can expose this registered
  bounded capability through native logits.
- A fixed, zero-learned-parameter frontend can compile the complete atomic
  behavior into the already frozen R11 package schema.
- That package can execute the registered compositional capability exactly
  without the source teacher at execution time.
- The result is not caused by evaluation-answer access: the extractor accessed
  zero training rows, zero evaluation rows, and zero answers.

## What remains open

- Exact teacher-relative public behavior: failed by two best-case rows.
- Any held-out R12 validation: not opened.
- Pre-existing knowledge extraction from an independently pretrained teacher.
- English fluency, natural-domain labeling, LayerCake product acceptance, and
  comparison with LoRA or distillation.
- Global or practical information minimality.

The fact that the structured package executor scored 512/512 while native
Qwen peaked at 510/512 is a property of this synthetic compilation task. It is
not evidence that ABI generally improves a teacher.

## Evidence

- V4 receipt evidence SHA-256:
  `73f99133010888b6d2be0626cf58df9f4bb6e8944f6f1bc48d15a144f40c01cb`
- V4 final adapter SHA-256:
  `cb5bf860a50637c046022a9affc75dd23ce05f9e665942c07b5bfb91966cb081`
- V4 package SHA-256:
  `4dc30545b34b45ab8fd68d6825100d02f9af99fcf4f6f24b99d8db4c6cb87a28`
- R11 transition SHA-256:
  `33fa23f180697c602adc0bb8b41fb6e37af58aba0770d3c85a4730409ee07637`

The receipts and content-addressed packages are preserved under
`results/foreign_teacher_r12/`. Model adapters are reproducible local
intermediates excluded from Git.
