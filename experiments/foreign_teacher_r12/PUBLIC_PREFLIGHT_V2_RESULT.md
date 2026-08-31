# R12-A public preflight v2: bounded frontend construction, strict-gate failure

The immutable balanced-source attempt completed on 2026-08-31. It did not
pass the strict Qwen-teacher prerequisite, so it is not R12 certification and
no held-out secret was created.

- Qwen BEFORE exact accuracy: `0 / 512 = 0.0`
- Qwen AFTER final exact accuracy: `282 / 512 = 0.55078125`
- best observed Qwen AFTER accuracy: `362 / 512 = 0.70703125` at step 2,900
- Qwen AFTER atomic accuracy: `24 / 24 = 1.0`
- extracted R11 package size: `2,053 bytes`
- extracted R11 package execution accuracy: `512 / 512 = 1.0`
- Qwen base state before/after: byte-identical by canonical tensor hash

This is positive bounded evidence for the foreign frontend construction: a
capability learned through ordinary Qwen LoRA was read through native logits
by a fixed zero-parameter extractor and emitted as an exact R11 package. The
package generalized more strongly than the source Qwen. It is not evidence of
lossless teacher-relative transplantation because the source teacher itself
was not exact on the registered depth-6/7 matrix.

Evidence identities:

- receipt evidence SHA-256:
  `ea20d893174d122d4f44e62b3aceba9d61ad12e107a84a5a455f812d3befcc1f`
- failure-analysis evidence SHA-256:
  `574c6fee3461edfe212d911b4bda8cc5577de51c9c7941031e7bbd9e225644a4`
- saved adapter SHA-256:
  `a9d1c61e48be777d0aa545da488d4c0a6fabc202ffb5dbf887919d00d4b0c262`
- exact extracted package SHA-256:
  `15d2f689e074b7f2bf4d1e922a171a175afa5c410863516c4cf9ef9595bbe934`

V3 addresses the isolated length-distribution bottleneck. It trains Qwen on
15,192 preregistered depth-1–7 prompts that are physically disjoint from the
unchanged 512 public evaluation prompts. The exact gates and all downstream
R11 components remain unchanged.
