# R12-A public preflight v4: exact ABI package, strict teacher gate failed

The preregistered bounded continuation completed on 2026-08-31. It did not
pass the exact conventional-teacher prerequisite, so no held-out secret was
created or revealed and this is not R12 certification.

- frozen source: Qwen2.5-0.5B at revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`
- training prompts: `15,192` unique depth-1 through depth-7 programs
- evaluation prompts: `512` unique depth-6/7 programs
- training/evaluation prompt overlap: `0`
- source BEFORE exact accuracy: `0 / 512 = 0.0`
- source AFTER best observed exact accuracy: `510 / 512 = 0.99609375`
  at step `600`
- source AFTER final exact accuracy: `509 / 512 = 0.994140625`
- source AFTER atomic accuracy: `24 / 24 = 1.0`
- fixed extractor learned parameters: `0`
- extracted R11 package size: `2,053 bytes`
- extracted R11 package execution accuracy: `512 / 512 = 1.0`
- source base weights: unchanged by canonical tensor hash
- training hardware: NVIDIA GeForce RTX 3080 Laptop GPU
- continuation wall time: `394.3050444999826` seconds

The ordinary LoRA-trained conventional source encoded the complete atomic
capability. The fixed zero-parameter extractor compiled that atomic behavior
into the unchanged R11 transition package, whose executor was exact on every
unseen prompt-disjoint composition. This is a bounded public
foreign-teacher-to-ABI frontend construction result. It is not a lossless
teacher-relative transplant result: the best saved evaluation missed two
source predictions, the final adapter missed three, and the registered source
gate required all 512.

This bounded source-training branch is closed. Another learning-rate,
step-count, data-size, or nearby LoRA sweep is not authorized by this result.
A successor must make a material change to source capability acquisition or
teacher formulation and must clear the exact public source prerequisite before
any held-out run.

Evidence identities:

- receipt evidence SHA-256:
  `73f99133010888b6d2be0626cf58df9f4bb6e8944f6f1bc48d15a144f40c01cb`
- config SHA-256:
  `53c6b0f40f9d16883d707be78e4e33deaa5eb29ca029b05890a4da4b011f8701`
- saved final adapter SHA-256:
  `cb5bf860a50637c046022a9affc75dd23ce05f9e665942c07b5bfb91966cb081`
- exact extracted package SHA-256:
  `4dc30545b34b45ab8fd68d6825100d02f9af99fcf4f6f24b99d8db4c6cb87a28`
- exact transition SHA-256:
  `33fa23f180697c602adc0bb8b41fb6e37af58aba0770d3c85a4730409ee07637`

The 35,229,008-byte adapter remains a local reproducible intermediate and is
not committed to Git. The content-addressed ABI package and evidence receipt
are preserved.
