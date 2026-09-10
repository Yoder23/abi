# R24 selective LayerCake domain composition

R24 is a host-ingestion and segregation experiment. It reuses the already
strictly verified R16 second factual extraction and the already strictly
verified R23 English-factor packages. It performs no new source-model query and
makes no autonomous-discovery claim.

For each of three new training seeds, R24 trains one signed LayerCake direct
decoder for the eight extracted chemistry facts and one for the eight extracted
geography facts. Only the 48 R16 extraction rows are training data. The 48
previously frozen R16 evaluation rows remain evaluation-only. The source
teacher is absent during training and execution.

The experiment passes only if:

- all six domain packages answer 48/48 evaluation questions exactly per seed;
- all answers equal the preserved source answers;
- namespace routing is 48/48;
- a package from the other namespace never emits the target answer;
- the six English-only R23 factors never emit any of the 48 domain answers;
- installing and removing both domain packages leaves all 120 R23 English
  outputs byte-identical for every domain seed;
- GPU and CPU domain outputs are byte-identical on all 144 seed/evaluation
  pairs;
- all packages are signed, removable, restorable, and reject targeted tensor
  corruption; and
- source parameters copied, source software at execution, and host training
  steps are all zero.

No architecture, width, step, data, seed, prompt, or gate sweep is authorized.
Failure is preserved. Passing certifies only registered two-domain LayerCake
ingestion and segregation; it does not certify autonomous labeling, open-world
domains, broad English, global minimality, or LoRA/distillation superiority.
