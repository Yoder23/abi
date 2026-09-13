# R25 direct canonical factual import into LayerCake

R25 is the preregistered successor to the strictly verified R24 negative
result. R24 showed that signed package mechanics, CPU/GPU identity, namespace
selection, and English-core immutability worked, but a newly trained neural
decoder degraded an already exact 1,240-byte R16 factual artifact to 115/144
held-out answers.

R25 therefore changes the representation, not the data or score. It imports
the exact R16 canonical relation/entity/value records into LayerCake's generic
signed canonical-factual decoder. It performs no training and no source-model
calls. Three independent builds must emit byte-identical packages.

The experiment passes only if the same two final package bytes:

- preserve all 16 imported records exactly;
- answer all 48 frozen held-out R16 questions on each of three fresh CPU and
  GPU hosts;
- match the preserved Qwen answers exactly;
- abstain when the other namespace is selected;
- leave all 120 R23 English outputs byte-identical with packages installed and
  after removal, on all three hosts;
- do not cause any R23 English factor to emit a domain target answer;
- remove and restore with exact behavior and archive identity;
- reject targeted tensor corruption; and
- copy zero source parameters, run zero receiver-training steps, and execute
  without the source teacher.

Passing proves only direct ingestion, exact execution, and segregation of two
registered finite factual domains. It does not prove autonomous discovery,
open-world labeling, broad English transfer, global minimality, or superiority
to LoRA or distillation.
