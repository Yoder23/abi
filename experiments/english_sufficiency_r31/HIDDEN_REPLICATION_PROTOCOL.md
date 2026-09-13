# R31 v4 frozen-package replication

This protocol is frozen before execution.  It is an additive replication of
the R31 v3 disclosed cascade; it may not modify, retrain, or replace any of the
36 v3 package archives.

## Frozen inputs

- R31 v3 result SHA-256:
  `9038c59287390d137665dfed2152893c61e7a721c0c668d07adcee2d1194ea39`
- R31 ladder result SHA-256:
  `4bee74452b1c4d6e6b7aacdbc2f896df2d75b3ba3ca12b88fab3f670b8f23cd9`
- R31 MDL result SHA-256:
  `d47286b0f10a884896f614dfd9188fab13bd3dfd85b48ec1970f5247b8bdbd51`
- variant order: `first24`, `first48`, `mdl24`
- 12 prompts for each of the 12 registered contracts, generated at frozen
  ordinals 10,000 through 10,011 with shuffle seed family 931031
- maximum 384 output actions per execution

The task-to-package map is the previously diagnosed registered ontology and is
embedded in the runner.  It is not rediscovered or changed in this test.

## Isolation

The execution path must not read the teacher corpus, teacher outputs, source
model, diagnosis artifact, public evaluation rows, or public selected outputs.
Runtime selection receives only a prompt and generated candidate.  It infers
the registered structural contract from those two values and applies the fixed
variant order.  Oracle contract names are used only after selection to audit
the generated test fixture.

All packages must match the hashes in the two frozen training results and must
install through separate live LayerCake registries.  The result records every
attempt and selected output, package inventories, resource use, and package
removal behavior.

## Gates

Pass requires all of the following without retries or tuning:

- at least 132/144 functional outputs;
- at least 11/12 for every registered contract;
- exactly 12/12 abstention;
- 144/144 generated-fixture contract audits;
- 144/144 routes;
- all 36 exact archive hashes and successful signed installs;
- 36/36 removed-package rejection; and
- no loaded `transformers` module and no teacher/reference artifact input.

A pass establishes replication only for the registered supplied-content
capability family.  It does not establish unrestricted English, autonomous
semantic discovery, global minimality, arbitrary domains, or superiority to
LoRA or distillation.
