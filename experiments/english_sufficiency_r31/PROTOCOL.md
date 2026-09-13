# R31 normalized English acquisition and sufficiency ladder

R30 established a source-derived twelve-cluster diagnosis and a real
teacher-to-LayerCake package path, but all 288-row host alternatives failed the
absolute quality gate.  Even unrestricted 61.7M-parameter training reached only
43/144, while the grounded pointer packages reached 103/144.  R31 therefore
changes data quality and amount, not nearby model widths or adapter settings.

The unchanged Qwen2-7B-Instruct source receives fresh synthetic supplied-content
prompts spanning the same twelve behavior families.  No task name or label is
shown.  A deterministic prospective scorer accepts the first of at most three
generic self-check prompt variants that is grounded, adherent, and noncollapsed.
All attempts and rejected candidates are retained.  Source-derived opaque R30-v3
clusters remain the only package labels.

R31 targets 48 accepted training rows and 12 accepted evaluation rows per
cluster.  Candidate generation stops and the source stage fails if any quota is
not met within three times the requested candidate rows.  Information accounting
includes every prompt, output byte, teacher token, rejection, and wall/GPU cost.

After acquisition, the fixed R30-v7 grounded-pointer architecture is trained at
24 and then 48 rows per cluster.  Each level receives approximately 533 example
exposures per row using batch size 24.  Evaluation uses the same 144 fresh rows.
The first level passing all gates is the bounded minimum among tested levels:

- at least 132/144 functional (91.7%);
- at least 11/12 per cluster;
- 12/12 abstention;
- exact opaque routing;
- all package-removal controls; and
- no teacher at training or inference.

This can certify only the disclosed twelve-family supplied-content substrate.
It is not general English, global data minimality, or superiority to LoRA and
distillation.  A pass requires a later committed hidden replication.

After source acquisition and before learner execution, the measured longest
accepted target contained 319 lossless lexeme actions.  The package action
ceiling is therefore fixed at 384 instead of R30's 256; model widths and layers
remain unchanged.  This is a representation conformance boundary, not a
quality-selected model change.

