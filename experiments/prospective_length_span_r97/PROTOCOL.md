# R97 prospective length/structure transfer protocol

R96 is frozen at checkpoint SHA-256 `9d3d6b38b1d437b5cbbed50c1470b4df1aa0371ffa51b1abd2adefbb645806e4`.
It may not be retrained, calibrated, or selected using R97. R97 contains 1,400
new prompts composed after the R96 checkpoint and its R95 development result
were committed. Exact prior prompt overlap is prohibited.

The live frozen teacher is captured and committed before a candidate binding
exists. Source authorization checks capture integrity only. The candidate must
score at least 1,330/1,400 and 315/350 per family, retain at least 95% of
source-correct rows, equal or exceed the source point score, and have a paired
bootstrap 95% lower confidence bound of at least -0.02. It must gain at least
50 percentage points over both the frozen parent and random bridge, with random
at or below 500, no collapse, all-row physical sparse execution, no source at
inference, and unchanged bound artifacts.

Passing proves a bounded prospective two-hop reasoning capability transfer.
It does not prove broad English, arbitrary-domain transfer, global minimality,
LoRA/distillation superiority, human quality, independent hardware, or the
full ABI moonshot.
