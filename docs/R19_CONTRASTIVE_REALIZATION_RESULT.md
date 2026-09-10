# R19 contrastive English realization

Status: `BOUNDED_REPLICATED_CONTRASTIVE_REALIZATION_PASSED_LOCAL`

R19 is the evidence-supported successor to failed R18. It removes unsafe
cross-number pooling. For questions and future declaratives, it jointly selects
same-number positive/negative templates whose structures differ by exactly one
learned literal token. No English auxiliary or negator is hard-coded.

The R17-v2 public rows and the failed R18 hidden rows are both disclosed
development evidence. The primary compiler was frozen before its physical
verdict and received only 72 anonymous extraction IDs, signatures, slots, and
free teacher outputs per dataset.

| Development set | Teacher | Modal | R19 package | Regressions | Control | Removal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R17-v2 public | 47/48 | 46/48 | 48/48 | 0 | 0/48 | 48/48 |
| R18 failed hidden | 45/48 | 46/48 | 48/48 | 0 | 0/48 | 48/48 |

Both 4,263-byte packages were created in pivot-root/no-network capsules with
zero source or host training and no teacher at compilation or execution. The
strict verifier recomputed 96/96 functional rows, the hostile audit rejected
12/12 mutations, and a fresh physical replay reproduced both packages, both
bundles, and all evaluations byte-for-byte.

The lineage preserves one initial WSL permission abort, a v1 control-protocol
failure, a v2 evidence-binding gap, a cross-filesystem publication race, and
three hostile-harness operational aborts before the complete audit. None is
silently overwritten or counted as a scientific pass.

## Fresh hidden replication

The implementation was frozen at `173ed0b`. The 256-bit selection commitment
was then committed at `4380a83`, the reveal followed at `65d4b8f`, and all 120
raw GPU teacher rows plus their strict source verification were sealed at
`c2100ec` before compilation. The 20-item lexical bank is exact-field disjoint
from the R17 and R18 banks.

| Hidden measure | Result |
| --- | ---: |
| Source rows | 120 |
| Extraction / evaluation | 72 / 48 |
| Parseable support | at least 2/3 for all 24 signatures |
| Teacher functional evaluation | 43/48 |
| Independent modal functional evaluation | 48/48 |
| R19 package functional evaluation | 48/48 |
| Regressions on teacher-correct rows | 0 |
| Mood-permutation control | 0/48; compiler rejected, no package |
| Removed-package abstention | 48/48 |
| Package size | 4,263 bytes |
| Source / host / bridge parameters in package | 0 / 0 / 0 |
| Hostile mutations rejected | 22/22 |
| Fresh physical replay | byte-exact package, bundles, and evaluation |

Teacher acquisition used the pinned unchanged Qwen2-7B-Instruct revision on
the RTX 3080 Laptop GPU. It consumed 120 prompts, 20,745 rendered input-token
instances, 1,325 generated teacher tokens, and 4,758 teacher-output bytes. The
source inference itself took 89.23 seconds; the final package stores no logits,
hidden activations, or copied source parameters. The compiler and its control
ran in separate Linux pivot-root/no-network capsules with the prompts,
evaluation rows, source model, reveal, expected outputs, and success IDs
physically absent.

R19 therefore promotes only the registered bounded compositional surface-
realization claim. It demonstrates that ABI can compile a tiny teacher-derived,
teacher-absent realization package and replicate its functional behavior on a
fresh lexical split. It does not establish unrestricted English, prompt
understanding, conversation, summarization, reasoning, autonomous labeling,
arbitrary-domain extraction, production LayerCake ingestion, global
minimality, or superiority to LoRA/distillation. The full ABI moonshot remains
open.
