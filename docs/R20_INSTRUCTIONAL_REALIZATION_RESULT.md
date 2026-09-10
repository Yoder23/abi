# R20 instruction-conditioned realization

Status: `PUBLIC_SOURCE_COMPILABILITY_FAILED; COMPILER_NOT_RUN`

R20 materially broadened R19 from explicit grammatical signatures to raw
natural-language instructions over six supplied-content behaviors: prose,
summary, email, bullets, clarification, and abstention. The fixed public design
used 72 extraction prompts and 120 distinct evaluation prompts, exactly 20 per
behavior. A zero-training compiler was prepared to infer output programs and
instruction character-ngram centroids without task labels, expected outputs,
evaluation rows, or oracle access.

The implementation was frozen at `d1e0b72` and the code/model/data/control/gate
configuration at `1154093` before the unchanged pinned Qwen2-7B-Instruct source
ran on GPU. All 192 raw outputs were preserved and strictly verified at
`a55944e`.

## Frozen result

| Measure | Result |
| --- | ---: |
| Source rows | 192/192 present and verified |
| Extraction registered-exact | 10/72 |
| Evaluation registered-exact | 19/120 |
| Prose evaluation exact | 11/20 |
| Summary evaluation exact | 0/20 |
| Email evaluation exact | 0/20 |
| Bullets evaluation exact | 8/20 |
| Clarification evaluation exact | 0/20 |
| Abstention evaluation exact | 0/20 |
| Teacher generated tokens | 6,057 |
| Teacher output bytes | 22,088 |
| Source inference | 335.47 seconds |
| Package / host execution | not run |

The source did not fail uniformly. All 12 prose extraction outputs preserved
the three supplied values, and bullet outputs also remained structurally
parseable. But summary outputs paraphrased the supplied values, email outputs
expanded and sometimes hit the 96-token cap, clarification frequently omitted
fields, and abstention often returned only `Abstain`. Across extraction, the
largest exact delexicalized-program supports were 10 for prose, 8 for bullets,
2 for clarification, and 3 for abstention; summary and email had no output that
preserved all three fields verbatim. The registered requirement for six
programs with support of at least six therefore failed.

This is an ABI source/representation failure, not a LayerCake failure:
LayerCake was never invoked, no package was produced, and no host or bridge was
trained. It also demonstrates that registered-string equality is not a valid
stand-in for general free-form quality. Canonicalizing these outputs with a
hand-written six-task renderer would import the oracle rather than extract the
teacher, so that repair is forbidden.

The exact delexicalized-template branch is closed. A successor must use a
materially generative representation and independently score semantic content
preservation, grammaticality, instruction adherence, hallucination, and
teacher-relative quality. It must compare the ABI route with matched sequence
distillation rather than relabel either approach. The full ABI moonshot remains
open.
