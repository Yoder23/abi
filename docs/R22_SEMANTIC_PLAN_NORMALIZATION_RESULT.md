# R22 semantic-plan normalization result

Updated: 2026-09-10

Status: `FAIL_NORMALIZED_SOURCE`

R22 tested one preregistered teacher-side normalization step after the R21
hidden failure. It preserved the 600 frozen R21 raw teacher responses and
registered labels, supplied an explicit task-and-field plan to the same pinned
Qwen2-7B-Instruct revision, and requested one normalized response per row. No
student training, packaging, or LayerCake execution was allowed until the
source prerequisite passed.

The source prerequisite failed and the downstream bakeoff therefore did not
run.

| Metric | Frozen gate | Result |
| --- | ---: | ---: |
| Functional normalized targets | at least 594/600 | 520/600 |
| Non-hallucinating targets | 600/600 | 600/600 |
| Non-collapsed targets | at least 594/600 | 600/600 |
| Every field verbatim once and in order | 600/600 | 255/600 |
| Registered labels unchanged | 600/600 | 600/600 |

Task-level recomputation localized the functional misses: prose 98/100,
summary 84/100, email 100/100, bullets 100/100, clarification 67/100, and
abstention 71/100. Exact field preservation was weakest for email (0/100),
summary (17/100), abstention (18/100), and clarification (40/100), despite the
explicit normalization instruction. This rejects the hypothesis that another
unconstrained teacher rewrite is a reliable semantic-transfer target.

The run made exactly 600 new GPU source calls, consumed 100,011 rendered input
token instances and 21,328 output tokens, used 0.32835 source-model inference
hours, peaked at 15,338,472,960 GPU bytes and 2,168,479,744 process-RSS bytes,
stored no logits or activations, and copied zero source parameters. The raw
rows and receipt are preserved under
`results/semantic_plan_r22/public_v1_source/`.

R22 is an ABI acquisition/source-interface failure. It is not a LayerCake
failure because LayerCake was never invoked. R21 remains a bounded public pass
and hidden failure. R7 remains the controlling release claim, and the full ABI
moonshot remains open.
