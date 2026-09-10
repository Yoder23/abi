# R23 semantic replication result

Updated: 2026-09-10

Status: `PASS_STRICTLY_VERIFIED_SEMANTIC_REPLICATION`

R23 tested the exact frozen R21 package bytes on a newly committed hidden split
under a task-aware semantic contract. The evaluator code and thresholds were
committed before a new 256-bit row seed was generated; its commitment was
committed before reveal. No student retraining or package change occurred.

The semantic contract corrected one known validity defect without rescoring or
promoting the failed R21 evidence: summary paraphrases such as “decreased” and
“fell” are equivalent only when the exact case identifier, latency amount, and
review day remain bound to the correct concepts. Opposite polarity, new
numbers, missing numbers, and swapped numeric roles fail.

| System | Seed 21021 | Seed 21022 | Seed 21023 |
| --- | ---: | ---: | ---: |
| ABI factorized | 120/120 | 120/120 | 120/120 |
| Labeled monolith | 99/120 | 100/120 | 110/120 |
| Raw sequence | 32/120 | 32/120 | 27/120 |
| Live teacher | 96/120 | 96/120 | 96/120 |

Every ABI-factorized seed scored 20/20 on each of prose, summary, email,
bullets, clarification, and abstention, with 120/120 non-hallucinating and
120/120 non-collapsed outputs. Paired bootstrap 95% intervals for the ABI
factorized system versus the teacher were `[0.1333, 0.2750]` at every seed.
Intervals versus the labeled monolith were `[0.1083, 0.2500]`,
`[0.1000, 0.2333]`, and `[0.0333, 0.1333]`.

Fresh physical replay then regenerated 1,080/1,080 GPU outputs byte-exactly
and 54/54 task-covering CPU outputs identically. All 24 packages passed
signature and contract validation, removal made each unavailable, restoration
recovered exact behavior, and 24/24 targeted tensor corruptions were rejected.
The replay process loaded no source-model software. Package provenance reports
zero source parameters copied and zero receiver training steps.

The first frozen live verifier launch failed before executing any candidate row
because it looked for package systems in the hidden summary instead of the
bound public engine. That failure is preserved. A separately frozen repair
joined only the already immutable package metadata and changed no candidate,
package, scorer, source row, or gate.

R23 certifies only this bounded six-task, supplied-content semantic transfer
mechanism. It does not establish unrestricted English, autonomous capability
discovery or labeling, arbitrary domains, global information minimality, or
superiority to LoRA/distillation. R7 remains the controlling public release and
the full ABI moonshot remains open.
