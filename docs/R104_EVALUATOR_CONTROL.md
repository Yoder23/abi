# R104: evaluator validation and English-control prerequisite

Date: 2026-11-10. Registered verdict: `CONTROL_QUALIFICATION_FAILED`.
Whole-moonshot verdict: `FULL_MOONSHOT_NOT_PROVEN`.

The requested recipient/interface comparison remains **unexecuted**. The pinned
control failed the prospective qualification gate, so no recipient evaluation,
training, new checkpoint, LayerCake source change or product promotion followed.
Finite evaluator regression tests passed, but live outputs exposed additional
scope limitations. Do not describe the evaluator as generally semantically validated.

## Design and custody

The [protocol](../experiments/evaluator_control_r104/PROTOCOL.md) was committed at
`bcc8c03`; the 729-input freeze at `cd8c222` preceded GPU execution. R103 is
preserved unchanged. Qwen2-7B-Instruct revision
`f2826a00ceef68f0f2b946d945ecc0477ce4450c` generated raw native-chat answers
to 48 new supplied-content cases, six per task. Another 48 comparison inputs
were prospectively frozen but not executed. These are new identifier/day/order
counterfactuals of known templates, not unseen semantic families.

The control had one attempt, greedy decoding, max640 tokens, GPU FP16. The fixed
user request clarified identifier suffixes, required fields, separate bullet
lines and no subject line/commentary. This compound input-contract intervention
and changed scorer prevent causal comparison against R103's score.

Qualification stayed >=44/48, >=5/6 per task, zero collapse and zero truncation.
The scorer repaired one known future-conditional pattern, retained the prior
content/form checks, and strengthened word-count (120 instead of 180), weekday
and requested sentence-count checks. Every raw row retains both scores. The
control is a feasibility screen, not a matched-size/compute transformer baseline.

## Results

| Task | Registered screen / 6 | Legacy screen / 6 (descriptive) |
| --- | ---: | ---: |
| Summarization | 2 | 6 |
| Rewriting | 6 | 6 |
| Email drafting | 0 | 5 |
| Conversation | 5 | 5 |
| Format control | 6 | 6 |
| Tone control | 0 | 6 |
| Clarification | 4 | 4 |
| Abstention | 6 | 6 |
| **Total / 48** | **29** | **44** |

All 48 outputs retain the required literals, with zero unknown identifiers,
extra numbers, unsupported weekdays, collapse or truncation. That is useful
bounded content retention, not a fluency certificate or an acquisition result.
Ten responses miss the requested sentence-count proxy; six exceed the word
bound; two clarification responses do not form the requested question. Two
approval flags are discussed below. The legacy score also fails qualification
because clarification is 4/6, below the unchanged 5/6 floor.

## Evaluator validity: finite pass, broader limits remain

All [155 author-specified formal-obligation cases](../results/evaluator_control_r104/evaluator_audit.json)
passed before generation. They cover literal omissions, unsupported content,
format, repetition, length and conditional/assertion minimal pairs. They are not
independent human ratings or a general semantic-quality benchmark.

Two live responses expose uncovered semantic scope:

- `r104-qualification-045-email_drafting` says "once the approval is granted".
  The new additional `approval is granted` pattern incorrectly flags this
  conditional. This response independently exceeds the 120-word bound.
- `r104-qualification-030-conversation` says the project is confirmed to be
  underway while approval is still pending. The lexical `is confirmed` check
  does not distinguish what proposition is confirmed. This is another scope
  limitation, not evidence of an approval assertion by itself.

These are disclosed code/meaning discrepancies, not independent fluency labels.
The frozen scorer and scores are NOT repaired after seeing outputs. The
[post-run sensitivity inventory](../results/evaluator_control_r104/postrun_error_inventory.json)
shows that removing **all** approval flags yields an upper bound of only 30/48,
still failing summary/email/tone/clarification floors. This is a ceiling, not
a new qualification score. The two scorer problems cannot rescue the gate.

The full objective of a well-validated evaluator and qualified English control
is therefore still unmet. Do not infer that Qwen lacks fluent English generally;
it did not satisfy this particular constrained generation contract.

## Information and resource accounting

| Item | Observed |
| --- | ---: |
| Raw source prompts / calls | 48 / 48 |
| Unique rendered-prompt UTF-8 bytes | 49,773 |
| Teacher input / generated tokens | 10,562 / 5,017 |
| Generated UTF-8 bytes (also unique) | 20,359 |
| Source parameters (external control only) | 7,615,616,512 |
| Generation wall seconds | 300.4178 |
| Complete control-process wall seconds | 335.9846 |
| CPU process seconds | 336.34375 |
| Peak CUDA allocated bytes | 15,337,734,144 |
| Sampled peak RSS bytes | 2,364,485,632 |
| New training / copied source parameters / bridges / checkpoints | 0 |
| Stored logits / activations | 0 / 0 |

Hardware: local RTX3080 Laptop GPU; no external hardware. Output token accounting
uses actual runtime token IDs including termination tokens. RSS is sampled,
not an OS high-water mark. Costs are descriptive on a shared development desktop
(build/review activity also occurred), not an isolated speed benchmark. The
inherited 87,710,428-parameter recipients and their acquisition costs are not
erased; their files were bound but they were not executed. There is no final
deployed English artifact or CPU inference certificate from R104.

## Verification and reproduction

The [operational receipt](../results/review_2026_11_10/r104_preflight_operations.md)
preserves the incorrect direct test invocation, sandbox failures and successful
supported-harness rerun. Use the exact historical fixture runner for old Phase7
tests; do not edit bindings to accommodate the current sibling host.

```bash
python -m experiments.evaluator_control_r104.run verify
python -m experiments.evaluator_control_r104.error_audit
python -m pytest -q tests/test_evaluator_control_r104.py tests/test_r104_completed_verifier.py
python scripts/run_review_tests.py
python scripts/check_docs.py
```

These commands require the pinned sibling LayerCake checkout, four existing
recipient payloads and pinned source tokenizer/config files. Live source weights
are required only for a separately authorized live-generation reproduction.
Do not run `pilot` on the occupied write-once evidence directory.

Key SHA-256 identities:

- Freeze: `b9af881a74405a7f9bdf0f3e10607e82d6614032ebae3dc5fceefe9e205d89cc`
- Decision: `915a7b7e37241770871cdf502ec7613639a1c371b5f50e1d4b74fc8a035516e2`
- Raw outputs: `16240051f13e8011111b428cd4f0776312bca56184ccf22595dacb4c76606b57`
- Evaluator audit: `500bea6720a53fbb678379e64dab76bd615c35c7e0cfdeda1b1c2c2c3b23e9e2`
- Descriptive error audit: `2e21279a9e541d618020378da169ad99bfda57bdc29ccbcb536e9c6f8753ce0e`

The focused completed-evidence audit passes 43 checks (25 preregistration tests
plus 18 additive checks, including 15 hostile mutations and two observed-scorer
limitation regressions). Both wheel and source builds pass; no package API changed.
Use the [review manifest guide](releases/R104_REVIEW.md) for public reconstruction.

Final local supported review: 322 default tests and 16 exact historical tests
passed, with four declared skips. The 18 additive completed-evidence checks are
separate from that default suite. Documentation discovery checks cover 26 canonical
documents; focused Ruff passes. None of these checks certifies model quality.

The public inventory contains 566 required ABI assets, four inherited LFS paths
and 729 frozen bindings (including the separately pinned host source). Manifest
SHA-256: `4be7dfc63be3204876da8aae6ceed8a9f90860f085da467e26f174a8149b7af4`.
Public-review helper SHA-256:
`4ea260b812cfbe5f438d4bd30d8d0e2e37d5fb7df1c311ded31841e78b45f9f7`.
The immutable [research release](https://github.com/Yoder23/abi/releases/tag/abi-research-r104-2026-11-10)
is public. Its tag `abi-research-r104-2026-11-10` resolves to source commit
`f9bff9ae52499145f083b526b32aa2ac59eeb4ea`, with annotated tag object
`e2f170d684d8b32f2347d5a523cd1c4026803cca`. GitHub lint and verification succeeded
for both branch and tag; the supported matrix passes on Python3.10, 3.11 and3.12.

## Published reconstruction

The [public receipt](../results/review_2026_11_10/public_r104_reconstruction.json)
records `BOUNDED_R104_DECISION_RECOMPUTED`: all 566 assets, four LFS paths and
729 bindings matched; the identical negative decision and descriptive audit
recomputed; all 43 focused checks passed. Both public checkouts stayed clean.
Receipt SHA-256:
`017b84072a2d0e04c79426c12e72d9f002099d06785b660ab2de656d62e9d696`.

This reused the public-only sibling checkouts and tokenizer cache originally
created for R102, fetched the new R104 tag and downloaded the manifest/helper
from GitHub. No development assets were copied into the review checkout. It is
the same machine and installed Python dependencies, not a brand-new clone,
fresh environment, live-generation replication, human review or independent
hardware. The additive receipt/docs commit does not move the scientific tag.

## Attribution and next decision

This is a control/input-contract/evaluation prerequisite failure. ABI acquisition
and native LayerCake execution were not tested. The two-interface matrix is
implemented but not scientifically observed. No new host failure, interface
rescue, ABI advantage or comparison with direct weight editing follows.

Do not launch another nearby prompt/seed sweep. Any successor must first address
proposition-level approval scope and the validation gap between handcrafted
fixtures and real responses, with frozen independent challenge expectations.
Only a qualified control opens recipient comparisons. Keep fluency, factual
grounding and exact format adherence as distinct measurements; do not silently
replace the locked joint gate. Human quality and independent hardware stay open.
The [weight-space assessment](WEIGHT_SPACE_RESEARCH_2026_11.md) records a future
competing-method design, not permission for an unrelated editing campaign.
