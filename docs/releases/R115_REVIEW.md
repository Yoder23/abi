# R115 review — structural decoding and prospective scope validation

Status: complete — **BOUNDED_CONSTRAINED_REVISION_IMPROVEMENT**.
Whole moonshot: **FULL_MOONSHOT_NOT_PROVEN**.

## Design and claim boundary

Read the [preregistered protocol](../../experiments/structural_decode_r115/PROTOCOL.md)
and [experiment guide](../R115_STRUCTURAL_DECODING.md). Preregistration commit:
`fb4c07366ed91c5a00b103ac4755780dd713da5d`.

Implementation `df547e9ccf061ca77e2038f70adf04d9247df6a3`; freeze
`1c4a35bd21f3ff394a7a22c5123dfc04fd440fa9a9295c956e8affbee8c50713`.
Published freeze/challenge commits: `24853cd` / `1ec0ad9`, before scoring.
All 800 prospective and 3,217 development scope cases passed, zero errors or
false-safe outcomes, byte-identical replay in distinct processes. Raw hash:
`60bb9f842c2ee770c3a0ee34647e987128fb1d8fd43fecc1e0eb92302955355b`.

One conventional structural-decoding comparison, not a new ABI extraction
mechanism. The same complete draft conditions two independent revisions with
identical prompts. Only the constrained arm masks structural tokens on 24
summary/tone rows; the other 72 rows have no masks and must match exactly.
The 96 instances are fresh aliases/days of known templates, not hidden semantic
families. Two fresh workers provide 576 calls, not 576 independent prompts.

The grammar sees token bytes and generated prefixes, never facts, answers,
scope labels, task scores or the original request. It constrains a two-or-three
string JSON array and terminal punctuation; internal ASCII .!? and most JSON
escapes are disallowed. This style restriction is part of the intervention.
It is not a general JSON decoder or native unconstrained prose generation.
The unchanged R114 codec independently validates and joins all unmodified
strings. Invalid outputs fail: no splitting, selection, retries or repair.

Absolute gates remain 44/48 per panel and paired, 5/6 per task, no collapse or
truncation, exact facts/literals/format, 3–120 words, scope and requested sentence
count. Relative improvement needs a positive lower CI for sentence adherence
(12 paired clusters), plus all-task quality noninferiority (48 clusters, -0.10
margin). Bootstrap: 10,000 draws, seed108001. Neither a format guarantee nor
relative gain replaces an absolute gate.

Finite scope is a separate prerequisite: 800 new construction cases and 3,217
development obligations, zero errors/false-safe classifications and byte-identical
fresh replay required. Author construction and six disclosed R114 clause labels
are not independent human judgments or general semantic validity. The ambiguous
`will proceed as approved` remains unresolved. First failure closes GPU work.

## Measured result and limitations

All 576 calls completed. Initial and fresh-process replay match exactly on
scientific fields (timing/storage fields are measured separately). All 72
unmasked controls per worker have identical prompt/input/output tokens. The
3,558 masked token steps across both workers recompute against the frozen
grammar; the actual generation callback traces agree with the stored traces.

| Arm | Panel A /48 | Panel B /48 | Paired /48 | Sentence /24 | Invalid arrays | Qualified mechanically |
| --- | --- | --- | --- | --- | --- | --- |
| Draft | 40 | 38 | 37 | 7 | 0 (plain prose) | No |
| Unconstrained revision | 40 | 38 | 37 | 9 | 15 | No |
| Constrained revision | 47 | 46 | 45 | 24 | 0 | Yes |

Every constrained task meets the locked 5/6 floor in both panels and paired;
all arms have zero collapse and truncation. The constrained workflow's tone,
conversation and clarification pairs are each 5/6; other tasks are 6/6.

- Primary sentence adherence: +0.625, 95% CI [+0.333333, +0.875], 12 paired clusters.
- All-task row quality: +0.15625, CI [+0.0625, +0.260417], 48 paired clusters;
  this passes the preregistered -0.10 noninferiority margin.
- Both intervals use 10,000 paired bootstrap draws, seed108001. Repeated
  execution does not double the independent sample count.

The three failed constrained rows remain failed: `r115-005-a` (clarification,
bare `provisional` in a bullet), `r115-011-b` (tone, `pending status of approval`),
and `r115-040-b` (conversation, `will not be confirmed until approval is granted`).
Their frozen classifier outcomes are UNRESOLVED, not proven factual errors.
No post-run repair, relabeling, hidden rescore or sentence substitution occurred.

The evaluator-version dependence is explicit. R114's unchanged evaluator scores
the new draft/unconstrained/constrained pairs 37/48, 35/48 and 41/48 respectively;
R115 scores 37/48, 37/48 and 45/48. Four constrained A-panel rows change under
R115's pre-frozen scope rules: `r115-010-a` and `r115-026-a` contain
`provisional arrangements pending approval`; `r115-012-a` and `r115-024-a`
contain `no confirmation of approval has been issued/granted`. These are new
outputs, not retrospective rescoring of R114. Both revision arms use the same
R115 evaluator in the primary comparison.

**Semantic coverage remains incomplete.** Same-agent inspection of the raw
`r115-012-a` constrained output finds "the draft invitation review is underway
as scheduled." Its supplied notes state a task and planned day, not that work
has started. The frozen mechanical screen does not catch this unsupported
progress assertion. This observation is not an independent human rating or a
replacement score; it prevents treating the mechanical pass as full factual
entailment, broad English qualification, human tone quality or an acquisition
certificate. Future semantic testing must be preregistered on new cases.

Live decision SHA-256:
`664f7a50cd646d5b62b8efad4333489a4f7d04f28cd20e6b98cfcebfce4b5d71`.
The [raw inventory](../../results/review_2026_11_20/r115_error_audit.json) preserves
all frozen failures; the [hostile audit](../../results/review_2026_11_20/r115_hostile.json)
rejects 15/15 in-memory actual-row mutations. This is same-agent targeted
verification, not independent blind review or independent hardware.

## Observed costs

The live controller took 2,607.547 seconds (43.46 minutes); summed worker wall
time was 2,579.157 seconds and summed CPU process time 2,410.421875 seconds
(0.66956 CPU hours). Peak 50-ms sampled process RSS was 4,711,976,960 bytes
(4.388 GiB); peak CUDA allocation was 8,331,224,576 bytes (7.759 GiB).
The full experiment produced 51,448 runtime-counted output tokens from 202,168
input tokens: 951,324 prompt UTF-8 bytes, 186,522 raw output bytes and 178,346
delivered output bytes. These totals include all arms and both workers.

Generation time across both workers: draft 805.906 seconds, unconstrained
revision 823.789, constrained revision 833.549. Complete draft-plus-revision
generation costs are 1,629.695 versus 1,639.455 seconds. The descriptive ratio
is 1.006x; no timing significance, p95/p99 or architecture speed claim follows
from two executions. Grammar initialization totals 0.797 seconds; lazy mask
construction and application are included in generation/worker timing.
Controller time includes runtime verification; source cache preparation was
reused, not zero-cost from scratch. Software development/test/publication time
is outside these scientific worker measurements.

There were zero training steps, recipient calls or new downloads. The full
pinned teacher was present for this source-interface experiment. No teacher
weights, activations or logits were exported as an English capability artifact.
No inference-free teacher removal, compact core or ABI/LayerCake speed result
is asserted. GPU settings stayed unchanged; no storage cleanup was performed.

## Preservation and failure ownership

R114 and all historical scores remain immutable. Its manifest is
`0bf4dcd7944a407fc6987bf98f7aa5e8a8f950af16498c893767e8b3edaa9fef`.
Historical matched LoRA/distillation controls remain preserved, not newly run.
Later learners must receive matched representation, grammar access, information
and full workflow costs. ABI acquisition and LayerCake are NOT_TESTED here;
no training, recipient call, source-independent artifact or host promotion.

Separate evaluator failures from source/interface prerequisites and resource/
execution failures. A complete negative result is not a host defect. Missing
evidence cannot pass. GPU hardware settings, quality gates and the 10 GiB disk
floor remain unchanged; there is no cleanup or automatic successor.

## Software checks

The first development collection failed due to an ancestry import typo; its
receipt is preserved. The corrected core suite passed 119 tests, publication
helpers passed six, and four known fixtures passed the actual pinned tokenizer
without model calls. The first full suite passed 1,045 tests with four existing
skips plus 16 tests in the exact historical-host fixture.

Review then found an output-directory naming mismatch before any experiment
evidence was created. It was corrected and explicit namespace/unchanged-codec
checks were added. The final focused suite passed all 125 tests. The final full
suite passed 1,045 tests with four existing skips in 559.31 seconds, plus all 16
exact historical-host tests in 46.96 seconds. The post-run focused suite passes
125 tests in 106.43 seconds. These software tests are not independent scientific
review. Strict raw recomputation and targeted hostile checks are separate.

## Reproduction requirements

Use exact Git history, Git LFS payloads, host dependency identities, pinned
source cache and package versions in the final manifest. The evidence ZIP will
not be a self-contained source-weight archive. Missing dependencies fail closed.

```powershell
python -m experiments.structural_decode_r115.run verify
python scripts/check_docs.py
python scripts/run_review_tests.py
```

The registered stages are complete. Never overwrite or resume
write-once freeze/scope/live paths. Public byte custody, same-machine replay and
GitHub software checks are not independent-hardware scientific reproduction.
