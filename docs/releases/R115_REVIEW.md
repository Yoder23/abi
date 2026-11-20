# R115 review — structural decoding and prospective scope validation

Status: frozen; finite scope passed, no generation result yet.
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
exact historical-host tests in 46.96 seconds. These tests are not source quality, scientific scope
results or independent review.

## Reproduction requirements

Use exact Git history, Git LFS payloads, host dependency identities, pinned
source cache and package versions in the final manifest. The evidence ZIP will
not be a self-contained source-weight archive. Missing dependencies fail closed.

```powershell
python -m experiments.structural_decode_r115.run verify
python scripts/check_docs.py
python scripts/run_review_tests.py
```

Use the verifier only after registered stages finish. Never overwrite or resume
write-once freeze/scope/live paths. Public byte custody, same-machine replay and
GitHub software checks are not independent-hardware scientific reproduction.
