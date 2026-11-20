# R115 — structural decoding and prospective scope validation

Status: complete — **BOUNDED_CONSTRAINED_REVISION_IMPROVEMENT** (2026-11-20).

All 576 calls completed with exact scientific replay and 72/72 unchanged-input
controls per worker. Constrained revision passes the registered **mechanical**
workflow gates: 47/48 and 46/48 panels, 45/48 pairs (floor 44), every task >=5/6,
no collapse/truncation. This is not broad fluent-English or acquisition certification.

| Arm | Panel A | Panel B | Both pass | Sentence adherence | Invalid codec |
| --- | --- | --- | --- | --- | --- |
| Draft | 40/48 | 38/48 | 37/48 | 7/24 | 0 |
| Same-prompt unconstrained revision | 40/48 | 38/48 | 37/48 | 9/24 | 15 |
| Structurally constrained revision | 47/48 | 46/48 | 45/48 | 24/24 | 0 |

Primary paired sentence effect: +0.625, 95% bootstrap CI [+0.33333, +0.875].
All-task row-quality effect: +0.15625, CI [+0.0625, +0.26042], passing the locked
-0.10 noninferiority margin. Twelve and 48 paired clusters respectively;
replays are not extra independent prompts. All 15 actual-row hostile mutations
reject, including altered grammar traces. Post-run focused tests: 125 passed.

Three constrained outputs remain failed/unresolved; none was repaired. R114's
evaluator applied descriptively to these new outputs gives 41/48 constrained
pairs, not 45: four rows depend on the preregistered scope corrections. A
same-agent reading also flags unscored semantics: `r115-012-a` says the review
is "underway as scheduled," which the supplied notes do not establish. The
finite evaluator does not certify all factual entailment, warmth or fluency.
See the [review](releases/R115_REVIEW.md) for exact boundaries and costs.

Source commit: `df547e9ccf061ca77e2038f70adf04d9247df6a3`.
Freeze SHA-256: `1c4a35bd21f3ff394a7a22c5123dfc04fd440fa9a9295c956e8affbee8c50713`.
Frozen code, tests, protocol and inputs are immutable.
The freeze and exact challenge were published before scoring. All 800 new and
3,217 development cases passed with zero errors or false-safe outcomes and
byte-identical fresh-process replay. Raw SHA-256:
`60bb9f842c2ee770c3a0ee34647e987128fb1d8fd43fecc1e0eb92302955355b`.
This finite construction test is not general semantic validation.

The 119 core development tests and six publication tests pass. Four disclosed
fixtures also pass through the actual pinned tokenizer, including UTF-8 and
escaped quotes, with zero model calls. These are software checks, not prospective
scope or generation evidence. The first collection failure (incorrect ancestry
import) is preserved in `r115-development-a1.xml` and was repaired before freeze.
The first full suite passed 1,045 tests with four existing skips, plus 16 exact
historical-host tests. Manual review then caught the new output directory's
naming mismatch; it was corrected before any freeze or prospective evidence.
The namespace/unchanged-codec regression check passes. The final focused suite
passes 125 tests; the final full rerun passes 1,045 tests with four existing skips
plus all 16 exact historical-host tests. No scientific case or quality gate changed.

Entry points: [design/codec](../experiments/structural_decode_r115/design.py),
[scope classifier](../experiments/structural_decode_r115/evaluator.py),
[challenge construction](../experiments/structural_decode_r115/challenge.py),
[token grammar](../experiments/structural_decode_r115/decoder.py), and
[write-once driver](../experiments/structural_decode_r115/run.py).
The grammar receives token bytes and generated prefixes only. Internal ASCII
sentence punctuation is restricted, so abbreviation/ellipsis style is part of
the intervention. The original independent codec and quality gates still apply.
Identical revision prompts isolate the decoding intervention; 72 unmasked controls
must match exactly. Full-suite checks and publication must precede any freeze.

The [protocol](../experiments/structural_decode_r115/PROTOCOL.md) preserves R114
and all gates. First validate a new finite scope grammar prospectively. Only
after a published pass compare identical revision prompts with and without a
token-level structural grammar on fresh known-template instances.

The grammar constrains serialization and sentence punctuation, never semantic
content. It cannot insert answers or repair outputs. Every original quality gate
still applies. There are 576 source calls including fresh replay; no training,
recipient or LayerCake work. GPU settings and resource floors stay unchanged.
Historical LoRA/distillation controls are retained, not rerun. Any future learners
must receive matched interface access and full cost accounting.

This conventional decoding prerequisite is not an ABI breakthrough, extracted
English core, general semantic certificate or independent review.
R114 remains closed and failed; no historical scores are rewritten.
Whole moonshot: **FULL_MOONSHOT_NOT_PROVEN**. No automatic successor.
