# R115 — structural decoding and prospective scope validation

Status: implementation validated; awaiting publication and freeze (2026-11-20).

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
