# R30 v8 shared-English control

R30 v7 proved exact routing and package causality but failed the frozen quality
gate at 103/144.  Its failures were concentrated in longer English realization:
email was 2/12 and planning was 0/12 even though every isolated model reached
essentially exact training fit.  This supports one bounded diagnosis: splitting
288 examples across twelve separately trained decoders prevents reusable English
structure from being learned.

V8 is a matched sequence-distillation control, not an ABI promotion candidate.
It trains one immutable LayerCake portable decoder over the same 288 validated
teacher rows.  Each source-discovered opaque capability ID is prepended to the
student prompt; the frozen word router supplies that ID at evaluation.  The
teacher is not called, source rows and labels are unchanged, grounded-pointer
encoding and the v7 scorer are unchanged, and no teacher parameters, logits, or
hidden activations enter the package.

The decoder architecture remains width 64, two encoder layers, two decoder
layers, feed-forward width 192, and pointer width 32.  V8 processes 153,600
training examples, exactly matching the aggregate example count of the twelve
v7 runs (12 * 1,600 * 8).  It uses batch size 32 for 4,800 updates.  This
isolates shared realization from a larger per-token neural architecture.

The preregistered pass gates remain:

- 144/144 exact routes;
- at least 132/144 functional outputs;
- at least 10/12 functional outputs in every task;
- exactly 12/12 abstentions;
- score at least as high as the live source under the same scorer; and
- live removal of the sole package must reject execution.

Even a pass proves only that a compact shared sequence-distilled English
realizer can host this disclosed synthetic supplied-content suite.  It does not
prove ABI superiority, general English, autonomous labeling, or minimality.

## Operational repair

The first launch failed before training because the hexadecimal capability ID
was lexed into numeric pieces that could duplicate grounded copy values.  The
failure is preserved at `public_v8_shared_control/failure.json`.  The sole
authorized v8b repair encodes each capability as one collision-checked
alphabetic lexeme.  No scientific input, target, architecture, seed, compute,
scorer, or gate changes.
