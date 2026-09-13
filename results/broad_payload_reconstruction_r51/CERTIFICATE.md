# R51 broad-payload reconstruction certificate

Verdict: `FAIL_R51_DISCLOSED_SCREEN`

R51 is a preserved architectural experiment, not a promoted ABI artifact.

The fixed 6,000-step endpoint trained the existing 82.9M LayerCake graph from
the R47 parent using 24,419 context-compatible rows from the immutable
2,784,714-token broad English archive, all 1,438 rows of the 46,306-token
anchor, and frozen-parent top-64 preservation.  The pinned Phi teacher was not
loaded.  Exactly two broad rows (80 teacher tokens) were excluded because
their prompts exceeded the unchanged 256-token host context; their identities
and aggregate hash are preserved in metadata.

Training completed in 1,579.712 seconds with 6,000 successful optimizer steps,
4 safely skipped AMP attempts, 3,591,433 supervised main tokens,
622,883 anchor tokens, 8,409,762 frozen-parent forward tokens, and
4,205,769,216 peak allocated CUDA bytes.  The emitted checkpoint SHA-256 is
`3a080529732deaf8a56b53572f7c5faf7d44584b5fab3494b5bf7f7acdba45b8`.

On the disclosed 1,400-row validation screen, R51 scored 1,271 functional
versus the parent’s 1,277 and the stored source’s 1,220.  Source-pass retention
was 95.6557%, routing and physical sparsity were 1,400/1,400, and collapse was
zero.  However, tone control scored only 52/100 against the frozen 65/100
minimum.  Both `parent_nondegradation` and `per_capability` failed.  The
disclosed final-test screen was therefore not run.

The result isolates shared-weight interference: broad training improved
domain-independent reasoning from the parent’s 74/100 to 93/100, but damaged
tone behavior.  R51 does not justify a nearby full-core learning-rate or
step-count sweep.  The next experiment must freeze the shared core and isolate
the broad update in sparse cakes.

Evidence:

- metadata SHA-256:
  `0b61419a095efa60a3970823cb76bcc23efffd4d7b3032c3b450037aebc6572f`;
- validation rows SHA-256:
  `3c964d8c6d48d474724928a00f54fdc3cb53fcebc8af1ac6f8ba014c0dcbed48`;
- validation result SHA-256:
  `32ad4e110b35f954e16fb88ba8a8f3c72a11d9308d571e542d96cb19acf983e6`.

The ABI moonshot remains `OPEN`.

