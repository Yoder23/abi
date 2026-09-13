# R65 preregistration: canonical host-prompt normalization

## Measured bottleneck

R64 proved that R60 has evaluator-lineage defects, but its unaffected 1,000
rows remain useful development evidence. R63 improved those rows to 804/1,000
against the live source's 816/1,000, while its prefix augmentation did not
address the dominant source-specific suffix in the training material.

The v3 bundle consumer already removes the Phi chat tokens.  A direct inventory
shows that 24,106 of 24,421 broad rows still end in the identical ABI extraction
policy.  The deployed host never receives that source-extraction instruction.
R65 therefore tests one normalization correction: remove that exact, audited
suffix before LayerCake tokenization. It changes neither source responses nor
the immutable archive.

## Frozen run

R65 restarts from untouched R47 and uses the R55 six-block rank-32 sparse
adapter topology and frozen optimization contract: CUDA, seed 65,001, 6,000
successful steps, batch 8, anchor batch 4, balanced capabilities, rates
2e-5/1e-4, classifier weight 0.25, prompt-overlap weight 1.0, parent-logit
preservation 0.5, weight decay 0.01, max 256 tokens, and autonomous recovery
400/every-8/[8,16,32]. The anchor is unchanged. No R60 output, teacher call,
logit, activation, evaluator result, or candidate output enters training.

An immutable receipt binds every original and normalized prompt hash, the exact
suffix hash, row counts, inputs, and checkpoint.

## Development decision

R60 remains non-promotional. A repaired screen may use its 1,000 unchanged and
200 post-hoc-diagnostic rows only to decide whether a new prospective split is
warranted. Coherence and format are non-rescorable and excluded from quality
claims, but all 1,400 rows remain in collapse and physical-sparsity checks.

To authorize a new prospective split, the candidate must:

* preserve at least 94% of the source-passing adjudicable rows;
* reach at least 65/100 on every unchanged capability;
* materially exceed R59 on the 1,000 unchanged rows;
* produce zero final collapse; and
* physically execute only the selected LayerCake route on all 1,400 rows.

Failure closes this normalization-only branch. R65 cannot promote, and the full
ABI moonshot remains OPEN.

