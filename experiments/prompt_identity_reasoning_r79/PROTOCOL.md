# R79/R80 protocol: bridge-only prompt-identity acquisition

R78 established two simultaneous host limitations on R77: conditional choice
683/1,400 and autonomous exact generation 83/1,400. Raw outputs frequently
copied the numeric suffix but corrupted an unseen three-letter stem. R74 and
R78 close nearby deep-adapter acquisition variants.

R79 makes one measured architectural change. It starts from the unchanged
three-block v51 route-control host (checkpoint SHA-256
`012c5443dca21fb73874f1329bcfb6a10526284092e15d7408fff15614dd563f`)
and trains only its existing 49,931-parameter selective prompt-identity bridge.
The bridge uses prompt-position attention plus a learned copy/language gate;
the source teacher remains absent. Main artifact SHA-256 is
`292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc`
and broad anchor SHA-256 is
`82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150`.

Training reuses the established v53 bridge configuration: CUDA seed 459824,
6,000 successful steps, main/anchor batches 4/4, bridge learning rate 3e-4,
prompt-overlap and direct prompt-identity weights 1.0, source and parent
distillation disabled, and autonomous prefix recovery from step 400 every
eight steps over horizons 4/8/16. Only the R76 reasoning route is supplied by
the main artifact; the balanced broad anchor preserves other behavior.

R80 contains 1,400 validation-only prompts, 200 per premise family, with
lexical stems, subject prefix, nonce range, and exact prompts disjoint from
R76 and R77. It is frozen before source scoring or candidate training. The
source must pass at least 1,330 overall, at least 180/family, with zero ties.
The live candidate must autonomously pass the same quality and family gates,
retain at least 95% of source-passing rows, exhibit zero collapse, improve at
least 50 percentage points over the unchanged v51 parent, execute one task
cake plus the shared pointer bridge, and contain no source model.

This remains a bounded nonce reasoning/copy test, not unrestricted English or
the complete ABI moonshot.

R80 failed before host training because family 0 reached 175/200. One
post-failure, non-promotional diagnostic is authorized: subtract each stored
actual candidate mean log probability from a new same-order, same-candidate,
no-relation control score. This tests candidate lexical/display prior without
changing R80's verdict. R79 remains blocked unless a wholly new prospective
source catalog passes a scorer frozen from this diagnostic.

The diagnostic completed over all 1,400 immutable R80 rows. Raw selection was
1,368/1,400; prior-corrected selection was 1,369/1,400, with family counts
181/200, 200/200, 200/200, 188/200, 200/200, 200/200, and 200/200. Seventeen
raw failures were corrected and sixteen raw passes regressed. Its verdict is
`PRIOR_CORRECTION_SUPPORTS_PROSPECTIVE_REPLICATION`; it remains explicitly
non-promotional and does not change `FAIL_R80_SOURCE`.
