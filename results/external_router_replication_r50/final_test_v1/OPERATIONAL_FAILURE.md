# R50 final-test v1 operational failure

Verdict: `FAIL_CLOSED_ZERO_ROWS_SOURCE_EVIDENCE_ABSENT`

The preregistered R50 runner was launched against the frozen R49 candidate,
router, catalog, and the two frozen Phi survey archives.  Before loading the
LayerCake model or generating any candidate response, it verified the inputs,
selected the 1,400 `final_test` probes, obtained 1,400/1,400 routes from the
frozen router, and then failed closed because the source archives contain no
`final_test` Phi outputs.

Consequences:

- candidate evaluation rows: 0;
- candidate model loads: 0;
- candidate generations: 0;
- training steps: 0;
- model or router mutations: 0; and
- scientific verdict: unavailable, not a candidate failure.

The missing source comparison may not be replaced with validation rows or
silently omitted.  The only authorized repair is the additive R50-v1a
protocol: execute the pinned local Phi source once on every frozen final-test
probe, seal those raw rows, and then evaluate the unchanged R49 system.

