# R50-v1a held-split certificate

Verdict: `FAIL_R50_HELD_SPLIT_LIVE_SOURCE`

This certificate records a correctly failed prospective replication.  It is
not an ABI moonshot certificate.

## Frozen construction

- unchanged R47 neural checkpoint:
  `65b1aae5e1aed947f2bc086281a3c7388b22a4260c562363aff2e8cc59f9661b`;
- unchanged R49 external router:
  `20ae6562432cc59ece5b9cb552caaa16da4824323e62460c6c33feb0579df863`;
- candidate training steps: 0;
- candidate teacher calls: 0;
- planner and symbolic output calls: 0; and
- source teacher absent from candidate inference.

The originally preregistered v1 launch failed closed with zero candidate rows
because its frozen source bundles lacked final-test evidence.  R50-v1a first
captured the pinned Phi-3 source once on all 1,400 final-test prompts, sealed
those rows, and only then evaluated the candidate.

## Recomputed result

- candidate: 1,261/1,400 functional;
- pinned Phi source: 1,154/1,400 functional;
- candidate minus source: +0.076429, paired bootstrap 95% CI
  [+0.055000, +0.099286];
- every capability: at least 72/100 candidate functional;
- collapse and generation errors: 0;
- exact external routes: 1,400/1,400;
- physically single selected cake: 1,400/1,400; and
- fresh exact neural replay: 1,400/1,400.

The candidate regressed on 74 source-passing rows, retaining
1,080/1,154 = 93.5875%.  The preregistered requirement was at least 94%.
`source_retention` is the sole failed gate.  The requirement is not rounded,
relaxed, or reinterpreted.

## Evidence

- source rows SHA-256:
  `8ead4c8fb110a9dff7f88122d6ce0271d9eb74c63afca485f43df4cc24213f95`;
- source receipt SHA-256:
  `fabfeb5e151b190721054f49644ebbf285741b19db76e492f65e6d824025843e`;
- candidate rows SHA-256:
  `6c68b0e3e4cec3f056dcc5bc1805f8d679e01716aded85378378457dca957548`;
- result SHA-256:
  `b9e50631736a98ee52f5155f9a90dbd49e5f71d65732a5c932a73220881d9bb9`;
- strict replay receipt SHA-256:
  `f98210e6b2a964af4ef5f82f2d0c77f422fe5f88eeb3ef408ff12c1a24009ac1`.

R50 therefore strengthens evidence for bounded teacher-derived neural
capability acquisition, sparse LayerCake execution, and same-lineage
generalization, but it does not pass its complete held-split contract and does
not prove unrestricted English, minimum information, independent hardware,
human quality, or superiority to LoRA/distillation.  The ABI moonshot remains
`OPEN`.

