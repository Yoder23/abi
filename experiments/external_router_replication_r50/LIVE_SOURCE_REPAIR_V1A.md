# R50-v1a additive repair: live held-out source capture

R50 v1 failed closed with zero candidate rows because the two preregistered
source survey archives do not contain the catalog's `final_test` split.  This
repair changes only the source-evidence acquisition step.

## Frozen inputs

- Catalog SHA-256:
  `8992c7de94d3733d66f2083f96ec8d3cba31be3943afe1849f42220f33ff8d08`.
- Source: `microsoft/Phi-3-mini-4k-instruct` at revision
  `f39ac1d28e925b323eae81227eaba4464caced4e`.
- Source manifest SHA-256:
  `3bac528a1825e77dcb35963f5c78946fb14400e1cd832e0db40c2d964360c310`.
- R49 neural checkpoint, metadata, router, and result hashes remain exactly as
  registered in `PROTOCOL.md`.
- All 1,400 pre-existing `final_test` prompts, evaluators, capability labels,
  token limits, routes, gates, and thresholds remain unchanged.

## Authorized source capture

The pinned local source snapshot is re-hashed through the Phase-2 source
verifier and loaded in BF16 on CUDA with eager attention.  Its own chat
template is applied to each raw catalog prompt.  Each probe receives exactly
one deterministic greedy sequence (`do_sample=False`) at its frozen
`max_new_tokens`.  Batching may parallelize sequences but may not change any
per-sequence generation setting.  Generated token IDs are authoritative.

For every row, preserve the raw prompt, rendered-prompt hash, input token
count, output text and token IDs, output hash, evaluator and recomputed score,
and collapse result.  Seal the complete JSONL, source inventory, catalog hash,
software/hardware identity, wall time, teacher tokens, bytes, and peak memory.

No generated source text may be used to train, tune, select, repair, or route
the candidate.  After the source rows are sealed, R50-v1a evaluates the
unchanged frozen R49 system on the same final-test rows.  The original gates
apply without relaxation.  A strict verifier must recompute every raw metric
and a fresh live candidate replay must match all 1,400 stored outputs exactly.

This remains a same-machine, same-catalog-lineage held-split replication.  It
cannot establish unrestricted English, global minimality, human quality,
independent hardware reproduction, or superiority to LoRA/distillation.
