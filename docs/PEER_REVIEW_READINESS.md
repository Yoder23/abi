# ABI peer-review readiness

Status date: 2026-09-18

## Verdict

`READY_FOR_BOUNDED_PEER_REVIEW_AFTER_TAG_PUBLICATION`

`FULL_MOONSHOT_NOT_PROVEN`

Independent reviewers can evaluate R7, V1089, R97, and R21-R23 as separate
bounded evidence lineages. They must not combine those lineages into a single
final-artifact proof.

## Recomputed candidate checks

- `python -m abi status --json`: 18/18 internal readiness gates; human ratings
  remain 0/21,000.
- `python -m abi_v2.verify_release --check-existing`:
  `TECHNICALLY_PROVEN_EXTERNAL_VALIDATION_PENDING`; 21 technical gates pass
  and all three external gates remain false.
- Default configured suite: 174 passed, 4 skipped.
- Human packet, workflow, binding, and hostile checks: 16 passed.
- Git LFS integrity: `Git LFS fsck OK`.
- Package build: sdist and wheel built successfully.
- Documentation: 15 canonical documents and all tracked Markdown links pass.

The four skips preserve declared exact-lineage boundaries. They do not turn
missing source-lineage context into a pass.

## Human-scoring binding repair

The still-unrun V1 human protocol bound the raw workspace bytes of
`pyproject.toml`, even though repository test collection is not part of human
scoring semantics. Later additive collection changes made that binding stale.

The additive
`ABI_CAPABILITY_COMPILER_PHASE2_HUMAN_SCORING_BINDING_REPAIR_V2.json` removes
only that non-scientific binding and directly binds the scoring implementation,
evidence verifier, sealed packet manifest, and repaired test. The original
protocol and failure remain preserved. Every custody, validation, scoring,
interpretation, threshold, seed, and sealed-packet rule is unchanged. All
21,000 production rows remain blind and unrated.

This repair creates no ratings, quality result, Phase 2 pass, Phase 8 pass,
release certification, or full-moonshot claim.

## Gates that local development cannot complete

1. Three independent humans must complete and attest the frozen 21,000-rating
   packet.
2. An independent operator must execute the eligible immutable release on
   different CPU and CUDA hardware.
3. The registered minimum-information program must be executed.
4. A full claim requires one coherent final artifact rather than measurements
   composed from R7, V1089, and R97.
5. Broad English acquisition, automatic discovery/segregation, and fair
   LoRA/distillation comparisons remain open.

## Immutable review identity

The intended tag is `abi-peer-review-candidate-v1-2026-09-18`. Before review,
verify that the annotated tag is public, resolve its peeled commit, clone it
into a new directory, pull Git LFS, and follow the exact lineage-specific host
requirements in the independent-review handoff.

## Reviewer entry points

1. [`PROJECT_STATUS.md`](PROJECT_STATUS.md)
2. [`../CLAIMS.md`](../CLAIMS.md)
3. [`INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md`](INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md)
4. [`../reviews/independent_2026-09-14/`](../reviews/independent_2026-09-14/)
5. [`../review_packet/00_READ_ME_FIRST.md`](../review_packet/00_READ_ME_FIRST.md)
6. [`PHASE2_HUMAN_RATING_HANDOFF_V1.md`](PHASE2_HUMAN_RATING_HANDOFF_V1.md)
7. [`PHASE8_EXTERNAL_REPRODUCTION_V1.md`](PHASE8_EXTERNAL_REPRODUCTION_V1.md)
8. [`../ABI_CAPABILITY_COMPILER_PHASE2_HUMAN_SCORING_BINDING_REPAIR_V2.json`](../ABI_CAPABILITY_COMPILER_PHASE2_HUMAN_SCORING_BINDING_REPAIR_V2.json)
