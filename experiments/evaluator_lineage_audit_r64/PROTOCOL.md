# R64 evaluator-lineage audit

## Purpose

R60 was generated prospectively, but its catalog builder imported the original
`natural_english_catalog.BUILDERS`.  Later source-eligibility repairs in the
v2/v3 catalog path were therefore absent.  R64 is a read-only development audit
that quantifies this defect without changing R60 or promoting a candidate.

## Frozen inputs

R64 reads the immutable R60 catalog, live Phi source rows, frozen-R59 screen,
and R63 screen.  Their SHA-256 digests are bound in `audit_v1.py`.

## Classification

* `unchanged`: the v1 evaluator and generation budget were not repaired later.
* `diagnostic_rescore`: abstention and nonce reasoning can be rescored from the
  recorded text with the later evaluator, but the result is post-hoc and is not
  promotion evidence.
* `censored`: coherence used the obsolete 40-token generation ceiling.
* `incompatible_contract`: format v2 changed both the requested format and its
  evaluator, so recorded JSON-task outputs cannot be rescored as the new task.

## Decision rule

This audit may identify which measured failures were caused by benchmark
lineage.  It cannot certify transfer, select a release, or turn R60/R63 into a
prospective pass.  Any promotion requires a new endpoint frozen before a new,
corrected, naturally unique prospective catalog is materialized.

