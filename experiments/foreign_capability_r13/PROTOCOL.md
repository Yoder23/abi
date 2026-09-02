# R13-B held-out finite capability-extraction protocol

## Purpose

R13-B tests Track B from `docs/ABI_TRANSFER_TRACKS.md`. It does not reopen the
failed R12-A behavioral-transplant gate and cannot certify Track A.

The registered claim is deliberately narrow:

> A conventional open-weight source, trained through ordinary next-token LoRA,
> can expose the complete atomic specification of a previously committed
> finite capability. A fixed zero-parameter compiler can convert that
> specification into an unchanged R11 package that exactly realizes the
> capability through frozen heterogeneous hosts without the source teacher or
> recipient training.

## Fixed scientific boundary

- Source model: Qwen2.5-0.5B revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- Source adaptation: R12 rank-16 LoRA, AdamW, learning rate `0.0002`,
  depth-balanced batches, and no LM-head or embedding update.
- Source training data: all 2,904 unique depth-1 through depth-5 programs for
  each capability.
- Source stopping rule: three consecutive 24/24 atomic evaluations, checked
  every 100 optimizer steps, with a 1,500-step maximum. No depth-6/7 held-out
  row is evaluated until the selected source state is frozen.
- Extractor: the exact R12 fixed 24-probe zero-parameter compiler.
- Package and execution: the exact sealed R11 package schema, recurrent
  executor, host codecs, model revisions, conditions, and native output heads.
- Held-out matrix: four committed capabilities, 512 unique depth-6/7 prompts
  per capability, and zero source-training/evaluation prompt overlap.

The 24 probes exhaust the `3 × 8` transition table. This is intentionally a
finite-table control. It does not test sparse recovery, English, factual
knowledge, or an unbounded capability.

## Track B gates

Every capability must satisfy:

- source BEFORE atomic accuracy at most `0.25`;
- frozen source AFTER atomic accuracy exactly `1.0`;
- atomic gain at least `0.70`;
- source base-weight hash unchanged;
- extractor training rows, evaluation rows, answers, and rule access all zero;
- package/oracle accuracy exactly `1.0` on all 512 held-out rows;
- Pythia, Qwen2, and T5 AFTER and RESTORED accuracy exactly `1.0`;
- every registered negative-control accuracy at most `0.30`;
- REMOVED, BACKEND_REMOVED, and CODEC_REMOVED predictions exactly equal BASE;
- zero recipient optimizer steps and unchanged recipient/codec hashes; and
- source capability adapters absent from recipient worker inputs and memory.

Native source depth-6/7 accuracy and exact package/source equality are always
reported but are not Track B gates. They remain Track A measurements.

## Custody and verification

The held-out secret is generated after this protocol and implementation are
complete. Only its SHA-256 commitment is committed before execution. The
reveal is copied into immutable evidence only after the preregistration commit.

The verifier must recompute commitments, hashes, row identities, prompt
disjointness, package behavior, source/package disagreement, recipient gates,
and R11 custody. Stored status booleans are untrusted. Missing artifacts or
unrecomputable claims fail closed. A live replay must reconstruct source
outputs from the saved adapters and recipient outputs from the exact packages.

## Claim ceiling

A pass is `BOUNDED_ENUMERABLE_CAPABILITY_EXTRACTION`. It is not lossless
teacher-function transplantation, extraction of pre-existing source knowledge,
English/domain extraction, universal host compatibility, LayerCake product
acceptance, information minimality, or superiority to LoRA/distillation.
