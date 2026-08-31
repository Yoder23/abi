# R11 native-neural ABI construction result

Date: 2026-08-31

Status: `BOUNDED_SYNTHETIC_CONSTRUCTION_PASS`

R11 is the first ABI experiment in this repository to satisfy the complete
copy/paste mechanism for a learned capability state: learn, extract one
content-addressed package, execute the identical bytes through frozen
heterogeneous hosts, remove the behavior, and restore it exactly without
recipient training.

This is a construction proof for an ABI-native synthetic teacher. It is not
evidence that ABI can yet extract English or a natural domain from an arbitrary
open-weight transformer.

## What ran

- Four capabilities were derived only after the held-out seed commitment,
  implementation, host identities, codec scales, data matrix, and gates were
  frozen in Git.
- Each teacher capability state had 192 learned float32 values and was learned
  from 288 depth-1-to-3 rows.
- Each immutable AFTER package was 2,053 bytes and contained a normalized
  `3 x 8 x 8` neural transition state plus provenance. It contained no prompt,
  answer, row ID, rule seed, capability ID, solver, model/tokenizer identity,
  host width, or recipient matrix.
- A zero-parameter recurrent neural ISA executed the package.
- Capability-blind codecs were frozen before reveal for DistilGPT2,
  Pythia-160M, Qwen2.5-0.5B, and T5-base. They saw zero capability examples and
  took zero optimizer steps.
- The source and every recipient used its own frozen full-vocabulary output
  head. Recipient base parameters and codecs were unchanged before/after.
- The source completed before recipient execution; recipient receipts record
  the source as absent and zero recipient optimizer steps.

## Recomputed result

| Gate | Result |
| --- | ---: |
| Unseen capabilities | 4 |
| Evaluation prompts per capability | 512, depths 4-7 |
| Conditions | 11 |
| Source raw rows | 22,528 |
| Recipient raw rows | 67,584 |
| Total native observations | 90,112 |
| Teacher AFTER | 1.0 for every capability |
| Teacher BEFORE | 0.0820-0.1113 |
| Pythia AFTER / RESTORED | 1.0 / 1.0 for every capability |
| Qwen2 AFTER / RESTORED | 1.0 / 1.0 for every capability |
| T5 AFTER / RESTORED | 1.0 / 1.0 for every capability |
| Largest negative-control accuracy | 0.19140625 |
| Teacher/recipient canonical UTF-8 mismatches | 0 / 12,288 comparisons |
| REMOVED / BACKEND_REMOVED / CODEC_REMOVED vs BASE | exact on every row |
| Frozen model/codec hash mismatches | 0 |
| Recipient optimizer steps | 0 |
| Fail-closed hostile controls | 7/7 rejected |

The independent verifier recomputed the neural state from package bytes and
prompts, recomputed native token accounting and all accuracies from raw rows,
and did not consume the run's stored verdict as evidence.

## Frozen-package live replay

The decisive replay did not retrain or regenerate the capability state. It
copied the exact revision-001 package files by SHA-256, loaded them into a new
live execution, and reran all four hosts and all interventions.

| Raw evidence | Original SHA-256 | Frozen-package replay SHA-256 |
| --- | --- | --- |
| Teacher JSONL | `c67a16afe1151b6b1595fbe1f761c2cbc1e0dfc6d055a297316afdc2dd5bc4b2` | identical |
| Recipient JSONL | `bb6fd12a49f4f0394890c1fdb0d37c5a8bbfff15dc5f1eee297d39f86366bd9f` | identical |

The replay's independent verification evidence SHA-256 is
`2286b276603a4f5af9bdea93b28f3583cbb9c9f0a08207d4afd29acaa297ff72`.

## Preserved reproducibility limitation

A second end-to-end acquisition run also passed every functional gate, but
GPU training produced different float32 transition bytes and therefore
different package hashes. This does not affect copying an already frozen
package: the exact-package replay above is bitwise identical. It does mean ABI
must not claim bit-reproducible package acquisition from the current GPU
training path. Machine comparison found zero behavior mismatches across all
90,112 rows, zero identical AFTER package pairs across the four acquisitions,
and 32,528 neural-state row mismatches. Its evidence SHA-256 is
`a4a5a97f286c076c31e29bf36246b87ef779534c419b9b56a0700aa5f9c1a08b`.
Deterministic acquisition remains open.

## Exact claim boundary

R11 establishes:

> For four held-out synthetic capabilities learned in the registered
> ABI-native teacher substrate, one compact learned neural state can be
> content-addressed and copied unchanged into three frozen heterogeneous host
> environments. Their native output heads reproduce the teacher's canonical
> UTF-8 output on every registered unseen prompt, without recipient training;
> removal removes the behavior and restoring the same bytes restores it.

R11 does not establish:

- extraction from a conventional pretrained transformer's weights;
- English fluency or natural-domain knowledge extraction;
- arbitrary-model compatibility;
- exact full-vocabulary distribution equivalence;
- semantic or parameter minimality;
- internalization into recipient transformer blocks;
- LayerCake product acceptance; or
- superiority to LoRA, distillation, or fine-tuning.

The next scientific boundary is a foreign-teacher acquisition front end that
maps a capability already encoded in an independently trained open-weight
model into this package ABI without capability-specific recipient training.
That front end must pass held-out teacher-relative behavior and reject
memorized-table or answer-solver substitutes before English/domain claims can
open.

## Evidence map

- Protocol: `experiments/native_isa_r11/PROTOCOL.md`
- Pre-codec preregistration:
  `experiments/native_isa_r11/configs/preregistered_codec_freeze_v1.json`
- Final preregistration:
  `experiments/native_isa_r11/configs/preregistered_transfer_v2.json`
- Held-out reveal: `experiments/native_isa_r11/reveals/heldout_v1.json`
- Codec freeze: `results/native_isa_r11/codec_freeze_v1/`
- Primary run and verification: `results/native_isa_r11/revision_001/`
- Independent acquisition replication: `results/native_isa_r11/revision_002/`
- Exact-package live replay:
  `results/native_isa_r11/frozen_package_replay_001/`
- Evidence inventory: `results/native_isa_r11/manifest.json` (31 files,
  211,483,521 bytes; evidence SHA-256
  `20584b6863a2d11766cab8adb0e405e7291bfe28d4d4e1a9e071a6f1fd3caa3d`)
