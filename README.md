# ABI

ABI is a research codebase for acquiring, labeling, packaging, validating, and
eventually minimizing capabilities extracted from open-weight teacher models.
It is designed to produce immutable capability artifacts that a separate
[LayerCake](https://github.com/Yoder23/layercake) runtime can install, compose,
route, and execute.

The repositories have deliberately separate responsibilities:

- **ABI:** teacher qualification, probing, labeling, segregation, provenance,
  information accounting, artifact construction, and verification.
- **LayerCake:** hosting, installation, transfer between compatible hosts,
  routing, orchestration, and inference performance.

## Current release status

The latest technical validation is R7:

<https://github.com/Yoder23/abi/releases/tag/abi-final-validation-v2-repaired-r7-2026-08-30>

R7 passes a bounded capability-runtime/conformance proof across LayerCake v25,
Qwen2.5-0.5B, and Pythia-160M using four immutable packages: English, Python,
civics, and chemistry.

This is not a claim that ABI already extracts a fluent minimal English core
from any teacher. Human ratings, different-hardware reproduction, registered
minimum-information certification, teacher-extraction quality, and comparison
with LoRA/distillation remain open.

The additive R8 native-neural-transfer campaign is a Level 0 negative result.
Its strongest package-only recipient interface produced AFTER−BASE +0.001953
across 1,024 paired Pythia rows (95% bootstrap CI -0.019531 to +0.024414).
Canonical extraction was exact, but recipient behavior did not transfer. The
held-out secret was never revealed, and R8 does not change the R7 release.
The final public verifier rejected 7/7 hostile mutations and ignored one forged
scientific boolean without changing the `NO` verdict (8/8 expected outcomes).

R9 then tested the backend/compiler hypothesis directly with an intentionally
capability-specific Pythia neural backend. V1 failed to fit its own training
rows. The preregistered v2 repair exposed both embedding and final recipient
states and trained a 1,039,880-parameter backend for 10,000 steps. It still
recomputed at only 12.85% training accuracy and 12.5% unseen-depth AFTER
accuracy, versus 7.32% BASE; ZERO also reached 12.5%, failing causality. All
8,192 evaluation and 288 training observations replayed live, and 5/5 hostile
controls were rejected. R9 therefore closes this recipient-state GRU branch
without opening the universal-backend experiment or changing R7.

R10 then connected the synthetic R8 extraction to a clean package slot and
three frozen recipients. Its runtime-owned copy/paste component passed: four
2,040-byte AFTER packages produced 1.0 accuracy after paste and restore across
61,440 Pythia, Qwen2, and T5 rows; removal exactly matched BASE, all negative
controls were at most 0.15625, recipient parameters were unchanged, and a fresh
live run reproduced every row byte-for-byte. The overall R10 contract is still
negative because the source model's native decoder reached only 0.59375 to
0.65234375 against the registered 0.99 source gate. See
[`docs/R10_COPY_PASTE_RESULT.md`](docs/R10_COPY_PASTE_RESULT.md). This proves a
bounded canonical-runtime component, not LayerCake integration, lossless source
behavior, English/domain extraction, or native neural transplantation.

## R7 at a glance

| Evidence | Result |
|---|---:|
| Public archive | 844,018,841 bytes |
| Archive SHA-256 | `fc50f423986149b5d4670ec9e28698540f64be96034efa26e5704c4469921e88` |
| Archive members | 1,193, all exact |
| Physical certification environments | 3/3 |
| Reachable inventory rows | 301,543 |
| Locked matrix | 5,043/5,043 rows |
| Live causality | 3,072 rows, 24 distinct processes |
| Live isolation | 2,100 rows, zero target successes |
| Fail-closed controls | 19/19 pre-public and 19/19 blind replay |
| Public reconstruction tests | 17/17 |
| Blind tar-prefix controls | 12/12 |
| Blind verdict | PASS, bounded scope |

The blind reconstruction succeeded with a 368-character extracted path while
Windows long paths were disabled.

## Install

ABI targets Python 3.10. For the lightweight package and CLI:

```bash
python -m pip install -e .
abi status
abi self-check
```

Teacher extraction and research workflows require the dependencies in
`requirements.txt`. Human-rating helpers use the optional human extra:

```bash
python -m pip install -e ".[human]"
```

## Public R7 verification

Download `public_release_assets_r7.json` from the R7 GitHub Release. It binds
the definitive archive and four capability packages by byte length, SHA-256,
content address, release tag, and commit.

From a clean checkout of the public tag:

```bash
python -m abi_v2.public_reconstruction \
  --manifest /path/to/public_release_assets_r7.json \
  --tag-clone /path/to/clean/tag-clone \
  --workspace /path/to/new/workspace \
  --output /path/to/public-reconstruction-receipt.json
```

The external different-hardware workflow is under `external_reproduction/`.
Its command sequence is:

```text
abi-reproduce verify
abi-reproduce certify-hosts
abi-reproduce capability-matrix
abi-reproduce causality
abi-reproduce isolation
abi-reproduce performance
abi-reproduce hostile-audit
abi-reproduce report
```

Executing this sequence on the development laptop is a rehearsal, not an
independent reproduction.

## Human review

The frozen packet contains 7,000 judgments for each of three independent
raters. The gate remains `0/21,000` until real raters complete and attest their
forms:

```bash
abi human-rate --rater R1
abi human-rate --rater R2
abi human-rate --rater R3
```

See `docs/PHASE2_HUMAN_RATING_HANDOFF_V1.md` for the exact handoff.

## Published artifacts

| Capability | Bytes | SHA-256 |
|---|---:|---|
| English | 253,216,208 | `acb787b3ffa0153c57d88cd37ba81c3f00b370d4ca4937e659cd4c775851f25d` |
| Python | 448,404 | `f1defaef2771ced336a332572a2d2f0e1e542399c877d182c48a6cd2e199231d` |
| Civics | 495,919 | `634ce66958859ec36dc1fbdf5ef34d6d2a9949d10cf2348a68c245d8c325d604` |
| Chemistry | 510,981 | `f9c9b2668fda5ef6b92844c1b7097fbdf8ff0daaae51f5b86f72d4a49000abeb` |

## Claim boundary

R7 supports only the bounded capability-runtime/conformance result described
above. It does not establish:

- arbitrary or universal model compatibility;
- tensor transplantation between unrelated architectures;
- complete diagnosis or extraction of teacher knowledge;
- fluent teacher-quality English generation;
- superiority over LoRA, distillation, or fine-tuning;
- human-rated quality;
- independent hardware reproducibility;
- global information minimality; or
- completion of the full ABI moonshot.

See `docs/ABI_TECHNICAL_CLAIMS.md`, `docs/ABI_FINAL_RESULTS.md`, and the
`review_packet/` directory before citing results.

## Repository map

- `abi/` — acquisition, labeling, artifact, and public CLI code.
- `abi_v2/` — canonical runtime/conformance, certification, strict verification,
  public reconstruction, and external reproduction tooling.
- `tests/` — supported automated checks.
- `results/abi_final_validation_v2/` — immutable R3-R7 validation lineage.
- `results/abi_moonshot/packages/` — published specialist packages.
- `external_reproduction/` — independent-operator workflow and environment
  lock.
- `review_packet/` — ordered technical and external-review handoff.
- `docs/` — architecture, claims, results, and review instructions.

Historical experimental evidence remains in Git for auditability. Generated
caches, model weights, temporary public reconstructions, and bulk reproducible
intermediates are intentionally excluded from the production tree.

## License and research status

This repository is a research release. Consult `LICENSE` and `NOTICE` before
redistribution. Scientific claims are governed by immutable evidence and the
claim ceiling above, not by roadmap language.
