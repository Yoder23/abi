# R102 research review: coverage and exposure

This is a **negative research prerelease**, not a qualified English-core or
production release. `FULL_MOONSHOT_NOT_PROVEN` remains controlling.

Four matched LoRA/adapter-distillation trajectories compared repeated original
data with invertible entity renaming, at 384 and 1,536 optimizer updates. Neither
registered primary factor met its support criterion; no checkpoint passed the
locked quality screen. All eight checkpoints, raw rows, costs, confidence
intervals and the initial corpus-construction failure are preserved.

| Control | Passes / 96 at 384 updates | At 1,536 updates |
| --- | ---: | ---: |
| LoRA, repeated | 51 | 46 |
| LoRA, renamed | 14 | 34 |
| Adapter distillation, repeated | 32 | 39 |
| Adapter distillation, renamed | 10 | 24 |

The teacher scored 90/96 raw and 95/96 after the registered normalization. The
quality gate requires at least 44/48 in each identifier stratum, plus per-task,
collapse and teacher-relative constraints. These literal/format screens do not
certify broad English fluency. Renaming changed more than one distributional
property; this is not evidence that data diversity generally harms acquisition.

The acquisition/recipient combination failed. No native LayerCake runtime defect
was localized and no LayerCake source or artifact was promoted. The next proposal
must test recipient/interface suitability against a qualified English control,
not continue the same renaming/step-count sweep.

## Immutable identity and assets

- Tag: `abi-research-r102-2026-11-10`.
- Scientific source commit: `d5a8957b152ce883779436103bd61a7ba4cd2f35`.
- Annotated tag object: `1ef35b43aa14c23e1f3836e07009dcf0bd5ee1b5`.
- Manifest SHA-256: `45a03f18d0f567e8a5c06fc08f40c31cbc0adb6ce5baf0c3d0b7e9cf4e1a39be`.
- Decision SHA-256: `177117d4cb59e79139366cb09eda856c82dbf257c50fd973d5758effb7de9a4e`.

`r102_manifest.json` inventories all 782 required repository files, including
14 direct/transitive LFS paths (12 unique checkpoint payloads), and separately
binds external LayerCake sources. Resolve every listed LFS path; GitHub source
archives alone may contain pointers rather than model payloads.

The additive `review_r102_public.py` helper can run from outside the exact tagged
checkout. It verifies public sibling identities and every asset, then recomputes
the scientific decision and runs focused tests. It does not train, generate new
model outputs, perform human review or establish independent hardware.

The tag's local software review passed 287 tests with four declared skips; its
separately run supported CI list passed 108. Local Git/LFS custody passed. These
software results do not promote the failed scientific gates.

See the current [full R102 packet](https://github.com/Yoder23/abi/blob/master/docs/R102_COVERAGE_EXPOSURE.md)
for reconstruction commands, receipts and later publication checks. Original
freeze and inventory fields describing local-only status preserve the state at
their creation; the user subsequently authorized public release. No historical
evidence, source decision or tag is rewritten.

## Public reconstruction completed

Fresh public-only sibling clones matched all 782 assets and 833 frozen bindings,
strictly recomputed the identical negative decision, and passed 42 focused tests.
All 14 LFS paths were downloaded from GitHub; the five tokenizer files were
downloaded separately from their pinned public source into a fresh cache. Both
checkouts remained clean. This reused the same machine and installed Python
dependencies, not development assets, independent hardware or fresh model runs.

Receipt SHA-256:
`7d19d8a936d2c311faab3f71c846ae86033a1a87bc53cecf76ce72bb7886c347`.
The receipt and operational record are attached as additive release assets.

The first tag/helper CI runs exposed a missing Transformers dependency. The
original failure remains visible; additive commit `48b7b19` installs pinned
research-test dependencies without changing/skipping tests. Its GitHub matrix
passes on Python 3.10, 3.11 and 3.12. Updated local review passes 293 tests with
four declared skips. This does not change the negative scientific verdict.
