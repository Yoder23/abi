# R107 public reconstruction

Published 2026-11-13. This is an additive reconstruction record, not a change
to the immutable scientific tag or a new English-quality certificate.

- Tag: `abi-research-r107-2026-11-13`
- Scientific commit: `e87212bd731aebc9486d08d4aa15f1402b034d4c`
- [GitHub prerelease](https://github.com/Yoder23/abi/releases/tag/abi-research-r107-2026-11-13)
- [Public input manifest](../results/review_2026_11_13/r107_manifest.json)
- Manifest SHA-256: `f74f1e073141690db25887c9e45c634a9d46d64d260045195a6015b77cf07236`
- [Reconstruction receipt](../results/review_2026_11_13/public_r107_reconstruction.json)
- Receipt SHA-256: `06819c352568c871652f851770e4e83d8c5be3ec046caeda2c127bd9fbf202fe`

The separate clean public-only checkout matched all **585 repository assets**,
**182 pinned host bindings**, **four published LFS checkpoints**, and **seven
upstream source files**. It recomputed the primary decision, all parameter
custody, and the descriptive tensor/route comparison, passed **55 focused tests**,
and passed documentation validation. Neither checkout changed during review.
No development-only training snapshots were copied into it.

Public checkouts, installed dependencies and the pinned source cache were reused
on this machine. This was evidence recomputation, not a new live inference run,
fresh environment, blind review or independent-hardware replication.

[Tagged CI](https://github.com/Yoder23/abi/actions/runs/43482271473) passed Python
3.10, 3.11 and 3.12 verification; tagged lint also passed. The separate LayerCake
documentation-only review is `layercake-review-2026-11-13`, commit
`2c3c03fd749616d253798bd33bdb01dc8cc02b01`; its tests and campaign-integrity CI
passed. R107 still binds the prior unchanged host-runtime commit, as its manifest
requires; the documentation commit is not substituted for that identity.

Read [the scientific result and limitations](R107_RETENTION_INTERVENTION.md)
before drawing conclusions. The controlling whole-moonshot verdict remains
`FULL_MOONSHOT_NOT_PROVEN`.
