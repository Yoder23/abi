# R15 public development ledger

This ledger is append-only. A failed public result remains a failed result even
when a later architecture passes. Bulk journals and rejected tensor artifacts
remain in the local research archive; the compact receipts are identified here
by both file and internal evidence hashes.

| Run | Source atomic exact | Development exact | Frontend | Result | Receipt SHA-256 | Evidence SHA-256 |
| --- | ---: | ---: | --- | --- | --- | --- |
| v1 | not promoted | 0/8 | kernel delta | failed | `ec6483aed5df99af7fba7d338fd4c3253bac7a86264ca5b06c25fad7ecf6080d` | `37b3533f8b41e54bdf4633736690065e845db1dfccf3527a2d6b77dac2ae4c72` |
| v2 | not promoted | 0/8 | kernel full-LoRA sketch | failed; source underfit | `2a64e15d0b52da6a171a63d02f32f8bf3e4a3f41fb051d77f8ba1dfb3ca87fd8` | `72770c777bfdbeec4b109d82b12ceed4e7f3bd894f01e1a14fd759108a3d0144` |
| v3 | 160/160 | 11/32 | kernel output delta | failed | `b955a00b6a9a385417409e9b7b6628a77409cec26c4ad34c9b48157097a8d644` | `fc506a20cede44ff5a0db7aadfdfdd9247c3d9b98453b5c443762fc92e1bd55e` |
| v4 | 160/160 | 31/32 | six-head bilinear | failed | `8bb160257da8a70c915485387478d35668604eea12f2b351f79758daa128d4a1` | `0153d0fa7e9530f94c269a09b5ea7b529062efeef8c5694a26d39c14d8d9cec3` |
| v5 | 320/320 | 60/64 | six-head bilinear | failed; four invalid affine pairs | `c3b481ad8bf141cf209c5fef4f575d30395ff608d1962dce5cd4ae2d33ee0e76` | `27e028a8106e90366b04744528dab7e1069c887a5537d1f7e0c67dc06a65c6d5` |
| v6 | 320/320 | 63/64 | valid-pair codebook | failed; one reproducible head error | `c31f43046e514e4217460846ef8a8b8b7e5912fc7ed5fc6cba33c101f1f2bda9` | `76f4376621e414fbbaf785fa992750c5513ae16f7d6014b5bb9773bfca7eb3d0` |
| v7 | 320/320 | 64/64 | full-table affine decoder | public qualification pass | `566d05e5c59c39b4773d789002eae12c633003fcb0b6890ac2ce387b94bc31db` | `2c607c2d52995cce864bf04d7e040f0c4d0c54dd105f88a772a81e347323b24f` |

The v6 failure was not reclassified. Increasing epochs from 2,000 to 5,000 and
testing weight decay at 0.001, 0.0001, and zero reproduced the same error. A
pairwise code loss also reproduced it. The v7 change was therefore specific:
decode the 18 atomic constraints that v6 discarded and aggregate all 24 before
choosing each affine operation.

The tracked v7 release inputs are:

- receipt: `results/foreign_neural_state_r15/public_preflight_v7.json`;
- frontend: `results/foreign_neural_state_r15/public_frontend_v7.safetensors`,
  SHA-256 `aaa88da40a1cdd4bf721c56c471867da87d8a396448936630e0d0d108b59864d`;
- consolidated 320-event delta dataset:
  `results/foreign_neural_state_r15/public_dataset_v7.safetensors`, SHA-256
  `d114e030727d22bf7f2563ee672d7049fd4db8addc26184f10b619909ec0997a`.

The original v7 641-file journal remains local and immutable. Consolidation
checks every journal receipt and tensor hash before producing the release
dataset. The strict verifier then reconstructs all capability labels, verifies
all 320 effective-delta hashes, retrains the v7 frontend exactly, and recomputes
64/64 development accuracy from the consolidated tensor.

## Secret execution ledger

Heldout v1 produced all eight source adapters, deltas, isolated extractions,
and packages, then failed closed before a verdict when Pythia BASE emitted a
non-canonical token and the inherited R14 summary code attempted `int(None)`.
The failure is an accounting exception, not a failed or passing scientific
result. Its compact receipt is
`results/foreign_neural_state_r15/heldout_v1_execution_failure.json`; the full
partial directory remains immutable locally. Because the bug was in code frozen
before reveal, the exposed v1 secret is never reused. The null-safe repair must
be committed with a new hidden commitment before a new campaign begins.

Heldout v2 completed all stages. The primary result was positive--8/8 exact
isolated extractions, 8/8 packages at 1.0 on both 10,000 unseen and 1,000
counterfactual rows, three passing recipients, and 0/8 R14 black-box
recoveries--but certification failed. CPU recomputation of CUDA matrix products
differed by at most `4.76837158203125e-7` while the verifier incorrectly
required bit equality. More importantly, the registered SHUFFLED-delta control
permuted only eight output rows; that structure-preserving transformation
reached 1.0 counterfactual and 0.5017 unseen accuracy on one capability. V2 is
therefore negative evidence, not a pass. A successor must preregister a bounded
cross-device tolerance and a full 7,168-element permutation control against a
new secret.
