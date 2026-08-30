# Repository layout

The default branch is curated for users, reviewers, and contributors.

| Path | Purpose |
| --- | --- |
| `abi/` | Installable research package and public alpha API |
| `abi_v2/` | Canonical ABI spec, host certification, reconstruction, and strict verification |
| `tests/` | Supported and adversarial automated tests |
| `examples/` | Small examples that do not require a model download |
| `docs/` | Architecture, status, claims, and review procedures |
| `review_packet/` | Human/independent reviewer entry point |
| `evidence/current/` | Compact controlling campaign state and certificates |
| `results/abi_final_validation_v2/` | R7 technical evidence and preserved failed lineages |
| `results/abi_v2/` | Earlier host/package matrix evidence |
| `experiments/` | Reusable research drivers outside the stable API |
| `scripts/` | Repository maintenance and validation utilities |

Large model weights, generated caches, temporary certification filesystems,
and public reconstruction workspaces do not belong in Git. Required immutable
capability packages and the definitive archive are published as hash-addressed
R7 GitHub Release assets.

The public `research-history-v1089` branch preserves the pre-curation research
tree. Use it to audit historical work; use the default branch and R7 release
for current claims.
