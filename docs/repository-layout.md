# Repository layout

The default branch is intentionally curated for readers and contributors.

| Path | Purpose |
| --- | --- |
| `abi/` | Installable Python package; only exports in `abi/__init__.py` are the stable alpha API |
| `tests/` | Supported public tests plus historical campaign tests; plain `pytest` selects the supported suite |
| `examples/` | Small supported examples that require no model download |
| `docs/` | Architecture, status, reproduction, and publication guidance |
| `evidence/current/` | Controlling compact state and bounded certificates |
| `experiments/` | Reusable experiment drivers outside the stable API |
| `artifacts/` | Small schemas and reference artifacts |
| `abi_v2/` | Canonical host specification, capability-blind certification, 3x4 matrix harness, and inference-free verifier |
| `results/abi_v2/` | Frozen V1 lineage, V2 adapters, raw matrix evidence, summaries, hostile audit, and technical certificate |
| `results/abi_v2/external_reproduction/` | Clean-room commands, raw evidence schema, and tracked archive receipt; generated zip stays out of Git |

## Research history

The public `research-history-v1089` branch preserves the exact pre-curation
tree: 5,191 tracked files, all versioned campaign ledgers, negative results,
catalogs, generated measurements, and Git LFS checkpoint references.

Use that branch when auditing a historical claim:

```bash
git fetch origin research-history-v1089
git switch --detach origin/research-history-v1089
git lfs pull
```

The history branch is large. The default branch does not require those LFS
payloads for installation or the supported self-check.

## Adding evidence

Do not add thousands of generated files back to the repository root. Put raw
outputs on a dated research-history branch or external content-addressed
artifact store. Add only a compact manifest, certificate, retrieval procedure,
and honest claim boundary to `evidence/current/`.
