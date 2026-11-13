# R107 research review and reproduction

Research tag: `abi-research-r107-2026-11-13`. Package API remains `0.4.0a1`.
This is a natural-text retention/intervention diagnostic, not a product release.
See [scope and evidence](../R107_RETENTION_INTERVENTION.md). The immutable tag is
created only after the full raw replay, custody checks and local tests complete.

Local validation completed: 55 focused checks (26 hostile complete-packet
mutations), 391 supported tests and 16 exact historical Phase 7 checks; four skips.
The result is `PARENT_RESIDUAL_INTERFERENCE_SUPPORTED` within the declared panel,
not an English-quality pass. Full raw replay and checkpoint-derived custody pass.

## Public inputs

Clone ABI and LayerCake as sibling directories named `abi_release` and
`layercake_release`. Check out the R107 tag in ABI and LayerCake commit
`865a2b9e59e6221caa1185f9b64479d2077953f3`. The newer LayerCake documentation commit
is not a replacement for that pinned runtime reference. Fetch the four existing
Git-LFS checkpoints named in `results/review_2026_11_13/r107_manifest.json`.
No development-only training snapshots are required by the final freeze.

The external source is `distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`; all seven required files/hashes are in
the manifest. The complete cached source is approximately 356 MB. Obtain these
public files from Hugging Face at that exact revision, not an unpinned alias.
The pinned WikiText validation parquet and attribution are included in Git.
The initial overbroad inventory is historical only and is not a live dependency.

Use the dependency versions recorded in `freeze.json`/the public manifest.
Evidence-only reconstruction needs no GPU or new model inference:

```powershell
python -m experiments.retention_intervention_r107.complete
python -m pytest -q tests/test_retention_intervention_r107.py tests/test_r107_completed_verifier.py
python scripts/check_docs.py
```

For a public-checkout receipt, run `scripts/review_r107_public.py review` with
`--root`, `--commit` (the tag's resolved commit), `--manifest-sha256` (verified
against the release asset) and `--output` (a new path outside both checkouts).
The helper requires clean public repositories, verifies all published assets,
recomputes raw results and parameter custody, and runs the focused tests.
It fails on missing inputs; unavailable evidence is not a pass.

## Live replay and limits

The published packet contains two independently launched processes per cell,
not independent machines. To perform another live execution, use a disposable
checkout/output location: immutable output directories deliberately refuse
overwrite. Preserve original evidence. Replaying execution requires the recorded
GPU/precision environment; cross-hardware numerical identity is not assumed.

Public reconstruction receipts are additive, separate from the immutable tag.
Reusing public checkouts, cached weights and installed dependencies is explicitly
not a fresh environment or independent-hardware replication. Neither raw replay
nor passing hostile verifier tests certifies fluent English or the full moonshot.
