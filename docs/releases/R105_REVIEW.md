# R105 research review: rejected semantic-scope evaluator

Tag: `abi-research-r105-2026-11-11`. API version remains `0.4.0a1`.
This is a negative research result, not a production or moonshot release.

The [result packet](../R105_EVALUATOR_SCOPE.md) records 779/1,200 external gold
scope cases and 62/64 approval cases. Both gates fail. All 1,264 raw token/logit/
prediction rows replay byte-identically in a fresh process. No teacher acquisition,
recipient generation, training or native LayerCake change occurred.

Required evidence is inventoried in
`results/review_2026_11_11/r105_manifest.json`, with a public review helper at
`scripts/review_r105_public.py`. The eight evaluator payload files are available
at the pinned public Hugging Face revision and checked by exact SHA-256. They
are evaluator dependencies, not imported LayerCake capability weights.

The full-study verifier is `experiments.evaluator_scope_r105.complete`, not the
initial-stage-only `run verify` command. The first missing-replay hostile-audit
failure is preserved alongside the repaired checks. The scientific code, raw
outputs, gates and rejected decision remain unchanged.

Local validation: 338 default tests plus 16 exact historical tests, four skips;
41 separate focused checks, including 23 hostile cases. No model quality claim
is implied by these test counts. The public helper does no model inference.

Human evaluation, qualified English control, recipient comparison and broad
English acquisition remain open. `FULL_MOONSHOT_NOT_PROVEN` remains authoritative.

Manifest: 40 repository assets and eight upstream model files.
SHA-256: `1f651c13866164009630f523735a959e24284c230cc694b59e3a5de06a25e2f5`.
Public reconstruction is recorded additively after publication, without moving
the source tag at `711f5a480bfab843ef7b2b1323bcc3a3aaef19f6`.

The release's `public_r105_reconstruction.json` receipt matches all 40 assets
and eight model files, recomputes the negative decision, and passes 41 focused
checks. Receipt SHA-256:
`1c994c811b835f76ac54fbe6a064cbc55bec51dd6ac8753bb5086a05d73f2e80`.
The public-only checkout and installed dependencies were reused on the same
machine; evaluator files were downloaded from the pinned public upstream.
No development assets were copied. This is not a brand-new environment, live
inference replication or independent-hardware review. Tagged CI passes on
Python 3.10-3.12 plus lint. See the result packet for exact replay instructions.
