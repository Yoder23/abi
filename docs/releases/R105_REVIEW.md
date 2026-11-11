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
Public reconstruction is recorded additively after publication; this source
snapshot itself does not attest that later operation.
