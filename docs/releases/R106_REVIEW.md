# R106 research review: grammatical-discrimination control failed

Tag: `abi-research-r106-2026-11-11`. Package API remains `0.4.0a1`.
This is a negative diagnostic, not a production or moonshot release.

The [result packet](../R106_GRAMMAR_INTERFACE.md) reports 178/200 external BLiMP
pairs versus the frozen 190/200 gate; irregular participles score 37/50 against
45/50. Exact fresh-process raw replay passed. No recipient comparison or training
ran. The diagnostic supplies all target tokens and cannot qualify autonomous
English generation. Original generation gates and all historical evidence remain.

An apparent upstream label reversal and other possible judgment ambiguities
are disclosed without changing any label or score. The post-run source-identity
check matches the four Qwen weight shards to public pinned metadata.

The manifest is `results/review_2026_11_11/r106_manifest.json` and the helper is
`scripts/review_r106_public.py`. Reproduction requires the declared LayerCake
sibling revision, four already-published LFS checkpoints and all ten pinned
Qwen files. The helper can fetch the exact public source files with
`--fetch-control`; it never executes model inference. A complete source cache
requires about 15.24 GB, not a new deployed ABI artifact.

Local validation: 52 focused checks, including 25 hostile mutations; supported
suite 364 plus 16 historical checks, with four skips. These certify evidence
handling, not English quality. Public reconstruction is additive after publication.
`FULL_MOONSHOT_NOT_PROVEN` remains authoritative.

Manifest: 581 repository assets, four inherited LFS paths, 182 sibling-host
bindings and ten pinned source files. SHA-256:
`3e0b15e1368cd71e9e51cdd4e28f3a9aac973bc618ea67fd7ff5398288d30233`.
