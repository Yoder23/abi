# R113 review — completed replication, qualification failed

Research tag: `abi-research-r113-2026-11-20`.
Result: **CONTROL_QUALIFICATION_FAILED**. Whole moonshot: **FULL_MOONSHOT_NOT_PROVEN**.

Read the [result and limits](../R113_STORAGE_RECOVERED_REPLICATION.md),
[protocol](../../experiments/count_feedback_r113/PROTOCOL.md),
[freeze](../../results/count_feedback_r113/freeze.json), and
[decision](../../results/count_feedback_r113/live/decision.json).

## Evidence and boundary

One new write-once replication, not a resume or fresh holdout: 96 prompts, three
shared-draft stages, two fresh worker processes, 576 calls. Source, evaluator,
greedy BF16 decoding and gates are unchanged. No old rows were spliced into R113.
R112 remains preserved and incomplete. All 288 initial scientific rows also
match R112 descriptively; these are not independent new observations.

R113 has complete exact scientific replay and 72 unchanged-input controls per
worker. Raw file hashes differ because timing/storage observations differ; do
not call the raw JSONL files byte-identical. Draft/revision/feedback paired
scores are 36/48, 38/48 and 40/48 versus 44. Feedback panels are both 43/48;
tone pairs 3/6 and conversation 4/6 miss the 5/6 floor. Sentence adherence is
7/24, 15/24, 18/24; the primary CI [-0.08333, +0.33333] crosses zero. No arm
qualifies. Every output is collapse-free and untruncated.

Finite scope coverage is not general semantic validation. The [failure inventory](../../results/review_2026_11_20/r113_error_audit.json)
discloses unresolved approval forms and an apparent imperative-scope false
positive alongside genuine sentence-count failures. Do not silently rescore,
label every flag a teacher hallucination or delete failing rows.

Registered failure owner: EXPERIMENTAL_PREREQUISITE. ABI acquisition and native
LayerCake: NOT_TESTED. Matched LoRA/distillation checkpoints remain preserved;
no training or recipient run. No English artifact or superiority claim.

The same-agent [hostile audit](../../results/review_2026_11_20/r113_hostile.json)
rejects 14/14 actual-row mutations, not a fresh independent review. Offline checks
also reject missing packets, stale hashes/receipts, same-PID replay and resource
violations. All original failed direct historical-host test receipts are retained.
The final supported offline suite passes 829 tests with four existing skips;
the exact historical Phase 7 fixture separately passes all 16 tests. These are
software checks, not model quality or independent scientific certification.

Controller loop: 2,559.875 seconds; process wall: 2,514.64; CPU process: 2,345.156.
Generated/input tokens: 51,302 / 196,168. Peak RSS: 4,606,885,888 bytes; CUDA:
8,288,964,608. Each workflow includes draft plus its revision; study cost includes
all arms and replay. These are observed costs, not isolated efficiency claims.
No downloads, source changes, training, recipients or GPU-setting modifications.

## Recompute without inference

Use full tagged Git history, required Git LFS payloads, exact host files and the
pinned source cache in the public manifest. The evidence ZIP is not self-contained.

```powershell
python -m experiments.count_feedback_r113.run verify
python scripts/check_docs.py
python scripts/run_review_tests.py
```

Do not rerun write-once `freeze` or `live` over existing evidence. The review
runner supplies 46 historical Phase 7 files from their exact LayerCake revision.
Direct tests against today's sibling correctly reject stale historical bindings;
the live host and historical verifier were not changed to force a pass.

Preregistration: `209dadea63572feee990454066d6e1ab603b2c8c`; execution source:
`5efaf9f`; published freeze commit: `2fa83a9bedec9b968a62cd50ec0b8e51e4463520`.
Freeze SHA-256: `54b113c52f9dbb1b3a344cee9a10567d3d46c157a9f06a0150c3e70144529bdc`.
Decision SHA-256: `6386bfb033dc421840758c794cdf655561e765dc84139ce73b643d0906b7c5e9`.

The research release provides `r113_manifest.json`, `r113_evidence.zip` and
`r113_archive.json`. Additive public-download verification establishes byte
custody only, not clean-clone scientific, independent-hardware or human
reproduction. No automatic successor is authorized by this negative result.
